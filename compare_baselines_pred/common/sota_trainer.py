# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# 动态将项目根目录添加到包路径，以便导入 config 和 untils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(PROJECT_ROOT)

from config import Config
from untils.data import SlidingWindowDataset, Normalizer, make_obs_rel, generate_pure_structural_missing_mask
from untils.losses import masked_prediction_loss
from compare_baselines_pred.common.sota_wrapper import get_sota_model

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_openstl_baseline(model_name, missing_rate=0.3):
    """
    通用 Baseline 训练流程
    """
    set_seed(42)
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Info] 启动 {model_name} 基线对比实验 (Missing Rate: {missing_rate})")

    # 1. 数据准备
    print("[Info] 正在加载数据并拟合归一化器...")
    train_data_raw = np.load(cfg.train_npy)
    obs, rel = make_obs_rel(train_data_raw, cfg.phys_channels)
    norm = Normalizer(cfg.normalize, cfg.clip_percentile, cfg.phys_channels)
    norm = norm.fit(train_data_raw, obs)
    del train_data_raw, obs, rel

    train_dataset = SlidingWindowDataset(
        cfg.train_npy, normalizer=norm,
        input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
        phys_channels=cfg.phys_channels
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # 2. 获取并包装基线模型
    model = get_sota_model(model_name, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_p)
    
    # 统一将基线模型的权重保存在各自独立的文件夹下
    output_ckpt = os.path.join(cfg.base_dir, f"compare_baselines_pred/outputs/{model_name.lower()}/{model_name}_best.pth")
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)

    best_loss = float('inf')

    # 3. 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        epoch_pred_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.epochs}] {model_name}")
        
        for step, (hx, hm, hr, fx, fm, fr) in enumerate(pbar):
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)
            B, C, Tin, H, W = hx.shape

            # 统一应用扇形/随机掩码模拟极端工况
            art_mask, _ = generate_pure_structural_missing_mask(B, H, W, device, drop_prob=missing_rate, valid_mask=(hm*hr))
            art_mask = art_mask.unsqueeze(1).expand(-1, 1, Tin, -1, -1)

            optimizer.zero_grad()
            
            # 使用 Wrapper 前向传播
            out_pred = model(hx, hm, hr, art_mask, cfg)
            
            # 真实未来标签与掩码 (目标只有 V 通道)
            tgt = fx[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            valid_mask_f = (fm * fr)[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            
            # 仅在未来真实有效区域计算 MAE，保证对比公平性
            pred_mae = masked_prediction_loss(out_pred, tgt, valid_mask_f)
            
            assert torch.isfinite(pred_mae).all(), f"Fatal: {model_name} 产生了 NaN 梯度！请检查 OpenSTL 模型内部。"
            
            pred_mae.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_pred_loss += pred_mae.item()
            pbar.set_postfix({'MAE': f'{pred_mae.item():.4f}'})

        avg_loss = epoch_pred_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model': model.state_dict(), 'normalizer': norm}, output_ckpt)
            print(f"[*] {model_name} 保存新模型! Best Validation MAE: {best_loss:.4f}")