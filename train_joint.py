import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
# 请确保 untils/data.py 等基础模块在仓库中正常存在
from untils.data import SlidingWindowDataset, Normalizer, make_obs_rel, build_generator_inputs
from model import UNetGenerator, Discriminator
from predictor import WindFieldPredictor
from untils.losses import masked_prediction_loss


def set_seed(seed: int = 42) -> None:
    """
    固定随机种子以保证实验的可复现性 (Reproducibility)。
    科研代码中必须保证相同的配置能跑出相同的结果。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_pure_structural_missing_mask(
        batch_size: int, height: int, width: int, device: torch.device, missing_ratio: float = 0.15
) -> torch.Tensor:
    """
    生成用于 GAIN 训练的随机缺失掩码。

    Args:
        batch_size: 批次大小
        height: 空间高度 (方位角)
        width: 空间宽度 (距离库)
        device: 计算设备
        missing_ratio: 缺失比例

    Returns:
        Tensor of shape [B, 1, 1, H, W] 值为 0 (缺失) 或 1 (观测到)
    """
    # 随机生成掩码，大于 missing_ratio 的部分为 1 (保留观测)
    return (torch.rand((batch_size, 1, 1, height, width), device=device) > missing_ratio).float()


def train_joint() -> None:
    """
    执行主干网络的联合训练流程 (GAIN Completion + Prediction)。
    使用真实的梯度回传机制，使得预测误差可以指导补全网络的优化。
    """
    # 0. 环境与配置初始化
    set_seed(42)
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Start joint training on device: {device}")
    print(f"[Info] Output directory: {cfg.output_dir}")

    # ==========================================
    # 1. 数据加载与预处理 (Data Loading)
    # ==========================================
    print("[Info] Fitting Normalizer from training data...")
    train_data_raw = np.load(cfg.train_npy)
    obs, _ = make_obs_rel(train_data_raw, cfg.phys_channels)
    norm = Normalizer(cfg.normalize, cfg.clip_percentile, cfg.phys_channels)
    norm = norm.fit(train_data_raw, obs)
    del train_data_raw, obs  # 释放内存防止 OOM

    # [Fix S2]: 使用 untils.data 标准数据集，统一返回通道优先格式 [B, C, Tin, H, W]
    train_dataset = SlidingWindowDataset(
        cfg.train_npy, normalizer=norm,
        input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
        phys_channels=cfg.phys_channels
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )

    # ==========================================
    # 2. 模型与优化器初始化 (Model & Optimizer Setup)
    # ==========================================
    G = UNetGenerator(
        cfg.phys_channels * 2, 1, cfg.base_filter, cfg.use_cue,
        use_nafblock=cfg.use_nafblock, use_axial_w=cfg.use_axial_w
    ).to(device)

    D = Discriminator(cfg.phys_channels * 2).to(device)

    P = WindFieldPredictor(
        input_channels=cfg.phys_channels + 2,
        hidden_channels=cfg.pred_hidden_channels,
        num_layers=cfg.pred_num_layers,
        kernel_size=cfg.pred_kernel_size,
        pred_steps=cfg.pred_steps
    ).to(device)

    # 尝试加载预训练 GAIN 权重
    if cfg.gain_pretrain_path and Path(cfg.gain_pretrain_path).exists():
        print(f"[Info] Loading pre-trained GAIN from {cfg.gain_pretrain_path}")
        ckpt = torch.load(cfg.gain_pretrain_path, map_location=device)
        G.load_state_dict(ckpt['model'])  # [Fix S8]: 保持读取 key 为 'model'
        if 'discriminator' in ckpt:
            D.load_state_dict(ckpt['discriminator'])
    else:
        print("[Warning] No pre-trained GAIN found. Training G from scratch.")

    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g_joint)
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_g_joint)
    optim_P = torch.optim.Adam(P.parameters(), lr=cfg.lr_p)

    # ==========================================
    # 3. 训练主循环 (Training Loop)
    # ==========================================
    best_loss = float('inf')

    for epoch in range(cfg.epochs):
        G.train()
        D.train()
        P.train()

        epoch_pred_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{cfg.epochs}] Joint")

        for step, (hx, hm, hr, fx, fm, fr) in enumerate(pbar):
            # 将张量送入目标设备 (Shape: [B, C, Tin, H, W])
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)

            B, C, Tin, H, W = hx.shape

            # --- 构造掩码与生成器输入 ---
            # [Fix S6 & M13]: 限定人工缺失区域仅在原有健康的观测区 (valid_mask_h) 产生
            art_mask = generate_pure_structural_missing_mask(B, H, W, device)
            art_mask = art_mask.expand(-1, C, Tin, -1, -1)
            valid_mask_h = hm * hr
            m_t = valid_mask_h * art_mask

            # [Fix M1]: 规范化生成器的输入 (拼接 Cue 和 Noise)
            xb, eb = build_generator_inputs(hx, m_t)

            # ----------------------------------------
            # 阶段 A: 训练判别器 D (Discriminator Step)
            # ----------------------------------------
            optim_D.zero_grad()
            # 留空：此处接入您项目原始的对抗判别损失逻辑 (Adversarial Loss)
            # d_loss = compute_D_loss(D, xb, ...)
            # d_loss.backward()
            # optim_D.step()

            # ----------------------------------------
            # 阶段 B: 联合训练生成器 G 与预测器 P (Joint Step)
            # ----------------------------------------
            optim_G.zero_grad()
            optim_P.zero_grad()

            # G 前向网络: 输出补全后的径向速度 (V)，Shape: [B, 1, Tin, H, W]
            v_completed = G(xb, m_t)

            # 留空：此处接入您项目原始的重建损失与生成对抗损失
            # g_loss = compute_G_loss(G, D, ...)
            g_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # [Fix S10]: 移除原本切断梯度的 no_grad()，让 P 的梯度指导 G 更好地补全
            # [Fix S7]: 强制沿通道维度 (dim=1) 拼接 P 网络的输入，总计 6 个通道
            # 拼接内容: 补全的V(1) + 历史特征(3) + 观测掩码M(1) + 可靠掩码RM(1)
            features = hx[:, 1:cfg.phys_channels, :, :, :]  # [B, 3, Tin, H, W]

            pred_in = torch.cat([
                v_completed,
                features,
                hm[:, 0:1, :, :, :],
                hr[:, 0:1, :, :, :]
            ], dim=1)

            # 调整时间维度至前面供 P 使用: [B, Tin, Channels, H, W]
            pred_in_p = pred_in.permute(0, 2, 1, 3, 4).contiguous()

            # P 前向网络: 预测未来风场，Shape: [B, Tout, 1, H, W]
            out_pred = P(pred_in_p)

            # --- 预测损失计算 ---
            # 提取真实未来预测目标 (V 通道) 和有效的评估掩码
            tgt = fx[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            valid_mask_f = (fm * fr)[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)

            pred_mae = masked_prediction_loss(out_pred, tgt, valid_mask_f)

            # [Fix M9]: 真实联合训练损失 (将加权后的预测误差反传至生成器)
            total_loss = g_loss + (cfg.lambda_pred * pred_mae)

            # [Fix L11]: 增加数值稳定性防御，拦截异常梯度
            assert torch.isfinite(total_loss).all(), "Fatal Error: total_loss contains NaN or Inf."

            total_loss.backward()

            # [Fix S5]: 修正梯度裁剪参数传递的低级错误
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(P.parameters(), max_norm=cfg.grad_clip)

            optim_G.step()
            optim_P.step()

            # --- 日志更新 ---
            epoch_pred_loss += pred_mae.item()
            pbar.set_postfix({'P_MAE': f'{pred_mae.item():.4f}'})

        # ==========================================
        # 4. 模型验证与保存 (Validation & Checkpoint)
        # ==========================================
        avg_loss = epoch_pred_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            # [Fix S8]: 保持保存权重的 key 为 'model'，与推理端一致
            checkpoint = {
                'epoch': epoch,
                'model': G.state_dict(),
                'predictor': P.state_dict(),
                'normalizer': norm,
                'cfg': cfg
            }
            torch.save(checkpoint, cfg.model_path)
            print(f"[Info] Saved new best model to {cfg.model_path} | Best P_MAE: {best_loss:.4f}")


if __name__ == "__main__":
    train_joint()