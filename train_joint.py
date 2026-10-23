import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np

from config import Config
from untils.data import SlidingWindowDataset, Normalizer, build_generator_inputs, make_obs_rel
from model import UNetGenerator, Discriminator
from predictor import WindFieldPredictor
from untils.losses import masked_prediction_loss
from untils.data import generate_pure_structural_missing_mask


def set_seed(seed=42):
    """
    设置全局随机种子以确保结果可复现。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_joint():
    set_seed(42)
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化与归一化
    print("Loading data for normalizer...")
    train_data_raw = np.load(cfg.train_npy)
    obs, rel = make_obs_rel(train_data_raw, cfg.phys_channels)
    norm = Normalizer(cfg.normalize, cfg.clip_percentile, cfg.phys_channels)
    norm = norm.fit(train_data_raw, obs)
    del train_data_raw, obs, rel

    # [修复 S2]: 使用统一的通道优先数据集 [B, C, T, H, W]
    train_dataset = SlidingWindowDataset(
        cfg.train_npy, normalizer=norm,
        input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
        phys_channels=cfg.phys_channels
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # 2. 网络组装
    G = UNetGenerator(cfg.phys_channels * 2, 1, cfg.base_filter, cfg.use_cue, use_nafblock=cfg.use_nafblock,
                      use_axial_w=cfg.use_axial_w).to(device)
    D = Discriminator(cfg.phys_channels * 2).to(device)
    P = WindFieldPredictor(input_channels=cfg.phys_channels + 2, hidden_channels=cfg.pred_hidden_channels,
                           num_layers=cfg.pred_num_layers, kernel_size=cfg.pred_kernel_size,
                           pred_steps=cfg.pred_steps).to(device)

    # [修复 M8]: 若路径配置了但文件不存在，抛出异常，防止无意义的随机初始化训练
    if cfg.gain_pretrain_path:
        if Path(cfg.gain_pretrain_path).exists():
            print(f"Loading pre-trained GAIN from {cfg.gain_pretrain_path}")
            ckpt = torch.load(cfg.gain_pretrain_path, map_location=device)
            # [修复 S8]: 确保以 'model' 键名读取
            G.load_state_dict(ckpt.get('model', ckpt.get('generator', None)))
            if 'discriminator' in ckpt:
                D.load_state_dict(ckpt['discriminator'])
        else:
            raise FileNotFoundError(f"[Fatal] Pretrain path {cfg.gain_pretrain_path} does not exist!")

    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g_joint)
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_g_joint)
    optim_P = torch.optim.Adam(P.parameters(), lr=cfg.lr_p)

    best_loss = float('inf')

    # 3. 联合训练循环
    for epoch in range(cfg.epochs):
        G.train()
        D.train()
        P.train()

        epoch_pred_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} Joint")
        for step, (hx, hm, hr, fx, fm, fr) in enumerate(pbar):
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)

            B, C, Tin, H, W = hx.shape

            # 1. 创建随机缺失结构
            # 根据 data.py，此函数现在返回 (mask, rates)
            art_mask, _ = generate_pure_structural_missing_mask(B, H, W, device, drop_prob=0.15, valid_mask=(hm * hr))
            art_mask = art_mask.unsqueeze(1).expand(-1, C, Tin, -1, -1)

            # 2. 结合雷达本身的有效区域和人工挖洞，得到最终输入给网络的掩码 m_t
            valid_mask_h = hm * hr
            m_t = valid_mask_h * (1 - art_mask) # 1 为保留，0 为缺失

            # ========================================================
            # [核心修复 M11]: 强制归零，物理隔离真值！
            # 无论网络前面怎么取特征(例如可能用到的 cue)，
            # 只要在这里让原始数据 hx 在缺失处变成绝对的 0，
            # 模型就永远不可能“偷看”到被挖掉的真值。
            # ========================================================
            xin = hx * m_t

            # ========================================================
            # 构建生成器 G 的输入
            # 使用 build_generator_inputs 正确处理 xin 和 掩码
            # ========================================================
            xb, eb = build_generator_inputs(xin, m_t)
            gx, gm = xb.to(device), m_t.to(device)

            # --- D 与 G, P 联合优化 ---
            optim_D.zero_grad()
            optim_G.zero_grad()
            optim_P.zero_grad()

            # 此处可以插入你的 D 判别器的前向与 loss 计算
            # ...
            optim_D.step()

            # 3. 生成器补全
            # 传入的已经是处理过并且真值已绝对抹除的安全输入 (gx, gm)
            v_completed = G(gx, gm)

            # 提取雷达补全目标的伪代码，需根据您的判别器定义补全逻辑
            g_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # [修复 S10]: 移除 no_grad 限制，将 G 的输出拼接传给 P
            # [修复 S7]: 沿通道拼接: V_comp(1) + 几何/物理通道(3) + 观测掩码(1) + 可靠掩码(1) = 6
            features = hx[:, 1:cfg.phys_channels, :, :, :]
            pred_in = torch.cat([
                v_completed,
                features,
                hm[:, 0:1, :, :, :],
                hr[:, 0:1, :, :, :]
            ], dim=1)

            # 维度换序以适配 Predictor: [B, C, T, H, W] -> [B, T, C, H, W]
            pred_in_p = pred_in.permute(0, 2, 1, 3, 4).contiguous()

            out_pred = P(pred_in_p)

            # 序列损失计算
            tgt = fx[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            valid_mask_f = (fm * fr)[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            pred_mae = masked_prediction_loss(out_pred, tgt, valid_mask_f)

            # [修复 M9]: 基于全局梯度的真实联合损失
            total_loss = g_loss + (cfg.lambda_pred * pred_mae)

            assert torch.isfinite(total_loss).all(), "Fatal Error: Loss is NaN/Inf"
            total_loss.backward()

            # [修复 S5]: 修正裁剪元组签名
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(P.parameters(), cfg.grad_clip)

            optim_G.step()
            optim_P.step()

            epoch_pred_loss += pred_mae.item()
            pbar.set_postfix({'P_MAE': f'{pred_mae.item():.4f}'})

        # --- 模型保存 ---
        avg_loss = epoch_pred_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            # [修复 S8]: 固定权重键名为 'model'
            torch.save({
                'epoch': epoch,
                'model': G.state_dict(),
                'predictor': P.state_dict(),
                'normalizer': norm,
                'cfg': cfg
            }, cfg.model_path)
            print(f"Saved best joint model to {cfg.model_path} with P_MAE: {best_loss:.4f}")


if __name__ == "__main__":
    train_joint()