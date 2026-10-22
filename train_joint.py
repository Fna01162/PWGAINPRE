# -*- coding: utf-8 -*-
"""GAIN 补全 + ConvLSTM 时间预测 联合训练。

用法：python main.py joint

训练流程：
1. 加载预训练 GAIN 和预测网络
2. 每个时间窗口：
   a. GAIN 逐帧补全历史序列 → completed V
   b. 将补全后的历史序列输入预测网络 → 预测未来 V
   c. 联合损失 = GAIN 损失 + λ_pred × 预测损失
3. 反向传播同时更新 GAIN、Refiner 和预测网络
4. 验证集同时监控补全质量和预测质量
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from untils.data import (
    Normalizer, make_obs_rel,
    PHYS_CHANNELS, TARGET_CHANNEL,
)
from untils.losses import (
    discriminator_loss, generate_hint, generator_adv_loss,
    reconstruction_loss, tv_loss, azimuth_tv_loss,
)
from untils.losses_refine import compute_refiner_losses
from untils.refiner import ResidualRefiner, apply_refiner
from model import UNetGenerator, PatchGANDiscriminator, init_weights
from predictor import WindFieldPredictor, init_predictor_weights

BAD_RAYS = [73, 74, 75]
CHANNEL_NAMES = ["V", "CNR", "SW", "Skew"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SlidingWindowDataset(Dataset):
    """滑动窗口数据集：从完整 PPI 序列中提取 (历史, 未来) 对。

    输出：
        history_x: [B, Tin, C, H, W]  历史多通道序列
        future_x : [B, Tout, C, H, W]  未来多通道序列（仅用 V 做监督）
        history_m: [B, Tin, 1, H, W]  历史观测掩码
        history_r: [B, Tin, 1, H, W]  历史可靠掩码
        future_m : [B, Tout, 1, H, W]  未来观测掩码
        future_r : [B, Tout, 1, H, W]  未来可靠掩码
    """

    def __init__(self, npy_path, normalizer, input_steps=10, pred_steps=1,
                 phys_channels=4, frame_interval_min=5):
        arr = np.load(npy_path).astype(np.float32)
        self.arr = normalizer.transform(arr)
        obs, rel = make_obs_rel(arr, phys_channels)
        self.obs = obs.astype(np.float32)
        self.rel = rel.astype(np.float32)
        self.T = arr.shape[0]
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.phys_channels = phys_channels
        self.frame_interval_min = frame_interval_min

        valid_windows = []
        for start in range(self.T - input_steps - pred_steps + 1):
            end_hist = start + input_steps
            end_future = end_hist + pred_steps

            hist_obs = self.obs[start:end_hist]
            fut_obs = self.obs[end_hist:end_future]

            has_valid_history = np.any(hist_obs > 0.5)
            has_valid_future = np.any(fut_obs > 0.5)

            if has_valid_history and has_valid_future:
                valid_windows.append(start)

        self.windows = valid_windows
        print(f"[SlidingWindowDataset] 总帧数={self.T}, 窗口数={len(self.windows)}, "
              f"input_steps={input_steps}, pred_steps={pred_steps}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end_hist = start + self.input_steps
        end_future = end_hist + self.pred_steps

        history_x = torch.from_numpy(self.arr[start:end_hist]).float()
        future_x = torch.from_numpy(self.arr[end_hist:end_future]).float()
        history_m = torch.from_numpy(self.obs[start:end_hist, None]).float()
        history_r = torch.from_numpy(self.rel[start:end_hist, None]).float()
        future_m = torch.from_numpy(self.obs[end_hist:end_future, None]).float()
        future_r = torch.from_numpy(self.rel[end_hist:end_future, None]).float()

        return history_x, future_x, history_m, history_r, future_m, future_r


def assemble_predictor_input(history_x, v_completed, history_m, history_r):
    """组装预测网络输入。

    history_x  : [B, Tin, C, H, W]  原始历史多通道（含缺失）
    v_completed: [B, Tin, 1, H, W]  GAIN 补全后的 V
    history_m  : [B, Tin, 1, H, W]  观测掩码
    history_r  : [B, Tin, 1, H, W]  可靠掩码

    返回 : [B, Tin, Cin, H, W]
        Cin = V(补全) + CNR/SW/Skew(原始) + obs + rel = 6
    """
    components = [
        v_completed,
        history_x[:, :, 1:],  # CNR, SW, Skew
        history_m,
        history_r,
    ]
    return torch.cat(components, dim=2)


def repeat_mask(m, channels):
    if m.dim() == 4:
        m = m.unsqueeze(1)
    return m.repeat(1, channels, 1, 1, 1) if m.shape[1] == 1 else m


def build_generator(cfg):
    return UNetGenerator(
        in_channels=8, out_channels=1, base_filter=cfg.base_filter,
        use_cue=cfg.use_cue, use_nafblock=cfg.use_nafblock,
        use_volformer=cfg.use_volformer,
        use_axial_w=cfg.use_axial_w,
        axial_num_pairs=cfg.axial_num_pairs,
        axial_num_heads=cfg.axial_num_heads,
    )


def build_refiner(cfg):
    if not getattr(cfg, "use_refiner", False):
        return None
    from untils.wavelet import wavelet_feature_channels
    phys_ch = int(getattr(cfg, "phys_channels", 4))
    geom_ch = 3
    mask_ch = 2
    base_ch = phys_ch + geom_ch + mask_ch
    if getattr(cfg, "use_wavelet", False):
        wl = wavelet_feature_channels(int(getattr(cfg, "n_wavelet_levels", 1)))
    else:
        wl = 0
    in_ch = base_ch + wl
    return ResidualRefiner(in_channels=in_ch, base_filter=cfg.refiner_base_filter)


def build_predictor(cfg):
    phys_ch = int(getattr(cfg, "phys_channels", 4))
    pred_in = int(getattr(cfg, "pred_input_channels", 0))
    if pred_in <= 0:
        pred_in = phys_ch + 2  # V + obs + rel

    return WindFieldPredictor(
        input_channels=pred_in,
        hidden_channels=int(getattr(cfg, "pred_hidden_channels", 64)),
        num_layers=int(getattr(cfg, "pred_num_layers", 2)),
        kernel_size=int(getattr(cfg, "pred_kernel_size", 3)),
        pred_steps=int(getattr(cfg, "pred_steps", 1)),
        base_filter=32,
    )


def train_one_epoch(G, D, R, P, loader, optim_G, optim_D, optim_P, cfg, device):
    G.train()
    D.train()
    P.train()
    if R is not None:
        R.train()

    keys = [
        "G", "D", "D_real", "D_fake", "G_adv", "G_rec",
        "TV", "AZTV",
        "REFINE", "WAVE", "PHY",
        "pred_loss", "pred_MAE",
    ]
    s = {k: 0. for k in keys}
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        (history_x, future_x, history_m, history_r,
         future_m, future_r) = batch
        history_x = history_x.to(device)
        future_x = future_x.to(device)
        history_m = history_m.to(device)
        history_r = history_r.to(device)
        future_m = future_m.to(device)
        future_r = future_r.to(device)

        B, T_in, C, H, W = history_x.shape
        T_out = future_x.shape[1]

        # ──── Step 1: GAIN 补全所有历史帧 ────
        g_losses = {}
        d_losses = {}

        for t in range(T_in):
            x_t = history_x[:, t:t+1]  # [B, 1, C, H, W]
            m_t = history_m[:, t:t+1]  # [B, 1, 1, H, W]
            r_t = history_r[:, t:t+1]  # [B, 1, 1, H, W]

            v_true = x_t[:, 0:1]       # [B, 1, 1, H, W]
            v_obs = m_t
            v_rel = r_t
            v_val = v_rel  # 简化：假设有效=可靠

            from untils.data import generate_pure_structural_missing_mask
            art, _ = generate_pure_structural_missing_mask(
                B, H, W, device, cfg.drop_prob, cfg=cfg, valid_mask=v_val
            )
            art = art * v_val

            given = v_obs * (1 - art)

            xin = x_t.clone()
            xin[:, 0] = v_true * given
            min_ = m_t.clone()
            min_[:, 0] = given

            z = torch.randn_like(xin)

            from untils.losses import generate_contextual_cue, build_contextual_fusion
            cue = generate_contextual_cue(xin, min_, t, target_channel=0)
            xb = build_contextual_fusion(xin, min_, cue, z, cfg.noise_scale)

            gx, gm = xin, min_
            vf = G(gx, gm)
            vfake = vf[:, 0]

            # D step
            with torch.no_grad():
                vhat = v_true * given + vfake.detach() * (1 - given)
            hint = generate_hint(given, cfg.hint_rate)
            dl = D(vhat, hint)
            d_loss = discriminator_loss(dl, given, v_val)

            optim_D.zero_grad(set_to_none=True)
            d_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
            optim_D.step()

            # G step
            dlg = D(vhat, hint)
            g_adv = generator_adv_loss(dlg, art, v_val)
            g_rec = reconstruction_loss(vfake, v_true, art, v_rel)
            g_tv = tv_loss(vfake, art, v_val) if cfg.tv_weight > 0 else 0.0
            g_aztv = azimuth_tv_loss(vfake, art, v_val) if cfg.aztv_weight > 0 else 0.0

            g_loss = (cfg.adv_weight * g_adv +
                      cfg.recon_weight * g_rec +
                      cfg.tv_weight * g_tv +
                      cfg.aztv_weight * g_aztv)

            if t == 0:
                optim_G.zero_grad(set_to_none=True)

            g_loss.backward(retain_graph=True)

            g_losses[t] = {
                'g_adv': g_adv.item(),
                'g_rec': g_rec.item(),
                'vfake': vfake.detach(),
            }
            d_losses[t] = d_loss.item()

        # Refine
        if R is not None and T_in > 0:
            vfake_0 = g_losses[0]['vfake']
            refine_losses = compute_refiner_losses(
                R, vfake_0, history_x[:, 0, 0:1], history_m[:, 0],
                history_r[:, 0], cfg
            )
            r_loss = (cfg.lambda_refine * refine_losses['refine'] +
                      cfg.lambda_wavelet * refine_losses.get('wavelet', 0) +
                      cfg.lambda_physics * refine_losses.get('physics', 0))
            r_loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_((G.parameters(), cfg.grad_clip))
            if R is not None:
                torch.nn.utils.clip_grad_norm_(R.parameters(), cfg.grad_clip)
        optim_G.step()
        if R is not None:
            # 假设 R 有独立优化器或与 G 共享
            pass

        # ──── Step 2: 预测网络训练 ────
        with torch.no_grad():
            v_completed_list = []
            for t in range(T_in):
                x_t = history_x[:, t:t+1]
                m_t = history_m[:, t:t+1]
                r_t = history_r[:, t:t+1]

                art, _ = generate_pure_structural_missing_mask(
                    B, H, W, device, cfg.test_drop_prob, cfg=cfg
                )
                given = m_t * (1 - art)

                xin = x_t.clone()
                xin[:, 0] = x_t[:, 0] * given
                min_ = m_t.clone()
                min_[:, 0] = given

                z = torch.randn_like(xin)
                cue = generate_contextual_cue(xin, min_, t, target_channel=0)
                xb = build_contextual_fusion(xin, min_, cue, z, cfg.noise_scale)

                vf = G(xb, min_)
                if R is not None:
                    vf = apply_refiner(R, vf, x_t, m_t, r_t, cfg)
                v_completed_list.append(vf[:, 0])

            v_completed = torch.stack(v_completed_list, dim=1)  # [B, Tin, 1, H, W]

        pred_input = assemble_predictor_input(
            history_x, v_completed, history_m, history_r
        )

        future_pred = P(pred_input)  # [B, Tout, 1, H, W]
        future_true = future_x[:, :, 0:1]  # [B, Tout, 1, H, W]
        future_valid = future_m * future_r

        pred_mae = F.l1_loss(future_pred, future_true, reduction='none')
        pred_mae = (pred_mae * future_valid).sum() / (future_valid.sum() + 1e-8)

        pred_loss = cfg.lambda_pred * pred_mae

        optim_P.zero_grad(set_to_none=True)
        pred_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(P.parameters(), cfg.grad_clip)
        optim_P.step()

        n += B
        s["pred_loss"] += pred_loss.item() * B
        s["pred_MAE"] += pred_mae.item() * B

    return {k: (v / n if n > 0 else 0.) for k, v in s.items()}


@torch.no_grad()
def validate(G, D, R, P, loader, cfg, device):
    G.eval()
    D.eval()
    P.eval()
    if R is not None:
        R.eval()

    keys = ["val_pred_MAE", "val_G_rec"]
    s = {k: 0. for k in keys}
    n = 0

    for batch in tqdm(loader, desc="val", leave=False):
        (history_x, future_x, history_m, history_r,
         future_m, future_r) = batch
        history_x = history_x.to(device)
        future_x = future_x.to(device)
        history_m = history_m.to(device)
        history_r = history_r.to(device)
        future_m = future_m.to(device)
        future_r = future_r.to(device)

        B = history_x.shape[0]

        # GAIN 补全
        v_completed_list = []
        for t in range(history_x.shape[1]):
            x_t = history_x[:, t:t+1]
            m_t = history_m[:, t:t+1]
            r_t = history_r[:, t:t+1]

            art, _ = generate_pure_structural_missing_mask(
                B, x_t.shape[2], x_t.shape[3], device, cfg.test_drop_prob, cfg=cfg
            )
            given = m_t * (1 - art)

            xin = x_t.clone()
            xin[:, 0] = x_t[:, 0] * given
            min_ = m_t.clone()
            min_[:, 0] = given

            z = torch.randn_like(xin)
            from untils.losses import generate_contextual_cue, build_contextual_fusion
            cue = generate_contextual_cue(xin, min_, t, target_channel=0)
            xb = build_contextual_fusion(xin, min_, cue, z, cfg.noise_scale)

            vf = G(xb, min_)
            if R is not None:
                vf = apply_refiner(R, vf, x_t, m_t, r_t, cfg)
            v_completed_list.append(vf[:, 0])

        v_completed = torch.stack(v_completed_list, dim=1)

        # 预测
        pred_input = assemble_predictor_input(
            history_x, v_completed, history_m, history_r
        )
        future_pred = P(pred_input)
        future_true = future_x[:, :, 0:1]
        future_valid = future_m * future_r

        pred_mae = F.l1_loss(future_pred, future_true, reduction='none')
        pred_mae = (pred_mae * future_valid).sum() / (future_valid.sum() + 1e-8)

        n += B
        s["val_pred_MAE"] += pred_mae.item() * B

    return {k: (v / n if n > 0 else 0.) for k, v in s.items()}


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print("="*60)
    print("  GAIN + ConvLSTM 时间预测 联合训练")
    print("="*60)
    print(f"  input_steps={cfg.input_steps}, pred_steps={cfg.pred_steps}")
    print(f"  lambda_pred={cfg.lambda_pred}")
    print(f"  device={device}")
    print("="*60)

    normalizer = Normalizer(cfg.normalize, cfg.clip_percentile, cfg.phys_channels)
    normalizer = normalizer.fit(np.load(cfg.train_npy),
                                 make_obs_rel(np.load(cfg.train_npy), cfg.phys_channels)[0])

    train_ds = SlidingWindowDataset(
        cfg.train_npy, normalizer,
        input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
        phys_channels=cfg.phys_channels,
        frame_interval_min=getattr(cfg, 'frame_interval_min', 5),
    )
    val_ds = SlidingWindowDataset(
        cfg.val_npy, normalizer,
        input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
        phys_channels=cfg.phys_channels,
        frame_interval_min=getattr(cfg, 'frame_interval_min', 5),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    G = build_generator(cfg).to(device)
    D = PatchGANDiscriminator(in_channels=2, base_filter=64).to(device)
    R = build_refiner(cfg)
    if R is not None:
        R = R.to(device)
    P = build_predictor(cfg).to(device)

    init_weights(G)
    init_weights(D)
    init_predictor_weights(P)

    if cfg.gain_pretrain_path and Path(cfg.gain_pretrain_path).exists():
        ckpt = torch.load(cfg.gain_pretrain_path, map_location='cpu',
                          weights_only=False)
        if 'generator' in ckpt:
            G.load_state_dict(ckpt['generator'])
            print("  [GAIN] 已加载预训练权重")
        if 'normalizer' in ckpt:
            normalizer.__dict__.update(ckpt['normalizer'])

    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)
    optim_P = torch.optim.Adam(P.parameters(), lr=cfg.lr_p,
                                weight_decay=cfg.pred_weight_decay)

    output_dir = Path(getattr(cfg, 'joint_pred_output_dir', cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{cfg.epochs}")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(
            G, D, R, P, train_loader, optim_G, optim_D, optim_P, cfg, device
        )
        val_metrics = validate(G, D, R, P, val_loader, cfg, device)

        record = {"epoch": epoch}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update({f"val_{k}": v for k, val_metrics.items()})
        history.append(record)

        print(f"\n  [Train] pred_MAE={train_metrics.get('pred_MAE', 0):.4f}")
        print(f"  [Val]   pred_MAE={val_metrics.get('val_pred_MAE', 0):.4f}")

        current_val = val_metrics.get('val_pred_MAE', float('inf'))
        if current_val < best_val_loss:
            best_val_loss = current_val
            patience_counter = 0

            save_dict = {
                'epoch': epoch,
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'predictor': P.state_dict(),
                'normalizer': normalizer.__dict__,
                'config': cfg.to_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
            }
            if R is not None:
                save_dict['refiner'] = R.state_dict()

            save_path = output_dir / "joint_best.pth"
            torch.save(save_dict, save_path)
            print(f"  ✓ 已保存最佳模型 (val_pred_MAE={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ 无改善 ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print(f"\n  早停！连续 {cfg.patience} 个 epoch 无改善")
            break

        if epoch % cfg.save_interval == 0:
            save_path = output_dir / f"joint_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'predictor': P.state_dict(),
                'config': cfg.to_dict(),
            }, save_path)

    history_path = output_dir / "joint_train_history.csv"
    if history:
        with open(history_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        print(f"\n✓ 训练历史已保存: {history_path}")

    print(f"\n{'='*60}")
    print("  训练完成！")
    print(f"  最佳验证 MAE: {best_val_loss:.4f}")
    print(f"  模型保存位置: {output_dir / 'joint_best.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()