# -*- coding: utf-8 -*-
"""径向量程扩展网络独立训练。

用法：python main.py ext_train

训练流程：
1. 加载 GAIN 预训练模型（可选，用于生成补全后的近距离输入）
2. 训练量程扩展网络，用真实远距离 V 作为监督
3. 验证集监控远距离 MAE/RMSE
4. 保存最佳模型
"""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from untils.data import (
    MultichannelClipDataset, Normalizer, make_obs_rel, pad_to, crop_to,
    ORI_H, ORI_W, PAD_H, PAD_W, PHYS_CHANNELS, TARGET_CHANNEL,
    RangeExtensionSequenceDataset,
)
from untils.losses import extension_mae_loss, compute_extension_metrics, extension_composite_loss
from extension_model import build_extension_model, init_extension_weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RangeExtensionDataset(torch.utils.data.Dataset):
    """径向量程扩展数据集。

    对每个 PPI 扫描帧，将近距离区作为输入，远距离区作为监督目标。

    返回：
        near_x    : [C, H_near, W]  近距离多通道数据（归一化后）
        near_m    : [C, H_near, W]  近距离观测掩码
        near_r    : [C, H_near, W]  近距离可靠性掩码
        far_v     : [H_far, W]      远距离 V 目标（归一化后）
        far_valid : [H_far, W]      远距离有效标签掩码
    """

    def __init__(self, npy_path, normalizer=None, phys_channels=PHYS_CHANNELS,
                 near_end=68, far_end=78):
        self.arr = np.load(npy_path).astype(np.float32)
        if self.arr.ndim != 4:
            raise ValueError(f"数据必须为[T,H,W,C]，当前shape={self.arr.shape}")
        self.normalizer = normalizer
        self.phys_channels = phys_channels
        self.near_end = near_end
        self.far_end = far_end
        self.obs, self.rel = make_obs_rel(self.arr, phys_channels)
        self.length = self.arr.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.arr[idx]
        obs = self.obs[idx]
        rel = self.rel[idx]

        near_x_raw = x[:self.near_end, :, :self.phys_channels]
        far_x_raw = x[self.near_end:self.far_end, :, :self.phys_channels]
        near_obs = obs[:self.near_end]
        far_obs = obs[self.near_end:self.far_end]
        near_rel = rel[:self.near_end]
        far_rel = rel[self.near_end:self.far_end]

        if self.normalizer is not None:
            near_x_raw = np.nan_to_num(near_x_raw, nan=0.0, posinf=0.0, neginf=0.0)
            near_x = self.normalizer.transform(near_x_raw)
            far_x_raw = np.nan_to_num(far_x_raw, nan=0.0, posinf=0.0, neginf=0.0)
            far_x = self.normalizer.transform(far_x_raw)
        else:
            near_x = np.nan_to_num(near_x_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            far_x = np.nan_to_num(far_x_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        near_x = np.where(near_obs[..., None] > 0.5, near_x, 0.0).astype(np.float32)
        far_v = far_x[..., 0].astype(np.float32)
        far_v = np.where(far_obs > 0.5, far_v, 0.0).astype(np.float32)

        near_x = np.transpose(near_x, (2, 0, 1))
        near_m = np.repeat(near_obs[None, ...], self.phys_channels, axis=0).astype(np.float32)
        near_r = np.repeat(near_rel[None, ...], self.phys_channels, axis=0).astype(np.float32)
        far_valid = (far_obs * far_rel).astype(np.float32)

        return (
            torch.from_numpy(near_x),
            torch.from_numpy(near_m),
            torch.from_numpy(near_r),
            torch.from_numpy(far_v),
            torch.from_numpy(far_valid),
        )


def assemble_extension_input(near_x, near_m, near_r):
    """组装量程扩展网络输入。

    near_x : [B, C, H_near, W]
    near_m : [B, C, H_near, W]
    near_r : [B, C, H_near, W]

    返回 : [B, C+2, H_near, W]
    """
    components = [near_x, near_m[:, 0:1], near_r[:, 0:1]]
    return torch.cat(components, dim=1)


def fit_normalizer(train_npy, method, clip, phys_channels):
    arr = np.load(train_npy)
    obs, _ = make_obs_rel(arr, phys_channels)
    return Normalizer(method, clip, phys_channels).fit(arr, obs)


def train_one_epoch(model, loader, optim, cfg, device, is_spatiotemporal=False):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        if is_spatiotemporal:
            x_cur, x_seq, far_v, far_valid = batch
            x_cur = x_cur.to(device)
            x_seq = x_seq.to(device)
            far_v = far_v.to(device)
            far_valid = far_valid.to(device)
            far_pred = model(x_cur, x_seq)
            # 提取近场最外圈锚点（通道0=V，通道4=obs，通道5=rel）
            near_v_last = x_cur[:, 0, -1, :]
            near_valid_last = x_cur[:, 4, -1, :] * x_cur[:, 5, -1, :]
        else:
            near_x, near_m, near_r, far_v, far_valid = batch
            near_x = near_x.to(device)
            near_m = near_m.to(device)
            near_r = near_r.to(device)
            far_v = far_v.to(device)
            far_valid = far_valid.to(device)
            ext_input = assemble_extension_input(near_x, near_m, near_r)
            far_pred = model(ext_input)
            # 提取近场最外圈锚点
            near_v_last = near_x[:, 0, -1, :]
            near_valid_last = near_m[:, 0, -1, :] * near_r[:, 0, -1, :]

            # [核心替换] 调用复合物理扩展损失
        loss = extension_composite_loss(far_pred, far_v, far_valid, near_v_last, near_valid_last)
        if not torch.isfinite(loss):
            continue

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.ext_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.ext_grad_clip)
        optim.step()

        with torch.no_grad():
            mae, rmse, _ = compute_extension_metrics(far_pred, far_v, far_valid)
            total_loss += loss.item()
            total_mae += mae
            total_rmse += rmse
            n += 1

    return {
        "loss": total_loss / max(n, 1),
        "MAE": total_mae / max(n, 1),
        "RMSE": total_rmse / max(n, 1),
    }


@torch.no_grad()
def validate(model, loader, cfg, device, is_spatiotemporal=False):
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    total_pts = 0.0
    n = 0

    for batch in loader:
        if is_spatiotemporal:
            x_cur, x_seq, far_v = batch
            x_cur = x_cur.to(device)
            x_seq = x_seq.to(device)
            far_v = far_v.to(device)
            far_pred = model(x_cur, x_seq)
            far_valid = (far_v != 0).float().to(device)
        else:
            near_x, near_m, near_r, far_v, far_valid = batch
            near_x = near_x.to(device)
            near_m = near_m.to(device)
            near_r = near_r.to(device)
            far_v = far_v.to(device)
            far_valid = far_valid.to(device)

            ext_input = assemble_extension_input(near_x, near_m, near_r)
            far_pred = model(ext_input)

        mae, rmse, pts = compute_extension_metrics(far_pred, far_v, far_valid)
        total_mae += mae * pts
        total_rmse += rmse ** 2 * pts
        total_pts += pts
        n += 1

    if total_pts > 0:
        return total_mae / total_pts, (total_rmse / total_pts) ** 0.5
    return 0.0, 0.0


def save_history(history, out_dir):
    if not history:
        return
    out_dir = Path(out_dir)
    keys = list(history[0].keys())
    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)

    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(10, 6))
    plt.plot(ep, [h["loss"] for h in history], label="Train Loss")
    plt.plot(ep, [h["val_MAE"] for h in history], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Extension Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ext_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ep, [h["val_MAE"] for h in history], label="Val MAE")
    plt.plot(ep, [h["val_RMSE"] for h in history], label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Extension Validation Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ext_metrics.png", dpi=180)
    plt.close()


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    near_end = cfg.near_end_gate
    far_end = cfg.far_end_gate
    far_gates = far_end - near_end
    exp_name = getattr(cfg, "ext_exp_name", f"ext_{cfg.ext_model_type}_near{near_end}_far{far_end}")
    out_dir = Path(f"outputs_single_v/outputs_{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    is_st = (cfg.ext_model_type == "spatiotemporal")
    temporal_steps = getattr(cfg, "temporal_in_steps", 3)

    print(f"径向量程扩展训练")
    print(f"  近距离: gate 0~{near_end - 1} ({660}m~{660 + 30 * (near_end - 1)}m)")
    print(f"  远距离: gate {near_end}~{far_end - 1} ({660 + 30 * near_end}m~{660 + 30 * (far_end - 1)}m)")
    print(f"  扩展跨度: {30 * far_gates}m ({far_gates} gates)")
    print(f"  模型: {cfg.ext_model_type}, base_filter={cfg.ext_base_filter}")
    if is_st:
        print(f"  时序模式: T={temporal_steps}帧")
    print(f"  设备: {device}, batch={cfg.ext_batch_size}")

    norm = fit_normalizer(cfg.train_npy, cfg.normalize, cfg.clip_percentile, cfg.phys_channels)

    gain_ckpt_path = getattr(cfg, "gain_pretrain_path", None)
    if gain_ckpt_path and Path(gain_ckpt_path).exists():
        ckpt = torch.load(gain_ckpt_path, map_location=device, weights_only=False)
        if "normalizer" in ckpt:
            norm_dict = ckpt["normalizer"]
            norm.lo = np.asarray(norm_dict["lo"])
            norm.hi = np.asarray(norm_dict["hi"])
            norm.mean = np.asarray(norm_dict["mean"])
            norm.std = np.asarray(norm_dict["std"])
            print(f"  加载预训练 normalizer: lo={norm.lo[0]:.2f}~{norm.lo[-1]:.2f}, hi={norm.hi[0]:.2f}~{norm.hi[-1]:.2f}")

    if is_st:
        train_ds = RangeExtensionSequenceDataset(
            cfg.train_npy, normalizer=norm, phys_channels=cfg.phys_channels,
            near_end=near_end, far_end=far_end, temporal_steps=temporal_steps,
        )
        val_ds = RangeExtensionSequenceDataset(
            cfg.val_npy, normalizer=norm, phys_channels=cfg.phys_channels,
            near_end=near_end, far_end=far_end, temporal_steps=temporal_steps,
        )
    else:
        train_ds = RangeExtensionDataset(
            cfg.train_npy, normalizer=norm, phys_channels=cfg.phys_channels,
            near_end=near_end, far_end=far_end,
        )
        val_ds = RangeExtensionDataset(
            cfg.val_npy, normalizer=norm, phys_channels=cfg.phys_channels,
            near_end=near_end, far_end=far_end,
        )
    train_loader = DataLoader(train_ds, batch_size=cfg.ext_batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.ext_batch_size, shuffle=False, num_workers=0)

    model = build_extension_model(cfg).to(device)
    if not is_st:
        init_extension_weights(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.ext_lr,
        betas=cfg.betas, weight_decay=cfg.ext_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=5, factor=0.5, min_lr=1e-6,
    )

    best = float("inf")
    pat = 0
    history = []

    for epoch in range(1, cfg.ext_epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, cfg, device, is_st)
        val_mae, val_rmse = validate(model, val_loader, cfg, device, is_st)
        scheduler.step(val_mae)

        row = {
            "epoch": epoch,
            "loss": tr["loss"],
            "train_MAE": tr["MAE"],
            "train_RMSE": tr["RMSE"],
            "val_MAE": val_mae,
            "val_RMSE": val_rmse,
        }
        history.append(row)

        print(f"\n[Epoch {epoch:03d}] "
              f"Loss={tr['loss']:.6f} "
              f"Train MAE={tr['MAE']:.6f} RMSE={tr['RMSE']:.6f} | "
              f"Val MAE={val_mae:.6f} RMSE={val_rmse:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "normalizer": norm.__dict__,
            "cfg": cfg.to_dict(),
            "epoch": epoch,
            "val_MAE": val_mae,
            "val_RMSE": val_rmse,
        }
        torch.save(ckpt, out_dir / "ext_last.pth")

        if val_mae < best:
            best = val_mae
            pat = 0
            torch.save(ckpt, out_dir / "ext_best.pth")
            print(f"  保存最佳模型，Val MAE={best:.6f}")
        else:
            pat += 1

        if pat >= cfg.ext_patience:
            print(f"早停：epoch={epoch}, best Val MAE={best:.6f}")
            break

    save_history(history, out_dir)
    print(f"\n训练完成。最佳模型：{out_dir / 'ext_best.pth'}")


if __name__ == "__main__":
    main()