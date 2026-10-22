# -*- coding: utf-8 -*-
"""CNNRangeExtension 通用训练器（协议照抄 train_extension.py）。

输入：已组装的 numpy 数组（各基线自行完成近门补全后组装）：
    x_train   [Ntr, 6, Hn, W]   扩展输入（归一化空间）
    fv_train  [Ntr, FarGates, W] 远门 V 目标（归一化）
    fvalid_*  [N, FarGates, W]   远门有效掩码 (far_obs*far_rel)
保存：ext_best.pth / ext_last.pth / ext_history.csv / run_meta.json
"""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import cfg


def set_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_ds(*arrays):
    return TensorDataset(*[torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                           for a in arrays])


def _train_epoch(model, loader, optim, device, grad_clip):
    model.train()
    tot = maes = n = 0.0
    for x, fv, fvm in loader:
        x, fv, fvm = x.to(device), fv.to(device), fvm.to(device)
        pred = model(x)
        pred3 = pred[:, 0]
        m = fvm
        denom = m.sum() + 1e-8
        loss = (torch.abs(pred3 - fv) * m).sum() / denom
        if not torch.isfinite(loss):
            continue
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        with torch.no_grad():
            tot += loss.item()
            maes += (torch.abs(pred3 - fv) * m).sum().item() / denom.item()
            n += 1
    return tot / max(n, 1), maes / max(n, 1)


@torch.no_grad()
def _validate(model, loader, device):
    model.eval()
    tot = cnt = 0.0
    sabs = sn = 0.0
    for x, fv, fvm in loader:
        x, fv, fvm = x.to(device), fv.to(device), fvm.to(device)
        pred3 = model(x)[:, 0]
        m = fvm
        sabs += (torch.abs(pred3 - fv) * m).sum().item()
        sn += m.sum().item()
        tot += 1
    return (sabs / max(sn, 1e-8)), sn


def train_extension(x_train, fv_train, fvm_train,
                    x_val, fv_val, fvm_val,
                    out_dir, device=None, epochs=None, patience=None,
                    batch_size=None, lr=None, seed=cfg.SEED):
    """训练 CNNRangeExtension。返回 (best_val_mae, out_dir)。"""
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    epochs = epochs or cfg.EXT_EPOCHS
    patience = patience or cfg.EXT_PATIENCE
    batch_size = batch_size or cfg.EXT_BATCH_SIZE
    lr = lr or cfg.EXT_LR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from extension_model import CNNRangeExtension, init_extension_weights
    model = CNNRangeExtension(in_channels=6, base_filter=cfg.EXT_BASE_FILTER,
                              far_gates=cfg.FAR_GATES).to(device)
    init_extension_weights(model)
    n_params = sum(p.numel() for p in model.parameters())

    tr_loader = DataLoader(_to_ds(x_train, fv_train, fvm_train),
                           batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    va_loader = DataLoader(_to_ds(x_val, fv_val, fvm_val),
                           batch_size=batch_size, shuffle=False, num_workers=0)

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=cfg.BETAS,
                             weight_decay=cfg.EXT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=cfg.EXT_SCHED_PATIENCE,
        factor=cfg.EXT_SCHED_FACTOR, min_lr=cfg.EXT_MIN_LR)

    best = float("inf")
    best_state = None
    pat = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_mae = _train_epoch(model, tr_loader, optim, device, cfg.EXT_GRAD_CLIP)
        va_mae, va_pts = _validate(model, va_loader, device)
        sched.step(va_mae)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_MAE": tr_mae,
                        "val_MAE": va_mae})
        print(f"  [ext {epoch:03d}] loss={tr_loss:.6f} trainMAE={tr_mae:.6f} "
              f"valMAE={va_mae:.6f} lr={optim.param_groups[0]['lr']:.2e}")
        if va_mae < best:
            best = va_mae
            pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                print(f"  [ext] 早停于 epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "val_MAE": best,
                "n_params": n_params, "in_channels": 6,
                "base_filter": cfg.EXT_BASE_FILTER, "far_gates": cfg.FAR_GATES},
               out_dir / "ext_best.pth")
    torch.save({"model": model.state_dict(), "epoch": epoch},
               out_dir / "ext_last.pth")

    with open(out_dir / "ext_history.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    meta = {"n_params": n_params, "best_val_MAE": best, "epochs_run": len(history),
            "elapsed_s": round(time.time() - t0, 1),
            "protocol": {"lr": lr, "betas": list(cfg.BETAS), "batch_size": batch_size,
                         "epochs": epochs, "patience": patience,
                         "grad_clip": cfg.EXT_GRAD_CLIP, "seed": seed,
                         "scheduler": "ReduceLROnPlateau(p5,f0.5,min1e-6)"}}
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return best, out_dir


@torch.no_grad()
def eval_extension(ext_best_path, x_test, fv_test, fvm_test, device=None):
    """加载 ext_best.pth，返回 (pred_phys [N,FarG,W], 每帧耗时ms)。

    预测为归一化空间；此处直接返回归一化 pred，物理转换由调用方做。
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ext_best_path, map_location=device, weights_only=False)
    from extension_model import CNNRangeExtension
    model = CNNRangeExtension(in_channels=ckpt.get("in_channels", 6),
                              base_filter=ckpt.get("base_filter", cfg.EXT_BASE_FILTER),
                              far_gates=ckpt.get("far_gates", cfg.FAR_GATES)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = torch.from_numpy(np.ascontiguousarray(x_test, dtype=np.float32)).to(device)
    preds = []
    t0 = time.time()
    bs = cfg.EXT_BATCH_SIZE
    for i in range(0, x.shape[0], bs):
        preds.append(model(x[i:i + bs]).cpu().numpy())
    elapsed = (time.time() - t0) * 1000.0 / max(x.shape[0], 1)
    return np.concatenate(preds, axis=0)[:, 0].astype(np.float32), elapsed
