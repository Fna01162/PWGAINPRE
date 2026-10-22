# -*- coding: utf-8 -*-
"""指标：物理空间补全/扩展指标 + 有效探测距离。

所有指标在物理空间（m/s）计算；归一化预测先 inverse_transform。
R_eff：从近门边界（gate 68）向远逐门检查 MAE<=TAU，最后一个连续通过门
对应距离为有效距离；gate 68 就失败则为 2.67（近门边界前一门）。
"""
import csv

import numpy as np

from . import cfg


def completion_metrics(pred_phys, true_phys, hole_mask):
    """洞像素上的 MAE/RMSE/Bias/Corr（物理 m/s）。

    pred/true: [T,Hn,W]；hole_mask: [T,Hn,W] bool。
    """
    m = hole_mask.astype(bool)
    if m.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Corr": np.nan, "n": 0}
    p = pred_phys[m].astype(np.float64)
    t = true_phys[m].astype(np.float64)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if p.std() > 1e-8 and t.std() > 1e-8:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = np.nan
    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "Corr": corr, "n": int(m.sum())}


def far_gate_maes(pred_phys, true_phys, valid, near_end=cfg.NEAR_GATES):
    """逐远门 MAE（物理）。

    pred/true: [T, H, W] 全场或 [T, FarGates, W]；valid 同形 [T,H,W]。
    返回 list[float] 长度 FAR_GATES。
    """
    if pred_phys.shape[1] >= near_end + cfg.FAR_GATES:
        p = pred_phys[:, near_end:near_end + cfg.FAR_GATES]
        t = true_phys[:, near_end:near_end + cfg.FAR_GATES]
        v = valid[:, near_end:near_end + cfg.FAR_GATES]
    else:
        p, t, v = pred_phys, true_phys, valid
    maes = []
    for g in range(p.shape[1]):
        m = v[:, g] > 0.5
        if m.sum() == 0:
            maes.append(np.nan)
        else:
            maes.append(float(np.mean(np.abs(p[:, g][m] - t[:, g][m]))))
    return maes


def r_eff_km(gate_maes, tau=cfg.TAU_MPS):
    """有效探测距离(km)。

    从第一个远门起，MAE 连续 <= tau 的最后一个门 -> 其距离；
    第一门即失败 -> 近门边界（gate NEAR_GATES-1 的距离）。
    """
    last_pass = None
    for i, mae in enumerate(gate_maes):
        if mae is not None and np.isfinite(mae) and mae <= tau:
            last_pass = i
        else:
            break
    if last_pass is None:
        return cfg.gate_dist_km(cfg.NEAR_GATES - 1)
    return cfg.gate_dist_km(cfg.NEAR_GATES + last_pass)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
