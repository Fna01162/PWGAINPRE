# -*- coding: utf-8 -*-
"""数据加载、归一化、构造 (输入窗口, 未来帧) 与人工遮挡。

每个窗口：
    x     : [Tin,H,W,C] 归一化，人工遮挡处置 0（给补全方法的输入）
    clean : [Tin,H,W,C] 归一化，遮挡前的完整场（补全真值，仅 valid 处有意义）
    art   : [Tin,H,W]   人工扇区遮挡（1=遮住，真值已知）
    valid : [Tin,H,W]   真实有效区域（obs & rel）
    y     : [Tout,H,W,C] 未来帧原始值（预测目标）
    y_valid: [Tout,H,W]
"""
import numpy as np

from untils.data import make_obs_rel, Normalizer
from . import cfg
from .masks import window_sector, window_seed

_NORM = None


def _load_raw(split):
    path = {
        "train": cfg.TRAIN_NPY,
        "val": cfg.VAL_NPY,
        "test": cfg.TEST_NPY,
    }[split]
    arr = np.load(path).astype(np.float32)
    obs, rel = make_obs_rel(arr, cfg.PHYS_CHANNELS)
    return arr[..., :cfg.PHYS_CHANNELS], obs, rel


def get_normalizer():
    """归一化器只在训练集上拟合一次，全局复用。"""
    global _NORM
    if _NORM is None:
        arr = np.load(cfg.TRAIN_NPY).astype(np.float32)
        obs, _ = make_obs_rel(arr, cfg.PHYS_CHANNELS)
        _NORM = Normalizer("minmax", [0.5, 99.5], cfg.PHYS_CHANNELS).fit(
            arr, obs)
    return _NORM


def build_windows(split, rate, limit=None):
    phys, obs, rel = _load_raw(split)
    valid = (obs * rel).astype(np.float32)

    norm = get_normalizer()
    clean_all = np.nan_to_num(phys, nan=0.0, posinf=0.0, neginf=0.0)
    xn = norm.transform(clean_all)

    t = phys.shape[0]
    n = t - cfg.INPUT_STEPS - cfg.PRED_STEPS + 1
    if limit is not None:
        n = min(n, int(limit))

    wins = []
    for i in range(n):
        b = i + cfg.INPUT_STEPS
        clean = xn[i:b].copy()
        art = window_sector(
            valid[i:b], rate, window_seed(split, rate, i))
        x = np.where(art[..., None] > 0.5, 0.0, clean).astype(np.float32)
        wins.append({
            "x": x,
            "clean": clean.astype(np.float32),
            "art": art,
            "valid": valid[i:b],
            "y": phys[b:b + cfg.PRED_STEPS],
            "y_valid": valid[b:b + cfg.PRED_STEPS],
        })
    return wins


def future_norm(window):
    """未来帧 V 的归一化值（预测器训练目标），[Tout,H,W]。"""
    norm = get_normalizer()
    y = np.nan_to_num(window["y"], nan=0.0, posinf=0.0, neginf=0.0)
    return norm.transform(y)[..., cfg.TARGET_CHANNEL].astype(np.float32)


def inverse_v(v_norm):
    """把归一化的 V 还原成物理单位（m/s）。"""
    norm = get_normalizer()
    lo, hi = float(norm.lo[0]), float(norm.hi[0])
    return (v_norm + 1.0) * 0.5 * (hi - lo) + lo
