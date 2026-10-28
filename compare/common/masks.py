# -*- coding: utf-8 -*-
"""扇区遮挡掩码生成。

扇区 = 一段连续的方位射线（W 维，列），方位向首尾相接，所以射线按循环取。
这对应真实的硬件坏射线（BAD_RAYS），比随机撒像素更贴近实际。
掩码只落在真实有效区域上，按目标缺失率校准，固定随机种子并缓存到 npy。
"""
import numpy as np

from . import cfg


def _wrap_cols(start, n, width):
    return (np.arange(n) + start) % width


def _one_frame(allowed, rate, rng):
    """在单帧 allowed 上生成连续射线扇区，尽量逼近 rate。"""
    h, w = allowed.shape
    ray_counts = allowed.sum(axis=0).astype(np.float64)
    total = ray_counts.sum()
    if total <= 0:
        return np.zeros((h, w), np.float32)

    target = rate * total
    best = None
    for _ in range(40):
        start = int(rng.integers(0, w))
        order = _wrap_cols(start, w, w)
        cum = np.cumsum(ray_counts[order])
        k = int(np.searchsorted(cum, target) + 1)
        k = max(1, min(k, w))
        m = np.zeros((h, w), np.float32)
        m[:, order[:k]] = 1.0
        m *= allowed
        got = float(m.sum())
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (m, got)
        if abs(got - target) <= max(1.0, 0.01 * total):
            break
    return best[0]


def window_sector(valid_window, rate, seed):
    """valid_window [T,H,W]：生成一个固定扇区，窗口内所有帧遮住相同射线。

    只遮挡在窗口所有帧都有效的射线，保证遮挡位置都有真值；固定扇区对应
    硬件物理遮挡（如 BAD_RAYS），而不是每帧随机跳变。
    """
    rng = np.random.default_rng(seed)
    allowed = np.all(valid_window > 0.5, axis=0)
    art0 = _one_frame(allowed, rate, rng)
    return np.repeat(
        art0[None].astype(np.float32), valid_window.shape[0], axis=0)


def window_seed(split, rate, window_index):
    """每个窗口的确定性种子（避免使用 hash()）。"""
    key = int(round(rate * 100))
    split_seed = {"train": 101, "val": 202, "test": 303}[split]
    return cfg.SEED + split_seed + key + int(window_index) * 7919


def actual_rate(art, valid):
    """实际遮挡率（相对真实有效区域）。"""
    v = valid > 0.5
    return float(art[v].sum()) / max(v.sum(), 1)
