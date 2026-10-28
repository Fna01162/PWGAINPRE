# -*- coding: utf-8 -*-
"""非训练基线：零填充、方位向线性插值、空间 KNN。逐帧处理 V 通道。

输入：
    x     : [T,H,W] 归一化 V，遮挡处置 0
    valid : [T,H,W] 真实有效区域
输出：补全后的 [T,H,W]
"""
import numpy as np
from scipy.spatial import cKDTree

from common import cfg


def zero_fill(x):
    return x


def _linear_azimuth(frame, valid):
    """沿方位角（列，循环）线性插值；整行无有效值时先留空，交给径向兜底。"""
    h, w = frame.shape
    out = frame.copy()
    for r in range(h):
        cols = np.where(valid[r])[0]
        if cols.size == 0:
            continue
        if cols.size == 1:
            out[r, :] = frame[r, cols[0]]
            continue
        # 方位向循环：把已知列复制到 [-W, 0, +W]
        cols_ext = np.concatenate([cols - w, cols, cols + w])
        vals_ext = np.tile(frame[r, cols], 3)
        for c in np.where(~valid[r])[0]:
            left = cols_ext[cols_ext <= c].max()
            right = cols_ext[cols_ext >= c].min()
            if right == left:
                out[r, c] = vals_ext[cols_ext == left][0]
            else:
                ratio = (c - left) / (right - left)
                out[r, c] = (
                    vals_ext[cols_ext == left][0] * (1 - ratio)
                    + vals_ext[cols_ext == right][0] * ratio
                )
    return out


def _radial_fallback(out, valid):
    """整行无有效值时，沿距离向（行）线性插值。"""
    h = out.shape[0]
    rows_with = np.where(valid.any(axis=1))[0]
    if rows_with.size == 0:
        return out
    for r in range(h):
        if valid[r].any():
            continue
        ri = np.searchsorted(rows_with, r)
        up = rows_with[max(0, ri - 1)]
        dn = rows_with[min(rows_with.size - 1, ri)]
        ratio = (r - up) / max(dn - up, 1)
        out[r] = out[up] * (1 - ratio) + out[dn] * ratio
    return out


def linear_interp(x, valid):
    out = []
    for i in range(x.shape[0]):
        v = valid[i] > 0.5
        f = _linear_azimuth(x[i], v)
        f = _radial_fallback(f, v)
        out.append(f)
    return np.stack(out).astype(np.float32)


def _knn_frame(frame, valid, k):
    h, w = frame.shape
    obs = np.argwhere(valid)
    if obs.shape[0] == 0:
        return frame
    # 方位向循环：把已知点复制到 ±W
    pts = np.concatenate([
        obs + np.array([0, w]),
        obs,
        obs - np.array([0, w]),
    ], axis=0)
    vals = np.tile(frame[valid], 3)
    tree = cKDTree(pts)

    out = frame.copy()
    miss = np.argwhere(~valid)
    if miss.shape[0] == 0:
        return out
    dist, idx = tree.query(miss, k=min(k, pts.shape[0]))
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    wgt = 1.0 / np.maximum(dist, 1e-6)
    wsum = wgt.sum(axis=1)
    out[miss[:, 0], miss[:, 1]] = (wgt * vals[idx]).sum(axis=1) / wsum
    return out


def knn(x, valid, k=None):
    if k is None:
        k = cfg.KNN_K
    out = []
    for i in range(x.shape[0]):
        out.append(_knn_frame(x[i], valid[i] > 0.5, k))
    return np.stack(out).astype(np.float32)
