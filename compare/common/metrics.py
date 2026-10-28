# -*- coding: utf-8 -*-
"""补全 / 预测指标：MAE、RMSE、SSIM。

SSIM 是带掩码的版本：每个 7x7 窗口只统计有效像素，窗口有效像素不足一半就丢弃，
避免把真实缺失区域算进去。输入均为 [T,H,W]。
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def mae_rmse(pred, target, mask):
    m = mask > 0.5
    p = pred[m].astype(np.float64)
    t = target[m].astype(np.float64)
    if p.size == 0:
        return float("nan"), float("nan")
    e = p - t
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def _ssim_frame(p, t, valid, win=7, k1=0.01, k2=0.03, data_range=1.0):
    pad = win // 2
    pp = np.pad(p, pad, mode="edge")
    tt = np.pad(t, pad, mode="edge")
    mm = np.pad(valid, pad, mode="constant", constant_values=0.0)

    vp = sliding_window_view(pp, (win, win))
    vt = sliding_window_view(tt, (win, win))
    vm = sliding_window_view(mm, (win, win))
    wsum = vm.sum(axis=(-2, -1))
    ok = wsum >= (win * win * 0.5)

    def wmean(a):
        return (a * vm).sum(axis=(-2, -1)) / np.maximum(wsum, 1e-8)

    mu_p = wmean(vp)
    mu_t = wmean(vt)
    var_p = wmean(vp * vp) - mu_p ** 2
    var_t = wmean(vt * vt) - mu_t ** 2
    cov = wmean(vp * vt) - mu_p * mu_t

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    s = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    s = s[ok]
    return float(np.mean(s)) if s.size else float("nan")


def ssim(pred, target, mask):
    m = mask > 0.5
    valid = m.astype(np.float64)
    pp = pred.astype(np.float64)
    tt = target.astype(np.float64)
    if m.any():
        data_range = float(tt[m].max() - tt[m].min())
    else:
        data_range = 1.0
    data_range = max(data_range, 1e-6)

    vals = []
    for i in range(pred.shape[0]):
        vals.append(_ssim_frame(pp[i], tt[i], valid[i], data_range=data_range))
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def all_metrics(pred, target, mask):
    mae, rmse = mae_rmse(pred, target, mask)
    return {"MAE": mae, "RMSE": rmse, "SSIM": ssim(pred, target, mask)}
