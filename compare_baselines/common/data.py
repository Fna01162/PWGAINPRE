# -*- coding: utf-8 -*-
"""数据装载、归一化、扩展输入组装。

归一化统计量只从主 ckpt G_best.pth 读取，禁止重拟合（方案公平性要求）。
扩展输入组装与 train_joint_ext.assemble_extension_input_from_completed 一致：
    [completed_v(1), CNR/SW/Skew 归一化并按obs掩零(3), obs(1), rel(1)] = 6ch
"""
import sys
from pathlib import Path

import numpy as np
import torch

from . import cfg

if str(cfg.ROOT) not in sys.path:
    sys.path.insert(0, str(cfg.ROOT))

from untils.data import Normalizer, make_obs_rel, BAD_RAYS  # noqa: E402


def load_normalizer_from_ckpt(ckpt_path=None):
    """从主 ckpt 还原 Normalizer（不重拟合）。"""
    ckpt_path = Path(ckpt_path) if ckpt_path else cfg.CKPT_GAIN
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ckpt["normalizer"]
    norm = Normalizer(str(st.get("method", cfg.NORMALIZE_METHOD)),
                      float(st.get("clip_percentile", cfg.CLIP_PERCENTILE)),
                      int(st.get("num_channels", cfg.PHYS_CHANNELS)))
    norm.__dict__.update(st)
    return norm


def load_split(name="test", npy_path=None):
    """加载 split：返回 arr[T,H,W,C]、obs、rel（float32 掩码）。"""
    p = Path(npy_path) if npy_path else {
        "train": cfg.TRAIN_NPY, "val": cfg.VAL_NPY, "test": cfg.TEST_NPY
    }[name]
    arr = np.load(p).astype(np.float32)
    obs, rel = make_obs_rel(arr, cfg.PHYS_CHANNELS)
    return arr, obs, rel


def normalize_phys(arr, norm, obs=None):
    """物理 [T,H,W,C] -> 归一化，obs=0 处置 0（与 test.py 一致）。"""
    phys = np.nan_to_num(arr[..., :cfg.PHYS_CHANNELS].astype(np.float32), nan=0.0)
    out = norm.transform(phys)
    if obs is not None:
        out = np.where(obs[..., None] == 1, out, 0.0)
    return out.astype(np.float32)


def denorm_v(v_norm, norm):
    """归一化 V -> 物理。支持 [...,] 任意形状，末维补通道后 inverse。"""
    v = np.asarray(v_norm, dtype=np.float32)
    stacked = np.stack([v, np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)], axis=-1)
    return norm.inverse_transform(stacked)[..., 0]


def assemble_ext_input_np(v_near_norm, cond_near_norm, obs_near, rel_near):
    """numpy 版扩展输入组装（与 assemble_extension_input_from_completed 一致）。

    v_near_norm    : [N, Hn, W]  补全后近门 V（归一化，obs 处保留原值）
    cond_near_norm : [N, 3, Hn, W] CNR/SW/Skew 归一化并按 obs 掩零
    obs_near       : [N, Hn, W]
    rel_near       : [N, Hn, W]
    返回 [N, 6, Hn, W]
    """
    return np.concatenate([
        v_near_norm[:, None, ...],
        cond_near_norm,
        obs_near[:, None, ...],
        rel_near[:, None, ...],
    ], axis=1).astype(np.float32)


def build_near_cond(norm_data, obs):
    """从归一化全场提取近门条件通道（CNR/SW/Skew，按 obs 掩零）。"""
    near = norm_data[:, :cfg.NEAR_GATES, :, 1:cfg.PHYS_CHANNELS]  # [N,Hn,W,3]
    near = near * obs[:, :cfg.NEAR_GATES, :, None]
    return np.transpose(near, (0, 3, 1, 2)).astype(np.float32)  # [N,3,Hn,W]


def oracle_complete_near(v_near_norm, obs_near):
    """obs 处保留原值、缺失处填 0（no_completion 基线 / 深度基线 oracle 保留）。"""
    return np.where(obs_near > 0.5, v_near_norm, 0.0).astype(np.float32)


def nearest_fill_nan(field2d, valid_mask):
    """最近邻填充 2D 场中的无效点（传统方法兜底）。"""
    from scipy import ndimage
    out = field2d.copy()
    invalid = ~valid_mask
    if not invalid.any() or valid_mask.all():
        return out
    _, idx = ndimage.distance_transform_edt(invalid, return_indices=True)
    out[invalid] = field2d[idx[0][invalid], idx[1][invalid]]
    return out


def near_slices():
    return slice(0, cfg.NEAR_GATES), slice(cfg.NEAR_GATES, cfg.NEAR_GATES + cfg.FAR_GATES)


def bad_rays_mask():
    """坏射线掩码 [W]（BAD_RAYS 在 ORI 坐标系）。"""
    m = np.zeros(cfg.ORI_W, dtype=np.float32)
    for b in BAD_RAYS:
        if 0 <= b < cfg.ORI_W:
            m[b] = 1.0
    return m
