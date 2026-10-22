# -*- coding: utf-8 -*-
"""基线 1：方位周期线性插值。

缺失点分两遍填充：
  1. 方位向（处理 0°/360° 周期）：np.unwrap 相位法 + interp1d
  2. 径向：普通线性插值
  3. 兜底：最近邻（scipy distance_transform）

用法（项目根）：
    python compare_baselines/linear_interp/run_linear.py [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # compare_baselines

from compare_baselines.common import cfg, data as cdata, pipeline  # noqa: E402


def _interp_1d_periodic(values, valid, period):
    """周期轴上的线性插值：把值映射到复相位 unwrap 后 interp1d。

    values: [W]；valid: [W] bool；period: 周期长度（=W，方位 bins）。
    返回 [W]，valid 处保留原值。
    """
    from scipy.interpolate import interp1d
    out = values.copy()
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[~valid] = values[idx[0]]
        return out
    # 展开：在有效点两侧复制一份以覆盖周期边界
    ext_idx = np.concatenate([idx - period, idx, idx + period])
    ext_vals = np.tile(values[idx], 3)
    order = np.argsort(ext_idx)
    f = interp1d(ext_idx[order], ext_vals[order], kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    out[~valid] = f(np.flatnonzero(~valid))
    return out


def _interp_linear_1d(values, valid):
    """非周期轴线性插值，端点外推用最近有效值。"""
    from scipy.interpolate import interp1d
    out = values.copy()
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[~valid] = values[idx[0]]
        return out
    f = interp1d(idx, values[idx], kind="linear",
                 bounds_error=False, fill_value=(values[idx[0]], values[idx[-1]]))
    out[~valid] = f(np.flatnonzero(~valid))
    return out


def linear_complete_frame(v_near, obs_near, rel_near):
    """单帧近门补全 [Hn, W]。有效 = obs∧rel；先方位后径向，最后最近邻兜底。"""
    valid = (obs_near > 0.5) & (rel_near > 0.5)
    H, W = v_near.shape
    out = v_near.copy()

    # 1) 方位向（周期）逐门
    for r in range(H):
        out[r] = _interp_1d_periodic(v_near[r], valid[r], W)

    # 2) 径向逐列（仍缺的点用上下有效门；此时方位已填，扩展有效集）
    valid2 = valid | ~np.isnan(out)
    for c in range(W):
        out[:, c] = _interp_linear_1d(out[:, c], valid2[:, c])

    # 3) 兜底：任何残余 NaN -> 最近邻
    bad = ~np.isfinite(out)
    if bad.any():
        out = cdata.nearest_fill_nan(out, ~bad)
    return out.astype(np.float32)


def make_complete_fn():
    """返回 complete_fn(v_in[N,Hn,W], obs, rel, cond) -> completed[N,Hn,W]。"""
    def complete_fn(v_in, obs_near, rel_near, cond):
        out = np.empty_like(v_in)
        for i in range(v_in.shape[0]):
            out[i] = linear_complete_frame(v_in[i], obs_near[i], rel_near[i])
        return out
    return complete_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 2 帧快速自检")
    args = ap.parse_args()

    t0 = time.time()
    if args.smoke:
        print("[smoke] 2 帧线性插值自检 ...")
        norm = cdata.load_normalizer_from_ckpt()
        arr, obs, rel = cdata.load_split("test")
        nd = cdata.normalize_phys(arr[:2], norm, obs[:2])
        ns, _ = cdata.near_slices()
        fn = make_complete_fn()
        comp = fn(nd[:, ns, :, 0], obs[:2, ns], rel[:2, ns], None)
        assert comp.shape == (2, cfg.NEAR_GATES, cfg.ORI_W)
        assert np.isfinite(comp).all()
        print(f"[smoke ok] {time.time() - t0:.1f}s")
        return

    pipeline.run_standard_pipeline("linear_interp", make_complete_fn())
    print(f"总用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
