# -*- coding: utf-8 -*-
"""基线 2：空间 KNN（自定义距离 d = Δr + λ·Δθ_wrap，逆距加权 k=8）。

距离设计（物理直觉）：
  Δr：距离门差（米）= |r1-r2| * GATE_STEP_M
  Δθ_wrap：方位角最小环绕差（bin）经半径折算成弧长（米）
           arc ≈ r_mid_m * Δθ_bins * (2π/W)
  d = Δr + arc，权重 w = 1/(d + eps)，取 k=8 近邻加权平均。

用法（项目根）：
    python compare_baselines/spatial_knn/run_knn.py [--smoke] [--k 8] [--lambda_arc 1.0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # compare_baselines

from compare_baselines.common import cfg, data as cdata, pipeline  # noqa: E402


def knn_complete_frame(v_near, obs_near, rel_near, k=8, lambda_arc=1.0):
    """单帧近门 KNN 补全 [Hn, W]。"""
    valid = (obs_near > 0.5) & (rel_near > 0.5)
    H, W = v_near.shape
    out = v_near.copy()

    vr, vc = np.nonzero(valid)
    mr, mc = np.nonzero(~valid)
    if len(vr) == 0 or len(mr) == 0:
        return out

    r_phys = cfg.GATE0_M + cfg.GATE_STEP_M * vr          # 有效点半径 (m)
    dtheta_bins = np.abs(vc[None, :] - mc[:, None])
    dtheta_bins = np.minimum(dtheta_bins, W - dtheta_bins)  # 环绕 [m_gap, bins]
    r_mid_m = cfg.GATE0_M + cfg.GATE_STEP_M * ((mr[:, None] + vr[None, :]) / 2.0)
    arc_m = lambda_arc * r_mid_m * dtheta_bins * (2.0 * np.pi / W)
    radial_m = np.abs(mr[:, None] - vr[None, :]) * cfg.GATE_STEP_M
    dist = radial_m + arc_m                                   # [Nm, Nv]

    kk = min(k, dist.shape[1])
    nn = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
    d_k = np.take_along_axis(dist, nn, axis=1)
    v_k = v_near[vr[nn], vc[nn]]
    w = 1.0 / (d_k + 1e-6)
    filled = (w * v_k).sum(axis=1) / w.sum(axis=1)
    out[mr, mc] = filled

    bad = ~np.isfinite(out)
    if bad.any():
        out = cdata.nearest_fill_nan(out, ~bad)
    return out.astype(np.float32)


def make_complete_fn(k=8, lambda_arc=1.0):
    def complete_fn(v_in, obs_near, rel_near, cond):
        out = np.empty_like(v_in)
        for i in range(v_in.shape[0]):
            out[i] = knn_complete_frame(v_in[i], obs_near[i], rel_near[i],
                                        k=k, lambda_arc=lambda_arc)
        return out
    return complete_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lambda_arc", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    if args.smoke:
        print("[smoke] 2 帧 KNN 自检 ...")
        norm = cdata.load_normalizer_from_ckpt()
        arr, obs, rel = cdata.load_split("test")
        nd = cdata.normalize_phys(arr[:2], norm, obs[:2])
        ns, _ = cdata.near_slices()
        fn = make_complete_fn(args.k, args.lambda_arc)
        comp = fn(nd[:, ns, :, 0], obs[:2, ns], rel[:2, ns], None)
        assert comp.shape == (2, cfg.NEAR_GATES, cfg.ORI_W)
        assert np.isfinite(comp).all()
        print(f"[smoke ok] {time.time() - t0:.1f}s")
        return

    pipeline.run_standard_pipeline(
        "spatial_knn", make_complete_fn(args.k, args.lambda_arc),
        run_meta_extra={"k": args.k, "lambda_arc": args.lambda_arc})
    print(f"总用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
