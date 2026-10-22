# -*- coding: utf-8 -*-
"""官方 GAIN（2D 卷积改造）完整 runner。

流程：训练 G2D 补全网络 -> 冻结 -> 标准 pipeline（扩展训练 + 双设定评估）。

用法（项目根）：
    python compare_baselines/official_gain/run_gain.py [--smoke] [--epochs N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # compare_baselines

from compare_baselines.common import cfg, data as cdata, deep_comp, pipeline  # noqa: E402
from compare_baselines.official_gain.gain2d import build_gain2d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 帧 + 1 epoch 自检")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.time()

    out_dir = cfg.method_out_dir("official_gain")

    if args.smoke:
        print("[smoke] GAIN-2D 前向 + 1 epoch 自检 ...")
        import torch
        G, D = build_gain2d()
        x = torch.randn(2, 6, cfg.NEAR_GATES, cfg.ORI_W)
        y = G(x)
        assert y.shape == (2, 1, cfg.NEAR_GATES, cfg.ORI_W), y.shape
        n = sum(p.numel() for p in G.parameters())
        print(f"[smoke] G params={n:,}  out={tuple(y.shape)}")
        norm = cdata.load_normalizer_from_ckpt()
        arr, obs, rel = cdata.load_split("val")
        v, cond, obs_n, rel_n = deep_comp.arrays_for_split(arr[:4], obs[:4], rel[:4], norm)
        G2, D2 = build_gain2d()
        G2 = deep_comp.train_completion(G2, D2, (v, cond, obs_n, rel_n),
                                        (v, cond, obs_n, rel_n),
                                        out_dir / "smoke", "cpu", epochs=1, patience=1)
        fn = deep_comp.make_complete_fn_from_G(G2, "cpu")
        comp = fn(v, obs_n, rel_n, cond)
        assert comp.shape == v.shape
        print(f"[smoke ok] {time.time()-t0:.1f}s")
        return

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    norm = cdata.load_normalizer_from_ckpt()
    arr_tr, obs_tr, rel_tr = cdata.load_split("train")
    arr_va, obs_va, rel_va = cdata.load_split("val")
    tr = deep_comp.arrays_for_split(arr_tr, obs_tr, rel_tr, norm)
    va = deep_comp.arrays_for_split(arr_va, obs_va, rel_va, norm)

    G, D = build_gain2d()
    G = deep_comp.train_completion(G, D, tr, va, out_dir, device,
                                   epochs=args.epochs, patience=args.patience)
    complete_fn = deep_comp.make_complete_fn_from_G(G, device)

    pipeline.run_standard_pipeline(
        "official_gain", complete_fn, device=device,
        run_meta_extra={
            "completion_net": "Gain2DGenerator (conv adaptation of jsyoon0823/GAIN)",
            "losses": "official GAIN: D=BCE(D(Hat_X,H),M); G=adv+10*MSE",
            "G_params": sum(p.numel() for p in G.parameters()),
        })
    print(f"总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
