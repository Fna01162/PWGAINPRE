# -*- coding: utf-8 -*-
"""Ours 基线：分离训练（ours_sep）与联合训练（ours_joint）。

分离：加载 G_best.pth（GAIN+Refiner，冻结）-> 补全近门 -> 训练 CNNRangeExtension
     -> 标准双设定评估（表1 挖洞用 G+R 补全；表2 扩展与其它基线同协议）。
联合：加载 joint_ext_best.pth（G+R+Ext spatiotemporal）-> 直接评估
     （表1 挖洞用 G+R；表2 用其自带 Ext，不重新训练——保留其联合特性）。

用法（项目根）：
    python compare_baselines/ours/run_ours.py --mode sep [--smoke]
    python compare_baselines/ours/run_ours.py --mode joint [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compare_baselines.common import (  # noqa: E402
    cfg, data as cdata, ext_trainer, holes, metrics, pipeline)


def _punch_holes_in_arr(arr, hole):
    """把 [T,Hn,W] 洞写进 arr 副本的 obs 掩码通道（ch7），V 置 0。"""
    out = arr.copy()
    T = hole.shape[0]
    out[:, :cfg.NEAR_GATES, :, 0][hole] = 0.0
    out[:, :cfg.NEAR_GATES, :, 7][hole] = 0.0
    return out


def _complete_with_GR(G, R, cfg_loaded, norm, arr_in, device, use_refiner=True):
    """用 G(+R) 补全整个 [T,H,W,C] 场 -> 物理 V [T,H,W]（obs 保留原值）。"""
    from test import impute_sequence
    out = impute_sequence(G, arr_in, norm, cfg_loaded, device,
                          deterministic=True, R=R if use_refiner else None)
    return out[..., 0]


def run_sep(args):
    from test import load_models
    t0 = time.time()
    method = "ours_sep"
    out_dir = cfg.method_out_dir(method)
    print(f"\n===== [{method}] G_best 冻结补全 + CNNRangeExtension =====")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    G, R, cfg_loaded, ckpt = load_models(str(cfg.CKPT_GAIN), device)
    norm = cdata.load_normalizer_from_ckpt()

    # ── 表1：挖洞补全（G+R 冻结）──
    print("  [表1] 挖洞补全评估 ...")
    arr_te, obs_te, rel_te = cdata.load_split("test")
    masks = holes.load_or_generate_masks(obs_te, rel_te)
    t1_rows = []
    for shape in cfg.HOLE_SHAPES:
        for ratio in cfg.HOLE_RATIOS:
            hole = masks[(shape, ratio)]
            arr_h = _punch_holes_in_arr(arr_te, hole)
            comp_phys = _complete_with_GR(G, R, cfg_loaded, norm, arr_h, device)
            true_phys = arr_te[..., 0]
            m = metrics.completion_metrics(comp_phys, true_phys, hole)
            t1_rows.append({"shape": shape, "ratio": ratio, **m})
            print(f"    [{shape} {ratio:.0%}] MAE={m['MAE']:.3f} n={m['n']}")
    metrics.write_csv(out_dir / f"table1_{method}.csv", t1_rows)

    # ── 表2：近门补全（真实缺失）-> CNNRangeExtension ──
    print("  [表2] 真实缺失补全 + 扩展训练 ...")
    ns, fs = cdata.near_slices()
    splits = []
    for name in ("train", "val", "test"):
        arr, obs, rel = cdata.load_split(name)
        comp_phys = _complete_with_GR(G, R, cfg_loaded, norm, arr, device)
        # 物理补全结果 -> 全场归一化（obs 处保留观测归一化值）
        nd = cdata.normalize_phys(arr, norm, obs)
        phys_full = arr.copy()
        phys_full[..., 0] = comp_phys
        nd2 = cdata.normalize_phys(phys_full, norm, np.ones_like(obs))
        v_near = np.where(obs[:, ns] > 0.5, nd[:, ns, :, 0], nd2[:, ns, :, 0])
        cond = cdata.build_near_cond(nd, obs)
        x = cdata.assemble_ext_input_np(v_near.astype(np.float32), cond,
                                        obs[:, ns], rel[:, ns])
        fv = nd[:, fs, :, 0].astype(np.float32)
        fvm = (obs[:, fs] * rel[:, fs]).astype(np.float32)
        splits.append((x, fv, fvm))
    (x_tr, fv_tr, fvm_tr), (x_va, fv_va, fvm_va), (x_te, fv_te, fvm_te) = splits

    best_val, _ = ext_trainer.train_extension(
        x_tr, fv_tr, fvm_tr, x_va, fv_va, fvm_va, out_dir, device=device,
        epochs=args.ext_epochs, patience=args.ext_patience)
    _eval_ext_table2(method, out_dir, x_te, fv_te, fvm_te, device, best_val)
    print(f"  [{method}] 完成 {time.time()-t0:.0f}s")


def run_joint(args):
    from test_ext import load_joint_ext_models, impute_and_extend
    t0 = time.time()
    method = "ours_joint"
    out_dir = cfg.method_out_dir(method)
    print(f"\n===== [{method}] joint_ext_best.pth 端到端 =====")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    G, R, Ext, cfg_loaded, norm = load_joint_ext_models(str(cfg.CKPT_JOINT_EXT), device)

    # ── 表1：挖洞（G+R 部分）──
    print("  [表1] 挖洞补全评估（联合 G+R）...")
    arr_te, obs_te, rel_te = cdata.load_split("test")
    masks = holes.load_or_generate_masks(obs_te, rel_te)
    t1_rows = []
    for shape in cfg.HOLE_SHAPES:
        for ratio in cfg.HOLE_RATIOS:
            hole = masks[(shape, ratio)]
            arr_h = _punch_holes_in_arr(arr_te, hole)
            comp_phys = _complete_with_GR(G, R, cfg_loaded, norm, arr_h, device)
            m = metrics.completion_metrics(comp_phys, arr_te[..., 0], hole)
            t1_rows.append({"shape": shape, "ratio": ratio, **m})
            print(f"    [{shape} {ratio:.0%}] MAE={m['MAE']:.3f} n={m['n']}")
    metrics.write_csv(out_dir / f"table1_{method}.csv", t1_rows)

    # ── 表2：自带 spatiotemporal Ext 直接评估 ──
    print("  [表2] 自带 Ext 端到端评估 ...")
    res = impute_and_extend(G, R, Ext, arr_te, norm, cfg_loaded, device,
                            test_drop_prob=0.0)
    ext_pred = res["ext_pred"]      # [T, far, W] 归一化
    ext_true = res["ext_true"]
    ext_valid = res["ext_valid"]
    pred_phys = cdata.denorm_v(ext_pred, norm)
    true_phys = cdata.denorm_v(ext_true, norm)
    gate_maes = metrics.far_gate_maes(pred_phys, true_phys, ext_valid, near_end=0)
    reff = metrics.r_eff_km(gate_maes)
    overall = metrics.completion_metrics(pred_phys, true_phys, ext_valid > 0.5)

    t2_rows = []
    for g in range(cfg.FAR_GATES):
        m = ext_valid[:, g] > 0.5
        p, t = pred_phys[:, g][m], true_phys[:, g][m]
        t2_rows.append({
            "gate": cfg.NEAR_GATES + g,
            "range_km": round(cfg.gate_dist_km(cfg.NEAR_GATES + g), 2),
            "MAE": float(np.mean(np.abs(p - t))) if m.any() else np.nan,
            "RMSE": float(np.sqrt(np.mean((p - t) ** 2))) if m.any() else np.nan,
            "Bias": float(np.mean(p - t)) if m.any() else np.nan,
            "Corr": float(np.corrcoef(p, t)[0, 1]) if m.any() and p.std() > 1e-8 and t.std() > 1e-8 else np.nan,
            "n": int(m.sum()),
        })
    t2_rows.append({"gate": "overall", "range_km": "", "MAE": overall["MAE"],
                    "RMSE": overall["RMSE"], "Bias": overall["Bias"],
                    "Corr": overall["Corr"], "n": overall["n"]})
    t2_rows.append({"gate": "R_eff_km", "range_km": "", "MAE": round(reff, 2),
                    "RMSE": "", "Bias": "", "Corr": "", "n": ""})
    t2_rows.append({"gate": "ext_model", "range_km": "",
                    "MAE": cfg_loaded.ext_model_type, "RMSE": "", "Bias": "",
                    "Corr": "", "n": ""})
    metrics.write_csv(out_dir / f"table2_{method}.csv", t2_rows)

    import json
    mp = out_dir / "run_meta.json"
    meta = {"method": method, "ckpt": str(cfg.CKPT_JOINT_EXT),
            "far_overall_MAE": overall["MAE"], "R_eff_km": reff,
            "ext_model": cfg_loaded.ext_model_type,
            "elapsed_s": round(time.time() - t0, 1)}
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{method}] R_eff={reff:.2f}km farMAE={overall['MAE']:.3f} "
          f"用时{meta['elapsed_s']}s")


def _eval_ext_table2(method, out_dir, x_te, fv_te, fvm_te, device, best_val):
    """与 pipeline.run_standard_pipeline 的表2 部分一致。"""
    pred_norm, ms = ext_trainer.eval_extension(
        out_dir / "ext_best.pth", x_te, fv_te, fvm_te, device=device)
    # 需要 normalizer 反归一化
    norm = cdata.load_normalizer_from_ckpt()
    fv_phys = cdata.denorm_v(fv_te, norm)
    pred_phys = cdata.denorm_v(pred_norm, norm)
    gate_maes = metrics.far_gate_maes(pred_phys, fv_phys, fvm_te, near_end=0)
    reff = metrics.r_eff_km(gate_maes)
    overall = metrics.completion_metrics(pred_phys, fv_phys, fvm_te > 0.5)
    rows = []
    for g in range(cfg.FAR_GATES):
        m = fvm_te[:, g] > 0.5
        p, t = pred_phys[:, g][m], fv_phys[:, g][m]
        rows.append({"gate": cfg.NEAR_GATES + g,
                     "range_km": round(cfg.gate_dist_km(cfg.NEAR_GATES + g), 2),
                     "MAE": float(np.mean(np.abs(p - t))),
                     "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
                     "Bias": float(np.mean(p - t)),
                     "Corr": float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 and t.std() > 1e-8 else np.nan,
                     "n": int(m.sum())})
    rows.append({"gate": "overall", "range_km": "", "MAE": overall["MAE"],
                 "RMSE": overall["RMSE"], "Bias": overall["Bias"],
                 "Corr": overall["Corr"], "n": overall["n"]})
    rows.append({"gate": "R_eff_km", "range_km": "", "MAE": round(reff, 2),
                 "RMSE": "", "Bias": "", "Corr": "", "n": ""})
    rows.append({"gate": "ext_val_MAE_norm", "range_km": "", "MAE": best_val,
                 "RMSE": "", "Bias": "", "Corr": "", "n": ""})
    rows.append({"gate": "ext_ms_per_frame", "range_km": "",
                 "MAE": round(ms, 2), "RMSE": "", "Bias": "", "Corr": "", "n": ""})
    metrics.write_csv(out_dir / f"table2_{method}.csv", rows)
    print(f"  [表2] R_eff={reff:.2f}km farMAE={overall['MAE']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sep", "joint"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ext-epochs", type=int, default=None)
    ap.add_argument("--ext-patience", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        print("[smoke] Ours 模式加载检查 ...")
        if args.mode == "sep":
            from test import load_models
            device = "cpu"
            G, R, cfg_loaded, ckpt = load_models(str(cfg.CKPT_GAIN), device)
            print("[smoke] G_best 加载 ok")
        else:
            from test_ext import load_joint_ext_models
            G, R, Ext, cfg_loaded, norm = load_joint_ext_models(
                str(cfg.CKPT_JOINT_EXT), "cpu")
            print("[smoke] joint_ext_best 加载 ok")
        return
    if args.mode == "sep":
        run_sep(args)
    else:
        run_joint(args)


if __name__ == "__main__":
    main()
