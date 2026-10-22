# -*- coding: utf-8 -*-
"""标准基线 pipeline：补全 -> 组装扩展输入 -> 训练 CNNRangeExtension -> 双设定评估。

各传统/深度基线只需提供 complete_fn（缺失填充函数）与可选的
train_complete_fn（深度基线训练补全网络），其余流程完全一致：

  1. 冻结补全器，对 train/val/test 的真实缺失（obs=0）做补全（obs 处 oracle 保留）
  2. 组装扩展输入 [N,6,Hn,W]，训练 CNNRangeExtension（协议同 train_extension.py）
  3. 设定 A（表1）：test 健康区挖 4 形态 × 15%/30% 洞，补全后在洞像素算物理 MAE/RMSE/Bias/Corr
  4. 设定 B（表2）：远门逐门物理 MAE/RMSE/Bias/Corr + R_eff（tau=1.0 m/s）

输出到 compare_baselines/outputs/{method}/：
  table1_{method}.csv  table2_{method}.csv  ext_best.pth  ext_history.csv  run_meta.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import cfg, data as cdata, ext_trainer, holes, metrics


def _complete_split(v_near, obs_near, rel_near, cond, complete_fn):
    """对一个 split 完成近门补全（obs 处保留原值）。"""
    v_in = np.where(obs_near > 0.5, v_near, 0.0).astype(np.float32)
    filled = complete_fn(v_in, obs_near, rel_near, cond)
    return np.where(obs_near > 0.5, v_near, filled).astype(np.float32)


def _build_ext_arrays(v_completed, norm_data, obs, rel):
    """由补全后近门 V 组装扩展输入与远门目标（归一化空间）。"""
    ns, fs = cdata.near_slices()
    N = v_completed.shape[0]
    cond = cdata.build_near_cond(norm_data, obs)  # [N,3,Hn,W]
    x = cdata.assemble_ext_input_np(v_completed, cond, obs[:, ns], rel[:, ns])
    far_v = norm_data[:, fs, :, 0]  # [N,FarG,W]
    far_valid = (obs[:, fs] * rel[:, fs]).astype(np.float32)
    return x, far_v.astype(np.float32), far_valid


def run_standard_pipeline(method, complete_fn, train_complete_fn=None,
                          device=None, ext_epochs=None, ext_patience=None,
                          run_meta_extra=None, punch_holes_fn=None):
    """标准基线流程。punch_holes_fn 供 Ours 类方法复用（表1 挖洞补全）。"""
    t_start = time.time()
    out_dir = cfg.method_out_dir(method)
    print(f"\n===== [{method}] pipeline =====")
    ext_trainer.set_seed(cfg.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    norm = cdata.load_normalizer_from_ckpt()
    arr_tr, obs_tr, rel_tr = cdata.load_split("train")
    arr_va, obs_va, rel_va = cdata.load_split("val")
    arr_te, obs_te, rel_te = cdata.load_split("test")

    nd_tr = cdata.normalize_phys(arr_tr, norm, obs_tr)
    nd_va = cdata.normalize_phys(arr_va, norm, obs_va)
    nd_te = cdata.normalize_phys(arr_te, norm, obs_te)

    ns, fs = cdata.near_slices()

    if train_complete_fn is not None:
        print("  [1/4] 训练补全网络 ...")
        train_complete_fn(arr_tr, obs_tr, rel_tr, arr_va, obs_va, rel_va,
                          norm, out_dir, device)

    print("  [2/4] 冻结补全器补全真实缺失 + 组装扩展输入 ...")
    splits = []
    for arr, obs, rel, nd in ((arr_tr, obs_tr, rel_tr, nd_tr),
                              (arr_va, obs_va, rel_va, nd_va),
                              (arr_te, obs_te, rel_te, nd_te)):
        v_near = nd[:, ns, :, 0]
        v_comp = _complete_split(v_near, obs[:, ns], rel[:, ns],
                                 cdata.build_near_cond(nd, obs), complete_fn)
        x, fv, fvm = _build_ext_arrays(v_comp, nd, obs, rel)
        splits.append((x, fv, fvm))
    (x_tr, fv_tr, fvm_tr), (x_va, fv_va, fvm_va), (x_te, fv_te, fvm_te) = splits

    print("  [3/4] 训练 CNNRangeExtension ...")
    best_val, _ = ext_trainer.train_extension(
        x_tr, fv_tr, fvm_tr, x_va, fv_va, fvm_va, out_dir,
        device=device, epochs=ext_epochs, patience=ext_patience)

    print("  [4/4] 评估（设定 A 补全 + 设定 B 扩展） ...")
    # ── 设定 B：远门扩展 ──
    pred_norm, ms_per_frame = ext_trainer.eval_extension(
        out_dir / "ext_best.pth", x_te, fv_te, fvm_te, device=device)
    v_te_phys = cdata.denorm_v(nd_te[..., 0], norm)
    fv_phys = cdata.denorm_v(fv_te, norm)
    pred_phys_far = cdata.denorm_v(pred_norm, norm)
    gate_maes = metrics.far_gate_maes(pred_phys_far, fv_phys, fvm_te, near_end=0)
    reff = metrics.r_eff_km(gate_maes)
    overall = metrics.completion_metrics(pred_phys_far, fv_phys,
                                         fvm_te > 0.5)
    t2_rows = []
    for g in range(cfg.FAR_GATES):
        m = fvm_te[:, g] > 0.5
        p, t = pred_phys_far[:, g][m], fv_phys[:, g][m]
        t2_rows.append({
            "gate": cfg.NEAR_GATES + g,
            "range_km": round(cfg.gate_dist_km(cfg.NEAR_GATES + g), 2),
            "MAE": float(np.mean(np.abs(p - t))),
            "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
            "Bias": float(np.mean(p - t)),
            "Corr": float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 and t.std() > 1e-8 else np.nan,
            "n": int(m.sum()),
        })
    t2_rows.append({"gate": "overall", "range_km": "", "MAE": overall["MAE"],
                    "RMSE": overall["RMSE"], "Bias": overall["Bias"],
                    "Corr": overall["Corr"], "n": overall["n"]})
    t2_rows.append({"gate": "R_eff_km", "range_km": "", "MAE": round(reff, 2),
                    "RMSE": "", "Bias": "", "Corr": "", "n": ""})
    t2_rows.append({"gate": "ext_val_MAE_norm", "range_km": "", "MAE": best_val,
                    "RMSE": "", "Bias": "", "Corr": "", "n": ""})
    t2_rows.append({"gate": "ext_ms_per_frame", "range_km": "",
                    "MAE": round(ms_per_frame, 2), "RMSE": "", "Bias": "", "Corr": "", "n": ""})
    metrics.write_csv(out_dir / f"table2_{method}.csv", t2_rows)

    # ── 设定 A：人工洞补全 ──
    if punch_holes_fn is not None:
        t1_rows = punch_holes_fn(nd_te, obs_te, rel_te, norm, method, out_dir)
    else:
        t1_rows = _eval_holes_standard(nd_te, obs_te, rel_te, norm, complete_fn)
    if t1_rows:
        metrics.write_csv(out_dir / f"table1_{method}.csv", t1_rows)

    meta = {"method": method, "best_ext_val_MAE_norm": best_val,
            "far_overall_MAE": overall["MAE"], "far_overall_RMSE": overall["RMSE"],
            "R_eff_km": reff, "ext_ms_per_frame": round(ms_per_frame, 2),
            "elapsed_s": round(time.time() - t_start, 1)}
    if run_meta_extra:
        meta.update(run_meta_extra)
    old = {}
    mp = out_dir / "run_meta.json"
    if mp.exists():
        try:
            old = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old.update(meta)
    mp.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{method}] 完成：R_eff={reff:.2f}km farMAE={overall['MAE']:.3f} "
          f"extVal={best_val:.4f} 用时{meta['elapsed_s']}s")
    return meta


def _eval_holes_standard(nd_te, obs_te, rel_te, norm, complete_fn):
    """设定 A：对每个 (shape, ratio) 挖洞 -> 补全 -> 洞像素物理指标。"""
    masks = holes.load_or_generate_masks(obs_te, rel_te)
    ns, _ = cdata.near_slices()
    rows = []
    for shape in cfg.HOLE_SHAPES:
        for ratio in cfg.HOLE_RATIOS:
            hole = masks[(shape, ratio)]  # [T,Hn,W] bool
            obs_near = obs_te[:, ns]
            # 挖洞：洞内 V 置 0、obs 置 0（健康区洞，rel 保持）
            v_in = nd_te[:, ns, :, 0].copy()
            v_in[hole] = 0.0
            obs_h = obs_near.copy()
            obs_h[hole] = 0.0
            rel_h = rel_te[:, ns].copy()
            cond = cdata.build_near_cond(nd_te, obs_te).copy()
            for ch in range(3):
                ch_data = cond[:, ch]
                ch_data[hole] = 0.0
                cond[:, ch] = ch_data
            filled = complete_fn(v_in, obs_h, rel_h, cond)
            comp = np.where(obs_h > 0.5, v_in, filled)
            pred_phys = cdata.denorm_v(comp, norm)
            true_phys = cdata.denorm_v(nd_te[:, ns, :, 0], norm)
            m = metrics.completion_metrics(pred_phys, true_phys, hole)
            rows.append({"shape": shape, "ratio": ratio, **m})
            print(f"    [表1] {shape} {ratio:.0%}: MAE={m['MAE']:.3f} "
                  f"RMSE={m['RMSE']:.3f} n={m['n']}")
    return rows


def average_table1(method, out_dir=None):
    """表2 近门补全 MAE = 表1 平均（真实缺失无真值，方案约定）。"""
    out_dir = Path(out_dir or cfg.method_out_dir(method))
    p = out_dir / f"table1_{method}.csv"
    if not p.exists():
        return None
    import csv as _csv
    with open(p, encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
    maes = [float(r["MAE"]) for r in rows if r.get("MAE") not in ("", None)]
    return float(np.mean(maes)) if maes else None
