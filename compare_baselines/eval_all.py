# -*- coding: utf-8 -*-
"""汇总所有基线结果：表1（设定 A 补全）+ 表2（设定 B 扩展）。

用法（项目根）：
    python compare_baselines/eval_all.py
输出：
    compare_baselines/outputs/summary_table1.csv
    compare_baselines/outputs/summary_table2.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_baselines.common import cfg  # noqa: E402

ORDER = ("no_completion", "linear_interp", "spatial_knn", "official_gain",
         "dsa_unet", "unetpp_cgan", "ours_sep", "ours_joint")

CN = {
    "no_completion": "无补全 (直接外推)",
    "linear_interp": "线性插值",
    "spatial_knn": "空间 KNN",
    "official_gain": "官方 GAIN (2D)",
    "dsa_unet": "DSA-UNet",
    "unetpp_cgan": "U-Net++ cGAN",
    "ours_sep": "本文补全 (分离训练)",
    "ours_joint": "本文完整框架 (联合训练)",
}


def _read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _mean_table1(method):
    p = cfg.OUTPUTS_DIR / method / f"table1_{method}.csv"
    if not p.exists():
        return None
    rows = _read_csv(p)
    maes = [float(r["MAE"]) for r in rows if r.get("MAE") not in ("", "-", None)]
    rmses = [float(r["RMSE"]) for r in rows if r.get("RMSE") not in ("", "-", None)]
    if not maes:
        return None
    return {"MAE_mean": float(np.mean(maes)), "RMSE_mean": float(np.mean(rmses))}


def _table2_overall(method):
    p = cfg.OUTPUTS_DIR / method / f"table2_{method}.csv"
    if not p.exists():
        return None
    rows = _read_csv(p)
    out = {}
    for r in rows:
        g = str(r["gate"])
        if g == "overall":
            out["far_MAE"] = float(r["MAE"])
            out["far_RMSE"] = float(r["RMSE"])
        elif g == "R_eff_km":
            out["R_eff_km"] = float(r["MAE"])
    return out if "R_eff_km" in out else None


def _per_gate(method):
    p = cfg.OUTPUTS_DIR / method / f"table2_{method}.csv"
    if not p.exists():
        return {}
    rows = _read_csv(p)
    return {f"gate_{r['gate']}": float(r["MAE"]) for r in rows
            if str(r["gate"]).isdigit()}


def main():
    t1_rows, t2_rows = [], []
    for m in ORDER:
        name = CN.get(m, m)
        t1 = _mean_table1(m)
        t1_rows.append({
            "method": name, "key": m,
            "MAE_mean": t1["MAE_mean"] if t1 else "-",
            "RMSE_mean": t1["RMSE_mean"] if t1 else "-",
        })
        t2 = _table2_overall(m)
        pg = _per_gate(m)
        row = {"method": name, "key": m,
               "near_comp_MAE": t1["MAE_mean"] if t1 else "-",
               "far_MAE": t2.get("far_MAE", "-") if t2 else "-",
               "far_RMSE": t2.get("far_RMSE", "-") if t2 else "-",
               "R_eff_km": t2.get("R_eff_km", "-") if t2 else "-"}
        row.update({k: v for k, v in pg.items()})
        t2_rows.append(row)

    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    p1 = cfg.OUTPUTS_DIR / "summary_table1.csv"
    p2 = cfg.OUTPUTS_DIR / "summary_table2.csv"
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(t1_rows[0].keys()))
        w.writeheader(); w.writerows(t1_rows)
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        keys = list(t2_rows[0].keys())
        for r in t2_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(t2_rows)

    def _display_width(s):
        return sum(2 if ord(c) > 127 else 1 for c in str(s))

    def _pad_center(s, width):
        s = str(s)
        dw = _display_width(s)
        padding = width - dw
        if padding <= 0:
            return s
        left = padding // 2
        right = padding - left
        return " " * left + s + " " * right

    def _print_table(title, path, rows, headers, formatters):
        widths = [max(_display_width(h), max(_display_width(formatters[h](r)) for r in rows)) for h in headers]
        widths = [w + 2 for w in widths]
        sep = "+" + "+".join(["-" * w for w in widths]) + "+"
        head = "|" + "|".join([_pad_center(h, w) for h, w in zip(headers, widths)]) + "|"
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"  文件: {path}")
        print(f"{'='*70}")
        print(sep)
        print(head)
        print(sep)
        for r in rows:
            line = "|" + "|".join([_pad_center(formatters[h](r), w) for h, w in zip(headers, widths)]) + "|"
            print(line)
        print(sep)

    _print_table(
        "表1: 设定 A - 雷达图像补全性能对比", p1, t1_rows,
        ["方法", "MAE ↓", "RMSE ↓"],
        {
            "方法": lambda r: r["method"],
            "MAE ↓": lambda r: r["MAE_mean"],
            "RMSE ↓": lambda r: r["RMSE_mean"],
        }
    )

    _print_table(
        "表2: 设定 B - 外推扩展性能对比", p2, t2_rows,
        ["方法", "近域MAE↓", "远域MAE↓", "远域RMSE↓", "有效半径(km)↑"],
        {
            "方法": lambda r: r["method"],
            "近域MAE↓": lambda r: r["near_comp_MAE"],
            "远域MAE↓": lambda r: r["far_MAE"],
            "远域RMSE↓": lambda r: r["far_RMSE"],
            "有效半径(km)↑": lambda r: r["R_eff_km"],
        }
    )
    print()


if __name__ == "__main__":
    main()