# -*- coding: utf-8 -*-
"""表 4-X：补全质量对比。

在固定扇区遮挡下，比较各方法补全区域的 MAE / RMSE / SSIM。
用法：
    python run_completion.py
    python run_completion.py --methods zero,linear,knn --limit 20
    python run_completion.py --rate 0.5 --epochs 30
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from common import cfg
from common.data import build_windows, inverse_v
from common import metrics
from baselines import api


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="all")
    p.add_argument("--rate", type=float, default=cfg.MAIN_RATE)
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--split", default="test")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.set_seed()
    methods = api.ALL if args.methods == "all" else args.methods.split(",")

    windows = build_windows(args.split, args.rate, args.limit)
    truth = np.stack([w["clean"][..., 0] for w in windows])
    art = np.stack([w["art"] for w in windows])

    truth_flat = truth.reshape(-1, *truth.shape[-2:])
    art_flat = art.reshape(-1, *art.shape[-2:])
    truth_raw = inverse_v(truth_flat)

    rows = []
    for method in methods:
        method = method.strip()
        print(f"== 补全方法: {method} ==")
        comp = api.completed_v(
            method, args.split, rate=args.rate,
            epochs=args.epochs, limit=args.limit,
            train_limit=args.train_limit)
        comp_flat = comp.reshape(-1, *comp.shape[-2:])
        comp_raw = inverse_v(comp_flat)
        m = metrics.all_metrics(comp_raw, truth_raw, art_flat)
        rows.append({
            "补全方法": method,
            "遮挡率": args.rate,
            "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "SSIM": round(m["SSIM"], 4),
        })
        print(f"   MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} "
              f"SSIM={m['SSIM']:.4f}")

    out = cfg.OUT_DIR / f"table4X_completion_{args.split}_{int(args.rate*100)}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已保存：{out}")


if __name__ == "__main__":
    main()
