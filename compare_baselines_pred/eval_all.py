# -*- coding: utf-8 -*-
import csv
import numpy as np
from pathlib import Path
from compare_baselines_pred.common.cfg import METHODS, OUTPUTS_DIR
from compare_baselines_pred.common.metrics import write_csv
CN = {
    "persistence": "Persistence",
    "linear_extrap": "Linear Extrapolation",
    "convlstm": "Pure ConvLSTM",
    "ours": "Ours (GAIN+ConvLSTM)",
}
def read_results(method):
    path = OUTPUTS_DIR / method / "results.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None
def main():
    print("\n" + "="*70)
    print("  Prediction Baseline Comparison Summary")
    print("="*70)
    rows = []
    for m in METHODS:
        name = CN.get(m, m)
        r = read_results(m)
        row = {"method": name, "MAE": r["MAE"] if r else "-", "RMSE": r["RMSE"] if r else "-", "Corr": r["Corr"] if r else "-"}
        rows.append(row)
    def _dw(s):
        return sum(2 if ord(c) > 127 else 1 for c in str(s))
    def _pc(s, w):
        s = str(s)
        dw = _dw(s)
        pad = w - dw
        if pad <= 0:
            return s
        l = pad // 2
        r = pad - l
        return " "*l + s + " "*r
    headers = ["Method", "MAE (down)", "RMSE (down)", "Corr (up)"]
    widths = [max(_dw(h), max(_dw(r[h]) for r in rows)) for h in headers]
    widths = [w+2 for w in widths]
    sep = "+" + "+".join(["-"*w for w in widths]) + "+"
    head = "|" + "|".join([_pc(h, w) for h, w in zip(headers, widths)]) + "|"
    print(sep)
    print(head)
    print(sep)
    for r in rows:
        line = "|" + "|".join([_pc(r[h], w) for h, w in zip(headers, widths)]) + "|"
        print(line)
    print(sep)
    out_path = OUTPUTS_DIR / "summary_prediction.csv"
    write_csv(out_path, rows)
    print(f"\nResults saved to: {out_path}")
if __name__ == "__main__":
    main()
