# -*- coding: utf-8 -*-
import csv
import numpy as np
def compute_metrics(pred, target, mask=None):
    if isinstance(pred, np.ndarray) and pred.ndim == 5:
        pred, target = pred[:, :, 0], target[:, :, 0]
        if mask is not None and mask.ndim == 5:
            mask = mask[:, :, 0]
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    else:
        mask = mask > 0.5
    p = pred[mask].flatten().astype(np.float64)
    t = target[mask].flatten().astype(np.float64)
    if len(p) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "Corr": float("nan")}
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "Corr": corr}
def write_csv(path, rows):
    if not rows:
        return
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
