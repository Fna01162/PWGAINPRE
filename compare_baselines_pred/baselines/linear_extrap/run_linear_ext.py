# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import numpy as np
from tqdm import tqdm
from compare_baselines_pred.common.cfg import TEST_NPY, INPUT_STEPS, PRED_STEPS, method_out_dir
from compare_baselines_pred.common.data import SlidingWindowDataset, load_normalizer
from compare_baselines_pred.common.metrics import compute_metrics, write_csv
def linear_extrapolate(history):
    history = history.numpy()
    B, T, H, W = history.shape
    x = np.arange(T).astype(np.float32)
    x_mean = x.mean()
    x_var = ((x - x_mean)**2).sum()
    slopes, intercepts = [], []
    for b in range(B):
        y = history[b]
        y_mean = y.mean(axis=0, keepdims=True)
        cov = ((x[:, None, None] - x_mean) * (y - y_mean)).sum(axis=0)
        slope = cov / (x_var + 1e-8)
        intercept = y_mean - slope * x_mean
        slopes.append(slope)
        intercepts.append(intercept)
    slopes = np.stack(slopes)
    intercepts = np.stack(intercepts)
    preds = []
    for step in range(PRED_STEPS):
        future_t = T + step
        pred = slopes * future_t + intercepts
        preds.append(pred)
    pred = np.stack(preds, axis=1)
    return torch.from_numpy(pred).float()[:, :, None]
def main():
    print("[LinearExtrap] Running...")
    device = torch.device("cpu")
    norm = load_normalizer()
    ds = SlidingWindowDataset(TEST_NPY, norm, INPUT_STEPS, PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    results = []
    for batch in tqdm(loader, desc="linear_extrap"):
        hx, fx, hm, fm = [v.to(device) for v in batch]
        pred = linear_extrapolate(hx[:, :, 0])
        true = fx[:, :, 0:1]
        valid = fm
        m = compute_metrics(pred.numpy(), true.numpy(), valid.numpy())
        results.append(m)
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    out_dir = method_out_dir("linear_extrap")
    write_csv(out_dir / "results.csv", [{"method": "LinearExtrap", **avg}])
    print(f"  MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, Corr={avg['Corr']:.4f}")
if __name__ == "__main__":
    main()
