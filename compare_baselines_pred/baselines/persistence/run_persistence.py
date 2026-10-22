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
def main():
    print("[Persistence] Running...")
    device = torch.device("cpu")
    norm = load_normalizer()
    ds = SlidingWindowDataset(TEST_NPY, norm, INPUT_STEPS, PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    results = []
    for batch in tqdm(loader, desc="persistence"):
        hx, fx, hm, fm = [v.to(device) for v in batch]
        last_frame = hx[:, -1:, 0:1]
        pred = last_frame.repeat(1, PRED_STEPS, 1, 1, 1)
        true = fx[:, :, 0:1]
        valid = fm
        m = compute_metrics(pred.numpy(), true.numpy(), valid.numpy())
        results.append(m)
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    out_dir = method_out_dir("persistence")
    write_csv(out_dir / "results.csv", [{"method": "Persistence", **avg}])
    print(f"  MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, Corr={avg['Corr']:.4f}")
if __name__ == "__main__":
    main()
