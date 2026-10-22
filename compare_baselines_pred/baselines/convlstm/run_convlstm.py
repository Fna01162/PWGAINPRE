# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import numpy as np
from tqdm import tqdm
from compare_baselines_pred.common.cfg import TEST_NPY, INPUT_STEPS, PRED_STEPS, PHYS_CHANNELS, method_out_dir
from compare_baselines_pred.common.data import SlidingWindowDataset, load_normalizer
from compare_baselines_pred.common.metrics import compute_metrics, write_csv
from predictor import WindFieldPredictor, init_predictor_weights
def main():
    print("[ConvLSTM] Running...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = load_normalizer()
    model = WindFieldPredictor(
        input_channels=PHYS_CHANNELS + 2,
        hidden_channels=64,
        num_layers=2,
        pred_steps=PRED_STEPS,
    ).to(device)
    init_predictor_weights(model)
    ds = SlidingWindowDataset(TEST_NPY, norm, INPUT_STEPS, PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="convlstm"):
            hx, fx, hm, fm = [v.to(device) for v in batch]
            hr = torch.ones_like(hm)
            inp = torch.cat([hx, hm, hr], dim=2)
            pred = model(inp)
            true = fx[:, :, 0:1]
            valid = fm
            m = compute_metrics(pred.cpu().numpy(), true.cpu().numpy(), valid.cpu().numpy())
            results.append(m)
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    out_dir = method_out_dir("convlstm")
    write_csv(out_dir / "results.csv", [{"method": "ConvLSTM", **avg}])
    print(f"  MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, Corr={avg['Corr']:.4f}")
if __name__ == "__main__":
    main()
