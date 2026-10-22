# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import numpy as np
from tqdm import tqdm
from compare_baselines_pred.common.cfg import TEST_NPY, INPUT_STEPS, PRED_STEPS, JOINT_PRED_CKPT, method_out_dir
from compare_baselines_pred.common.data import SlidingWindowDataset, load_normalizer
from compare_baselines_pred.common.metrics import compute_metrics, write_csv
from model import UNetGenerator
from predictor import WindFieldPredictor
from untils.data import generate_pure_structural_missing_mask
def main():
    print("[Ours] GAIN+ConvLSTM Running...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = load_normalizer()
    ckpt_path = JOINT_PRED_CKPT
    if not ckpt_path.exists():
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        print("  Please run: python main.py joint")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G = UNetGenerator(in_channels=8, out_channels=1, base_filter=48, use_cue=True, use_nafblock=True, use_axial_w=True).to(device)
    G.load_state_dict(ckpt["generator"])
    P = WindFieldPredictor(input_channels=6, hidden_channels=64, num_layers=2, pred_steps=PRED_STEPS).to(device)
    P.load_state_dict(ckpt["predictor"])
    G.eval()
    P.eval()
    ds = SlidingWindowDataset(TEST_NPY, norm, INPUT_STEPS, PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ours"):
            hx, fx, hm, fm = [v.to(device) for v in batch]
            B, Tin, C, H, W = hx.shape
            completed_list = []
            for t in range(Tin):
                x_t = hx[:, t:t+1]
                m_t = hm[:, t:t+1]
                art, _ = generate_pure_structural_missing_mask(B, H, W, device, 0.1)
                given = m_t * (1 - art)
                xin = x_t.clone()
                xin[:, 0] = x_t[:, 0] * given
                min_ = m_t.clone()
                min_[:, 0] = given
                from train_joint import generate_contextual_cue, build_contextual_fusion
                z = torch.randn_like(xin)
                cue = generate_contextual_cue(xin, min_, t, 0)
                xb = build_contextual_fusion(xin, min_, cue, z, 0.1)
                vf = G(xb, min_)
                completed_list.append(vf[:, 0])
            v_completed = torch.stack(completed_list, dim=1)
            hr = torch.ones_like(hm)
            pred_input = torch.cat([v_completed.unsqueeze(2), hx[:, :, 1:], hm, hr], dim=2)
            future_pred = P(pred_input)
            future_true = fx[:, :, 0:1]
            valid = fm
            m = compute_metrics(future_pred.cpu().numpy(), future_true.cpu().numpy(), valid.cpu().numpy())
            results.append(m)
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    out_dir = method_out_dir("ours")
    write_csv(out_dir / "results.csv", [{"method": "Ours (GAIN+ConvLSTM)", **avg}])
    print(f"  MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, Corr={avg['Corr']:.4f}")
if __name__ == "__main__":
    main()
