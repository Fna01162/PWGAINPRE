# -*- coding: utf-8 -*-
"""运行此脚本生成时间预测对比实验框架"""
import os

BASE = "compare_baselines_pred"

def w(path, content):
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [OK] {path}")

print("\n" + "="*60)
print("  Creating Prediction Comparison Framework")
print("="*60 + "\n")

# === 1. common/cfg.py ===
w("common/cfg.py", '''# -*- coding: utf-8 -*-
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset" / "split_32"
OUTPUTS_DIR = ROOT / "compare_baselines_pred" / "outputs"
TRAIN_NPY = DATA_DIR / "train.npy"
TEST_NPY = DATA_DIR / "test.npy"
INPUT_STEPS = 10
PRED_STEPS = 1
PHYS_CHANNELS = 4
JOINT_PRED_CKPT = ROOT / "outputs_single_v" / "outputs_joint_pred_Tin10_Tout1_lp0.1" / "joint_best.pth"
METHODS = ["persistence", "linear_extrap", "convlstm", "ours"]
def method_out_dir(method):
    d = OUTPUTS_DIR / method
    d.mkdir(parents=True, exist_ok=True)
    return d
''')

# === 2. common/data.py ===
w("common/data.py", '''# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_joint import SlidingWindowDataset
from untils.data import Normalizer, make_obs_rel
import numpy as np
from .cfg import TRAIN_NPY
def load_normalizer():
    normalizer = Normalizer("minmax", 99.5, 4)
    arr = np.load(TRAIN_NPY)
    obs, _ = make_obs_rel(arr, 4)
    return normalizer.fit(arr, obs)
''')

# === 3. common/metrics.py ===
w("common/metrics.py", '''# -*- coding: utf-8 -*-
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
''')

# === 4. baselines/persistence/run_persistence.py ===
w("baselines/persistence/run_persistence.py", '''# -*- coding: utf-8 -*-
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
''')

# === 5. baselines/linear_extrap/run_linear_ext.py ===
w("baselines/linear_extrap/run_linear_ext.py", '''# -*- coding: utf-8 -*-
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
''')

# === 6. baselines/convlstm/run_convlstm.py ===
w("baselines/convlstm/run_convlstm.py", '''# -*- coding: utf-8 -*-
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
''')

# === 7. baselines/ours/run_ours.py ===
w("baselines/ours/run_ours.py", '''# -*- coding: utf-8 -*-
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
''')

# === 8. eval_all.py ===
w("eval_all.py", '''# -*- coding: utf-8 -*-
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
    print("\\n" + "="*70)
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
    print(f"\\nResults saved to: {out_path}")
if __name__ == "__main__":
    main()
''')

# === 9. run_all.bat ===
w("run_all.bat", '''@echo off
cd /d "%~dp0.."
if "%~1"=="" goto :usage
if /i "%~1"=="persistence" goto :persistence
if /i "%~1"=="linear_extrap" goto :linear_extrap
if /i "%~1"=="convlstm" goto :convlstm
if /i "%~1"=="ours" goto :ours
if /i "%~1"=="summary" goto :summary
if /i "%~1"=="all" goto :all
goto :usage
:persistence
echo === [1/4] Persistence ===
python compare_baselines_pred\\baselines\\persistence\\run_persistence.py
goto :eof
:linear_extrap
echo === [2/4] Linear Extrapolation ===
python compare_baselines_pred\\baselines\\linear_extrap\\run_linear_ext.py
goto :eof
:convlstm
echo === [3/4] Pure ConvLSTM ===
python compare_baselines_pred\\baselines\\convlstm\\run_convlstm.py
goto :eof
:ours
echo === [4/4] Ours (GAIN+ConvLSTM) ===
python compare_baselines_pred\\baselines\\ours\\run_ours.py
goto :eof
:summary
echo === [SUMMARY] ===
python compare_baselines_pred\\eval_all.py
goto :eof
:all
call "%~f0" persistence
call "%~f0" linear_extrap
call "%~f0" convlstm
call "%~f0" ours
call "%~f0" summary
goto :eof
:usage
echo Usage: run_all.bat [persistence^|linear_extrap^|convlstm^|ours^|summary^|all]
goto :eof
''')

print("\n" + "="*60)
print("  SUCCESS! Framework created!")
print("="*60)
print("\nNext steps:")
print("  1. Train your model:     python main.py joint")
print("  2. Run all baselines:    compare_baselines_pred\\run_all.bat all")
print("  3. View summary:         compare_baselines_pred\\run_all.bat summary")
print("="*60 + "\n")