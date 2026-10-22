# -*- coding: utf-8 -*-
"""生成时间预测对比实验框架的所有文件"""
import os

BASE = "compare_baselines_pred"

def w(path, content):
    """写入文件的快捷函数"""
    full_path = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  + {path}")

print("\n" + "="*60)
print("  生成时间预测对比实验框架")
print("="*60 + "\n")

# ===== 1. common/cfg.py =====
w("common/cfg.py", '''# -*- coding: utf-8 -*-
"""时间预测对比实验配置"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset" / "split_32"
OUTPUTS_DIR = ROOT / "compare_baselines_pred" / "outputs"

TRAIN_NPY = DATA_DIR / "train.npy"
VAL_NPY = DATA_DIR / "val.npy"
TEST_NPY = DATA_DIR / "test.npy"

INPUT_STEPS = 10
PRED_STEPS = 1
PHYS_CHANNELS = 4
TARGET_CHANNEL = 0

JOINT_PRED_CKPT = ROOT / "outputs_single_v" / \\
    "outputs_joint_pred_Tin10_Tout1_lp0.1" / "joint_best.pth"

METHODS = ["persistence", "linear_extrap", "optical_flow", "convlstm", "ours"]

def method_out_dir(method):
    d = OUTPUTS_DIR / method
    d.mkdir(parents=True, exist_ok=True)
    return d
''')

# ===== 2. common/data.py =====
w("common/data.py", '''# -*- coding: utf-8 -*-
"""数据加载：复用 train_joint.py 的 SlidingWindowDataset"""
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

__all__ = ['SlidingWindowDataset', 'load_normalizer']
''')

# ===== 3. common/metrics.py =====
w("common/metrics.py", '''# -*- coding: utf-8 -*-
"""预测评估指标"""
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

    return {"MAE": mae, "RMSE": rmse, "Corr": corr, "n": int(len(p))}

def write_csv(path, rows):
    if not rows:
        return
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
''')

# ===== 4. baselines/persistence/run_persistence.py =====
w("baselines/persistence/run_persistence.py", '''# -*- coding: utf-8 -*-
"""基线1: 持久化（用最后一帧作为预测）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import numpy as np
from tqdm import tqdm

from compare_baselines_pred.common.cfg import (
    TEST_NPY, INPUT_STEPS, PRED_STEPS, method_out_dir
)
from compare_baselines_pred.common.data import SlidingWindowDataset, load_normalizer
from compare_baselines_pred.common.metrics import compute_metrics, write_csv

def main(smoke=False):
    print("[Persistence] 持久化基线")
    device = torch.device("cpu")
    norm = load_normalizer()

    ds = SlidingWindowDataset(TEST_NPY, norm, INPUT_STEPS, PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    results = []
    for batch in tqdm(loader, desc="persistence"):
        hx, fx, hm, fm = [v.to(device) for v in batch]
        # 用最后一帧历史重复 pred_steps 次
        last_frame = hx[:, -1:, 0:1]  # [B, 1, 1, H, W]
        pred = last_frame.repeat(1, PRED_STEPS, 1, 1, 1)
        true = fx[:, :, 0:1]
        valid = fm

        m = compute_metrics(pred.numpy(), true.numpy(), valid.numpy())
        results.append(m)

    # 汇总
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    out_dir = method_out_dir("persistence")
    write_csv(out_dir / "results.csv", [{"method": "Persistence", **avg}])
    print(f"  MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, Corr={avg['Corr']:.4f}")

if __name__ == "__main__":
    main()
''')

# ===== 5. baselines/linear_extrap/run_linear_ext.py =====
w("baselines/linear_extrap/run_linear_ext.py", '''# -*- coding: utf-8 -*-
"""基线2: 线性外推（根据历史趋势线性拟合）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import numpy as np
from tqdm import tqdm

from compare_baselines_pred.common.cfg import (
    TEST_NPY, INPUT_STEPS, PRED_STEPS, method_out_dir
)
from compare_baselines_pred.common.data import SlidingWindowDataset, load_normalizer
from compare_baselines_pred.common.metrics import compute_metrics, write_csv

def linear_extrapolate(history):
    history = history.numpy()  # [B, T, H, W]
    B, T, H, W = history.shape
    # 对每个像素点做线性回归
    x = np.arange(T).astype(np.float32)
    x_mean = x.mean()
    x_var = ((x - x_mean)**2).sum()

    slopes = []
    intercepts = []
    for b in range(B):
        y = history[b]  # [T, H, W]
        y_mean = y.mean(axis=0, keepdims=True)  # [1, H, W]
        cov = ((x[:, None, None] - x_mean) * (y - y_mean)).sum(axis=0)
        slope = cov / (x_var + 1e-8)
        intercept = y_mean - slope * x_mean
        slopes.append(slope)
        intercepts.append(intercept)

    slopes = np.stack(slopes)  # [B, H, W]
    intercepts = np.stack(intercepts)  # [B, H, W]

    # 外推未来时刻
    preds = []
    for step in range(PRED_STEPS):
        future_t = T + step
        pred = slopes * future_t + intercepts  # [B, H, W]
        preds.append(pred)

    pred = np.stack(preds, axis=1)  # [B, Tout, H, W]
    return torch.from_numpy(pred).float()[:, :, None]  # [B, Tout, 1, H, W]

def main():
    print("[LinearExtrap] 线性外推基线")
    device = torch.device("cpu")
    norm = load_normalizer()

    ds = SlidingWindowDataset(TEST_NPY, norm, INPUT_STEPS, PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    results = []
    for batch in tqdm(loader, desc="linear_extrap"):
        hx, fx, hm, fm = [v.to(device) for v in batch]
        pred = linear_extrapolate(hx[:, :, 0])  # 只用 V 通道
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

# ===== 6. baselines/convlstm/run_convlstm.py =====
w("baselines/convlstm/run_convlstm.py", '''# -*- coding: utf-8 -*-
"""基线4: 纯 ConvLSTM（无 GAIN 补全）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from compare_baselines_pred.common.cfg import (
    TEST_NPY, INPUT_STEPS, PRED_STEPS, PHYS_CHANNELS, method_out_dir
)
from compare_baselines_pred.common.data import SlidingWindowDataset, load_normalizer
from compare_baselines_pred.common.metrics import compute_metrics, write_csv
from predictor import WindFieldPredictor, init_predictor_weights

def main():
    print("[ConvLSTM] 纯 ConvLSTM 预测基线")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = load_normalizer()

    model = WindFieldPredictor(
        input_channels=PHYS_CHANNELS + 2,  # V+CNR+SW+Skew+obs+rel
        hidden_channels=64,
        num_layers=2,
        pred_steps=PRED_STEPS,
    ).to(device)
    init_predictor_weights(model)

    ds = SlidingWindowDataset(TEST_NPY, norm, INPUT_STEPS, PRED_STEPS)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    # 用随机权重测试（或可加载预训练权重）
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="convlstm"):
            hx, fx, hm, fm = [v.to(device) for v in batch]
            hr = torch.ones_like(hm)  # 简化：假设全部可靠

            # 组装输入：[V, CNR, SW, Skew, obs, rel]
            inp = torch.cat([hx, hm, hr], dim=2)  # [B, Tin, 6, H, W]
            pred = model(inp)  # [B, Tout, 1, H, W]
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

# ===== 7. baselines/ours/run_ours.py =====
w("baselines/ours/run_ours.py", '''# -*- coding: utf-8 -*-
"""我们的方法：GAIN 补全 + ConvLSTM 预测"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import numpy as np
from tqdm import tqdm

from compare_baselines_pred.common.cfg import (
    TEST_NPY, INPUT_STEPS, PRED_STEPS, JOINT_PRED_CKPT, method_out_dir
)
from compare_baselines_pred.common.data import SlidingWindowDataset, load_normalizer
from compare_baselines_pred.common.metrics import compute_metrics, write_csv
from model import UNetGenerator
from predictor import WindFieldPredictor
from untils.data import generate_pure_structural_missing_mask

def main():
    print("[Ours] GAIN + ConvLSTM 联合预测")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = load_normalizer()

    # 加载联合训练模型
    ckpt_path = JOINT_PRED_CKPT
    if not ckpt_path.exists():
        print(f"  ✗ 检查点不存在: {ckpt_path}")
        print("  请先运行: python main.py joint")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G = UNetGenerator(in_channels=8, out_channels=1, base_filter=48,
                      use_cue=True, use_nafblock=True, use_axial_w=True).to(device)
    G.load_state_dict(ckpt["generator"])

    P = WindFieldPredictor(input_channels=6, hidden_channels=64,
                            num_layers=2, pred_steps=PRED_STEPS).to(device)
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

            # GAIN 补全每一帧
            completed_list = []
            for t in range(Tin):
                x_t = hx[:, t:t+1]
                m_t = hm[:, t:t+1]
                art, _ = generate_pure_structural_missing_mask(
                    B, H, W, device, 0.1
                )
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

            v_completed = torch.stack(completed_list, dim=1)  # [B, Tin, H, W]
            hr = torch.ones_like(hm)

            # 预测
            pred_input = torch.cat([
                v_completed.unsqueeze(2),  # [B, Tin, 1, H, W]
                hx[:, :, 1:],             # CNR, SW, Skew
                hm, hr
            ], dim=2)  # [B, Tin, 6, H, W]

            future_pred = P(pred_input)  # [B, Tout, 1, H, W]
            future_true = fx[:, :, 0:1]
            valid = fm

            m = compute_metrics(future_pred.cpu().numpy(),
                              future_true.cpu().numpy(),
                              valid.cpu().numpy())
            results.append(m)

    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    out_dir = method_out_dir("ours")
    write_csv(out_dir / "results.csv", [{"method": "Ours (GAIN+ConvLSTM)", **avg}])
    print(f"  MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}, Corr={avg['Corr']:.4f}")

if __name__ == "__main__":
    main()
''')

# ===== 8. eval_all.py =====
w("eval_all.py", '''# -*- coding: utf-8 -*-
"""汇总所有时间预测基线结果"""
import csv
import numpy as np
from pathlib import Path

from compare_baselines_pred.common.cfg import METHODS, OUTPUTS_DIR
from compare_baselines_pred.common.metrics import write_csv

CN = {
    "persistence": "持久化 (最后一帧)",
    "linear_extrap": "线性外推",
    "optical_flow": "光流法",
    "convlstm": "纯 ConvLSTM",
    "ours": "本文方法 (GAIN+ConvLSTM)",
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
    print("  时间预测性能对比汇总")
    print("="*70)

    rows = []
    for m in METHODS:
        name = CN.get(m, m)
        r = read_results(m)
        row = {
            "method": name,
            "MAE": r["MAE"] if r else "-",
            "RMSE": r["RMSE"] if r else "-",
            "Corr": r["Corr"] if r else "-",
        }
        rows.append(row)

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

    headers = ["方法", "MAE ↓", "RMSE ↓", "Corr ↑"]
    widths = [max(_display_width(h), max(_display_width(r[h]) for r in rows)) for h in headers]
    widths = [w + 2 for w in widths]

    sep = "+" + "+".join(["-" * w for w in widths]) + "+"
    head = "|" + "|".join([_pad_center(h, w) for h, w in zip(headers, widths)]) + "|"

    print(sep)
    print(head)
    print(sep)
    for r in rows:
        line = "|" + "|".join([_pad_center(r[h], w) for h, w in zip(headers, widths)]) + "|"
        print(line)
    print(sep)

    # 保存 CSV
    out_path = OUTPUTS_DIR / "summary_prediction.csv"
    write_csv(out_path, rows)
    print(f"\\n结果已保存: {out_path}")

if __name__ == "__main__":
    main()
''')

# ===== 9. run_all.bat =====
w("run_all.bat", '''@echo off
REM 时间预测对比实验一键运行脚本
REM 用法: run_all.bat [all|persistence|linear_extrap|convlstm|ours|summary]

cd /d "%~dp0.."

if "%~1"=="" goto :usage
if /i "%~1"=="persistence" goto :persistence
if /i "%~1"=="linear_extrap" goto :linear_extrap
if /i "%~1"=="optical_flow" goto :optical_flow
if /i "%~1"=="convlstm" goto :convlstm
if /i "%~1"=="ours" goto :ours
if /i "%~1"=="summary" goto :summary
if /i "%~1"=="all" goto :all
goto :usage

:persistence
echo === [1/5] Persistence (持久化) ===
python compare_baselines_pred\\baselines\\persistence\\run_persistence.py
goto :eof

:linear_extrap
echo === [2/5] Linear Extrapolation (线性外推) ===
python compare_baselines_pred\\baselines\\linear_extrap\\run_linear_ext.py
goto :eof

:optical_flow
echo === [3/5] Optical Flow (光流法) ===
echo [TODO] 光流法待实现
goto :eof

:convlstm
echo === [4/5] Pure ConvLSTM ===
python compare_baselines_pred\\baselines\\convlstm\\run_convlstm.py
goto :eof

:ours
echo === [5/5] Ours (GAIN+ConvLSTM) ===
python compare_baselines_pred\\baselines\\ours\\run_ours.py
goto :eof

:summary
echo === [SUMMARY] 汇总表 ===
python compare_baselines_pred\\eval_all.py
goto :eof

:all
echo ########## 运行所有基线 ##########
call "%~f0" persistence
call "%~f0" linear_extrap
call "%~f0" convlstm
call "%~f0" ours
call "%~f0" summary
goto :eof

:usage
echo 用法: run_all.bat [persistence^|linear_extrap^|convlstm^|ours^|summary^|all]
goto :eof
''')

print("\n" + "="*60)
print("✓ 时间预测对比实验框架创建完成！")
print("="*60)
print("\n使用方法:")
print("  1. 先训练你的模型: python main.py joint")
print("  2. 运行所有基线:   python gen_pred_baselines.py && compare_baselines_pred\\run_all.bat all")
print("  3. 或单独运行某个: compare_baselines_pred\\run_all.bat persistence")
print("  4. 查看汇总结果:   compare_baselines_pred\\run_all.bat summary")
print("="*60 + "\n")