# -*- coding: utf-8 -*-
"""一键生成时间预测对比实验框架 - 运行此脚本即可"""
import os

BASE = "compare_baselines_pred"

def w(path, content):
    """写入文件"""
    full_path = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [OK] {path}")

print("\n" + "="*60)
print("  创建时间预测对比实验框架")
print("="*60 + "\n")

# ===== 1. common/cfg.py =====
w("common/cfg.py", '''# -*- coding: utf-8 -*-
"""时间预测对比实验配置"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset" / "split_32"
OUTPUTS_DIR = ROOT / "compare_baselines_pred" / "outputs"

TRAIN_NPY = DATA_DIR / "train.npy"
TEST_NPY = DATA_DIR / "test.npy"

INPUT_STEPS = 10
PRED_STEPS = 1
PHYS_CHANNELS = 4
TARGET_CHANNEL = 0

JOINT_PRED_CKPT = ROOT / "outputs_single_v" / \\
    "outputs_joint_pred_Tin10_Tout1_lp0.1" / "joint_best.pth"

METHODS = ["persistence", "linear_extrap", "convlstm", "ours"]

def method_out_dir(method):
    d = OUTPUTS_DIR / method
    d.mkdir(parents=True, exist_ok=True)
    return d
''')

# ===== 2. common/data.py =====
w("common/data.py", '''# -*- coding: utf-8 -*-
"""数据加载模块"""
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
    """计算 MAE, RMSE, Corr"""
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
    """写入CSV"""
    if not rows:
        return
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

__all__ = ['compute_metrics', 'write_csv']
''')

print("\n✓ 核心公共模块创建完成！")
print("="*60)