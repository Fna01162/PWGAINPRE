# -*- coding: utf-8 -*-
"""一键生成时间预测对比实验框架"""
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✓ {path}")

# 1. common/cfg.py
write_file("compare_baselines_pred/common/cfg.py", '''# -*- coding: utf-8 -*-
"""时间预测对比实验配置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "compare_baselines_pred"
DATA_DIR = ROOT / "dataset" / "split_32"

TRAIN_NPY = DATA_DIR / "train.npy"
VAL_NPY = DATA_DIR / "val.npy"
TEST_NPY = DATA_DIR / "test.npy"

JOINT_PRED_CKPT = ROOT / "outputs_single_v" / \\
    "outputs_joint_pred_Tin10_Tout1_lp0.1" / "joint_best.pth"

GAIN_PRETRAIN_CKPT = ROOT / "outputs_single_v" / \\
    "outputs_PPNAF_AZI_34km_CIR_aztv0.02_h0.3_bot_wavelet_phy_0.1" / "G_best.pth"

OUTPUTS_DIR = PRED_DIR / "outputs"

INPUT_STEPS = 10
PRED_STEPS = 1
FRAME_INTERVAL_MIN = 5

PHYS_CHANNELS = 4
TARGET_CHANNEL = 0

NORMALIZE_METHOD = "minmax"
CLIP_PERCENTILE = 99.5

BASELINE_METHODS = [
    "persistence",
    "linear_extrap",
    "optical_flow",
    "convlstm",
]

ALL_METHODS = BASELINE_METHODS + ["ours"]


def method_out_dir(method):
    d = OUTPUTS_DIR / method
    d.mkdir(parents=True, exist_ok=True)
    return d
''')

# 2. common/data.py
write_file("compare_baselines_pred/common/data.py", '''# -*- coding: utf-8 -*-
"""滑动窗口数据集 + 归一化工具。"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from untils.data import Normalizer, make_obs_rel


class SlidingWindowDataset(Dataset):
    """滑动窗口数据集。

    输出：
        history_x: [B, Tin, C, H, W]
        future_x : [B, Tout, C, H, W]
        history_m: [B, Tin, 1, H, W]
        future_m : [B, Tout, 1, H, W]
    """

    def __init__(self, npy_path, normalizer, input_steps=10, pred_steps=1,
                 phys_channels=4):
        arr = np.load(npy_path).astype(np.float32)
        self.arr = normalizer.transform(arr)
        obs, rel = make_obs_rel(arr, phys_channels)
        self.obs = obs.astype(np.float32)
        self.T = arr.shape[0]
        self.input_steps = input_steps
        self.pred_steps = pred_steps

        valid_windows = []
        for start in range(self.T - input_steps - pred_steps + 1):
            end_hist = start + input_steps
            end_future = end_hist + pred_steps
            hist_obs = self.obs[start:end_hist]
            fut_obs = self.obs[end_hist:end_future]
            if np.any(hist_obs > 0.5) and np.any(fut_obs > 0.5):
                valid_windows.append(start)

        self.windows = valid_windows
        print(f"[SlidingWindowDataset] {Path(npy_path).name}: "
              f"{len(self.windows)} windows (T={self.T}, "
              f"in={input_steps}, out={pred_steps})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end_hist = start + self.input_steps
        end_future = end_hist + self.pred_steps

        return (
            torch.from_numpy(self.arr[start:end_hist]).float(),
            torch.from_numpy(self.arr[end_hist:end_future]).float(),
            torch.from_numpy(self.obs[start:end_hist, None]).float(),
            torch.from_numpy(self.obs[end_hist:end_future, None]).float(),
        )


def load_normalizer(train_npy=None, method="minmax", clip=99.5, phys_channels=4):
    """加载或拟合归一化器。"""
    from .cfg import TRAIN_NPY
    if train_npy is None:
        train_npy = TRAIN_NPY

    normalizer = Normalizer(method, clip, phys_channels)
    arr = np.load(train_npy)
    obs, _ = make_obs_rel(arr, phys_channels)
    normalizer = normalizer.fit(arr, obs)
    print(f"[Normalizer] method={method}, clip={clip}%, from {train_npy}")
    return normalizer
''')

# 3. common/metrics.py
write_file("compare_baselines_pred/common/metrics.py", '''# -*- coding: utf-8 -*-
"""预测评估指标：MAE, RMSE, SSIM, Corr, Bias。"""
import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def compute_prediction_metrics(pred, target, valid_mask=None):
    """计算预测指标。

    Args:
        pred: [B, T, 1, H, W] 或 [B, T, H, W]
        target: 同上
        valid_mask: [B, T, 1, H, W] 或 None（全部有效）

    Returns:
        dict: {"MAE": float, "RMSE": float, "SSIM": float, "Corr": float, "Bias": float}
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu().numpy()

    # 确保形状一致
    if pred.ndim == 5:
        pred = pred[:, :, 0]  # [B, T, H, W]
    if target.ndim == 5:
        target = target[:, :, 0]

    if valid_mask is not None:
        if valid_mask.ndim == 5:
            valid_mask = valid_mask[:, :, 0]
        mask = valid_mask > 0.5
    else:
        mask = np.ones_like(pred, dtype=bool)

    p = pred[mask].flatten().astype(np.float64)
    t = target[mask].flatten().astype(np.float64)

    if len(p) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "SSIM": np.nan,
                "Corr": np.nan, "Bias": np.nan, "n": 0}

    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    # 相关系数
    if p.std() > 1e-8 and t.std() > 1e-8:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = np.nan

    # SSIM (简化版，逐样本计算后平均)
    ssim = _compute_ssim_batch(pred, target, mask)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "SSIM": ssim,
        "Corr": corr,
        "Bias": bias,
        "n": int(mask.sum()),
    }


def _compute_ssim_batch(pred, target, mask):
    """计算批量 SSIM。"""
    ssim_values = []
    for i in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            p = pred[i, t][mask[i, t]]
            tr = target[i, t][mask[i, t]]
            if len(p) < 2:
                continue
            # 简化 SSIM
            mean_p, mean_t = p.mean(), tr.mean()
            std_p, std_t = p.std(), tr.std()
            cov = ((p - mean_p) * (tr - mean_t)).mean()
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            num = (2 * mean_p * mean_t + c1) * (2 * cov + c2)
            den = (mean_p ** 2 + mean_t ** 2 + c1) * (std_p ** 2 + std_t ** 2 + c2)
            ssim_values.append(num / (den + 1e-8))

    return float(np.mean(ssim_values)) if ssim_values else np.nan


def write_csv(path, rows, fieldnames=None):
    """写入 CSV 文件。"""
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
''')

print("\\n" + "="*60)
print("✓ 时间预测对比实验框架核心模块创建完成！")
print("="*60)