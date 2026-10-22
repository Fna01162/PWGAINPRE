# -*- coding: utf-8 -*-
"""时间预测对比实验配置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "compare_baselines_pred"
DATA_DIR = ROOT / "dataset" / "split_32"

TRAIN_NPY = DATA_DIR / "train.npy"
VAL_NPY = DATA_DIR / "val.npy"
TEST_NPY = DATA_DIR / "test.npy"

JOINT_PRED_CKPT = ROOT / "outputs_single_v" / \
    "outputs_joint_pred_Tin10_Tout1_lp0.1" / "joint_best.pth"

GAIN_PRETRAIN_CKPT = ROOT / "outputs_single_v" / \
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