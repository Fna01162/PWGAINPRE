# -*- coding: utf-8 -*-
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
