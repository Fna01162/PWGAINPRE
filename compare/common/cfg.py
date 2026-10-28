# -*- coding: utf-8 -*-
"""补全对比实验的统一配置：路径、时间窗口、遮挡率、训练超参。

所有方法都在同一套固定扇区掩码下比较，保证公平。
"""
import sys
import random
from pathlib import Path

import numpy as np
import torch

# 项目根目录（compare/common/cfg.py 的上两级）
ROOT = Path(__file__).resolve().parents[2]
COMPARE_DIR = Path(__file__).resolve().parents[1]

# 让 compare 下的脚本能 import 项目根目录的模块
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "dataset" / "split_32"
OUT_DIR = COMPARE_DIR / "outputs"
MASK_DIR = OUT_DIR / "masks"
for _d in (OUT_DIR, MASK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_NPY = DATA_DIR / "train.npy"
VAL_NPY = DATA_DIR / "val.npy"
TEST_NPY = DATA_DIR / "test.npy"

# 数据尺寸 / 通道
PHYS_CHANNELS = 4
TARGET_CHANNEL = 0
ORI_H, ORI_W = 85, 125
PAD_H, PAD_W = 91, 128
PAD_TOP, PAD_BOTTOM = 3, 3
PAD_LEFT, PAD_RIGHT = 1, 2
BAD_RAYS = [73, 74, 75]

# 时间窗口
INPUT_STEPS = 10
PRED_STEPS = 1

# 遮挡率：主场景 30%，另做 10%/50% 扫描
MAIN_RATE = 0.30
RATES = [0.10, 0.30, 0.50]

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 训练超参（与主项目config.py统一）
EPOCHS = 100
BATCH_SIZE = 4
LR = 2e-4
# 表格型时序方法（BRITS/SAITS）的局部补丁
PATCH = 16
PATCH_STRIDE = 8
PATCH_TRAIN_PER_WIN = 8
KNN_K = 8
# 端到端联合训练里预测损失的权重
LAMBDA_PRED = 0.1
# Ours 生成器基础通道（与主项目统一）
OURS_BASE_FILTER = 48
HIDDEN_CHANNELS = 64
# 其他训练参数（与主项目统一）
BETAS = (0.5, 0.999)
GRAD_CLIP = 1.0
RECON_WEIGHT = 10.0


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)