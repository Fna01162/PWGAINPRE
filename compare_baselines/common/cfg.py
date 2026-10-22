# -*- coding: utf-8 -*-
"""共享配置：路径、门号、协议超参。

协议值全部写死并注明出处（config.py / train.py / train_extension.py），
不依赖主项目运行时状态，保证对比实验可独立复现。
"""
from pathlib import Path

# compare_baselines/common/cfg.py -> parents[0]=common, [1]=compare_baselines, [2]=项目根
ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = ROOT / "compare_baselines"
DATA_DIR = ROOT / "dataset" / "split_32"

TRAIN_NPY = DATA_DIR / "train.npy"
VAL_NPY = DATA_DIR / "val.npy"
TEST_NPY = DATA_DIR / "test.npy"

# 主 ckpt：只读 normalizer（deep 基线归一化、所有物理空间指标统一用它）
CKPT_GAIN = ROOT / "outputs_single_v" / \
    "outputs_PPNAF_AZI_34km_CIR_aztv0.02_h0.3_bot_wavelet_phy_0.1" / "G_best.pth"
CKPT_JOINT_EXT = ROOT / "outputs_single_v" / \
    "outputs_joint_ext_cnn_near68_far78_le0.1" / "joint_ext_best.pth"

EVAL_MASKS_DIR = BASELINES_DIR / "eval_masks"
OUTPUTS_DIR = BASELINES_DIR / "outputs"

# ── 几何常量（untils/data.py ORI_H/W=85/125；PROJECT.md：起始660m、门距30m）──
ORI_H, ORI_W = 85, 125
NEAR_GATES = 68          # config.near_end_gate：近门 0..67
FAR_GATES = 10           # config.far_end_gate - near_end_gate：远门 68..77
GATE0_M = 660            # 第 0 门距离
GATE_STEP_M = 30         # 门距
PHYS_CHANNELS = 4        # config.phys_channels
TARGET_CHANNEL = 0       # V
COND_CHANNELS = ("CNR", "SW", "Skew")


def gate_dist_km(gate_idx):
    """门索引 -> 距离(km)。gate 68=2.70km, gate 77=2.97km。"""
    return (GATE0_M + GATE_STEP_M * gate_idx) / 1000.0


# ── 归一化（config.normalize="minmax", clip_percentile=99.5）──
NORMALIZE_METHOD = "minmax"
CLIP_PERCENTILE = 99.5

# ── 补全训练协议（config.py，与主模型一致，公平性要求）──
SEED = 42
ADV_WEIGHT = 0.2         # config.adv_weight
RECON_WEIGHT = 1.0       # config.recon_weight
ALPHA = 10.0             # config.alpha（GAIN 式 adv+alpha*rec）
DROP_PROB = 0.3          # config.drop_prob：补全网络训练人工洞比例
HINT_RATE = 0.30         # config.hint_rate
NOISE_SCALE = 0.10       # config.noise_scale
LR_G = 2e-4              # config.lr_g
LR_D = 5e-5              # config.lr_d
BETAS = (0.5, 0.999)     # config.betas
GRAD_CLIP = 5.0          # config.grad_clip
BATCH_SIZE = 8           # config.batch_size
EPOCHS = 100             # config.epochs
PATIENCE = 20            # config.patience

# ── 扩展训练协议（config.py ext_* / train_extension.py）──
EXT_LR = 2e-4            # config.ext_lr
EXT_WEIGHT_DECAY = 0.0   # config.ext_weight_decay
EXT_EPOCHS = 100         # config.ext_epochs
EXT_PATIENCE = 20        # config.ext_patience
EXT_BATCH_SIZE = 8       # config.ext_batch_size
EXT_GRAD_CLIP = 5.0      # config.ext_grad_clip
EXT_BASE_FILTER = 32     # config.ext_base_filter
EXT_SCHED_PATIENCE = 5   # train_extension.py ReduceLROnPlateau
EXT_SCHED_FACTOR = 0.5
EXT_MIN_LR = 1e-6

# ── 评估 ──
TAU_MPS = 1.0            # R_eff 阈值：逐门 MAE <= 1.0 m/s 视为有效
HOLE_RATIOS = (0.15, 0.30)
HOLE_SHAPES = ("scatter", "sector", "strip", "block")

METHODS_WITH_COMPLETION = (
    "linear_interp", "spatial_knn", "official_gain",
    "dsa_unet", "unetpp_cgan", "ours_sep", "ours_joint",
)
ALL_METHODS = ("no_completion",) + METHODS_WITH_COMPLETION

# 输出目录约定
def method_out_dir(method):
    d = OUTPUTS_DIR / method
    d.mkdir(parents=True, exist_ok=True)
    return d
