# -*- coding: utf-8 -*-
"""
PWGAIN_PRE 项目全局配置文件 (精简版)

用法:
    python main.py joint      # 时间预测任务
"""

import os
import sys
import torch


class Config:
    """项目全局配置 - 只保留核心可调参数"""

    def __init__(self):
        # ==================== 基础设置 ====================
        self.random_seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ==================== 数据路径 ====================
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # 自动检测项目根目录
        self.data_dir = os.path.join(self.base_dir, "dataset/split_32")
        self.train_npy = os.path.join(self.data_dir, "train.npy")
        self.val_npy = os.path.join(self.data_dir, "val.npy")
        self.test_npy = os.path.join(self.data_dir, "test.npy")

        # ==================== 数据参数 ====================
        self.phys_channels = 4              # 物理通道数 [V, R, AZ, EL]
        self.target_channel = 0             # 目标通道 (0=V速度场)
        self.normalize = "minmax"           # 归一化策略
        self.clip_percentile = [0.5, 99.5]  # 截断分位数
        self.missing_rate = 0.15            # 缺失比例 (GAIN训练)

        # ==================== 时间窗口 ====================
        self.input_steps = 10               # 输入历史长度 (Tin)
        self.pred_steps = 1                 # 预测未来长度 (Tout)

        # ==================== 训练超参数 ====================
        self.epochs = 100
        self.batch_size = 4
        self.num_workers = 4
        self.lr = 2e-4                       # 统一学习率 (G/D/P)
        self.betas = (0.5, 0.999)
        self.grad_clip = 1.0

        # ==================== 损失权重 ====================
        self.lambda_pred = 0.1              # 预测损失权重
        self.recon_weight = 10.0            # 重建损失权重

        # ==================== 模型架构 (关键参数) ====================
        self.base_filter = 48               # UNet基础通道数
        self.hidden_channels = 64           # ConvLSTM隐藏通道
        self.num_layers = 2                 # ConvLSTM层数

        # ==================== 预训练模型路径 ====================
        self.gain_pretrain_path = os.path.join(
            self.base_dir,
            "outputs_single_v/outputs_PPNAF_AZI_34km_CIR_aztv0.02_h0.3_bot_wavelet_phy_0.1/G_best.pth"
        )  # 预训练GAIN补全网络 (空字符串=不加载)

        # ==================== 训练策略 ====================
        self.early_stopping_patience = 10
        self.save_every_n_epochs = 5

        # ==================== 动态路径设置 ====================
        self._setup_paths()

    def _setup_paths(self):
        mode = sys.argv[1] if len(sys.argv) > 1 else "joint"
        task_name = f"joint_pred_Tin{self.input_steps}_Tout{self.pred_steps}_lp{self.lambda_pred}"
        self.output_dir = os.path.join(self.base_dir, "outputs_single_v", task_name)
        self.model_path = os.path.join(self.output_dir, "joint_best.pth")
        os.makedirs(self.output_dir, exist_ok=True)