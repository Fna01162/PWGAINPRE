import os

import sys


class Config:
    """
    项目全局配置类 (Configuration for PWGAIN_PRE)
    集中管理所有的路径、超参数和模型架构开关。
    """

    def __init__(self) -> None:
        # ==========================================
        # 1. 基础目录与数据路径 (Directory & Paths)
        # ==========================================
        # 请确保此基础路径与您的实际科研服务器环境一致
        self.base_dir = "D:/windnet/GAIN/PWGAIN_PRE"
        self.data_dir = os.path.join(self.base_dir, "dataset/split_32")

        self.train_npy = os.path.join(self.data_dir, "train.npy")
        self.val_npy = os.path.join(self.data_dir, "val.npy")
        self.test_npy = os.path.join(self.data_dir, "test.npy")

        # ==========================================
        # 2. 数据与特征参数 (Data & Feature Configs)
        # ==========================================
        self.phys_channels = 4  # 物理通道数: V, CNR, SW, Skew
        self.normalize = "minmax"  # 归一化策略: 必须为 minmax 以适配 Tanh
        self.clip_percentile = [0.5, 99.5]
        self.frame_interval_min = 5  # 帧间隔 (分钟)

        # ==========================================
        # 3. 时空窗口参数 (Spatio-Temporal Windows)
        # ==========================================
        self.input_steps = 10  # 输入历史序列长度 (Tin)
        self.pred_steps = 1  # 预测未来序列长度 (Tout)

        # ==========================================
        # 4. 优化器与训练超参数 (Optimization & Hyperparams)
        # ==========================================
        self.epochs = 50
        self.batch_size = 4
        self.lr_g_joint = 1e-5  # 联合训练时生成器的学习率
        self.lr_p = 2e-4  # 预测器的学习率
        self.grad_clip = 1.0  # 梯度裁剪阈值 (防止梯度爆炸)

        # [Fix S1]: 确保 lambda_pred 先于 exp_name 定义，解决实例化崩溃问题
        self.lambda_pred = 0.1  # 预测误差在联合损失中的权重 (\lambda)

        # ==========================================
        # 5. 模型架构超参数 (Model Architecture)
        # ==========================================
        self.base_filter = 64
        self.pred_hidden_channels = 64
        self.pred_num_layers = 3
        self.pred_kernel_size = 3

        self.use_nafblock = True
        self.use_axial_w = True
        self.use_cue = True

        # ==========================================
        # 6. 实验命名与输出路径 (Experiment Naming & Output)
        # ==========================================
        self.exp_name = "PPNAF_AZI_32km_CIR_aztv0.02_h0.3_bot_wavelet_phy_pre_0.1"
        self.gain_pretrain_path = ""  # 预训练 GAIN 权重的绝对路径 (需要时填入)

        # [Fix L4]: 统一实验命名逻辑，避免由于字符串嵌套导致极长的目录名
        self.base_joint_exp_name = (
            f"outputs_joint_{self.exp_name}_"
            f"Tin{self.input_steps}_Tout{self.pred_steps}_lp{self.lambda_pred}"
        )
        self.joint_pred_exp_name = self.base_joint_exp_name
        self.ext_exp_name = "outputs_joint_ext_spatiotemporal_near68_far78_le0.1"

        # [Fix S9]: 根据 sys.argv 动态路由输出路径，解决路径硬编码覆盖导致测试找不到模型的问题
        self._setup_dynamic_paths()

    def _setup_dynamic_paths(self) -> None:
        """根据执行参数设置当前任务的输出目录和模型保存路径。"""
        if len(sys.argv) > 1 and "ext" in sys.argv[1]:
            # 量程扩展任务路径
            self.output_dir = os.path.join(self.base_dir, "outputs_single_v", self.ext_exp_name)
            self.model_path = os.path.join(self.output_dir, "joint_ext_best.pth")
        else:
            # 标准联合预测任务路径
            self.output_dir = os.path.join(self.base_dir, "outputs_single_v", self.joint_pred_exp_name)
            self.model_path = os.path.join(self.output_dir, "joint_best.pth")

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)