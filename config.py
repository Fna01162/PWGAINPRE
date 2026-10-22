import torch


class Config:
    def __init__(self):
        self.train_npy = r"dataset/split_32/train.npy"
        self.val_npy = r"dataset/split_32/val.npy"
        self.test_npy = r"dataset/split_32/test.npy"

        # 人工缺失
        self.drop_prob = 0.3
        self.hint_rate = 0.30
        self.noise_scale = 0.10

        self.exp_name = f"joint_PPNAF_AZI_32km_CIR_aztv0.02_h0.3_bot_wavelet_phy_pre_0.1_Tin10_Tout1_lp0.1"
        self.input_npy = self.test_npy
        self.output_dir = f"outputs_single_v/outputs_joint_ext_cnn_near68_far78_le0.1"
        # self.model_path = f"outputs_single_v/outputs_{self.exp_name}/joint_best.pth"
        self.model_path = f"outputs_single_v/outputs_joint_ext_cnn_near68_far78_le0.1/joint_ext_best.pth"

        self.use_cue = True
        self.normalizer_refit_npy = self.test_npy

        # GAIN_NAF
        self.use_nafblock = True
        self.use_volformer = False
        self.use_wavelet_attn = False
        self.use_fits = False
        self.use_sed_disc = False
        self.use_axial_w = True
        self.axial_num_pairs = 1
        self.axial_num_heads = 4

        # 第二阶段：Wavelet-guided Residual Refinement
        self.use_refiner = True
        self.use_wavelet = True
        self.use_physics = True
        self.refiner_base_filter = 16
        self.refiner_max_delta = 0.30

        # 推理端坏射线(73-75)处理：Refiner 只在人工缺失区(有真值)做残差监督，
        # 坏射线无真值、从未在训练 hole 中出现，推理时其 delta 属无监督外推，
        # 默认保护坏射线（视为已知可靠区、保持 GAIN 输出），避免被 Refiner 破坏。
        # 若后期想让 Refiner 也作用于此区，可置 False，但要先确认其补全更优。
        self.protect_bad_rays_refiner = True

        # Refiner / Wavelet / Physics loss
        self.lambda_refine = 1.0
        self.lambda_wavelet = 0.5
        self.lambda_physics = 0.1

        self.wavelet_ll_weight = 0.25
        self.wavelet_lh_weight = 0.25
        self.wavelet_hl_weight = 0.25
        self.wavelet_hh_weight = 0.25
        self.n_wavelet_levels = 1
        self.wavelet_depth_decay = 1.0

        self.radial_weight = 1.0
        self.azimuth_weight = 1.0
        self.gradient_weight = 0.0

        # 数据与模型
        self.phys_channels = 4
        self.target_idx = 0
        self.time_steps = 1
        self.base_filter = 48
        self.stride = 1



        # GAIN
        self.alpha = 10.0
        self.adv_weight = 0.2
        self.tv_weight = 0
        self.temp_weight = 0
        self.recon_weight = 1.0
        self.ring_weight = 0.01
        self.aztv_weight = 0.02
        self.curv_weight = 0.0

        # 优化器
        self.lr_g = 2e-4
        self.lr_d = 5e-5
        self.betas = (0.5, 0.999)
        self.grad_clip = 5.0

        # 训练
        self.batch_size = 8
        self.epochs = 100
        self.seed = 42
        self.save_interval = 50
        self.patience = 20
        self.deterministic = True

        # 归一化
        self.normalize = "minmax"
        self.clip_percentile = 99.5

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ──────────── 预测网络参数 ────────────
        self.input_steps = 10
        self.pred_steps = 1
        self.frame_interval_min = 5
        self.target_channel = 0
        self.pred_input_channels = 0
        self.pred_hidden_channels = 64
        self.pred_num_layers = 2
        self.pred_kernel_size = 3

        # ──────────── 联合预测训练参数 ────────────
        self.joint_pred_exp_name = f"joint_pred_Tin{self.input_steps}_Tout{self.pred_steps}_lp{self.lambda_pred}"
        self.joint_pred_output_dir = f"outputs_single_v/outputs_{self.joint_pred_exp_name}"
        self.joint_pred_model_path = f"{self.joint_pred_output_dir}/joint_best.pth"

        # ──────────── 联合训练参数 ────────────
        self.joint_training = False
        self.lambda_pred = 0.1
        self.lr_p = 2e-4
        self.freeze_g_epochs = 0
        self.pred_weight_decay = 0.0

        # ──────────── 径向量程扩展参数 ────────────
        self.near_end_gate = 68
        self.far_end_gate = 78
        self.ext_model_type = "spatiotemporal"
        self.ext_base_filter = 32
        self.ext_in_channels = 0
        self.ext_lr = 2e-4
        self.ext_weight_decay = 0.0
        self.ext_epochs = 100
        self.ext_patience = 20
        self.ext_batch_size = 8
        self.ext_grad_clip = 5.0
        self.lambda_ext = 0.1
        self.lr_ext_joint = 1e-4
        self.lr_g_joint = 1e-5
        self.gain_pretrain_path = r"outputs_single_v/outputs_PPNAF_AZI_34km_CIR_aztv0.02_h0.3_bot_wavelet_phy_0.1/G_best.pth"
        self.ext_exp_name = f"ext_{self.ext_model_type}_near{self.near_end_gate}_far{self.far_end_gate}"
        self.ext_pretrain_path = f"outputs_single_v/outputs_{self.ext_exp_name}/ext_best.pth"
        self.output_dir = f"outputs_single_v/outputs_joint_ext_{self.ext_model_type}_near{self.near_end_gate}_far{self.far_end_gate}_le{self.lambda_ext}"
        self.model_path = f"{self.output_dir}/joint_ext_best.pth"
        self.test_drop_prob = 0.1
        # ──────────── 时空扩展网络参数 ────────────
        self.temporal_in_steps = 3
        self.use_temporal_branch = True
        self.fusion_type = "gated"          # "concat" | "gated"
        self.ext_base_filter = 32
        self.ext_temporal_channels = 32

    def to_dict(self):
        return dict(self.__dict__)