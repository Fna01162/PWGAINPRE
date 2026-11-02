import os
import sys
import csv
import types
import platform
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

from config import Config
from untils.data import SlidingWindowDataset, Normalizer, build_generator_inputs, make_obs_rel
from model import UNetGenerator, PatchGANDiscriminator
from predictor import WindFieldPredictor
from untils.losses import masked_prediction_loss, reconstruction_loss, generator_adv_loss, discriminator_loss, tv_loss
from untils.data import generate_pure_structural_missing_mask, _continuous_ray_mask, _calibrate_missing
from untils.refiner import ResidualRefiner, apply_refiner
from test_joint import evaluate_on_test, azimuth_linear_interpolation
from train import make_training_cue, build_contextual_fusion


def set_seed(seed=42):
    """
    设置全局随机种子以确保结果可复现。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_refiner_from_checkpoint(ckpt_path, cfg, device):
    """从外部 checkpoint 加载与当前配置通道匹配的 Refiner（供 C 链路复用 D 链路的 R）。

    Returns:
        (R, refiner_phys_channels, refiner_n_wavelet_levels)；
        路径不存在 / 无 refiner 键 / 通道不匹配时返回 (None, cfg 值, cfg 值)。
    """
    phys = int(cfg.phys_channels)
    wl = int(getattr(cfg, 'n_wavelet_levels', 1))

    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, phys, wl
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"⚠️  Failed to load Refiner source {ckpt_path}: {e}")
        return None, phys, wl

    ref_state = ck.get('refiner', None)
    if ref_state is None:
        print(f"ℹ️  No refiner in {ckpt_path}, Refiner skipped")
        return None, phys, wl

    enc1 = ref_state.get('enc1.0.conv.weight', None)
    if enc1 is None:
        print(f"⚠️  Cannot infer in_channels from {ckpt_path}, Refiner skipped")
        return None, phys, wl

    from untils.wavelet import wavelet_feature_channels as _wfc
    in_ch = int(enc1.shape[1])
    expected = cfg.phys_channels + 5 + _wfc(wl)
    if in_ch != expected:
        print(f"⚠️  Refiner from {ckpt_path}: in_channels={in_ch} != expected {expected}, Refiner skipped")
        return None, phys, wl

    base_filter = int(enc1.shape[0])
    R = ResidualRefiner(in_channels=in_ch, base_filter=base_filter).to(device)
    R.load_state_dict(ref_state)
    R.eval()
    print(f"✅ Refiner loaded from {ckpt_path}: in_channels={in_ch}, base_filter={base_filter}")
    return R, phys, wl


def save_checkpoint(epoch, G, D, P, R, norm, cfg, is_best=False, is_last=False, save_epoch_ckpt=False):
    """保存模型检查点。

    [P0-4 FIX] 保存粒度由调用方控制：
      - is_best=True  → joint_best.pth
      - is_last=True  → joint_last.pth
      - save_epoch_ckpt=True → joint_epoch_{epoch+1:03d}.pth（仅每 save_every_n_epochs 保存）
    不再每 epoch 无条件写全量 checkpoint。
    """
    state = {
        'epoch': epoch,
        'model': G.state_dict(),
        'discriminator': D.state_dict(),
        'predictor': P.state_dict(),
        'normalizer': norm,
        'cfg': cfg
    }

    # [FIX] 保存 Refiner 权重（如果存在）
    if R is not None:
        state['refiner'] = R.state_dict()

    # 保存最佳模型
    if is_best:
        best_path = os.path.join(cfg.output_dir, 'joint_best.pth')
        torch.save(state, best_path)
        print(f"✅ Saved best model to {best_path}")

    # 保存最后一个模型
    if is_last:
        last_path = os.path.join(cfg.output_dir, 'joint_last.pth')
        torch.save(state, last_path)

    # 定期保存checkpoint：仅当调用方显式要求时才保存
    if save_epoch_ckpt:
        checkpoint_path = os.path.join(cfg.output_dir, f'joint_epoch_{epoch+1:03d}.pth')
        torch.save(state, checkpoint_path)
        print(f"💾 Saved periodic checkpoint to {checkpoint_path}")


def plot_training_curves(history_path, output_dir):
    """绘制训练曲线图"""
    epochs = []
    pred_losses = []

    with open(history_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            pred_losses.append(float(row['pred_mae']))

    # 绘制预测损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, pred_losses, 'b-', label='Prediction MAE', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Joint Training - Prediction Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    loss_plot_path = os.path.join(output_dir, 'loss_joint.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved training curves to {loss_plot_path}")


def train_joint(chain_mode=None):
    # 链路模式统一由 Config 管理（决定输出目录，训练与测试必须一致）
    cfg = Config(chain_mode=chain_mode)
    set_seed(cfg.random_seed)
    device = cfg.device

    # ==================== 链路配置 (A3: 四链路开关) ====================
    chain_mode = cfg.chain_mode
    # 支持的链路模式（见 Config.CHAIN_MODES）:
    # "A" - 直接预测（无补全，缺失处归零）
    # "B" - 线性插值补全 + 预测
    # "C" - GAIN补全（冻结G）+ 预测
    # "D" - 联合训练（Ours，G和P都训练）

    freeze_G = (chain_mode == "C")  # C链路冻结G
    use_completion = (chain_mode in ["B", "C", "D"])  # 是否使用补全
    use_gan = (chain_mode in ["C", "D"])  # 是否使用GAIN（非线性补全）
    train_G = (chain_mode == "D") and not freeze_G  # D链路且未冻结时训练G

    print(f"\n{'='*60}")
    print(f"Joint Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Chain Mode: {chain_mode} ({'Direct' if chain_mode=='A' else 'Linear' if chain_mode=='B' else 'Frozen-GAIN' if chain_mode=='C' else 'Joint-Training(Ours)'})")
    print(f"  Freeze G: {freeze_G}, Train G: {train_G}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  Learning Rate: {cfg.lr}")
    print(f"  Input Steps: {cfg.input_steps}, Pred Steps: {cfg.pred_steps}")
    print(f"  Lambda Pred: {cfg.lambda_pred}, Recon Weight: {cfg.recon_weight}")
    print(f"  Output Dir: {cfg.output_dir}")
    print(f"{'='*60}\n")

    # 1. 数据加载与归一化
    print("Loading data for normalizer...")
    train_data_raw = np.load(cfg.train_npy)
    obs, rel = make_obs_rel(train_data_raw, cfg.phys_channels)
    norm = Normalizer(cfg.normalize, cfg.clip_percentile, cfg.phys_channels)
    norm = norm.fit(train_data_raw, obs)
    del train_data_raw, obs, rel

    # [FIX] 必须在创建dataset之前定义actual_pred_steps！
    actual_pred_steps = max(cfg.multi_step_predictions) if hasattr(cfg, 'multi_step_predictions') else cfg.pred_steps

    # [FIX] Windows 兼容性：num_workers=0 避免 shm.dll 加载失败
    effective_num_workers = 0 if platform.system() == 'Windows' else cfg.num_workers
    if effective_num_workers == 0 and cfg.num_workers > 0:
        print(f"⚠️  Windows 系统检测到，num_workers 从 {cfg.num_workers} 强制设为 0（避免 DLL 加载错误）")

    train_dataset = SlidingWindowDataset(
        cfg.train_npy, normalizer=norm,
        input_steps=cfg.input_steps, pred_steps=actual_pred_steps,  # [FIX] 使用多步最大值
        phys_channels=cfg.phys_channels
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=effective_num_workers
    )

    # 尝试加载验证集（用于early stopping和最佳模型选择）
    val_loader = None
    if os.path.exists(cfg.val_npy):
        val_dataset = SlidingWindowDataset(
            cfg.val_npy, normalizer=norm,
            input_steps=cfg.input_steps, pred_steps=actual_pred_steps,
            phys_channels=cfg.phys_channels
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=effective_num_workers
        )
        print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
    else:
        print(f"⚠️  Validation dataset not found: {cfg.val_npy}")

    # 测试集评估已拆分到 test_joint.py（训练结束后请运行 python test_joint.py <链模式>）

    # 2. 网络初始化
    G = UNetGenerator(cfg.phys_channels * 2, 1, cfg.base_filter, use_cue=True, use_nafblock=True,
                      use_axial_w=True).to(device)
    # D 的输入是 cat(v, hint) -> 2通道，不是 phys_channels*2
    D = PatchGANDiscriminator(in_channels=2).to(device)
    # P 支持多步预测（actual_pred_steps已在前面定义）
    P = WindFieldPredictor(input_channels=cfg.phys_channels + 2, hidden_channels=cfg.hidden_channels,
                           num_layers=cfg.num_layers, kernel_size=3,
                           pred_steps=actual_pred_steps).to(device)

    # Refiner：小波+几何/物理条件引导的残差修正网络
    R = None
    refiner_ckpt = None  # 先保存 refiner 权重，用于推断 in_channels

    # 3. 加载预训练GAIN模型 (C/D链路需要)
    if use_gan and cfg.gain_pretrain_path and os.path.exists(cfg.gain_pretrain_path):
        print(f"\n📦 Loading pre-trained GAIN from: {cfg.gain_pretrain_path}")
        ckpt = torch.load(cfg.gain_pretrain_path, map_location=device, weights_only=False)

        if 'model' in ckpt:
            G.load_state_dict(ckpt['model'])
        elif 'generator' in ckpt:
            G.load_state_dict(ckpt['generator'])
        else:
            G.load_state_dict(ckpt)

        if 'discriminator' in ckpt:
            D.load_state_dict(ckpt['discriminator'])

        # [FIX] 先提取 refiner 权重（如果存在）
        if 'refiner' in ckpt:
            refiner_ckpt = ckpt['refiner']
            print(f"✅ Found refiner weights in checkpoint: {len(refiner_ckpt)} tensors")

        print("✅ Pre-trained GAIN loaded successfully!")
    elif use_gan and cfg.gain_pretrain_path:
        print(f"⚠️  Warning: Pretrain path not found: {cfg.gain_pretrain_path}")
        print("   Training from scratch...")

    # [FIX] 创建 Refiner（支持从 checkpoint 加载或根据配置新建）
    refiner_phys_channels = cfg.phys_channels  # Refiner实际使用的物理通道数
    refiner_n_wavelet_levels = int(getattr(cfg, 'n_wavelet_levels', 1))  # 小波层数
    use_checkpoint_refiner = True  # 是否尝试加载预训练权重

    if use_gan:
        if refiner_ckpt is not None:
            # 从预训练权重的 enc1.conv.weight 推断输入通道数
            enc1_weight = refiner_ckpt.get('enc1.0.conv.weight', None)
            if enc1_weight is not None:
                in_ch_from_ckpt = enc1_weight.shape[1]
                print(f"📐 Inferring Refiner in_channels from checkpoint: {in_ch_from_ckpt}")

                # 计算当前配置期望的 in_channels
                from untils.wavelet import wavelet_feature_channels as _wfc
                current_wl = _wfc(refiner_n_wavelet_levels)
                expected_in_ch = cfg.phys_channels + 5 + current_wl

                print(f"📊 Current config expects: in_channels={expected_in_ch}")
                print(f"   (phys={cfg.phys_channels} + geom+mask=5 + wavelet_{refiner_n_wavelet_levels}lvl={current_wl})")

                # 检查是否完全匹配
                if in_ch_from_ckpt == expected_in_ch:
                    # ✅ 完全匹配：直接加载
                    print("✅ Checkpoint matches current config perfectly!")
                    R = ResidualRefiner(in_channels=in_ch_from_ckpt,
                                       base_filter=getattr(cfg, 'refiner_base_filter', 16)).to(device)
                    R.load_state_dict(refiner_ckpt)
                    print(f"✅ Refiner loaded: {len(refiner_ckpt)} tensors, in_channels={in_ch_from_ckpt}")
                else:
                    # ❌ 不匹配
                    print(f"\n⚠️  Configuration mismatch detected!")
                    print(f"     Checkpoint:  in_channels={in_ch_from_ckpt} (trained with different config)")
                    print(f"     Current:     in_channels={expected_in_ch} (based on your config)")

                    if freeze_G:
                        # [P0-2 FIX] C链路：不使用随机初始化的冻结 Refiner。
                        # 随机 R 被冻结会向 GAIN 输出叠加随机扰动，污染 C 链路补全结果；
                        # 直接降级为 R=None，C 链路使用 GAIN 原始补全输出。
                        print(f"\n🔒 Chain C: falling back to R=None (no Refiner)")
                        print(f"   → 随机 Refiner 被冻结会向 GAIN 输出叠加随机扰动，")
                        print(f"     C 链路将直接使用 GAIN 原始补全输出\n")
                        R = None
                    else:
                        # D链路：保持随机初始化并从头训练（参与联合优化）
                        print(f"\n🔧 Decision: Creating NEW Refiner with current config (ignoring checkpoint weights)")
                        print(f"   → This means the Refiner will be randomly initialized and trained from scratch")
                        print(f"   → You may need more epochs for convergence\n")

                        R = ResidualRefiner(in_channels=expected_in_ch,
                                           base_filter=getattr(cfg, 'refiner_base_filter', 16)).to(device)
                        print(f"✅ New Refiner created: in_channels={expected_in_ch} (randomly initialized)")
                        print(f"   Wavelet levels: {refiner_n_wavelet_levels}, Phys channels: {cfg.phys_channels}")

                    # 保持配置值不变
                    refiner_phys_channels = cfg.phys_channels
                    refiner_n_wavelet_levels = refiner_n_wavelet_levels
            else:
                print("⚠️  Cannot infer in_channels from checkpoint, using config")
                R = None
        else:
            print("ℹ️  No refiner weights in checkpoint")
            R = None

        # 如果还没有创建 R（没有 checkpoint 或无法推断）：
        #   C链路：优先复用 D 链路（联合训练）训好的 Refiner，通道匹配才加载；
        #          加载不到则跳过 Refiner（绝不用随机初始化的冻结 Refiner 污染补全结果）
        #   D链路：按配置新建随机 Refiner，参与联合训练
        if R is None:
            if freeze_G:
                # 自动定位同任务 D 链路的 joint_best.pth（output_dir 以 "_C" 结尾）
                d_ckpt_path = os.path.join(cfg.output_dir[:-2] + '_D', 'joint_best.pth') \
                    if cfg.output_dir.endswith('_C') else None
                user_src = getattr(cfg, 'refiner_pretrain_path', '')
                src = user_src if user_src else d_ckpt_path
                R, refiner_phys_channels, refiner_n_wavelet_levels = \
                    load_refiner_from_checkpoint(src, cfg, device)
                if R is None:
                    print("ℹ️  Chain C: Refiner skipped (no compatible checkpoint) — using raw GAIN output")
            else:
                phys_ch = cfg.phys_channels
                geom_ch = 3  # Range, sinθ, cosθ
                mask_ch = 2  # given_mask, rel_mask
                base_ch = phys_ch + geom_ch + mask_ch
                if getattr(cfg, 'use_wavelet', True):
                    from untils.wavelet import wavelet_feature_channels as _wfc2
                    wl = _wfc2(int(getattr(cfg, 'n_wavelet_levels', 1)))
                else:
                    wl = 0
                in_ch = base_ch + wl
                R = ResidualRefiner(in_channels=in_ch, base_filter=getattr(cfg, 'refiner_base_filter', 16)).to(device)
                print(f"✅ ResidualRefiner created from config: in_channels={in_ch}")
                # 使用config创建，无需覆盖
                refiner_phys_channels = cfg.phys_channels
                refiner_n_wavelet_levels = int(getattr(cfg, 'n_wavelet_levels', 1))

    # A3: C链路冻结G和R的参数
    if freeze_G:
        print("🔒 Freezing Generator G (Chain Mode C)...")
        for param in G.parameters():
            param.requires_grad = False
        G.eval()
        if R is not None:
            print("🔒 Freezing Refiner R (Chain Mode C)...")
            for param in R.parameters():
                param.requires_grad = False
            R.eval()

    # 4. 优化器 (A3: 根据链路模式决定是否训练G/D/R)
    if train_G:
        # D链路：G和R一起训练（联合优化）
        trainable_params = list(filter(lambda p: p.requires_grad, G.parameters()))
        if R is not None and R.training:
            trainable_params += list(filter(lambda p: p.requires_grad, R.parameters()))
            print(f"✅ Optimizer will update G + R ({len(trainable_params)} parameter groups)")
        optim_G = torch.optim.Adam(trainable_params, lr=cfg.lr, betas=cfg.betas)
    else:
        optim_G = None  # A/B链路或C链路（冻结）不需要G优化器

    if use_gan and not freeze_G:  # D链路需要训练D
        optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)
    else:
        optim_D = None  # A/B/C链路不需要D优化器

    optim_P = torch.optim.Adam(P.parameters(), lr=cfg.lr, betas=cfg.betas)

    # 学习率衰减：只对 G（含 R）和 P，D 不衰减（避免对抗失衡）
    def _make_scheduler(optim):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=cfg.lr_factor,
            patience=cfg.lr_patience, min_lr=cfg.lr_min)

    sch_G = _make_scheduler(optim_G) if optim_G is not None else None
    sch_P = _make_scheduler(optim_P)

    # 5. 初始化训练日志
    history_path = os.path.join(cfg.output_dir, 'training_history.csv')
    csv_file = open(history_path, 'w', newline='')
    fieldnames = ['epoch', 'pred_mae', 'val_mae', 'recon_loss', 'adv_loss', 'total_loss', 'learning_rate']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_val_mae = float('inf')
    patience_counter = 0  # 早停计数器
    val_mae = float('inf')  # 当前epoch的验证集MAE

    print(f"\n🚀 Starting Joint Training for {cfg.epochs} epochs...\n")

    # 6. 联合训练循环
    for epoch in range(cfg.epochs):
        if use_gan:
            G.train() if not freeze_G else G.eval()  # A3: C链路保持eval
        if optim_D is not None:
            D.train()
        P.train()

        epoch_pred_loss = 0
        epoch_recon_loss = 0
        epoch_adv_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [{chain_mode}]")

        for step, (hx, hm, hr, fx, fm, fr) in enumerate(pbar):
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)

            B, C, Tin, H, W = hx.shape

            # ==================== [口径对齐 train.py] 结构楔 mask + 只遮 target 帧 + cue 填充 ====================
            valid_mask_h = (hm * hr)  # [B, C, Tin, H, W]
            target_t = Tin - 1

            # 结构楔（train.py generate_pure_structural_missing_mask 同源），只作用于 target 帧
            vrel_t = hr[:, 0:1, target_t]
            vval_t = valid_mask_h[:, 0:1, target_t]
            ve_t = vrel_t * vval_t
            art_target, _rates = generate_pure_structural_missing_mask(
                B, H, W, device, cfg.drop_prob, cfg=cfg, valid_mask=ve_t)
            art_target = art_target * ve_t  # [B,1,H,W] target 帧遮挡

            art_mask = torch.zeros(B, C, Tin, H, W, device=device)
            art_mask[:, :, target_t:target_t + 1] = art_target.unsqueeze(1)

            # 只对 target 帧打人工缺失 + cue 填充（train.py build_contextual_fusion 口径）
            m_t = valid_mask_h * (1 - art_mask)
            given_all = hm * (1 - art_mask)
            xin = hx.clone()
            xin[:, :, target_t:target_t + 1] = (
                hx[:, :, target_t:target_t + 1]
                * given_all[:, :, target_t:target_t + 1])
            min_ = hm.clone()
            min_[:, :, target_t:target_t + 1] = (
                given_all[:, :, target_t:target_t + 1])
            if train_G:
                z = torch.randn_like(xin)
                noise_scale = cfg.noise_scale
            else:
                z = torch.zeros_like(xin)
                noise_scale = 0.0
            cue, em = make_training_cue(xin, min_, hr, hm, target_t)
            if not getattr(cfg, "use_cue", True):
                cue = torch.zeros(B, 1, H, W, device=device, dtype=hx.dtype)
            xb = build_contextual_fusion(xin, em, cue, z, noise_scale)

            # ==================== 补全阶段 ====================
            if chain_mode == "A":
                # A链路：无补全，直接用缺失归零的数据
                v_completed = xin[:, 0:1, :, :, :]  # 只取V通道 [B, 1, Tin, H, W]
            elif chain_mode == "B":
                # [P1-2 FIX] B链路：使用统一的方位向线性插值函数
                missing_mask_b = (m_t[:, 0:1, :, :, :] < 0.5)  # [B, 1, Tin, H, W]
                v_completed = azimuth_linear_interpolation(xin[:, 0:1, :, :, :], missing_mask_b)
            else:
                # C/D链路：使用GAIN补全（整帧前向，与预训练 time_steps=10 一致）
                with torch.set_grad_enabled(train_G):
                    v_completed = G(xb, em)  # [B, 1, Tin, H, W]

                    # [口径对齐] Refiner 只 refine target 帧（其余帧无人工缺失）
                    if R is not None:
                        # [向下兼容] 准备 Refiner 专用的 cfg（可能需要降维/调整小波层数）
                        needs_override = (refiner_phys_channels != cfg.phys_channels or
                                          refiner_n_wavelet_levels != int(getattr(cfg, 'n_wavelet_levels', 1)))
                        if needs_override:
                            refiner_cfg = types.SimpleNamespace(**cfg.__dict__)
                            refiner_cfg.phys_channels = refiner_phys_channels
                            refiner_cfg.n_wavelet_levels = refiner_n_wavelet_levels
                        else:
                            refiner_cfg = cfg

                        v_t = v_completed[:, :, target_t, :, :]  # [B, 1, H, W]
                        x_t = hx[:, :, target_t:target_t + 1, :, :]  # 5D 保持时间维
                        if x_t.shape[1] > refiner_phys_channels:
                            x_t = x_t[:, :refiner_phys_channels, :, :]
                        given_t = hm[:, :, target_t, :, :]  # [B, 1, H, W]
                        rel_t = hr[:, :, target_t, :, :]  # [B, 1, H, W]
                        valid_t = valid_mask_h[:, :, target_t, :, :]  # [B, 1, H, W]

                        with torch.set_grad_enabled(train_G):
                            v_refined_t = apply_refiner(
                                R, v_t, x_t, given_t, rel_t, valid_t,
                                refiner_cfg, device)
                        v_completed = torch.cat([
                            v_completed[:, :, :target_t],
                            v_refined_t.unsqueeze(2),
                            v_completed[:, :, target_t + 1:]], dim=2)

            # ==================== [P0-1+P0-2 FIX] 完整的联合训练（仅D链路）====================
            # 核心改进：
            # 1. D使用逐帧2D调用：D(v_4d, hint_4d)，其中v和hint都是[B,1,H,W]
            # 2. 合并成一次backward：total = recon + adv + tv + pred_mae，让pred_mae能反传到G

            if use_gan and train_G and optim_D is not None and optim_G is not None:
                # ====== 第一步：更新判别器D（用detach后的G输出）======
                optim_D.zero_grad()

                # [P0-1 FIX] 取最后一帧的V和hint，转为4D [B,1,H,W]
                v_last_real = hx[:, 0:1, -1, :, :]        # [B, 1, H, W]
                hint_last_real = valid_mask_h[:, 0:1, -1, :, :]  # [B, 1, H, W]

                # 真实数据过D
                d_real = D(v_last_real, hint_last_real)
                d_real_loss = discriminator_loss(d_real,
                                                  (1 - art_mask[:, 0, -1, :, :]).unsqueeze(1),  # [B,1,H,W]
                                                  valid_mask_h[:, 0:1, -1, :, :])

                # 假数据（G生成的）过D（注意：v_completed要detach）
                v_last_fake = v_completed[:, 0:1, -1, :, :].detach()  # [B, 1, H, W]
                hint_last_fake = valid_mask_h[:, 0:1, -1, :, :]       # [B, 1, H, W]
                d_fake = D(v_last_fake, hint_last_fake)
                d_fake_loss = discriminator_loss(d_fake,
                                                 art_mask[:, 0, -1, :, :].unsqueeze(1),  # [B,1,H,W]
                                                 valid_mask_h[:, 0:1, -1, :, :])

                d_total_loss = (d_real_loss + d_fake_loss) * 0.5
                d_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
                optim_D.step()

                # ====== 第二步：联合更新G和P（一次backward，让pred_mae反传到G）======
                optim_G.zero_grad()
                optim_P.zero_grad()

                # G的重建损失（在人工缺失区域）
                recon_loss = reconstruction_loss(v_completed,
                                                  hx[:, 0:1, :, :, :],
                                                  art_mask[:, 0:1, :, :, :],
                                                  valid_mask_h[:, 0:1, :, :, :])

                # G的对抗损失（欺骗D）
                v_last_for_g = v_completed[:, 0:1, -1, :, :]  # [B, 1, H, W]，不detach！保持梯度
                hint_for_g = valid_mask_h[:, 0:1, -1, :, :]
                d_logits_g = D(v_last_for_g, hint_for_g)
                adv_loss = generator_adv_loss(d_logits_g,
                                              art_mask[:, 0, -1, :, :].unsqueeze(1),
                                              valid_mask_h[:, 0:1, -1, :, :])

                # TV平滑损失
                tv = tv_loss(v_completed, art_mask[:, 0:1, :, :, :], valid_mask_h[:, 0:1, :, :, :])

                # 构建预测器输入
                features = hx[:, 1:cfg.phys_channels, :, :, :]
                pred_in = torch.cat([
                    v_completed,  # 这里v_completed会继续传递梯度到G！
                    features,
                    hm[:, 0:1, :, :, :],
                    hr[:, 0:1, :, :, :]
                ], dim=1)

                pred_in_p = pred_in.permute(0, 2, 1, 3, 4).contiguous()
                out_pred = P(pred_in_p)  # [B, Tout, 1, H, W]

                # [P1-1 FIX] 多步预测损失：对所有配置的步长都计算loss
                # [FIX] fx布局是[B,C,T,H,W]，时间在dim=2，必须在dim=2切+固定通道0
                pred_mae_total = torch.tensor(0.0, device=device)
                for pred_step in cfg.multi_step_predictions:
                    if pred_step <= out_pred.shape[1]:
                        # 正确切片：通道0(V) + 时间维pred_step-1:pred_step
                        tgt_step = fx[:, 0:1, pred_step-1:pred_step, :, :].permute(0, 2, 1, 3, 4)
                        mask_step = (fm * fr)[:, 0:1, pred_step-1:pred_step, :, :].permute(0, 2, 1, 3, 4)
                        pred_step_out = out_pred[:, pred_step-1:pred_step, :, :, :]
                        pred_mae_step = masked_prediction_loss(pred_step_out, tgt_step, mask_step)
                        pred_mae_total = pred_mae_total + pred_mae_step

                # 平均多步loss
                pred_mae = pred_mae_total / len(cfg.multi_step_predictions)

                # [P0-2 FIX] 关键：合并所有损失，一次backward！
                # 这样pred_mae的梯度可以同时流向G（通过v_completed）和P
                total_loss = (cfg.recon_weight * recon_loss) + \
                             adv_loss + \
                             (0.1 * tv) + \
                             (cfg.lambda_pred * pred_mae)

                assert torch.isfinite(total_loss).all(), "Fatal Error: Loss is NaN/Inf"
                total_loss.backward()

                # 分别裁剪并更新G和P
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
                torch.nn.utils.clip_grad_norm_(P.parameters(), cfg.grad_clip)
                optim_G.step()  # G被recon+adv+tv+pred_mae共同更新 ✅
                optim_P.step()  # P只被pred_mae更新（其他项不经过P）✅

                recon_loss_val = recon_loss.detach()
                adv_loss_val = adv_loss.detach()

            else:
                # A/B/C链路：只有P的训练（无GAN）
                optim_P.zero_grad()

                features = hx[:, 1:cfg.phys_channels, :, :, :]
                pred_in = torch.cat([
                    v_completed,
                    features,
                    hm[:, 0:1, :, :, :],
                    hr[:, 0:1, :, :, :]
                ], dim=1)

                pred_in_p = pred_in.permute(0, 2, 1, 3, 4).contiguous()
                out_pred = P(pred_in_p)

                # 多步预测损失
                # [FIX] fx布局是[B,C,T,H,W]，时间在dim=2，必须在dim=2切+固定通道0
                pred_mae_total = torch.tensor(0.0, device=device)
                for pred_step in cfg.multi_step_predictions:
                    if pred_step <= out_pred.shape[1]:
                        # 正确切片：通道0(V) + 时间维pred_step-1:pred_step
                        tgt_step = fx[:, 0:1, pred_step-1:pred_step, :, :].permute(0, 2, 1, 3, 4)
                        mask_step = (fm * fr)[:, 0:1, pred_step-1:pred_step, :, :].permute(0, 2, 1, 3, 4)
                        pred_step_out = out_pred[:, pred_step-1:pred_step, :, :, :]
                        pred_mae_step = masked_prediction_loss(pred_step_out, tgt_step, mask_step)
                        pred_mae_total = pred_mae_total + pred_mae_step

                pred_mae = pred_mae_total / len(cfg.multi_step_predictions)
                total_loss = cfg.lambda_pred * pred_mae

                assert torch.isfinite(total_loss).all(), "Fatal Error: Loss is NaN/Inf"
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(P.parameters(), cfg.grad_clip)
                optim_P.step()

                recon_loss_val = torch.tensor(0.0, device=device)
                adv_loss_val = torch.tensor(0.0, device=device)

            epoch_pred_loss += pred_mae.item()
            epoch_recon_loss += recon_loss_val.item()
            epoch_adv_loss += adv_loss_val.item()
            pbar.set_postfix({
                'P_MAE': f'{pred_mae.item():.4f}',
                'Recon': f'{recon_loss_val.item():.4f}' if use_gan else 'N/A',
                'Adv': f'{adv_loss_val.item():.4f}' if use_gan else 'N/A'
            })

        # --- Epoch 结束处理 ---
        avg_pred_loss = epoch_pred_loss / max(len(train_loader), 1)
        avg_recon_loss = epoch_recon_loss / max(len(train_loader), 1)
        avg_adv_loss = epoch_adv_loss / max(len(train_loader), 1)
        current_lr = optim_P.param_groups[0]['lr']  # 使用P的lr（P总是训练的）

        # 写入训练日志
        writer.writerow({
            'epoch': epoch + 1,
            'pred_mae': f'{avg_pred_loss:.6f}',
            'val_mae': f'{val_mae:.6f}',
            'recon_loss': f'{avg_recon_loss:.6f}',
            'adv_loss': f'{avg_adv_loss:.6f}',
            'total_loss': f'{cfg.lambda_pred * avg_pred_loss + cfg.recon_weight * avg_recon_loss + avg_adv_loss:.6f}',
            'learning_rate': f'{current_lr:.8f}'
        })
        csv_file.flush()

        # 选最佳 / 衰减 / 早停都在"验证后"统一进行，计数单位是"验证次数"，
        # 这样衰减（lr_patience）一定早于早停（early_stopping_patience），
        # 给降低后的学习率留出改善窗口，而不是还没衰减就被早停。
        is_best = False
        stop_now = False
        do_val = val_loader is not None and \
            (epoch + 1) % cfg.eval_val_epochs == 0

        if do_val:
            _, val_mae = evaluate_on_test(
                G, P, R, refiner_phys_channels, refiner_n_wavelet_levels,
                val_loader, norm, cfg, device, chain_mode=chain_mode
            )
            print(f"  Val P_MAE: {val_mae:.6f} (Best: {best_val_mae:.6f})")
            is_best = val_mae < best_val_mae
            if is_best:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
            # 学习率衰减（按验证次数，内部计数）
            if sch_G is not None:
                sch_G.step(val_mae)
            sch_P.step(val_mae)
            if patience_counter >= cfg.early_stopping_patience:
                stop_now = True
        elif val_loader is None:
            # 无验证集：回退到训练 P_MAE
            is_best = avg_pred_loss < best_val_mae
            if is_best:
                best_val_mae = avg_pred_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                stop_now = True

        is_last = (epoch == cfg.epochs - 1)
        # [P0-4 FIX] 保存粒度三档：best / last / 每 save_every_n_epochs 的周期 checkpoint，
        # 其余 epoch 不再写任何全量文件
        is_periodic = ((epoch + 1) % cfg.save_every_n_epochs == 0)
        save_checkpoint(epoch, G, D, P, R, norm, cfg, is_best=is_best, is_last=is_last,
                        save_epoch_ckpt=is_periodic and not is_best and not is_last)

        print(f"Epoch {epoch+1}/{cfg.epochs} [{chain_mode}] - Train P_MAE: {avg_pred_loss:.6f} | Val P_MAE: {val_mae:.6f} (Best: {best_val_mae:.6f})")
        if use_gan and train_G:
            print(f"  Recon Loss: {avg_recon_loss:.6f}, Adv Loss: {avg_adv_loss:.6f}")

        # 早停检查
        if stop_now:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs "
                  f"(patience: {cfg.early_stopping_patience} 次验证)")
            break

    # 7. 关闭日志文件
    csv_file.close()

    # 8. 生成训练曲线图
    print("\n📊 Generating training curves...")
    plot_training_curves(history_path, cfg.output_dir)

    # 9. 测试集评估已拆分到 test_joint.py
    print(f"\n💡 测试集评估请运行: python test_joint.py {chain_mode}")

    print(f"\n{'='*60}")
    print(f"✅ Training Completed!")
    print(f"   Best Model: {os.path.join(cfg.output_dir, 'joint_best.pth')}")
    print(f"   Training Log: {history_path}")
    print(f"   Output Directory: {cfg.output_dir}")
    print(f"{'='*60}\n")


def main():
    # 命令行仅负责选择链路模式，具体行为与输出目录统一由 Config 决定
    # 只取合法的链路模式参数，兼容 `python train_joint.py D` 与 `python main.py joint D`
    chain_mode = None
    for arg in sys.argv[1:]:
        if arg.upper() in Config.CHAIN_MODES:
            chain_mode = arg.upper()
    train_joint(chain_mode=chain_mode)


if __name__ == "__main__":
    main()