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


def save_checkpoint(epoch, G, D, P, R, norm, cfg, is_best=False, is_last=False):
    """保存模型检查点"""
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

    # 定期保存checkpoint
    checkpoint_path = os.path.join(cfg.output_dir, f'joint_epoch_{epoch+1:03d}.pth')
    torch.save(state, checkpoint_path)


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
                    # ❌ 不匹配：询问用户选择（这里默认使用新配置）
                    print(f"\n⚠️  Configuration mismatch detected!")
                    print(f"     Checkpoint:  in_channels={in_ch_from_ckpt} (trained with different config)")
                    print(f"     Current:     in_channels={expected_in_ch} (based on your config)")
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

        # 如果还没有创建 R（没有 checkpoint 或无法推断），用配置计算
        if R is None:
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

            # ==================== A4: 改用真扇区mask + 精确校准 ====================
            valid_mask_h = (hm * hr)  # [B, C, Tin, H, W]

            # 使用连续射线掩码（真20%方位扇区）+ 校准到目标缺失率
            art_mask_list = []
            for b in range(B):
                allowed = valid_mask_h[b, 0, 0].cpu().numpy()  # 取第一个时间步、第一个通道的掩码
                # 生成连续射线扇区（约25条射线对应20%方位，W=125时）
                n_rays = max(1, int(cfg.missing_rate * W))  # 0.15*125 ≈ 19条射线
                raw_mask = _continuous_ray_mask(H, W, n_rays)
                # 校准到精确的目标缺失率
                calibrated_mask = _calibrate_missing(raw_mask, cfg.missing_rate, allowed > 0.5)
                art_mask_list.append(calibrated_mask)

            art_mask_np = np.stack(art_mask_list)  # [B, H, W]
            art_mask = torch.from_numpy(art_mask_np).unsqueeze(1).float().to(device)  # [B, 1, H, W]
            art_mask = art_mask.unsqueeze(1).expand(-1, C, Tin, -1, -1)  # [B, C, Tin, H, W]

            m_t = valid_mask_h * (1 - art_mask)  # 有效且非人工缺失的区域

            # 强制归零缺失区域
            xin = hx * m_t

            # ==================== 补全阶段 ====================
            if chain_mode == "A":
                # A链路：无补全，直接用缺失归零的数据
                v_completed = xin[:, 0:1, :, :, :]  # 只取V通道 [B, 1, Tin, H, W]
            elif chain_mode == "B":
                # [P1-2 FIX] B链路：使用统一的方位向线性插值函数
                missing_mask_b = (m_t[:, 0:1, :, :, :] < 0.5)  # [B, 1, Tin, H, W]
                v_completed = azimuth_linear_interpolation(xin[:, 0:1, :, :, :], missing_mask_b)
            else:
                # C/D链路：使用GAIN补全
                xb, eb = build_generator_inputs(xin, m_t)
                gx, gm = xb.to(device), m_t.to(device)

                with torch.set_grad_enabled(train_G):  # A3: 只有D链路才计算G的梯度
                    v_completed = G(gx, gm)  # [B, 1, Tin, H, W]

                    # [FIX] GAIN补全后，应用Refiner做残差修正（逐帧）
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

                        v_refined_list = []
                        for t in range(Tin):
                            v_t = v_completed[:, :, t, :, :]  # [B, 1, H, W]
                            x_t = hx[:, :, t:t+1, :, :]  # [FIX] 5D保持时间维，build_refiner_input需要

                            # 切片到 Refiner 能处理的通道数
                            if x_t.shape[1] > refiner_phys_channels:
                                x_t = x_t[:, :refiner_phys_channels, :, :]
                            given_t = hm[:, :, t, :, :]  # [B, 1, H, W]
                            rel_t = hr[:, :, t, :, :]  # [B, 1, H, W]
                            valid_t = valid_mask_h[:, :, t, :, :]  # [B, 1, H, W]

                            with torch.set_grad_enabled(train_G):
                                # 使用 refiner_cfg（phys_channels 已调整）
                                v_refined_t = apply_refiner(R, v_t, x_t, given_t, rel_t, valid_t, refiner_cfg, device)
                            v_refined_list.append(v_refined_t)

                        v_completed = torch.stack(v_refined_list, dim=2)  # [B, 1, Tin, H, W]

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

        # 定期在验证集上评估（用于early stopping和选最佳模型）
        if val_loader is not None and (epoch + 1) % cfg.eval_val_epochs == 0:
            _, val_mae = evaluate_on_test(
                G, P, R, refiner_phys_channels, refiner_n_wavelet_levels,
                val_loader, norm, cfg, device, chain_mode=chain_mode
            )
            print(f"  Val P_MAE: {val_mae:.6f} (Best: {best_val_mae:.6f})")

        # 基于验证集选最佳模型（仅在验证已执行过后更新计数器）
        # 如果没有验证集，回退到使用训练集P_MAE
        if val_mae != float('inf'):
            is_best = val_mae < best_val_mae
        else:
            is_best = avg_pred_loss < best_val_mae

        if is_best:
            best_val_mae = val_mae if val_mae != float('inf') else avg_pred_loss
            patience_counter = 0
        else:
            patience_counter += 1

        is_last = (epoch == cfg.epochs - 1)
        save_checkpoint(epoch, G, D, P, R, norm, cfg, is_best=is_best, is_last=is_last)

        # 定期保存checkpoint
        if (epoch + 1) % cfg.save_every_n_epochs == 0 and not is_best and not is_last:
            save_checkpoint(epoch, G, D, P, R, norm, cfg, is_best=False, is_last=False)

        print(f"Epoch {epoch+1}/{cfg.epochs} [{chain_mode}] - Train P_MAE: {avg_pred_loss:.6f} | Val P_MAE: {val_mae:.6f} (Best: {best_val_mae:.6f})")
        if use_gan and train_G:
            print(f"  Recon Loss: {avg_recon_loss:.6f}, Adv Loss: {avg_adv_loss:.6f}")

        # 早停检查
        if patience_counter >= cfg.early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs (patience: {cfg.early_stopping_patience})")
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
    chain_mode = sys.argv[1] if len(sys.argv) > 1 else None
    train_joint(chain_mode=chain_mode)


if __name__ == "__main__":
    main()