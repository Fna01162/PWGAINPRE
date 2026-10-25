import os
import csv
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
from untils.losses import masked_prediction_loss
from untils.data import generate_pure_structural_missing_mask


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


def save_checkpoint(epoch, G, D, P, norm, cfg, is_best=False, is_last=False):
    """保存模型检查点"""
    state = {
        'epoch': epoch,
        'model': G.state_dict(),
        'discriminator': D.state_dict(),
        'predictor': P.state_dict(),
        'normalizer': norm,
        'cfg': cfg
    }

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


def evaluate_on_test(G, P, test_loader, norm, cfg, device):
    """在测试集上评估模型"""
    G.eval()
    P.eval()

    total_pred_mae = 0
    total_samples = 0

    with torch.no_grad():
        for hx, hm, hr, fx, fm, fr in tqdm(test_loader, desc="Testing"):
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)

            B, C, Tin, H, W = hx.shape

            # 生成缺失掩码
            art_mask, _ = generate_pure_structural_missing_mask(
                B, H, W, device,
                drop_prob=cfg.missing_rate,
                valid_mask=(hm * hr)
            )
            art_mask = art_mask.unsqueeze(1).expand(-1, C, Tin, -1, -1)

            valid_mask_h = hm * hr
            m_t = valid_mask_h * (1 - art_mask)
            xin = hx * m_t

            # 补全
            xb, eb = build_generator_inputs(xin, m_t)
            v_completed = G(xb.to(device), m_t.to(device))

            # 预测
            features = hx[:, 1:cfg.phys_channels, :, :, :]
            pred_in = torch.cat([
                v_completed,
                features,
                hm[:, 0:1, :, :, :],
                hr[:, 0:1, :, :, :]
            ], dim=1)

            pred_in_p = pred_in.permute(0, 2, 1, 3, 4).contiguous()
            out_pred = P(pred_in_p)

            # 计算预测误差
            tgt = fx[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            valid_mask_f = (fm * fr)[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            pred_mae = masked_prediction_loss(out_pred, tgt, valid_mask_f)

            total_pred_mae += pred_mae.item() * B
            total_samples += B

    avg_pred_mae = total_pred_mae / max(total_samples, 1)

    # 保存测试结果
    test_results = {
        'test_pred_mae': avg_pred_mae,
        'test_samples': total_samples
    }

    test_path = os.path.join(cfg.output_dir, 'test_completed.npy')
    np.save(test_path, test_results)
    print(f"📝 Test Results saved to {test_path}")
    print(f"   Test Prediction MAE: {avg_pred_mae:.6f}")

    return avg_pred_mae


def train_joint():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = cfg.device

    print(f"\n{'='*60}")
    print(f"Joint Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  Learning Rate: {cfg.lr}")
    print(f"  Input Steps: {cfg.input_steps}, Pred Steps: {cfg.pred_steps}")
    print(f"  Lambda Pred: {cfg.lambda_pred}")
    print(f"  Output Dir: {cfg.output_dir}")
    print(f"{'='*60}\n")

    # 1. 数据加载与归一化
    print("Loading data for normalizer...")
    train_data_raw = np.load(cfg.train_npy)
    obs, rel = make_obs_rel(train_data_raw, cfg.phys_channels)
    norm = Normalizer(cfg.normalize, cfg.clip_percentile, cfg.phys_channels)
    norm = norm.fit(train_data_raw, obs)
    del train_data_raw, obs, rel

    train_dataset = SlidingWindowDataset(
        cfg.train_npy, normalizer=norm,
        input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
        phys_channels=cfg.phys_channels
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    # 尝试加载测试集
    test_loader = None
    if os.path.exists(cfg.test_npy):
        test_dataset = SlidingWindowDataset(
            cfg.test_npy, normalizer=norm,
            input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
            phys_channels=cfg.phys_channels
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

    # 2. 网络初始化
    G = UNetGenerator(cfg.phys_channels * 2, 1, cfg.base_filter, use_cue=True, use_nafblock=True,
                      use_axial_w=True).to(device)
    D = PatchGANDiscriminator(cfg.phys_channels * 2).to(device)
    P = WindFieldPredictor(input_channels=cfg.phys_channels + 2, hidden_channels=cfg.hidden_channels,
                           num_layers=cfg.num_layers, kernel_size=3,
                           pred_steps=cfg.pred_steps).to(device)

    # 3. 加载预训练GAIN模型
    if cfg.gain_pretrain_path and os.path.exists(cfg.gain_pretrain_path):
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

        print("✅ Pre-trained GAIN loaded successfully!")
    elif cfg.gain_pretrain_path:
        print(f"⚠️  Warning: Pretrain path not found: {cfg.gain_pretrain_path}")
        print("   Training from scratch...")

    # 4. 优化器
    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)
    optim_P = torch.optim.Adam(P.parameters(), lr=cfg.lr, betas=cfg.betas)

    # 5. 初始化训练日志
    history_path = os.path.join(cfg.output_dir, 'training_history.csv')
    csv_file = open(history_path, 'w', newline='')
    fieldnames = ['epoch', 'pred_mae', 'learning_rate']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_loss = float('inf')
    patience_counter = 0  # 早停计数器

    print(f"\n🚀 Starting Joint Training for {cfg.epochs} epochs...\n")

    # 6. 联合训练循环
    for epoch in range(cfg.epochs):
        G.train()
        D.train()
        P.train()

        epoch_pred_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} Joint")

        for step, (hx, hm, hr, fx, fm, fr) in enumerate(pbar):
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)

            B, C, Tin, H, W = hx.shape

            # 创建缺失掩码
            art_mask, _ = generate_pure_structural_missing_mask(
                B, H, W, device,
                drop_prob=cfg.missing_rate,
                valid_mask=(hm * hr)
            )
            art_mask = art_mask.unsqueeze(1).expand(-1, C, Tin, -1, -1)

            valid_mask_h = hm * hr
            m_t = valid_mask_h * (1 - art_mask)

            # 强制归零缺失区域
            xin = hx * m_t

            # 构建生成器输入
            xb, eb = build_generator_inputs(xin, m_t)
            gx, gm = xb.to(device), m_t.to(device)

            # --- 联合优化 ---
            optim_D.zero_grad()
            optim_G.zero_grad()
            optim_P.zero_grad()

            # 判别器训练（简化版）
            # TODO: 添加完整的判别器训练逻辑
            optim_D.step()

            # 生成器补全
            v_completed = G(gx, gm)

            # 构建预测器输入
            features = hx[:, 1:cfg.phys_channels, :, :, :]
            pred_in = torch.cat([
                v_completed,
                features,
                hm[:, 0:1, :, :, :],
                hr[:, 0:1, :, :, :]
            ], dim=1)

            # 维度调整 [B, C, T, H, W] -> [B, T, C, H, W]
            pred_in_p = pred_in.permute(0, 2, 1, 3, 4).contiguous()
            out_pred = P(pred_in_p)

            # 预测损失
            tgt = fx[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            valid_mask_f = (fm * fr)[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            pred_mae = masked_prediction_loss(out_pred, tgt, valid_mask_f)

            # 联合损失
            g_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_loss = g_loss + (cfg.lambda_pred * pred_mae)

            assert torch.isfinite(total_loss).all(), "Fatal Error: Loss is NaN/Inf"
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(P.parameters(), cfg.grad_clip)

            optim_G.step()
            optim_P.step()

            epoch_pred_loss += pred_mae.item()
            pbar.set_postfix({'P_MAE': f'{pred_mae.item():.4f}'})

        # --- Epoch 结束处理 ---
        avg_loss = epoch_pred_loss / len(train_loader)
        current_lr = optim_G.param_groups[0]['lr']

        # 写入训练日志
        writer.writerow({
            'epoch': epoch + 1,
            'pred_mae': f'{avg_loss:.6f}',
            'learning_rate': f'{current_lr:.8f}'
        })
        csv_file.flush()

        # 保存模型
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        is_last = (epoch == cfg.epochs - 1)
        save_checkpoint(epoch, G, D, P, norm, cfg, is_best=is_best, is_last=is_last)

        # 定期保存checkpoint
        if (epoch + 1) % cfg.save_every_n_epochs == 0 and not is_best and not is_last:
            save_checkpoint(epoch, G, D, P, norm, cfg, is_best=False, is_last=False)

        print(f"Epoch {epoch+1}/{cfg.epochs} - Avg P_MAE: {avg_loss:.6f} (Best: {best_loss:.6f})")

        # 早停检查
        if patience_counter >= cfg.early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs (patience: {cfg.early_stopping_patience})")
            break

    # 7. 关闭日志文件
    csv_file.close()

    # 8. 生成训练曲线图
    print("\n📊 Generating training curves...")
    plot_training_curves(history_path, cfg.output_dir)

    # 9. 测试集评估
    if test_loader is not None:
        print("\n🧪 Evaluating on test set...")
        evaluate_on_test(G, P, test_loader, norm, cfg, device)

    print(f"\n{'='*60}")
    print(f"✅ Training Completed!")
    print(f"   Best Model: {os.path.join(cfg.output_dir, 'joint_best.pth')}")
    print(f"   Training Log: {history_path}")
    print(f"   Output Directory: {cfg.output_dir}")
    print(f"{'='*60}\n")


def main():
    train_joint()


if __name__ == "__main__":
    main()