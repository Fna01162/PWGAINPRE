import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import Config
from untils.data import SlidingWindowDataset, build_generator_inputs
from model import UNetGenerator
from predictor import WindFieldPredictor
from untils.losses import masked_prediction_loss, compute_prediction_metrics


def assemble_predictor_input(
        v_completed: torch.Tensor,
        history_x: torch.Tensor,
        history_m: torch.Tensor,
        history_r: torch.Tensor,
        cfg: Config
) -> torch.Tensor:
    """
    组装预测网络 P 的输入特征。

    [Fix S7]: 修正切分维度错误，强制沿通道维度 (dim=1) 拼接。

    Args:
        v_completed: G 网络补全后的径向速度，Shape [B, 1, Tin, H, W]
        history_x: 原始历史输入数据，Shape [B, C, Tin, H, W]
        history_m: 历史数据的观测掩码，Shape [B, C, Tin, H, W]
        history_r: 历史数据的可靠性掩码，Shape [B, C, Tin, H, W]
        cfg: 全局配置对象

    Returns:
        转换布局后的预测器输入张量，Shape [B, Tin, Channels, H, W]
    """
    # 提取物理特征通道 (忽略未补全的 V 通道 0) -> Shape: [B, 3, Tin, H, W]
    features = history_x[:, 1:cfg.phys_channels, :, :, :]

    # 拼接: v_comp(1) + features(3) + obs_mask(1) + rel_mask(1) = 6 个通道
    pred_in = torch.cat([
        v_completed,
        features,
        history_m[:, 0:1, :, :, :],
        history_r[:, 0:1, :, :, :]
    ], dim=1)

    # 将时间步 (Tin) 调至第 1 维供预测器使用 -> Shape: [B, Tin, Channels, H, W]
    pred_in = pred_in.permute(0, 2, 1, 3, 4).contiguous()
    return pred_in


def evaluate_prediction(
        model_G: nn.Module,
        model_P: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        cfg: Config,
        device: torch.device
) -> dict:
    """
    在测试集上评估预测链路的整体性能（C5: 多指标版本）。

    Args:
        model_G: 训练好的生成器/补全网络
        model_P: 训练好的预测网络
        dataloader: 测试集加载器
        cfg: 配置对象
        device: 计算设备

    Returns:
        dict: 包含 MAE, RMSE, Corr, SSIM 等指标的字典
    """
    model_G.eval()
    model_P.eval()

    all_metrics = {
        'mae': [], 'rmse': [], 'corr': [], 'ssim': [], 'valid_ratio': []
    }
    total_samples = 0

    with torch.no_grad():
        for step, (hx, hm, hr, fx, fm, fr) in enumerate(tqdm(dataloader, desc="Evaluating")):
            # 将所有张量送入设备
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)
            B = hx.shape[0]

            # --- 1. 补全阶段 (Imputation Stage) ---
            valid_mask_h = hm * hr
            xb, eb = build_generator_inputs(hx, valid_mask_h)

            v_completed = model_G(xb, valid_mask_h)

            # [Note] Refiner (ResidualRefiner) 已在 train.py 中完整训练并保存在 G_best.pth 的 'refiner' 字段中
            # 此测试脚本为简化版本，暂未加载 Refiner；如需完整的 GAIN+Refiner 补全效果，请使用 train_joint.py

            # --- 2. 预测阶段 (Prediction Stage) ---
            pred_in = assemble_predictor_input(v_completed, hx, hm, hr, cfg)
            out_pred = model_P(pred_in)

            # --- 3. 多指标评估 (C5: 完整指标) ---
            tgt = fx[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            valid_mask_f = (fm * fr)[:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)

            mae, rmse, corr, ssim_val, valid_pts, valid_ratio = compute_prediction_metrics(
                out_pred, tgt, valid_mask_f
            )

            all_metrics['mae'].append(mae * B)
            all_metrics['rmse'].append(rmse * B)
            all_metrics['corr'].append(corr * B)
            all_metrics['ssim'].append(ssim_val * B)
            all_metrics['valid_ratio'].append(valid_ratio * B)
            total_samples += B

    # 计算加权平均
    avg_results = {}
    for key in all_metrics:
        avg_results[key] = sum(all_metrics[key]) / max(total_samples, 1)

    return avg_results


def test_main() -> None:
    """
    测试脚本入口点。
    负责加载配置、检查点，并输出归一化后的预测 MAE。
    """
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"[Error] Checkpoint not found at: {cfg.model_path}. Please train first.")

    print(f"[Info] Loading checkpoint from {cfg.model_path}")
    ckpt = torch.load(cfg.model_path, map_location=device)
    norm = ckpt['normalizer']

    # ==========================================
    # 实例化数据集 (Dataset & DataLoader)
    # ==========================================
    test_dataset = SlidingWindowDataset(
        cfg.test_npy, normalizer=norm,
        input_steps=cfg.input_steps, pred_steps=cfg.pred_steps,
        phys_channels=cfg.phys_channels
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4
    )

    # ==========================================
    # 初始化并加载模型 (Model Setup)
    # ==========================================
    G = UNetGenerator(
        cfg.phys_channels * 2, 1, cfg.base_filter, use_cue=True,
        use_nafblock=True, use_axial_w=True
    ).to(device)

    P = WindFieldPredictor(
        input_channels=cfg.phys_channels + 2,
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        kernel_size=3,
        pred_steps=cfg.pred_steps
    ).to(device)

    # [Fix S8]: 保持读取 key 名称为 'model'，杜绝 KeyError
    G.load_state_dict(ckpt['model'])
    P.load_state_dict(ckpt['predictor'])

    print("[Info] Starting prediction evaluation workflow...")
    test_metrics = evaluate_prediction(G, P, test_loader, cfg, device)

    print("=" * 70)
    print(f" 📊 Test Evaluation Result (C5: Full Metrics)")
    print("=" * 70)
    print(f" > Model: {cfg.output_dir}")  # [B2] 修复：使用output_dir替代不存在的exp_name
    print(f" > MAE:          {test_metrics['mae']:.6f}")
    print(f" > RMSE:         {test_metrics['rmse']:.6f}")
    print(f" > Correlation:  {test_metrics['corr']:.6f}  (↑ 越高越好)")
    print(f" > SSIM:         {test_metrics['ssim']:.6f}  (↑ 越高越好)")
    print(f" > Valid Pixels: {test_metrics['valid_ratio']*100:.2f}%  (D1: 有效评估区域占比)")
    print("=" * 70)


if __name__ == "__main__":
    test_main()