# -*- coding: utf-8 -*-
"""联合训练模型的独立测试脚本。

测试相关逻辑（固定mask生成、多步指标评估、链路补全）原本位于 train_joint.py，
现拆分到本文件；train_joint.py 只保留训练流程，并从这里复用评估函数。

用法:
    python test_joint.py              # 默认测试 D 链路
    python test_joint.py C            # 测试 C 链路
    python test_joint.py D --oracle   # 额外计算 Oracle 上界
    python test_joint.py D --no-oracle  # 只测试 D 链路（不计算 Oracle 上界）
    python main.py joint_test D       # 通过统一入口调用（等价写法）

链路模式由 Config 统一管理（Config.CHAIN_MODES），训练与测试只要传入相同的
chain_mode，就会自动定位到同一个输出目录。
"""

import os
import sys
import types
import platform

import torch
import numpy as np
from tqdm import tqdm

from config import Config
from untils.data import SlidingWindowDataset, build_generator_inputs
from model import UNetGenerator
from predictor import WindFieldPredictor
from untils.losses import compute_prediction_metrics
from untils.data import _continuous_ray_mask, _calibrate_missing
from untils.refiner import ResidualRefiner, apply_refiner
from untils.wavelet import wavelet_feature_channels


def azimuth_linear_interpolation(v_in, missing_mask):
    """[P1-2 FIX] 方位向（W维）线性插值，用于B链路

    向量化实现：用 cummax/cummin 一次性找到每个缺失像素的左右有效邻居，
    然后通过 gather + 线性插值公式在 GPU 上并行完成整批插值。

    Args:
        v_in: [B, 1, Tin, H, W] 输入数据（缺失处已归零）
        missing_mask: [B, 1, Tin, H, W] 缺失掩码（True=缺失）

    Returns:
        v_out: [B, 1, Tin, H, W] 插值后的数据
    """
    v_out = v_in.clone()

    if not missing_mask.any():
        return v_out

    B, C, Tin, H, W = v_in.shape
    device = v_in.device

    # 沿 W 维度的坐标索引: [1, 1, 1, 1, W]
    w_coords = torch.arange(W, device=device, dtype=torch.float32)
    w_coords = w_coords.view(1, 1, 1, 1, W).expand(B, C, Tin, H, W)

    # ── 左邻居搜索 ──
    # 缺失位置设为 -1（哨兵），cummax 向右传播最近的有效坐标
    left_idx = w_coords.clone()
    left_idx[missing_mask] = -1.0
    left_idx, _ = torch.cummax(left_idx, dim=-1)

    # ── 右邻居搜索 ──
    # 缺失位置设为 W（超出范围的哨兵），从右向左 cummin
    right_idx = w_coords.clone()
    right_idx[missing_mask] = float(W)
    right_idx = torch.flip(right_idx, [-1])
    right_idx, _ = torch.cummin(right_idx, dim=-1)
    right_idx = torch.flip(right_idx, [-1])

    # ── 邻域有效性判断 ──
    has_left = left_idx >= 0
    has_right = right_idx < float(W)
    both = has_left & has_right
    only_left = has_left & (~has_right)
    only_right = (~has_left) & has_right

    # ── 安全 gather ──
    left_idx_safe = left_idx.long().clamp(0, W - 1)
    right_idx_safe = right_idx.long().clamp(0, W - 1)

    left_vals = torch.gather(v_out, dim=-1, index=left_idx_safe)
    right_vals = torch.gather(v_out, dim=-1, index=right_idx_safe)

    # ── 线性插值（左右邻居都存在） ──
    eps = torch.tensor(1e-8, device=device, dtype=torch.float32)
    alpha = (w_coords - left_idx) / torch.max(right_idx - left_idx, eps)
    interp_vals = (1.0 - alpha) * left_vals + alpha * right_vals
    v_out = torch.where(missing_mask & both, interp_vals, v_out)

    # ── 仅左邻居（右边界无有效值，外推为左值） ──
    v_out = torch.where(missing_mask & only_left, left_vals, v_out)

    # ── 仅右邻居（左边界无有效值，外推为右值） ──
    v_out = torch.where(missing_mask & only_right, right_vals, v_out)

    return v_out


def generate_fixed_test_mask(test_loader, cfg, device):
    """A4: 预生成固定测试mask，确保可复现"""
    all_masks = []
    for hx, hm, hr, fx, fm, fr in test_loader:
        hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
        B, C, Tin, H, W = hx.shape
        valid_mask_h = hm * hr

        batch_masks = []
        for b in range(B):
            allowed = valid_mask_h[b, 0, 0].cpu().numpy()
            n_rays = max(1, int(cfg.missing_rate * W))
            raw_mask = _continuous_ray_mask(H, W, n_rays)
            calibrated_mask = _calibrate_missing(raw_mask, cfg.missing_rate, allowed > 0.5)
            batch_masks.append(calibrated_mask)

        batch_masks_np = np.stack(batch_masks)
        all_masks.append(batch_masks_np)

    # 合并所有batch的masks
    full_masks = np.concatenate(all_masks, axis=0)
    return torch.from_numpy(full_masks).float()


def evaluate_on_test(G, P, R, refiner_phys_channels, refiner_n_wavelet_levels, test_loader, norm, cfg, device, chain_mode="D", is_oracle=False):
    """在测试集上评估模型，使用预生成的固定mask确保可复现性 (A4)

    改进：
    - C4: 支持多步预测 [1, 3, 5]
    - C5: 计算完整指标 (MAE, RMSE, Corr, SSIM)
    - D1: 报告有效像素占比
    - C6: 支持 Oracle 上界模式
    - [FIX] 添加 R, refiner_phys_channels, refiner_n_wavelet_levels 参数用于 Refiner 兼容性处理
    """
    if chain_mode in ["C", "D"]:
        G.eval()
    P.eval()

    # A4: 预生成固定测试mask
    print(f"🔧 Pre-generating fixed test masks for reproducibility... {'(Oracle Mode)' if is_oracle else ''}")
    fixed_masks = generate_fixed_test_mask(test_loader, cfg, device)
    mask_idx = 0

    # C4: 多步预测结果存储
    multi_step_results = {step: {'mae': [], 'rmse': [], 'corr': [], 'ssim': [], 'valid_ratio': []}
                          for step in cfg.multi_step_predictions}

    total_samples = 0

    with torch.no_grad():
        for hx, hm, hr, fx, fm, fr in tqdm(test_loader, desc=f"Testing [{chain_mode}{'-Oracle' if is_oracle else ''}]"):
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)

            B, C, Tin, H, W = hx.shape

            # 使用预生成的固定mask
            art_mask = fixed_masks[mask_idx:mask_idx+B].to(device)
            art_mask = art_mask.unsqueeze(1).unsqueeze(1).expand(-1, C, Tin, -1, -1)
            mask_idx += B

            valid_mask_h = hm * hr
            m_t = valid_mask_h * (1 - art_mask)
            xin = hx * m_t

            # 根据链路模式补全
            if is_oracle:
                # C6: Oracle 模式 - 使用完整无缺失数据
                v_completed = hx[:, 0:1, :, :, :]  # 完整的真实数据
            elif chain_mode == "A":
                v_completed = xin[:, 0:1, :, :, :]
            elif chain_mode == "B":
                # [FIX] 使用统一的方位向线性插值函数
                missing_mask_b = (m_t[:, 0:1, :, :, :] < 0.5)
                v_completed = azimuth_linear_interpolation(xin[:, 0:1, :, :, :], missing_mask_b)
            else:
                xb, eb = build_generator_inputs(xin, m_t)
                v_completed = G(xb.to(device), m_t.to(device))

                # [FIX] 测试时也应用Refiner做残差修正（逐帧）
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

                        with torch.no_grad():
                            # 使用 refiner_cfg（phys_channels 已调整）
                            v_refined_t = apply_refiner(R, v_t, x_t, given_t, rel_t, valid_t, refiner_cfg, device)
                        v_refined_list.append(v_refined_t)

                    v_completed = torch.stack(v_refined_list, dim=2)  # [B, 1, Tin, H, W]

            # 构建预测器输入
            features = hx[:, 1:cfg.phys_channels, :, :, :]
            pred_in = torch.cat([
                v_completed,
                features,
                hm[:, 0:1, :, :, :],
                hr[:, 0:1, :, :, :]
            ], dim=1)

            pred_in_p = pred_in.permute(0, 2, 1, 3, 4).contiguous()
            out_pred = P(pred_in_p)  # [B, Tout, 1, H, W]

            # C4+C5: 对每个预测步长分别计算指标
            # [FIX] fx布局是[B,C,T,H,W]，时间在dim=2，必须在dim=2切+固定通道0
            for pred_step in cfg.multi_step_predictions:
                if pred_step > out_pred.shape[1]:
                    continue

                # 取第 pred_step 步的预测（out_pred是[B,T,1,H,W]，在dim=1切）
                pred_step_pred = out_pred[:, pred_step-1:pred_step, :, :, :]  # [B, 1, H, W]
                # 正确切片：通道0(V) + 时间维pred_step-1:pred_step
                tgt_step = fx[:, 0:1, pred_step-1:pred_step, :, :].permute(0, 2, 1, 3, 4)  # [B, 1, H, W]
                valid_mask_f_step = (fm * fr)[:, 0:1, pred_step-1:pred_step, :, :].permute(0, 2, 1, 3, 4)

                # 使用新的多指标计算函数
                mae, rmse, corr, ssim_val, valid_pts, valid_ratio = compute_prediction_metrics(
                    pred_step_pred, tgt_step, valid_mask_f_step
                )

                multi_step_results[pred_step]['mae'].append(mae * B)
                multi_step_results[pred_step]['rmse'].append(rmse * B)
                multi_step_results[pred_step]['corr'].append(corr * B)
                multi_step_results[pred_step]['ssim'].append(ssim_val * B)
                multi_step_results[pred_step]['valid_ratio'].append(valid_ratio * B)

            total_samples += B

    # 汇总结果并打印
    print(f"\n{'='*70}")
    print(f" 📊 Test Evaluation Results [{chain_mode}{'-Oracle' if is_oracle else ''}]")
    print(f"{'='*70}")
    print(f" Total Samples: {total_samples}")
    print(f"{'-'*70}")

    all_results = {}
    for pred_step in sorted(multi_step_results.keys()):
        results = multi_step_results[pred_step]
        if len(results['mae']) == 0:
            continue

        avg_mae = sum(results['mae']) / max(total_samples, 1)
        avg_rmse = sum(results['rmse']) / max(total_samples, 1)
        avg_corr = sum(results['corr']) / max(total_samples, 1)
        avg_ssim = sum(results['ssim']) / max(total_samples, 1)
        avg_valid_ratio = sum(results['valid_ratio']) / max(total_samples, 1)

        time_min = pred_step * 5  # 假设每步5分钟
        print(f"\n ⏱️  Prediction Horizon: {pred_step} step(s) ({time_min} min)")
        print(f"    MAE:          {avg_mae:.6f}")
        print(f"    RMSE:         {avg_rmse:.6f}")
        print(f"    Correlation:  {avg_corr:.6f}  (↑ 越高越好)")
        print(f"    SSIM:         {avg_ssim:.6f}  (↑ 越高越好)")
        print(f"    Valid Pixels: {avg_valid_ratio*100:.2f}%  (D1: 有效评估区域占比)")

        all_results[f'step{pred_step}'] = {
            'mae': avg_mae,
            'rmse': avg_rmse,
            'corr': avg_corr,
            'ssim': avg_ssim,
            'valid_ratio': avg_valid_ratio
        }

    print(f"\n{'='*70}")

    # 保存完整结果
    test_results = {
        'chain_mode': chain_mode,
        'is_oracle': is_oracle,
        'total_samples': total_samples,
        **all_results
    }

    mode_str = f"{chain_mode}_oracle" if is_oracle else chain_mode
    test_path = os.path.join(cfg.output_dir, f'test_{mode_str}.npy')
    np.save(test_path, test_results)
    print(f"💾 Results saved to: {test_path}")

    # 返回多步平均MAE，用于early stopping
    multi_step_mae = {step: all_results[f'step{step}']['mae']
                      for step in cfg.multi_step_predictions
                      if f'step{step}' in all_results}
    avg_mae = sum(multi_step_mae.values()) / max(len(multi_step_mae), 1)

    return all_results, avg_mae


def parse_args():
    """解析命令行参数。

    链路模式只作为 Config 的输入，不再改写 sys.argv：
    输出目录由 Config(chain_mode=...) 统一决定，训练与测试因此天然一致。
    """
    chain_mode = None
    force_oracle = False
    disable_oracle = False

    for arg in sys.argv[1:]:
        low = arg.lower()
        if low in ("--oracle", "-o"):
            force_oracle = True
        elif low in ("--no-oracle",):
            disable_oracle = True
        elif arg.upper() in Config.CHAIN_MODES:
            chain_mode = arg.upper()
        else:
            print(f"⚠️  忽略未知参数: {arg}")

    return chain_mode, force_oracle, disable_oracle


def build_test_loader(cfg, norm, actual_pred_steps):
    """构建测试集 DataLoader（多步预测取最大步长，与训练时一致）。"""
    # [FIX] Windows 兼容性：num_workers=0 避免 shm.dll 加载失败
    effective_num_workers = 0 if platform.system() == 'Windows' else cfg.num_workers
    if effective_num_workers == 0 and cfg.num_workers > 0:
        print(f"⚠️  Windows 系统检测到，num_workers 从 {cfg.num_workers} 强制设为 0（避免 DLL 加载错误）")

    test_dataset = SlidingWindowDataset(
        cfg.test_npy, normalizer=norm,
        input_steps=cfg.input_steps, pred_steps=actual_pred_steps,
        phys_channels=cfg.phys_channels
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=effective_num_workers
    )
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    return test_loader


def _infer_refiner_config(in_ch, cfg):
    """根据 Refiner 输入通道数反推 (phys_channels, n_wavelet_levels)。

    Refiner 输入 = phys_channels + 5(几何3 + 掩码2) + wavelet_feature_channels(levels)
    """
    default_phys = int(cfg.phys_channels)
    default_levels = int(getattr(cfg, 'n_wavelet_levels', 1))

    if default_phys + 5 + wavelet_feature_channels(default_levels) == in_ch:
        return default_phys, default_levels

    for levels in range(1, 5):
        phys = in_ch - 5 - wavelet_feature_channels(levels)
        if 1 <= phys <= default_phys:
            print(f"ℹ️  Inferred Refiner config from checkpoint: phys_channels={phys}, n_wavelet_levels={levels}")
            return phys, levels

    print(f"⚠️  无法从 in_channels={in_ch} 反推 Refiner 配置，回退到 config 默认值")
    return default_phys, default_levels


def build_refiner_from_checkpoint(ckpt, cfg, device):
    """从 checkpoint 中重建 ResidualRefiner。

    Returns:
        (R, refiner_phys_channels, refiner_n_wavelet_levels)
    """
    refiner_state = ckpt.get('refiner', None)
    if refiner_state is None:
        print("ℹ️  No refiner weights in checkpoint, Refiner will be skipped")
        return None, cfg.phys_channels, int(getattr(cfg, 'n_wavelet_levels', 1))

    enc1_weight = refiner_state.get('enc1.0.conv.weight', None)
    if enc1_weight is None:
        print("⚠️  Cannot infer Refiner in_channels from checkpoint, Refiner will be skipped")
        return None, cfg.phys_channels, int(getattr(cfg, 'n_wavelet_levels', 1))

    in_ch = int(enc1_weight.shape[1])
    base_filter = int(enc1_weight.shape[0])
    print(f"📐 Inferring Refiner from checkpoint: in_channels={in_ch}, base_filter={base_filter}")

    R = ResidualRefiner(in_channels=in_ch, base_filter=base_filter).to(device)
    R.load_state_dict(refiner_state)
    R.eval()

    refiner_phys_channels, refiner_n_wavelet_levels = _infer_refiner_config(in_ch, cfg)
    print(f"✅ Refiner loaded: {len(refiner_state)} tensors, "
          f"phys_channels={refiner_phys_channels}, wavelet_levels={refiner_n_wavelet_levels}")
    return R, refiner_phys_channels, refiner_n_wavelet_levels


def test_joint(chain_mode=None, force_oracle=False, disable_oracle=False):
    """加载联合训练权重，在测试集上评估指定链路。"""
    cfg = Config(chain_mode=chain_mode)
    chain_mode = cfg.chain_mode
    device = cfg.device

    print(f"\n{'='*60}")
    print(f"Joint Model Testing Configuration:")
    print(f"  Device: {device}")
    print(f"  Chain Mode: {chain_mode}")
    print(f"  Output Dir: {cfg.output_dir}")
    print(f"{'='*60}\n")

    # 1. 加载 checkpoint（包含 normalizer / G / P / R）
    model_path = os.path.join(cfg.output_dir, 'joint_best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Checkpoint not found: {model_path}\n"
            f"   请先运行: python train_joint.py {chain_mode}"
        )

    print(f"📦 Loading checkpoint from: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    norm = ckpt['normalizer']

    # [FIX] 测试集使用多步预测的最大步长，确保能评估所有配置的步长
    actual_pred_steps = max(cfg.multi_step_predictions) if hasattr(cfg, 'multi_step_predictions') else cfg.pred_steps

    # 2. 构建测试集
    test_loader = build_test_loader(cfg, norm, actual_pred_steps)

    # 3. 构建并加载模型
    G = UNetGenerator(cfg.phys_channels * 2, 1, cfg.base_filter, use_cue=True, use_nafblock=True,
                      use_axial_w=True).to(device)
    P = WindFieldPredictor(input_channels=cfg.phys_channels + 2, hidden_channels=cfg.hidden_channels,
                           num_layers=cfg.num_layers, kernel_size=3,
                           pred_steps=actual_pred_steps).to(device)

    G.load_state_dict(ckpt['model'])
    P.load_state_dict(ckpt['predictor'])
    G.eval()
    P.eval()
    print("✅ G, P loaded")

    R, refiner_phys_channels, refiner_n_wavelet_levels = build_refiner_from_checkpoint(ckpt, cfg, device)

    # 4. 在测试集上评估
    print(f"\n🧪 Evaluating on test set (Chain: {chain_mode})...")
    all_results, avg_mae = evaluate_on_test(
        G, P, R, refiner_phys_channels, refiner_n_wavelet_levels,
        test_loader, norm, cfg, device, chain_mode=chain_mode
    )

    # 5. C6: Oracle 上界评估（D链路默认执行，可用 --no-oracle 关闭）
    run_oracle = force_oracle or (chain_mode == "D" and not disable_oracle)
    if run_oracle:
        print(f"\n🔮 Computing Oracle Upper Bound (complete data, no missing)...")
        evaluate_on_test(
            G, P, R, refiner_phys_channels, refiner_n_wavelet_levels,
            test_loader, norm, cfg, device, chain_mode=chain_mode, is_oracle=True
        )

    print(f"\n{'='*60}")
    print(f"✅ Testing Completed!")
    print(f"   Chain Mode: {chain_mode}")
    print(f"   Multi-step Avg MAE: {avg_mae:.6f}")
    print(f"   Checkpoint: {model_path}")
    print(f"   Output Directory: {cfg.output_dir}")
    print(f"{'='*60}\n")


def main():
    chain_mode, force_oracle, disable_oracle = parse_args()
    test_joint(chain_mode=chain_mode, force_oracle=force_oracle, disable_oracle=disable_oracle)


if __name__ == "__main__":
    main()