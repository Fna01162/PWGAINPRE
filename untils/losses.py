# -*- coding: utf-8 -*-
"""GAIN 风速 V 补全损失函数。"""

import torch
import torch.nn.functional as F


def generate_hint(observed_mask, hint_rate=0.9):
    """生成 GAIN Hint，1=观测区域，0=缺失区域。"""
    b = torch.bernoulli(torch.full_like(observed_mask, float(hint_rate)))
    return b * observed_mask + 0.5 * (1.0 - b)


def _resize_mask(mask, size):
    """将 mask 调整到判别器输出尺寸。"""
    if mask.dim() == 5:
        mask = mask[:, :, 0]
    return F.interpolate(mask.float(), size=size, mode="nearest")


def discriminator_loss(d_logits, observed_mask, valid_mask):
    """PatchGAN 判别器损失。"""
    target = _resize_mask(observed_mask, d_logits.shape[-2:])
    valid = _resize_mask(valid_mask, d_logits.shape[-2:])
    target = target * 0.9 + (1.0 - target) * 0.1
    loss = F.binary_cross_entropy_with_logits(
        d_logits, target, reduction="none"
    )
    return (loss * valid).sum() / (valid.sum() + 1e-8)


def generator_adv_loss(d_logits, missing_mask, valid_mask):
    """生成器对抗损失，只约束 V 的缺失区域。"""
    miss = _resize_mask(missing_mask, d_logits.shape[-2:])
    valid = _resize_mask(valid_mask, d_logits.shape[-2:])
    loss = F.binary_cross_entropy_with_logits(
        d_logits, torch.ones_like(d_logits), reduction="none"
    )
    weight = miss * valid
    return (loss * weight).sum() / (weight.sum() + 1e-8)


def reconstruction_loss(v_fake, v_true, artificial_mask, reliable_mask):
    """V 人工缺失区域的监督重建损失。"""
    mask = artificial_mask * reliable_mask
    if mask.dim() == 5:
        mask = mask[:, :, 0]
    if v_fake.dim() == 5:
        v_fake = v_fake[:, :, 0]
    if v_true.dim() == 5:
        v_true = v_true[:, :, 0]
    diff = (v_fake - v_true) ** 2
    return (diff * mask).sum() / (mask.sum() + 1e-8)


def tv_loss(v_fake, missing_mask, valid_mask=None):
    """空间 TV 平滑约束，只作用于缺失区域。"""
    if v_fake.dim() == 5:
        v_fake = v_fake[:, :, 0]
    if missing_mask.dim() == 5:
        missing_mask = missing_mask[:, :, 0]
    if valid_mask is not None and valid_mask.dim() == 5:
        valid_mask = valid_mask[:, :, 0]

    m = missing_mask
    if valid_mask is not None:
        m = m * valid_mask

    dx = torch.abs(v_fake[:, :, 1:, :] - v_fake[:, :, :-1, :])
    dy = torch.abs(v_fake[:, :, :, 1:] - v_fake[:, :, :, :-1])

    mx = m[:, :, 1:, :]
    my = m[:, :, :, 1:]

    numerator = (dx * mx).sum() + (dy * my).sum()
    denominator = mx.sum() + my.sum() + 1e-8
    return numerator / denominator


def temporal_consistency_loss(v_fake, t_target, missing_mask, valid_mask=None):
    """V 的时间一致性约束。time_steps=1 时返回 0。"""
    if v_fake.shape[2] <= 1:
        return v_fake.new_zeros(())

    t_target = max(0, min(int(t_target), v_fake.shape[2] - 1))
    mask = missing_mask
    if valid_mask is not None:
        mask = mask * valid_mask

    denominator = mask.sum() + 1e-8
    target = v_fake[:, :, t_target]
    left_idx = max(0, t_target - 1)
    right_idx = min(v_fake.shape[2] - 1, t_target + 1)

    left_loss = (
        ((target - v_fake[:, :, left_idx]) ** 2) * mask
    ).sum() / denominator

    right_loss = (
        ((target - v_fake[:, :, right_idx]) ** 2) * mask
    ).sum() / denominator

    return 0.5 * (left_loss + right_loss)


def azimuth_tv_loss(v, mask):
    """方位角一阶连续性损失，支持 PPI 360° 首尾连接。"""
    if v.dim() == 5:
        v = v[:, :, 0]
    if mask.dim() == 5:
        mask = mask[:, :, 0]

    diff = (v[..., 1:] - v[..., :-1]).abs()
    m = mask[..., 1:] * mask[..., :-1]

    loss_inner = (diff * m).sum()
    count_inner = m.sum()

    wrap_diff = (v[..., 0] - v[..., -1]).abs()
    wrap_mask = mask[..., 0] * mask[..., -1]

    loss_wrap = (wrap_diff * wrap_mask).sum()
    count_wrap = wrap_mask.sum()

    return (loss_inner + loss_wrap) / (count_inner + count_wrap).clamp_min(1.0)


def azimuth_boundary_loss(v, missing_mask, reliable_mask=None):
    """缺失区域与两侧可靠区域之间的方位角边界连续性。"""
    if v.dim() == 5:
        v = v[:, :, 0]
    if missing_mask.dim() == 5:
        missing_mask = missing_mask[:, :, 0]
    if reliable_mask is not None and reliable_mask.dim() == 5:
        reliable_mask = reliable_mask[:, :, 0]

    m = missing_mask.float()
    if reliable_mask is None:
        r = 1.0 - m
    else:
        r = reliable_mask.float()

    v_left = v[..., :-1]
    v_right = v[..., 1:]
    m_left = m[..., :-1]
    m_right = m[..., 1:]
    r_left = r[..., :-1]
    r_right = r[..., 1:]

    w_lr = m_left * r_right
    w_rl = r_left * m_right

    diff = (v_left - v_right).abs()
    weight = w_lr + w_rl

    loss_inner = (diff * weight).sum()
    count_inner = weight.sum()

    v_first = v[..., 0]
    v_last = v[..., -1]
    m_first = m[..., 0]
    m_last = m[..., -1]
    r_first = r[..., 0]
    r_last = r[..., -1]

    wrap_weight = m_first * r_last + r_first * m_last
    wrap_diff = (v_first - v_last).abs()

    loss_wrap = (wrap_diff * wrap_weight).sum()
    count_wrap = wrap_weight.sum()

    return (loss_inner + loss_wrap) / (count_inner + count_wrap).clamp_min(1.0)


def azimuth_curvature_loss(v, mask):
    """方位角二阶差分损失，抑制缺失区域中的突变和折线伪影。"""
    if v.dim() == 5:
        v = v[:, :, 0]
    if mask.dim() == 5:
        mask = mask[:, :, 0]

    v_prev = torch.roll(v, shifts=1, dims=-1)
    v_next = torch.roll(v, shifts=-1, dims=-1)
    d2 = v_next - 2.0 * v + v_prev

    m = mask.float()
    return (d2.abs() * m).sum() / m.sum().clamp_min(1.0)


def masked_prediction_loss(pred, target, valid_mask):
    """未来风速预测的 masked MAE 损失。

    参数：
        pred       : [B, Tout, 1, H, W]  预测的 V 风速场
        target     : [B, Tout, 1, H, W]  真实 V 风速场
        valid_mask : [B, Tout, 1, H, W]  有效标签掩码（未来真实观测且可靠）

    返回：
        标量损失，在有效标签点上计算 MAE。
    """
    if pred.dim() == 4:
        pred = pred.unsqueeze(2)
    if target.dim() == 4:
        target = target.unsqueeze(2)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.unsqueeze(2)

    diff = (pred - target).abs()
    mask = valid_mask.float()
    numerator = (diff * mask).sum()
    denominator = mask.sum()

    if denominator == 0:
        return pred.new_zeros(())

    return numerator / denominator.clamp_min(1.0)


def masked_prediction_mse_loss(pred, target, valid_mask):
    """未来风速预测的 masked MSE 损失。

    参数：
        pred       : [B, Tout, 1, H, W]
        target     : [B, Tout, 1, H, W]
        valid_mask : [B, Tout, 1, H, W]

    返回：
        标量损失，在有效标签点上计算 MSE。
    """
    if pred.dim() == 4:
        pred = pred.unsqueeze(2)
    if target.dim() == 4:
        target = target.unsqueeze(2)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.unsqueeze(2)

    diff = (pred - target) ** 2
    mask = valid_mask.float()
    numerator = (diff * mask).sum()
    denominator = mask.sum()

    if denominator == 0:
        return pred.new_zeros(())

    return numerator / denominator.clamp_min(1.0)


def compute_prediction_metrics(pred, target, valid_mask):
    """计算预测的 MAE 和 RMSE（反归一化前在归一化空间计算）。

    参数：
        pred       : [B, Tout, 1, H, W]
        target     : [B, Tout, 1, H, W]
        valid_mask : [B, Tout, 1, H, W]

    返回：
        (mae, rmse, valid_points) 元组。
    """
    if pred.dim() == 4:
        pred = pred.unsqueeze(2)
    if target.dim() == 4:
        target = target.unsqueeze(2)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.unsqueeze(2)

    mask = valid_mask.float()
    diff = pred - target
    mae = ((diff.abs() * mask).sum() / mask.sum().clamp_min(1.0)).item()
    mse = ((diff ** 2 * mask).sum() / mask.sum().clamp_min(1.0)).item()
    rmse = mse ** 0.5
    valid_points = mask.sum().item()

    return mae, rmse, valid_points


def extension_mae_loss(pred, target, valid_mask):
    """径向量程扩展的 masked MAE 损失。

    参数：
        pred       : [B, 1, H_far, W]  预测的远距离 V
        target     : [B, H_far, W]     真实远距离 V
        valid_mask : [B, H_far, W]     有效标签掩码

    返回：
        标量损失。
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    diff = (pred - target).abs()
    mask = valid_mask.float()
    numerator = (diff * mask).sum()
    denominator = mask.sum()
    if denominator == 0:
        return pred.new_zeros(())
    return numerator / denominator.clamp_min(1.0)


def compute_extension_metrics(pred, target, valid_mask):
    """计算量程扩展的 MAE 和 RMSE。

    参数：
        pred       : [B, 1, H_far, W]
        target     : [B, H_far, W]
        valid_mask : [B, H_far, W]

    返回：
        (mae, rmse, valid_points) 元组。
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    mask = valid_mask.float()
    diff = pred - target
    mae = ((diff.abs() * mask).sum() / mask.sum().clamp_min(1.0)).item()
    mse = ((diff ** 2 * mask).sum() / mask.sum().clamp_min(1.0)).item()
    rmse = mse ** 0.5
    valid_points = mask.sum().item()
    return mae, rmse, valid_points

def extension_composite_loss(pred, target, valid_mask, near_v_last=None, near_valid_last=None):
    """物理引导的径向量程扩展复合损失函数（涨点核心）。

    参数：
        pred            : [B, 1, H_far, W]  预测的远距离 V
        target          : [B, H_far, W]     真实远距离 V
        valid_mask      : [B, H_far, W]     远距离有效掩码
        near_v_last     : [B, W] (可选)      近场最后一门 (near_end-1) 的 V
        near_valid_last : [B, W] (可选)      近场最后一门的有效掩码
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred_2d = pred[:, 0]
    else:
        pred_2d = pred

    mask = valid_mask.float()
    denom = mask.sum().clamp_min(1.0)
    if mask.sum() == 0:
        return pred.new_zeros(())

    # 1. 主损失：80% MAE + 20% Smooth L1（兼顾抗噪性与大误差惩罚）
    diff = (pred_2d - target).abs()
    mae_loss = (diff * mask).sum() / denom
    huber_loss = (F.smooth_l1_loss(pred_2d, target, beta=0.05, reduction="none") * mask).sum() / denom
    loss_main = 0.8 * mae_loss + 0.2 * huber_loss

    # 2. 径向梯度匹配损失：保持远距离径向风速切变结构
    if pred_2d.shape[1] > 1:
        dp_r = pred_2d[:, 1:, :] - pred_2d[:, :-1, :]
        dt_r = target[:, 1:, :] - target[:, :-1, :]
        m_r = mask[:, 1:, :] * mask[:, :-1, :]
        loss_rad_grad = ((dp_r - dt_r).abs() * m_r).sum() / m_r.sum().clamp_min(1.0)
    else:
        loss_rad_grad = pred.new_zeros(())

    # 3. 方位角周期梯度匹配损失：保持 0°~360° 方位角风场波形
    dp_a = torch.roll(pred_2d, shifts=-1, dims=-1) - pred_2d
    dt_a = torch.roll(target, shifts=-1, dims=-1) - target
    m_a = torch.roll(mask, shifts=-1, dims=-1) * mask
    loss_azi_grad = ((dp_a - dt_a).abs() * m_a).sum() / m_a.sum().clamp_min(1.0)

    # 4. 近-远交界处 (gate 67 -> gate 68) 跨门连续性约束
    loss_boundary = pred.new_zeros(())
    if near_v_last is not None and near_valid_last is not None:
        m_bd = mask[:, 0, :] * near_valid_last.float()
        pred_step = pred_2d[:, 0, :] - near_v_last
        true_step = target[:, 0, :] - near_v_last
        loss_boundary = ((pred_step - true_step).abs() * m_bd).sum() / m_bd.sum().clamp_min(1.0)

    return loss_main + 0.15 * loss_rad_grad + 0.10 * loss_azi_grad + 0.15 * loss_boundary