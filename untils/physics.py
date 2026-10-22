# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def _to_4d(x):
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"输入必须为 [B,H,W] 或 [B,1,H,W]，当前 shape={tuple(x.shape)}")


def _circular_diff_w(x):
    """
    方位角方向周期差分。
    W 方向满足 0° 与 360° 首尾相接。
    """
    x = _to_4d(x)
    return torch.roll(x, shifts=-1, dims=-1) - x


def _forward_diff_h(x):
    """
    径向方向前向差分。
    最后一层没有下一层，因此复制最后一层差分。
    """
    x = _to_4d(x)
    d = x[..., 1:, :] - x[..., :-1, :]
    if d.shape[-2] > 0:
        d = torch.cat([d, d[..., -1:, :]], dim=-2)
    return d


def _backward_diff_h(x):
    """
    径向方向后向差分。
    """
    x = _to_4d(x)
    d = x[..., 1:, :] - x[..., :-1, :]
    if d.shape[-2] > 0:
        d = torch.cat([d[..., :1, :], d], dim=-2)
    return d


def _boundary_context(reference, reliable_mask):
    """
    根据可靠观测构造物理一致性参考。

    reference:
        真实可靠 V 或训练阶段可用的真实观测。

    reliable_mask:
        1 = 可靠观测
        0 = 不可靠/缺失

    返回:
        context value
        context mask

    对每个位置，从径向和方位方向寻找最近的可靠观测。
    这里只构造边界/邻域参考，不把 V_gain 当作真实值。
    """
    reference = _to_4d(reference)
    reliable_mask = _to_4d(reliable_mask).float()

    B, C, H, W = reference.shape

    if C != 1:
        raise ValueError(
            f"physics 当前只处理单通道 V，C={C}"
        )

    left_v = torch.roll(reference, shifts=1, dims=-1)
    right_v = torch.roll(reference, shifts=-1, dims=-1)
    left_m = torch.roll(reliable_mask, shifts=1, dims=-1)
    right_m = torch.roll(reliable_mask, shifts=-1, dims=-1)

    up_v = F.pad(
        reference[..., :-1, :],
        (0, 0, 1, 0),
        mode="replicate",
    )
    down_v = F.pad(
        reference[..., 1:, :],
        (0, 0, 0, 1),
        mode="replicate",
    )
    up_m = F.pad(
        reliable_mask[..., :-1, :],
        (0, 0, 1, 0),
        mode="replicate",
    )
    down_m = F.pad(
        reliable_mask[..., 1:, :],
        (0, 0, 0, 1),
        mode="replicate",
    )

    candidates_v = torch.stack(
        [reference, left_v, right_v, up_v, down_v],
        dim=1,
    )
    candidates_m = torch.stack(
        [reliable_mask, left_m, right_m, up_m, down_m],
        dim=1,
    )

    weights = candidates_m / (
        candidates_m.sum(dim=1, keepdim=True) + 1e-8
    )

    context = (
        candidates_v * weights
    ).sum(dim=1)

    context_mask = candidates_m.max(dim=1).values

    return context, context_mask


def _masked_mean(value, mask):
    mask = mask.float()

    if mask.sum() <= 1e-8:
        return value.new_zeros(())

    return (
        value.abs() * mask
    ).sum() / (mask.sum() + 1e-8)


def _gradient_difference(pred, ref, mask, direction):
    if direction == "radial":
        dp = _forward_diff_h(pred)
        dr = _forward_diff_h(ref)
    elif direction == "azimuth":
        dp = _circular_diff_w(pred)
        dr = _circular_diff_w(ref)
    else:
        raise ValueError(
            f"未知方向: {direction}"
        )

    return _masked_mean(
        dp - dr,
        mask,
    )


def radial_consistency_loss(
    pred,
    reference,
    reliable_mask,
    target_mask=None,
):
    """
    径向方向物理一致性。

    约束补全区域与可靠观测边界之间的径向变化趋势。
    """
    pred = _to_4d(pred)
    reference = _to_4d(reference)
    reliable_mask = _to_4d(reliable_mask)

    if target_mask is None:
        target_mask = 1.0 - reliable_mask
    else:
        target_mask = _to_4d(target_mask)

    context, context_mask = _boundary_context(
        reference,
        reliable_mask,
    )

    # 目标区域周围至少存在一个可靠参考时才参与。
    mask = target_mask * context_mask

    return _gradient_difference(
        pred,
        context,
        mask,
        direction="radial",
    )


def azimuth_consistency_loss(
    pred,
    reference,
    reliable_mask,
    target_mask=None,
):
    """
    方位角方向物理一致性。

    使用 Circular 差分，保证 0°/360° 连续。
    """
    pred = _to_4d(pred)
    reference = _to_4d(reference)
    reliable_mask = _to_4d(reliable_mask)

    if target_mask is None:
        target_mask = 1.0 - reliable_mask
    else:
        target_mask = _to_4d(target_mask)

    context, context_mask = _boundary_context(
        reference,
        reliable_mask,
    )

    mask = target_mask * context_mask

    return _gradient_difference(
        pred,
        context,
        mask,
        direction="azimuth",
    )


def local_gradient_consistency_loss(
    pred,
    reference,
    reliable_mask,
    target_mask=None,
):
    """
    综合径向 + 方位局部梯度一致性。
    """
    radial = radial_consistency_loss(
        pred,
        reference,
        reliable_mask,
        target_mask,
    )

    azimuth = azimuth_consistency_loss(
        pred,
        reference,
        reliable_mask,
        target_mask,
    )

    return 0.5 * (radial + azimuth)


def physics_consistency_loss(
    pred,
    reference,
    target_mask,
    radial_weight=1.0,
    azimuth_weight=1.0,
    gradient_weight=1.0,
    reliable_mask=None,
):
    """
    PPI 风场物理一致性损失。

    参数:
        pred:
            V_final

        reference:
            真实可靠观测 V。

            注意：
            这里不是 V_gain。

        target_mask:
            需要进行物理约束的区域，通常为：
                real_missing + artificial_missing

        reliable_mask:
            真实可靠观测区域。

    设计原则：
        1. 不要求缺失区直接等于 V_gain。
        2. 只利用真实可靠观测作为上下文。
        3. 方位方向使用周期差分。
        4. 径向方向使用普通差分。
        5. 没有可靠邻域的区域不参与该项损失。
    """
    pred = _to_4d(pred)
    reference = _to_4d(reference)
    target_mask = _to_4d(target_mask)

    if reliable_mask is None:
        reliable_mask = 1.0 - target_mask
    else:
        reliable_mask = _to_4d(reliable_mask)

    reliable_mask = reliable_mask.float().clamp(0.0, 1.0)
    target_mask = target_mask.float().clamp(0.0, 1.0)

    radial = radial_consistency_loss(
        pred,
        reference,
        reliable_mask,
        target_mask,
    )

    azimuth = azimuth_consistency_loss(
        pred,
        reference,
        reliable_mask,
        target_mask,
    )

    gradient = local_gradient_consistency_loss(
        pred,
        reference,
        reliable_mask,
        target_mask,
    )

    total = (
        float(radial_weight) * radial
        + float(azimuth_weight) * azimuth
        + float(gradient_weight) * gradient
    )

    denom = (
        float(radial_weight)
        + float(azimuth_weight)
        + float(gradient_weight)
    )

    if denom <= 0:
        return pred.new_zeros(())

    return total / denom


def ppi_periodic_consistency_loss(pred, mask=None):
    """
    PPI 0°/360° 首尾连续性损失。

    只约束方位角首尾边界。
    """
    pred = _to_4d(pred)

    first = pred[..., 0]
    last = pred[..., -1]

    diff = torch.abs(first - last)

    if mask is None:
        return diff.mean()

    mask = _to_4d(mask)

    first_m = mask[..., 0]
    last_m = mask[..., -1]

    weight = first_m * last_m

    if weight.sum() <= 1e-8:
        return pred.new_zeros(())

    return (
        diff * weight
    ).sum() / (weight.sum() + 1e-8)


def boundary_value_consistency_loss(
    pred,
    reference,
    reliable_mask,
    target_mask=None,
):
    """
    缺失区域与可靠边界之间的数值连续性约束。

    与直接使用 V_gain 不同，这里参考值来自真实可靠观测。
    """
    pred = _to_4d(pred)
    reference = _to_4d(reference)
    reliable_mask = _to_4d(reliable_mask)

    if target_mask is None:
        target_mask = 1.0 - reliable_mask
    else:
        target_mask = _to_4d(target_mask)

    context, context_mask = _boundary_context(
        reference,
        reliable_mask,
    )

    mask = target_mask * context_mask

    return _masked_mean(
        pred - context,
        mask,
    )