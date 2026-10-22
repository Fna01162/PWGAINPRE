# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def _circular_pad_2d(x, pad_h=0, pad_w=1):
    """
    2D PPI 周期填充。
    H 为径向方向，不周期；
    W 为方位角方向，0°/360° 周期连接。
    """
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="replicate")

    if pad_w > 0:
        left = x[..., -pad_w:]
        right = x[..., :pad_w]
        x = torch.cat([left, x, right], dim=-1)

    return x


def _haar_dwt2(x):
    """
    单层 2D Haar DWT。

    输入:
        x: [B,1,H,W]

    输出:
        LL, LH, HL, HH
    """
    if x.ndim != 4:
        raise ValueError(
            f"_haar_dwt2 输入必须为 [B,C,H,W]，当前 shape={tuple(x.shape)}"
        )

    if x.shape[1] != 1:
        raise ValueError(
            f"_haar_dwt2 当前只处理单通道 V，输入 C={x.shape[1]}"
        )

    # H/W 奇数时补一行/列
    H, W = x.shape[-2:]
    if H % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1), mode="replicate")
    if W % 2 != 0:
        x = _circular_pad_2d(x, 0, 1)

    # 为了保持方位角周期性，先复制周期边界后再取局部差分
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    inv_sqrt2 = 0.7071067811865476

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    return ll, lh, hl, hh


def haar_dwt2(x):
    """
    对 [B,1,H,W] 计算单层 Haar 小波。
    """
    return _haar_dwt2(x)


def haar_dwt2_multi(x, levels=1):
    """
    递归多层 2D Haar DWT。

    输入:
        x: [B,1,H,W]
        levels: 分解层数 (>= 1)

    输出:
        list of length `levels`，每个元素是 (LL, LH, HL, HH) tuple。
        第 0 层对应最细尺度，第 levels-1 层对应最粗尺度。

    例: levels=2
        returns [
            (LL_1, LH_1, HL_1, HH_1),      # 最细, H/2 × W/2
            (LL_2, LH_2, HL_2, HH_2),      # 较粗, H/4 × W/4
        ]
        其中 LL_2 由 LL_1 再次分解得到。
    """
    if levels < 1:
        raise ValueError(f"levels 必须 >= 1，当前 {levels}")

    result = []
    current = x
    for _ in range(levels):
        ll, lh, hl, hh = _haar_dwt2(current)
        result.append((ll, lh, hl, hh))
        current = ll
    return result


def _ensure_4d_v(v):
    """
    将 V 统一成 [B,1,H,W]。
    支持:
        [B,H,W]
        [B,1,H,W]
        [B,T,H,W]，此时 T 必须为 1
    """
    if v.ndim == 4:
        if v.shape[1] != 1:
            raise ValueError(
                f"V 必须为单通道，当前 shape={tuple(v.shape)}"
            )
        return v

    if v.ndim == 3:
        return v.unsqueeze(1)

    raise ValueError(
        f"V 输入必须为 [B,H,W] 或 [B,1,H,W]，当前 shape={tuple(v.shape)}"
    )


def wavelet_features(v, levels=1):
    """
    多层 Haar 小波特征，全部上采样回原始尺寸后拼接。

    输入:
        v: [B,1,H,W]
        levels: DWT 层数

    输出:
        [B, 3*levels+1, H, W]

    通道顺序 (以 levels=2 为例):
        LH_1, HL_1, HH_1,   (最细尺度 detail)
        LH_2, HL_2, HH_2,   (次细尺度 detail)
        LL_2                (最深层 approximation)

    levels=1 时输出 4 通道: [LH_1, HL_1, HH_1, LL_1]
    """
    v = _ensure_4d_v(v)
    size = v.shape[-2:]

    coeffs = haar_dwt2_multi(v, levels=levels)

    bands = []
    for ll, lh, hl, hh in coeffs:
        bands.append(lh)
        bands.append(hl)
        bands.append(hh)
    bands.append(coeffs[-1][0])

    upsampled = []
    for b in bands:
        if b.shape[-2:] != size:
            b = F.interpolate(
                b,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        upsampled.append(b)

    return torch.cat(upsampled, dim=1)


def wavelet_feature_channels(levels=1):
    """
    wavelet_features 输出的通道数。
    供 Refiner / build_refiner_input 动态计算 in_channels。
    """
    return 3 * int(levels) + 1


def wavelet_decompose(v):
    """
    返回原始尺寸恢复后的四个小波分量。

    用于训练、可视化以及后续损失计算。
    """
    v = _ensure_4d_v(v)

    ll, lh, hl, hh = haar_dwt2(v)
    size = v.shape[-2:]

    return (
        F.interpolate(ll, size=size, mode="bilinear", align_corners=False),
        F.interpolate(lh, size=size, mode="bilinear", align_corners=False),
        F.interpolate(lh * 0 + hl, size=size, mode="bilinear", align_corners=False),
        F.interpolate(hh, size=size, mode="bilinear", align_corners=False),
    )


def _prepare_mask(mask, target_size):
    """
    将人工缺失 mask 调整到小波系数尺寸。
    使用 area pooling，使一个小波系数对应的像素区域
    中只要存在较明显人工缺失，就得到较高权重。
    """
    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if mask.ndim != 4:
        raise ValueError(
            f"mask 必须为 [B,H,W] 或 [B,1,H,W]，当前 {tuple(mask.shape)}"
        )

    mask = mask.float()

    if mask.shape[-2:] == target_size:
        return mask

    mask = F.interpolate(
        mask,
        size=target_size,
        mode="area",
    )

    return mask.clamp(0.0, 1.0)


def _masked_l1(pred, target, weight):
    """
    带空间权重的 L1。
    """
    if weight is None:
        return F.l1_loss(pred, target)

    if weight.sum() <= 1e-8:
        return pred.new_zeros(())

    return (
        torch.abs(pred - target) * weight
    ).sum() / (weight.sum() + 1e-8)


def wavelet_loss(
    pred,
    target,
    missing_mask=None,
    ll_weight=0.25,
    lh_weight=0.25,
    hl_weight=0.25,
    hh_weight=0.25,
    levels=1,
    depth_decay=1.0,
):
    """
    人工缺失区域的多层 Haar 小波结构损失。

    每层独立计算 masked L1，再按 depth_decay^(depth) 加权聚合。

    当 levels=1 时，行为与原 wavelet_loss 完全一致（只做单层）。
    当 levels>1 时，每层共享相同的子带权重 (ll/lh/hl/hh_weight)。

    pred / target:
        模型最终 V_fake / 真实 V，单通道 [B,1,H,W]

    missing_mask:
        人工缺失 mask (不能传 real_missing)

    depth_decay:
        深度衰减因子 (>= 0, <= 1)。
        depth_decay=1.0  → 每层权重相同
        depth_decay=0.5  → 越粗的层权重越低 (衰减)
        depth_decay=2.0  → 越粗的层权重越高 (膨胀，少用)
    """
    pred = _ensure_4d_v(pred)
    target = _ensure_4d_v(target)

    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target 尺寸不一致: "
            f"{tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    p_coeffs = haar_dwt2_multi(pred, levels=levels)
    t_coeffs = haar_dwt2_multi(target, levels=levels)

    total = pred.new_zeros(())
    total_w = 0.0

    for depth in range(levels):
        p_ll, p_lh, p_hl, p_hh = p_coeffs[depth]
        t_ll, t_lh, t_hl, t_hh = t_coeffs[depth]

        if missing_mask is None:
            m_ll = m_lh = m_hl = m_hh = None
        else:
            m = _prepare_mask(
                missing_mask,
                p_ll.shape[-2:],
            )
            m = (m > 0.0).float()
            m_ll = m_lh = m_hl = m_hh = m

        l_ll = _masked_l1(p_ll, t_ll, m_ll)
        l_lh = _masked_l1(p_lh, t_lh, m_lh)
        l_hl = _masked_l1(p_hl, t_hl, m_hl)
        l_hh = _masked_l1(p_hh, t_hh, m_hh)

        band_sum = (
            float(ll_weight) * l_ll
            + float(lh_weight) * l_lh
            + float(hl_weight) * l_hl
            + float(hh_weight) * l_hh
        )
        band_w = (
            float(ll_weight) + float(lh_weight)
            + float(hl_weight) + float(hh_weight)
        )

        level_w = float(depth_decay) ** depth
        total = total + level_w * band_sum / max(band_w, 1e-12)
        total_w += level_w

    if total_w <= 1e-12:
        return pred.new_zeros(())

    return total / total_w


def wavelet_loss_full(
    pred,
    target,
    ll_weight=0.25,
    lh_weight=0.25,
    hl_weight=0.25,
    hh_weight=0.25,
    levels=1,
    depth_decay=1.0,
):
    """
    不带 mask 的完整多层小波损失。
    仅用于特殊实验或诊断，不建议作为主训练损失。
    """
    return wavelet_loss(
        pred,
        target,
        missing_mask=None,
        ll_weight=ll_weight,
        lh_weight=lh_weight,
        hl_weight=hl_weight,
        hh_weight=hh_weight,
        levels=levels,
        depth_decay=depth_decay,
    )


def wavelet_gradient_loss(pred, target, missing_mask=None):
    """
    小波高频结构损失。
    重点约束 LH/HL/HH。

    当需要进一步强调局部风速变化时使用。
    当前主训练可以不单独调用。
    """
    pred = _ensure_4d_v(pred)
    target = _ensure_4d_v(target)

    p_ll, p_lh, p_hl, p_hh = haar_dwt2(pred)
    t_ll, t_lh, t_hl, t_hh = haar_dwt2(target)

    if missing_mask is None:
        return (
            F.l1_loss(p_lh, t_lh)
            + F.l1_loss(p_hl, t_hl)
            + F.l1_loss(p_hh, t_hh)
        ) / 3.0

    m = _prepare_mask(
        missing_mask,
        p_ll.shape[-2:],
    )
    m = (m > 0.0).float()

    l1 = _masked_l1(p_lh, t_lh, m)
    l2 = _masked_l1(p_hl, t_hl, m)
    l3 = _masked_l1(p_hh, t_hh, m)

    return (l1 + l2 + l3) / 3.0