# -*- coding: utf-8 -*-
import torch

from untils.losses import reconstruction_loss
from untils.physics import physics_consistency_loss
from untils.refiner import refine_delta_and_gate, merge_v_final
from untils.wavelet import wavelet_loss


def _to_4d_mask(x):
    if x.ndim == 5:
        return x[:, 0:1]
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(1)
    raise ValueError(f"Mask shape错误: {tuple(x.shape)}")


def _target_mask(missing, rel, valid):
    missing = _to_4d_mask(missing).float()
    rel = _to_4d_mask(rel).float()
    valid = _to_4d_mask(valid).float()
    return missing * rel * valid


def _known_mask(given, rel, valid):
    given = _to_4d_mask(given).float()
    rel = _to_4d_mask(rel).float()
    valid = _to_4d_mask(valid).float()
    return given * rel * valid


def _reliable_mask(given, rel, valid):
    return _known_mask(given, rel, valid)


def _get_v_true(v_true, target_t=0):
    if v_true.ndim == 5:
        return v_true[:, :, target_t]
    if v_true.ndim == 4:
        return v_true
    if v_true.ndim == 3:
        return v_true.unsqueeze(1)
    raise ValueError(f"v_true shape错误: {tuple(v_true.shape)}")


def compute_refiner_losses(
    R,
    v_gain,
    x,
    obs_orig,
    given,
    rel,
    valid,
    v_true,
    missing,
    t,
    cfg,
    device,
):
    delta, gate = refine_delta_and_gate(
        R,
        v_gain,
        x,
        given,
        rel,
        cfg,
        device,
        use_wavelet=getattr(cfg, "use_wavelet", True),
    )

    known = _known_mask(given, rel, valid)
    target = _target_mask(missing, rel, valid)

    v_final = merge_v_final(
        v_gain,
        delta,
        known,
    )

    gt_v = _get_v_true(
        v_true,
        getattr(cfg, "target_idx", t),
    )

    # 1. Refiner 重建损失
    g_refine = reconstruction_loss(
        v_final,
        gt_v,
        target,
        torch.ones_like(target),
    )

    # 2. 小波损失
    g_wave = v_final.new_zeros(())

    if getattr(cfg, "use_wavelet", True):
        g_wave = wavelet_loss(
            v_final,
            gt_v,
            missing_mask=target,
            ll_weight=getattr(cfg, "wavelet_ll_weight", 0.25),
            lh_weight=getattr(cfg, "wavelet_lh_weight", 0.25),
            hl_weight=getattr(cfg, "wavelet_hl_weight", 0.25),
            hh_weight=getattr(cfg, "wavelet_hh_weight", 0.25),
            levels=int(getattr(cfg, "n_wavelet_levels", 1)),
            depth_decay=float(getattr(cfg, "wavelet_depth_decay", 1.0)),
        )

    # 3. 物理一致性损失
    g_phy = v_final.new_zeros(())

    if getattr(cfg, "use_physics", False):
        reliable = _reliable_mask(
            given,
            rel,
            valid,
        )

        # 所有有效但不是可靠已知值的位置：
        # 包括真实缺失、人工缺失以及不可靠观测。
        valid_4d = _to_4d_mask(valid).float()
        physics_target = (
            (1.0 - reliable) * valid_4d
        ).clamp(0.0, 1.0)

        g_phy = physics_consistency_loss(
            pred=v_final,
            reference=gt_v,
            target_mask=physics_target,
            radial_weight=getattr(cfg, "radial_weight", 1.0),
            azimuth_weight=getattr(cfg, "azimuth_weight", 1.0),
            gradient_weight=getattr(cfg, "gradient_weight", 0.0),
            reliable_mask=reliable,
        )

    return (
        v_final,
        g_refine,
        g_wave,
        g_phy,
        delta,
        gate,
    )


def compute_total_refiner_loss(
    g_refine,
    g_wave,
    g_phy,
    cfg,
):
    w_ref = float(
        getattr(cfg, "lambda_refine", 1.0)
    )
    w_wave = float(
        getattr(cfg, "lambda_wavelet", 0.1)
    )
    w_phy = float(
        getattr(cfg, "lambda_physics", 0.1)
    )

    return (
        w_ref * g_refine
        + w_wave * g_wave
        + w_phy * g_phy
    )


def compute_refiner_losses_simple(
    R,
    v_gain,
    x,
    given,
    rel,
    valid,
    v_true,
    missing,
    cfg,
    device,
):
    target_t = int(
        getattr(cfg, "target_idx", 0)
    )

    gt_v = _get_v_true(
        v_true,
        target_t,
    )

    delta, gate = refine_delta_and_gate(
        R,
        v_gain,
        x,
        given,
        rel,
        cfg,
        device,
        use_wavelet=getattr(cfg, "use_wavelet", True),
    )

    known = _known_mask(
        given,
        rel,
        valid,
    )

    target = _target_mask(
        missing,
        rel,
        valid,
    )

    v_final = merge_v_final(
        v_gain,
        delta,
        known,
    )

    g_refine = reconstruction_loss(
        v_final,
        gt_v,
        target,
        torch.ones_like(target),
    )

    g_wave = v_final.new_zeros(())
    g_phy = v_final.new_zeros(())

    if getattr(cfg, "use_wavelet", True):
        g_wave = wavelet_loss(
            v_final,
            gt_v,
            missing_mask=target,
            ll_weight=getattr(cfg, "wavelet_ll_weight", 0.25),
            lh_weight=getattr(cfg, "wavelet_lh_weight", 0.25),
            hl_weight=getattr(cfg, "wavelet_hl_weight", 0.25),
            hh_weight=getattr(cfg, "wavelet_hh_weight", 0.25),
            levels=int(getattr(cfg, "n_wavelet_levels", 1)),
            depth_decay=float(getattr(cfg, "wavelet_depth_decay", 1.0)),
        )

    if getattr(cfg, "use_physics", False):
        reliable = _reliable_mask(
            given,
            rel,
            valid,
        )

        valid_4d = _to_4d_mask(valid).float()
        physics_target = (
            (1.0 - reliable) * valid_4d
        ).clamp(0.0, 1.0)

        g_phy = physics_consistency_loss(
            pred=v_final,
            reference=gt_v,
            target_mask=physics_target,
            radial_weight=getattr(cfg, "radial_weight", 1.0),
            azimuth_weight=getattr(cfg, "azimuth_weight", 1.0),
            gradient_weight=getattr(cfg, "gradient_weight", 0.0),
            reliable_mask=reliable,
        )

    total = compute_total_refiner_loss(
        g_refine,
        g_wave,
        g_phy,
        cfg,
    )

    return {
        "v_final": v_final,
        "delta": delta,
        "gate": gate,
        "loss_refine": g_refine,
        "loss_wave": g_wave,
        "loss_phy": g_phy,
        "loss_total": total,
    }