# -*- coding: utf-8 -*-
"""GAIN 补全 + 径向量程扩展 推理。

用法：python main.py ext_test
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import Config
from untils.data import (
    Normalizer, make_obs_rel, pad_to, crop_to,
    ORI_H, ORI_W, generate_pure_structural_missing_mask,
)
from train import generate_contextual_cue
from untils.refiner import apply_refiner
from extension_model import build_extension_model
from train_joint_ext import (
    build_generator, build_refiner,
    assemble_extension_input_from_completed,
    build_contextual_fusion, build_generator_inputs,
)
from untils.losses import compute_extension_metrics


def load_joint_ext_models(model_path, device):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in ckpt["cfg"].items(): setattr(cfg, k, v)

    G = build_generator(cfg).to(device)
    G.load_state_dict(ckpt["model"])
    G.eval()

    R = None
    if getattr(cfg, "use_refiner", False) and "refiner" in ckpt:
        R = build_refiner(cfg)
        if R is not None:
            R = R.to(device)
            R.load_state_dict(ckpt["refiner"])
            R.eval()
            print("  [Refiner] 已加载")

    Ext = build_extension_model(cfg).to(device)
    if "ext_model" in ckpt:
        Ext.load_state_dict(ckpt["ext_model"])
        Ext.eval()
        print("  [Ext] 已加载量程扩展网络")
    else:
        print("  [Ext] 检查点中无扩展网络，随机初始化")
        Ext.eval()

    normalizer = Normalizer(cfg.normalize, cfg.clip_percentile, cfg.phys_channels)
    norm_state = ckpt["normalizer"]
    if isinstance(norm_state, dict):
        normalizer.__dict__.update(norm_state)
    print("  [Normalizer] 从 checkpoint 加载")

    return G, R, Ext, cfg, normalizer


@torch.no_grad()
def impute_and_extend(G, R, Ext, arr, normalizer, cfg, device, test_drop_prob=0.0):
    T, H, W = arr.shape[:3]
    C = cfg.phys_channels
    near_end = cfg.near_end_gate
    far_end = cfg.far_end_gate

    obs, rel = make_obs_rel(arr, C)
    phys = np.nan_to_num(arr[..., :C].astype(np.float32), nan=0.0)
    norm_data = normalizer.transform(phys)
    norm_data = np.where(obs[..., None] == 1, norm_data, 0.0)

    completed_out = arr.copy()
    gain_only_out = arr.copy()

    all_ext_pred = []
    all_ext_true = []
    all_ext_valid = []
    all_comp_final = []
    all_art_mask = []

    seq_buffer = []  # 缓存时序数据窗口

    G.eval()
    if R is not None: R.eval()
    Ext.eval()

    for i in tqdm(range(T), desc="推理"):
        x_t = torch.from_numpy(pad_to(norm_data[i:i + 1].transpose(3, 0, 1, 2))).float().unsqueeze(0).to(device)
        m_t = torch.from_numpy(pad_to(obs[i:i + 1])).float().unsqueeze(0).to(device)
        r_t = torch.from_numpy(pad_to(rel[i:i + 1])).float().unsqueeze(0).to(device)

        if m_t.ndim == 4: m_t = m_t.unsqueeze(1).repeat(1, C, 1, 1, 1)
        if r_t.ndim == 4: r_t = r_t.unsqueeze(1).repeat(1, C, 1, 1, 1)

        B = x_t.shape[0]
        vobs_pad = m_t[:, 0, 0]
        vrel_pad = r_t[:, 0, 0]

        art_mask_pad = torch.zeros_like(vobs_pad)
        if test_drop_prob > 0:
            valid_eff = (vobs_pad * vrel_pad).unsqueeze(0)
            art, _ = generate_pure_structural_missing_mask(
                1, vobs_pad.shape[1], vobs_pad.shape[2],
                device, test_drop_prob, cfg=cfg, valid_mask=valid_eff
            )
            art_mask_pad = art[0]

        vgiven_pad = vobs_pad * (1 - art_mask_pad)
        m_given = m_t.clone()
        m_given[:, 0] = vgiven_pad.unsqueeze(1).expand_as(m_t[:, 0])
        if C > 1: m_given[:, 1:] = m_t[:, 1:] * (1 - art_mask_pad.unsqueeze(1).unsqueeze(1))

        xin = x_t.clone()
        given_all = vgiven_pad.unsqueeze(1).unsqueeze(1).expand_as(x_t[:, 0:1])
        if C > 1:
            given_all = torch.cat([given_all, m_given[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)], dim=1)
            given_all = (m_given > 0).float()
        xin = x_t * given_all
        z = torch.zeros_like(xin)

        use_cue = getattr(cfg, "use_cue", False)
        cue = generate_contextual_cue(xin, m_given, 0, 0) if use_cue else torch.zeros(B, 1, xin.shape[3], xin.shape[4],
                                                                                      device=device)

        xb = build_contextual_fusion(xin, m_given, cue, z, 0.0)
        gx, gm = build_generator_inputs(xb, m_given)

        v_gain = G(gx, gm)[:, 0, 0:1]
        v_final = v_gain.clone()
        if R is not None and getattr(cfg, "use_refiner", False):
            obs_2d = m_given[:, 0:1, 0]
            rel_2d = r_t[:, 0:1, 0]
            valid_2d = torch.ones_like(obs_2d)
            if getattr(cfg, "protect_bad_rays_refiner", True):
                from untils.data import BAD_RAYS as _BAD, PAD_LEFT as _PL
                obs_2d, rel_2d = obs_2d.clone(), rel_2d.clone()
                for br in _BAD:
                    j = int(br) + _PL
                    if 0 <= j < obs_2d.shape[-1]:
                        obs_2d[..., j] = 1.0;
                        rel_2d[..., j] = 1.0
            v_final = apply_refiner(R, v_gain, x_t, obs_2d, rel_2d, valid_2d, cfg, device)

        vobs_2d = vobs_pad
        v_completed = vobs_2d * x_t[:, 0, 0] + (1 - vobs_2d) * v_final[:, 0]

        final_crop = crop_to(v_final[0, 0], ORI_H, ORI_W).cpu().numpy()
        all_comp_final.append(final_crop)

        final_denorm = normalizer.inverse_transform(np.stack([final_crop], axis=-1))[..., 0]
        gain_only_out[i, :ORI_H, :ORI_W, 0] = np.where(
            obs[i].astype(bool), arr[i, :ORI_H, :ORI_W, 0], final_denorm[:ORI_H, :ORI_W]
        )

        v_completed_crop = crop_to(v_completed, ORI_H, ORI_W)
        vobs_crop = crop_to(vobs_2d, ORI_H, ORI_W)
        vrel_crop = crop_to(vrel_pad, ORI_H, ORI_W)
        vtrue_crop = crop_to(x_t[:, 0, 0], ORI_H, ORI_W)
        art_crop = crop_to(art_mask_pad, ORI_H, ORI_W)

        v_near = v_completed_crop[:, :near_end]
        near_obs = crop_to(vgiven_pad, ORI_H, ORI_W)[:, :near_end]
        near_rel = vrel_crop[:, :near_end]
        near_cond = crop_to(xin[:, 1:, 0], ORI_H, ORI_W)[:, :, :near_end]

        ext_input = assemble_extension_input_from_completed(v_near, near_obs, near_rel, near_cond)

        # [核心修复]：支持 spatiotemporal 时序预测推理
        if getattr(cfg, "ext_model_type", "cnn") == "spatiotemporal":
            temporal_steps = getattr(cfg, "temporal_in_steps", 3)
            if i == 0:
                seq_buffer = [ext_input] * temporal_steps
            else:
                seq_buffer.pop(0)
                seq_buffer.append(ext_input)
            x_seq_input = torch.stack(seq_buffer, dim=1)
            far_pred = Ext(ext_input, x_seq_input)
        else:
            far_pred = Ext(ext_input)

        far_v_true = vtrue_crop[:, near_end:far_end]
        far_valid = vobs_crop[:, near_end:far_end] * vrel_crop[:, near_end:far_end]

        all_ext_pred.append(far_pred[0].cpu().numpy())
        all_ext_true.append(far_v_true[0].cpu().numpy())
        all_ext_valid.append(far_valid[0].cpu().numpy())
        all_art_mask.append(art_crop[0].cpu().numpy())

        v_full = v_completed_crop.clone()
        if v_full.shape[1] >= far_end:
            v_full[:, near_end:far_end] = (
                    far_valid * far_pred[0] + (1 - far_valid) * v_full[:, near_end:far_end]
            )

        v_full_np = v_full[0].cpu().numpy()
        v_full_denorm = normalizer.inverse_transform(np.stack([v_full_np], axis=-1))[..., 0]
        completed_out[i, :ORI_H, :ORI_W, 0] = np.where(
            obs[i].astype(bool), arr[i, :ORI_H, :ORI_W, 0], v_full_denorm[:ORI_H, :ORI_W]
        )

    return {
        "completed": completed_out, "gain_only": gain_only_out,
        "ext_pred": np.stack(all_ext_pred), "ext_true": np.stack(all_ext_true),
        "ext_valid": np.stack(all_ext_valid), "comp_final": np.stack(all_comp_final),
        "vobs": obs, "vrel": rel, "art_mask": np.stack(all_art_mask),
    }


def plot_comparison(result, arr, normalizer, cfg, out_dir, stem, test_drop_prob):
    near_end, far_end = cfg.near_end_gate, cfg.far_end_gate
    T = arr.shape[0]
    mid_t = T // 2
    obs_t, rel_t, art_t = result["vobs"][mid_t], result["vrel"][mid_t], result["art_mask"][mid_t]
    true_v = arr[mid_t, :ORI_H, :ORI_W, 0]
    full_v = result["completed"][mid_t, :ORI_H, :ORI_W, 0]

    missing_mask = (obs_t == 0)
    art_eval_mask = (art_t > 0.5) & (obs_t == 1) & (rel_t == 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].pcolormesh(true_v, cmap="RdBu_r", shading="auto")
    axes[0].set_title("Original V");
    axes[0].axhline(y=near_end, color="green", linestyle="--")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(full_v, cmap="RdBu_r", shading="auto")
    axes[1].set_title("GAIN + Extension")
    axes[1].axhline(y=near_end, color="green", linestyle="--");
    axes[1].axhline(y=far_end, color="red", linestyle="--")
    plt.colorbar(im1, ax=axes[1])

    diff_all = np.abs(full_v - true_v)
    diff_show = diff_all.copy();
    diff_show[obs_t == 1] = 0
    im2 = axes[2].pcolormesh(diff_show, cmap="hot", shading="auto")
    axes[2].set_title("|Error| at real missing points");
    axes[2].axhline(y=near_end, color="green", linestyle="--")
    plt.colorbar(im2, ax=axes[2])

    if test_drop_prob > 0:
        im3 = axes[3].pcolormesh(diff_all * art_eval_mask, cmap="hot", shading="auto")
        axes[3].set_title(f"|Error| at art. masked ({test_drop_prob:.0%})")
    else:
        im3 = axes[3].pcolormesh(diff_all * missing_mask, cmap="hot", shading="auto")
        axes[3].set_title("|Error| at missing points")
    axes[3].axhline(y=near_end, color="green", linestyle="--");
    plt.colorbar(im3, ax=axes[3])

    for ax in axes: ax.set_xlabel("Azimuth"); ax.set_ylabel("Gate")
    plt.tight_layout();
    fig_path = out_dir / f"{stem}_comparison.png"
    plt.savefig(fig_path, dpi=150);
    plt.close()


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    model_path = getattr(cfg, "model_path", "")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"联合扩展模型检查点不存在: {model_path}")
    test_drop_prob = getattr(cfg, "test_drop_prob", 0.0)

    print(f"[模型] {model_path}\n[输入] {cfg.input_npy}\n[人工遮挡率] {test_drop_prob:.1%}")
    G, R, Ext, cfg_loaded, normalizer = load_joint_ext_models(model_path, device)

    refit_npy = getattr(cfg, "normalizer_refit_npy", None)
    if refit_npy:
        refit_arr = np.load(refit_npy)
        refit_obs, _ = make_obs_rel(refit_arr, cfg_loaded.phys_channels)
        normalizer = Normalizer(cfg_loaded.normalize, cfg_loaded.clip_percentile, cfg_loaded.phys_channels).fit(
            refit_arr, refit_obs)

    near_end, far_end = cfg_loaded.near_end_gate, cfg_loaded.far_end_gate
    print(f"  扩展门: {near_end}~{far_end - 1}")

    arr = np.load(cfg.input_npy)
    result = impute_and_extend(G, R, Ext, arr, normalizer, cfg_loaded, device, test_drop_prob=test_drop_prob)

    obs_raw, _ = make_obs_rel(arr, cfg_loaded.phys_channels)
    for key in ("completed", "gain_only"): result[key][obs_raw == 0, 0] = np.nan

    out_dir = Path(cfg.output_dir);
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.input_npy).stem
    np.save(out_dir / f"{stem}_completed_ext.npy", result["completed"])
    np.save(out_dir / f"{stem}_completed_gain.npy", result["gain_only"])

    ext_pred_t = torch.from_numpy(result["ext_pred"]).float()
    ext_true_t = torch.from_numpy(result["ext_true"]).float()
    ext_valid_t = torch.from_numpy(result["ext_valid"]).float()
    ext_mae, ext_rmse, ext_pts = compute_extension_metrics(ext_pred_t, ext_true_t, ext_valid_t)

    # [核心修复]：修复截断反归一化时的数值范围
    clip_min = -1.0 if getattr(cfg_loaded, "normalize", "minmax") == "minmax" else -10.0
    clip_max = 1.0 if getattr(cfg_loaded, "normalize", "minmax") == "minmax" else 10.0
    ext_pred_clipped = np.clip(result["ext_pred"], clip_min, clip_max)
    ext_true_clipped = np.clip(result["ext_true"], clip_min, clip_max)

    ext_pred_denorm = normalizer.inverse_transform(np.stack([ext_pred_clipped], axis=-1))[..., 0]
    ext_true_denorm = normalizer.inverse_transform(np.stack([ext_true_clipped], axis=-1))[..., 0]
    ext_valid_np = result["ext_valid"]
    ext_mae_real = (np.abs(ext_pred_denorm - ext_true_denorm) * ext_valid_np).sum() / max(ext_valid_np.sum(), 1.0)
    ext_rmse_real = np.sqrt(
        ((ext_pred_denorm - ext_true_denorm) ** 2 * ext_valid_np).sum() / max(ext_valid_np.sum(), 1.0))

    comp_mae = comp_rmse = comp_pts = comp_mae_real = comp_rmse_real = 0.0
    if test_drop_prob > 0:
        obs_np, rel_np, art_np, comp_final = result["vobs"], result["vrel"], result["art_mask"], result["comp_final"]
        true_norm = normalizer.transform(np.nan_to_num(arr[:, :ORI_H, :ORI_W, 0:1], nan=0.0))[..., 0]
        true_norm = np.where(obs_np[:, :ORI_H, :ORI_W] == 1, true_norm, 0.0)
        art_eval = (art_np[:, :ORI_H, :ORI_W] > 0.5) & (obs_np[:, :ORI_H, :ORI_W] == 1) & (
                    rel_np[:, :ORI_H, :ORI_W] == 1)
        comp_diff = np.abs(comp_final - true_norm) * art_eval
        comp_mae = comp_diff.sum() / max(art_eval.sum(), 1.0)
        comp_rmse = np.sqrt(((comp_final - true_norm) ** 2 * art_eval).sum() / max(art_eval.sum(), 1.0))
        comp_pts = art_eval.sum()

        true_real = arr[:, :ORI_H, :ORI_W, 0]
        pred_real = normalizer.inverse_transform(np.stack([comp_final], axis=-1))[..., 0]
        comp_diff_real = np.abs(pred_real - true_real) * art_eval
        comp_mae_real = comp_diff_real.sum() / max(art_eval.sum(), 1.0)
        comp_rmse_real = np.sqrt(((pred_real - true_real) ** 2 * art_eval).sum() / max(art_eval.sum(), 1.0))

    print(f"\n{'=' * 60}\n评估结果\n{'=' * 60}")
    if test_drop_prob > 0:
        print(
            f"\n[补全效果] 归一化 MAE/RMSE = {comp_mae:.4f} / {comp_rmse:.4f} | 真实值 = {comp_mae_real:.4f} m/s / {comp_rmse_real:.4f} m/s")
    print(
        f"\n[扩展效果] 归一化 MAE/RMSE = {ext_mae:.4f} / {ext_rmse:.4f} | 真实值 = {ext_mae_real:.4f} m/s / {ext_rmse_real:.4f} m/s")

    metrics = {
        "comp_MAE": comp_mae, "comp_RMSE": comp_rmse, "comp_MAE_real": comp_mae_real,
        "comp_RMSE_real": comp_rmse_real,
        "ext_MAE": ext_mae, "ext_RMSE": ext_rmse, "ext_pts": ext_pts,
        "ext_MAE_real": ext_mae_real, "ext_RMSE_real": ext_rmse_real,
    }
