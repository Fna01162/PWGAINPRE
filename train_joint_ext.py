# -*- coding: utf-8 -*-
"""GAIN 补全 + 径向量程扩展 联合微调。

用法：python main.py joint_ext

训练流程：
1. 加载预训练 GAIN 和量程扩展网络
2. 每个 PPI 扫描：
   a. GAIN 逐帧补全 → completed V (85×125)
   b. 提取近距离 completed V (68×125) → 量程扩展网络 → 远距离 V 预测 (10×125)
   c. 联合损失 = GAIN 损失 + λ_ext × 扩展损失
3. 反向传播同时更新 GAIN、Refiner 和扩展网络
4. 验证集同时监控补全质量和扩展质量
"""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from untils.data import (
    MultichannelClipDataset, Normalizer,
    generate_pure_structural_missing_mask, make_obs_rel,
    pad_to, crop_to, ORI_H, ORI_W, PAD_H, PAD_W,
    PHYS_CHANNELS, TARGET_CHANNEL,
)
from untils.losses import (
    azimuth_curvature_loss, azimuth_tv_loss,
    discriminator_loss, generate_hint, generator_adv_loss,
    reconstruction_loss, temporal_consistency_loss, tv_loss,
    extension_mae_loss, compute_extension_metrics, extension_composite_loss,
)
from untils.losses_refine import compute_refiner_losses
from untils.refiner import ResidualRefiner, apply_refiner
from untils.wavelet import wavelet_feature_channels
from model import UNetGenerator, PatchGANDiscriminator, init_weights
from untils.exp_utils import print_model_stats
from extension_model import build_extension_model, init_extension_weights

BAD_RAYS = [73, 74, 75]
CHANNEL_NAMES = ["V", "CNR", "SW", "Skew"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_normalizer(train_npy, method, clip, phys_channels):
    arr = np.load(train_npy)
    obs, _ = make_obs_rel(arr, phys_channels)
    return Normalizer(method, clip, phys_channels).fit(arr, obs)


def repeat_mask(m, channels):
    if m.dim() == 4:
        m = m.unsqueeze(1)
    return m.repeat(1, channels, 1, 1, 1) if m.shape[1] == 1 else m


def generate_contextual_cue(x, effective_mask, target_t, target_channel=0):
    B, C, T, H, W = x.shape
    v = x[:, target_channel:target_channel + 1]
    if effective_mask.dim() == 4:
        effective_mask = effective_mask.unsqueeze(1)
    mask_v = effective_mask[:, target_channel:target_channel + 1]
    ti = torch.arange(T, device=x.device, dtype=torch.float32)
    dist = torch.abs(ti - float(target_t)).view(1, T, 1, 1).expand(B, T, H, W).clone()
    valid = mask_v[:, 0] > 0.5
    dist[~valid] = float("inf")
    nt = torch.argmin(dist, dim=1)
    md = torch.amin(dist, dim=1)
    nv = torch.gather(v[:, 0], 1, nt.unsqueeze(1))
    hc = torch.isfinite(md)
    return torch.where(hc.unsqueeze(1), nv, torch.zeros_like(nv))


def build_contextual_fusion(x_bar, effective_mask, cue, z, noise_scale=0.1):
    B, C, T, H, W = x_bar.shape
    m = repeat_mask(effective_mask, C)
    xb = x_bar.clone()
    mv = m[:, 0:1]
    cue5 = cue.unsqueeze(2).expand(-1, -1, T, -1, -1)
    zv = z[:, 0:1]
    xb[:, 0:1] = mv * x_bar[:, 0:1] + (1 - mv) * cue5 + (1 - mv) * zv * noise_scale
    if C > 1:
        xb[:, 1:] = m[:, 1:] * x_bar[:, 1:] + (1 - m[:, 1:]) * z[:, 1:] * noise_scale
    return xb


def build_generator_inputs(x_bar, m):
    m = repeat_mask(m, x_bar.shape[1])
    return x_bar, m


def make_training_cue(x, m, r, valid, target_t):
    m5 = repeat_mask(m, x.shape[1])
    r5 = repeat_mask(r, x.shape[1])
    v5 = repeat_mask(valid, x.shape[1])
    em = m5 * r5 * v5
    return generate_contextual_cue(x, em, target_t, TARGET_CHANNEL), em


def prepare_artificial_mask(B, H, W, device, cfg, valid_eff):
    art, rates = generate_pure_structural_missing_mask(
        B, H, W, device, cfg.drop_prob, cfg=cfg, valid_mask=valid_eff
    )
    return (art if art.dim() == 4 else art.unsqueeze(1)).float(), rates


def build_generator(cfg):
    return UNetGenerator(
        in_channels=8, out_channels=1, base_filter=cfg.base_filter,
        use_cue=cfg.use_cue, use_nafblock=cfg.use_nafblock,
        use_volformer=cfg.use_volformer,
        use_axial_w=cfg.use_axial_w,
        axial_num_pairs=cfg.axial_num_pairs,
        axial_num_heads=cfg.axial_num_heads,
    )


def build_refiner(cfg):
    if not getattr(cfg, "use_refiner", False):
        return None
    phys_ch = int(getattr(cfg, "phys_channels", 4))
    geom_ch = 3
    mask_ch = 2
    base_ch = phys_ch + geom_ch + mask_ch
    if getattr(cfg, "use_wavelet", False):
        wl = wavelet_feature_channels(int(getattr(cfg, "n_wavelet_levels", 1)))
    else:
        wl = 0
    in_ch = base_ch + wl
    return ResidualRefiner(in_channels=in_ch, base_filter=cfg.refiner_base_filter)


def assemble_extension_input_from_completed(v_near, near_obs, near_rel, near_cond):
    """从 GAIN 补全后的近距离 V 组装扩展网络输入。

    v_near    : [B, H_near, W]      补全后的近距离 V
    near_obs  : [B, H_near, W]      近距离 V 观测掩码
    near_rel  : [B, H_near, W]      近距离 V 可靠掩码
    near_cond : [B, C-1, H_near, W] 近距离条件通道 (CNR, SW, Skew)

    返回 : [B, C+2, H_near, W]
    """
    v_near_4d = v_near.unsqueeze(1)
    components = [v_near_4d, near_cond, near_obs.unsqueeze(1), near_rel.unsqueeze(1)]
    return torch.cat(components, dim=1)


def train_one_epoch(G, D, R, Ext, loader, optim_G, optim_D, optim_Ext, cfg, device):
    G.train()
    D.train()
    Ext.train()
    if R is not None:
        R.train()

    near_end = cfg.near_end_gate
    far_end = cfg.far_end_gate
    is_st = (cfg.ext_model_type == "spatiotemporal")
    temporal_steps = getattr(cfg, "temporal_in_steps", 3) if is_st else 1

    keys = [
        "G", "D", "D_real", "D_fake", "G_adv", "G_rec",
        "TV", "RING", "AZTV", "CURV", "TEMP",
        "REFINE", "WAVE", "PHY",
        "ext_loss", "ext_MAE",
        "train_miss_rate",
    ]
    s = {k: 0. for k in keys}
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x, m, r, valid = batch
        x, m, r, valid = [v.to(device) for v in (x, m, r, valid)]
        B, C, T, H, W = x.shape
        t = min(int(cfg.target_idx), T - 1)

        vobs = m[:, 0:1, t]
        vrel = r[:, 0:1, t]
        vval = valid[:, 0:1, t]
        ve = vrel * vval

        art, rates = prepare_artificial_mask(B, H, W, device, cfg, ve)
        art = art * ve
        if isinstance(rates, list):
            rv = float(np.mean(rates))
        elif torch.is_tensor(rates):
            rv = float(rates.float().mean().item())
        else:
            rv = float(rates)
        s["train_miss_rate"] += rv

        obs_all = m[:, :, t]
        rel_all = r[:, :, t]
        valid_all = valid[:, :, t]
        given_all = obs_all * (1 - art)
        vgiven = vobs * (1 - art)

        xin = x.clone()
        xin[:, :, t] = x[:, :, t] * given_all
        min_ = m.clone()
        min_[:, :, t] = given_all
        z = torch.randn_like(xin)

        cue, em = make_training_cue(xin, min_, r, valid, t)
        if not cfg.use_cue:
            cue = torch.zeros(B, 1, H, W, device=device, dtype=x.dtype)
        xb = build_contextual_fusion(xin, em, cue, z, cfg.noise_scale)
        gx, gm = build_generator_inputs(xb, em)

        # ──── D step ────
        with torch.no_grad():
            vf = G(gx, gm)
        vfake = vf[:, 0, t:t + 1]
        vtrue = x[:, 0, t:t + 1]
        gv = vgiven
        vhat = vtrue * gv + vfake.detach() * (1 - gv)
        hint = generate_hint(gv, cfg.hint_rate)
        dl = D(vhat, hint)
        d_loss = discriminator_loss(dl, gv, vval)

        optim_D.zero_grad(set_to_none=True)
        d_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
        optim_D.step()

        # ──── G + Ext step ────
        vf = G(gx, gm)
        vfake = vf[:, 0, t:t + 1]
        vhat = vtrue * gv + vfake * (1 - gv)
        dlg = D(vhat, hint)

        g_adv = generator_adv_loss(dlg, art, vval)
        g_rec = reconstruction_loss(vfake, vtrue, art, vrel)
        g_tv = tv_loss(vfake, art, vval) if cfg.tv_weight > 0 else x.new_zeros(())
        g_temp = temporal_consistency_loss(vf, t, art, vval) if cfg.temp_weight > 0 and T > 1 else x.new_zeros(())
        g_ring = (crop_to(vfake, ORI_H, ORI_W)[..., 0] - crop_to(vfake, ORI_H, ORI_W)[..., -1]).abs().mean() if cfg.ring_weight > 0 else x.new_zeros(())
        mask_az = torch.clamp(art + (1 - vrel) * vval, 0, 1)
        g_aztv = azimuth_tv_loss(vfake, mask_az) if cfg.aztv_weight > 0 else x.new_zeros(())
        g_curv = azimuth_curvature_loss(vfake, mask_az) if cfg.curv_weight > 0 else x.new_zeros(())

        gt = (
            cfg.adv_weight * g_adv
            + cfg.alpha * cfg.recon_weight * g_rec
            + cfg.tv_weight * g_tv
            + cfg.temp_weight * g_temp
            + cfg.ring_weight * g_ring
            + cfg.aztv_weight * g_aztv
            + cfg.curv_weight * g_curv
        )

        rv_ref = wvl = phy = 0.
        v_final = vfake
        if R is not None and getattr(cfg, "use_refiner", False):
            try:
                vf_r, GR, GW, GP, _, _ = compute_refiner_losses(
                    R, vfake, x, obs_all, vgiven, vrel, vval, vtrue, art, t, cfg, device,
                )
                gt = gt + cfg.lambda_refine * GR + cfg.lambda_wavelet * GW + cfg.lambda_physics * GP
                v_final = vf_r
                rv_ref, wvl, phy = GR.item(), GW.item(), GP.item()
            except Exception:
                pass

        # ──── 量程扩展损失 ────
        # vgiven = vobs*(1-art)：真实缺失+人工遮挡点用G预测，观测点用真值
        # 扩展损失通过 (1-vgiven) 区域回传到G，覆盖范围由 drop_prob 控制
        v_given_for_ext = vgiven
        v_completed = vtrue * v_given_for_ext + v_final * (1 - v_given_for_ext)
        v_completed_crop = crop_to(v_completed[:, 0], ORI_H, ORI_W)

        v_near = v_completed_crop[:, :near_end]
        near_obs_crop = crop_to(v_given_for_ext[:, 0], ORI_H, ORI_W)[:, :near_end]
        near_rel_crop = crop_to(vrel[:, 0], ORI_H, ORI_W)[:, :near_end]
        near_cond_crop = crop_to(xin[:, 1:, t], ORI_H, ORI_W)[:, :, :near_end]

        if is_st:
            x_cur_input = assemble_extension_input_from_completed(
                v_near, near_obs_crop, near_rel_crop, near_cond_crop
            )
            half_t = temporal_steps // 2
            t_start = max(0, t - half_t)
            t_end = min(T, t + temporal_steps - half_t)
            seq_list = []
            for st in range(t_start, t_end):
                st_v = crop_to(v_completed_crop if st == t else crop_to(x[:, 0, st:st+1] if st < T else v_completed_crop, ORI_H, ORI_W)[:, :near_end], ORI_H, ORI_W)[:, :near_end]
                st_obs = crop_to(m[:, 0, st] if st < T else near_obs_crop, ORI_H, ORI_W)[:near_end]
                st_rel = crop_to(r[:, 0, st] if st < T else near_rel_crop, ORI_H, ORI_W)[:near_end]
                st_cond = crop_to(xin[:, 1:, st] if st < T else near_cond_crop, ORI_H, ORI_W)[:, :, :near_end]
                seq_list.append(assemble_extension_input_from_completed(st_v, st_obs, st_rel, st_cond))
            while len(seq_list) < temporal_steps:
                seq_list.append(seq_list[0])
            x_seq_input = torch.stack(seq_list[:temporal_steps], dim=1)
            far_pred = Ext(x_cur_input, x_seq_input)
        else:
            ext_input = assemble_extension_input_from_completed(
                v_near, near_obs_crop, near_rel_crop, near_cond_crop
            )
            far_pred = Ext(ext_input)

            # ──── 量程扩展损失 ────
            far_v_true = crop_to(vtrue[:, 0], ORI_H, ORI_W)[:, near_end:far_end]
            far_obs_mask = crop_to(vobs[:, 0], ORI_H, ORI_W)[:, near_end:far_end]
            far_rel_mask = crop_to(vrel[:, 0], ORI_H, ORI_W)[:, near_end:far_end]
            far_valid = (far_obs_mask * far_rel_mask)

            # [锚点提取] 获取实时补全结果的近场最外圈
            near_v_last = v_near[:, -1, :]
            near_valid_last = near_obs_crop[:, -1, :] * near_rel_crop[:, -1, :]

            # [核心替换] 调用复合物理扩展损失
            ext_loss = extension_composite_loss(far_pred, far_v_true, far_valid, near_v_last, near_valid_last)

        # near_v_last = x_cur[:, 0, -1, :] if is_spatiotemporal else near_x[:, 0, -1, :]
        # near_valid_last = (x_cur[:, 4, -1, :] * x_cur[:, 5, -1, :]) if is_spatiotemporal else (
        #             near_m[:, 0, -1, :] * near_r[:, 0, -1, :])
        # loss = extension_composite_loss(far_pred, far_v, far_valid, near_v_last, near_valid_last)

        g_joint = gt + cfg.lambda_ext * ext_loss

        optim_G.zero_grad(set_to_none=True)
        optim_Ext.zero_grad(set_to_none=True)
        g_joint.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
            if R is not None:
                torch.nn.utils.clip_grad_norm_(R.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(Ext.parameters(), cfg.ext_grad_clip)
        optim_G.step()
        optim_Ext.step()

        with torch.no_grad():
            ext_mae, _, _ = compute_extension_metrics(far_pred, far_v_true, far_valid)
            p = torch.sigmoid(dl)
            op = F.interpolate(gv, size=p.shape[-2:], mode="nearest")
            mp = F.interpolate(art, size=p.shape[-2:], mode="nearest")
            vp = F.interpolate(vval, size=p.shape[-2:], mode="nearest")
            rw = op * vp
            fw = mp * vp
            dr = (p * rw).sum().item() / (rw.sum().item() + 1e-8)
            df = (p * fw).sum().item() / (fw.sum().item() + 1e-8)

        vals = {
            "G": gt.item(), "D": d_loss.item(),
            "D_real": dr, "D_fake": df,
            "G_adv": g_adv.item(), "G_rec": g_rec.item(),
            "TV": g_tv.item(), "RING": g_ring.item(),
            "AZTV": g_aztv.item(), "CURV": g_curv.item(),
            "TEMP": g_temp.item(),
            "REFINE": rv_ref, "WAVE": wvl, "PHY": phy,
            "ext_loss": ext_loss.item(), "ext_MAE": ext_mae,
        }
        for k, v in vals.items():
            s[k] += v
        n += 1

    return {k: v / max(n, 1) for k, v in s.items()}


@torch.no_grad()
def validate(G, R, Ext, loader, cfg, device):
    G.eval()
    Ext.eval()
    if R is not None:
        R.eval()

    near_end = cfg.near_end_gate
    far_end = cfg.far_end_gate
    is_st = (cfg.ext_model_type == "spatiotemporal")
    temporal_steps = getattr(cfg, "temporal_in_steps", 3) if is_st else 1

    comp_mae_sum = comp_mse_sum = comp_cnt = 0.
    ext_mae_sum = ext_mse_sum = ext_cnt = 0.

    for batch in loader:
        x, m, r, valid = batch
        x, m, r, valid = [v.to(device) for v in (x, m, r, valid)]
        B, C, T, H, W = x.shape
        t = min(int(cfg.target_idx), T - 1)

        vobs = m[:, 0:1, t]
        vrel = r[:, 0:1, t]
        vval = valid[:, 0:1, t]
        ve = vrel * vval

        art, rates = prepare_artificial_mask(B, H, W, device, cfg, ve)
        art = art * ve
        vgiven = vobs * (1 - art)
        given_all = m[:, :, t] * (1 - art)

        xin = x.clone()
        xin[:, :, t] = x[:, :, t] * given_all
        min_ = m.clone()
        min_[:, :, t] = given_all
        z = torch.zeros_like(xin)

        cue, em = make_training_cue(xin, min_, r, valid, t)
        if not cfg.use_cue:
            cue = torch.zeros(B, 1, H, W, device=device, dtype=x.dtype)
        xb = build_contextual_fusion(xin, em, cue, z, 0.)
        gx, gm = build_generator_inputs(xb, em)
        pred = G(gx, gm)[:, 0, t:t + 1]

        if R is not None and getattr(cfg, "use_refiner", False):
            try:
                pred = apply_refiner(R, pred, x, vgiven, vrel, vval, cfg, device)
            except Exception:
                pass

        comp_mask = art * vrel * vval
        comp_err = pred - x[:, 0, t:t + 1]
        comp_mae_sum += (comp_err.abs() * comp_mask).sum().item()
        comp_mse_sum += (comp_err ** 2 * comp_mask).sum().item()
        comp_cnt += comp_mask.sum().item()

        v_completed = x[:, 0, t:t + 1] * vobs + pred * (1 - vobs)
        v_completed_crop = crop_to(v_completed[:, 0], ORI_H, ORI_W)

        v_near = v_completed_crop[:, :near_end]
        near_obs_crop = crop_to(vobs[:, 0], ORI_H, ORI_W)[:, :near_end]
        near_rel_crop = crop_to(vrel[:, 0], ORI_H, ORI_W)[:, :near_end]
        near_cond_crop = crop_to(x[:, 1:, t], ORI_H, ORI_W)[:, :, :near_end]

        if is_st:
            x_cur_input = assemble_extension_input_from_completed(
                v_near, near_obs_crop, near_rel_crop, near_cond_crop
            )
            half_t = temporal_steps // 2
            t_start = max(0, t - half_t)
            t_end = min(T, t + temporal_steps - half_t)
            seq_list = []
            for st in range(t_start, t_end):
                st_v = crop_to(v_completed_crop if st == t else x[:, 0, st:st+1] if st < T else v_completed_crop, ORI_H, ORI_W)[:, :near_end]
                st_obs = crop_to(m[:, 0, st] if st < T else near_obs_crop, ORI_H, ORI_W)[:near_end]
                st_rel = crop_to(r[:, 0, st] if st < T else near_rel_crop, ORI_H, ORI_W)[:near_end]
                st_cond = crop_to(x[:, 1:, st] if st < T else near_cond_crop, ORI_H, ORI_W)[:, :, :near_end]
                seq_list.append(assemble_extension_input_from_completed(st_v, st_obs, st_rel, st_cond))
            while len(seq_list) < temporal_steps:
                seq_list.append(seq_list[0])
            x_seq_input = torch.stack(seq_list[:temporal_steps], dim=1)
            far_pred = Ext(x_cur_input, x_seq_input)
        else:
            ext_input = assemble_extension_input_from_completed(
                v_near, near_obs_crop, near_rel_crop, near_cond_crop
            )
            far_pred = Ext(ext_input)

        far_v_true = crop_to(x[:, 0, t], ORI_H, ORI_W)[:, near_end:far_end]
        far_obs_mask = crop_to(vobs[:, 0], ORI_H, ORI_W)[:, near_end:far_end]
        far_rel_mask = crop_to(vrel[:, 0], ORI_H, ORI_W)[:, near_end:far_end]
        far_valid = far_obs_mask * far_rel_mask

        ext_err = far_pred[:, 0] - far_v_true
        ext_mae_sum += (ext_err.abs() * far_valid).sum().item()
        ext_mse_sum += (ext_err ** 2 * far_valid).sum().item()
        ext_cnt += far_valid.sum().item()

    comp_mae = comp_mae_sum / max(comp_cnt, 1.)
    comp_rmse = (comp_mse_sum / max(comp_cnt, 1.)) ** 0.5
    ext_mae = ext_mae_sum / max(ext_cnt, 1.)
    ext_rmse = (ext_mse_sum / max(ext_cnt, 1.)) ** 0.5

    return comp_mae, comp_rmse, ext_mae, ext_rmse


def save_history(history, out_dir):
    if not history:
        return
    out_dir = Path(out_dir)
    keys = list(history[0].keys())
    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)

    ep = [h["epoch"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(ep, [h["G"] for h in history], label="G")
    plt.plot(ep, [h["D"] for h in history], label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Joint G / D Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_G_D.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ep, [h["comp_MAE"] for h in history], label="Comp MAE")
    plt.plot(ep, [h["comp_RMSE"] for h in history], label="Comp RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Completion Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "completion_metrics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ep, [h["ext_MAE"] for h in history], label="Ext MAE")
    plt.plot(ep, [h["ext_RMSE"] for h in history], label="Ext RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Extension Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "extension_metrics.png", dpi=180)
    plt.close()


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    near_end = cfg.near_end_gate
    far_end = cfg.far_end_gate
    far_gates = far_end - near_end
    exp_name = f"joint_ext_{cfg.ext_model_type}_near{near_end}_far{far_end}_le{cfg.lambda_ext}"
    out_dir = Path(f"outputs_single_v/outputs_{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GAIN + 径向量程扩展 联合训练")
    print(f"  近距离: gate 0~{near_end - 1} ({660}m~{660 + 30 * (near_end - 1)}m)")
    print(f"  远距离: gate {near_end}~{far_end - 1} ({660 + 30 * near_end}m~{660 + 30 * (far_end - 1)}m)")
    print(f"  λ_ext={cfg.lambda_ext}, lr_g_joint={cfg.lr_g_joint}, lr_ext_joint={cfg.lr_ext_joint}")
    print(f"  设备: {device}, batch={cfg.batch_size}")

    norm = fit_normalizer(cfg.train_npy, cfg.normalize, cfg.clip_percentile, cfg.phys_channels)

    gain_ckpt_path = getattr(cfg, "gain_pretrain_path", None)
    if gain_ckpt_path and Path(gain_ckpt_path).exists():
        ckpt_gain = torch.load(gain_ckpt_path, map_location=device, weights_only=False)
        if "normalizer" in ckpt_gain:
            norm_dict = ckpt_gain["normalizer"]
            norm.lo = np.asarray(norm_dict["lo"])
            norm.hi = np.asarray(norm_dict["hi"])
            norm.mean = np.asarray(norm_dict["mean"])
            norm.std = np.asarray(norm_dict["std"])
            print(f"  加载预训练 normalizer: lo={norm.lo[0]:.2f}~{norm.lo[-1]:.2f}, hi={norm.hi[0]:.2f}~{norm.hi[-1]:.2f}")

    train_ds = MultichannelClipDataset(cfg.train_npy, normalizer=norm, time_steps=cfg.time_steps, phys_channels=cfg.phys_channels)
    val_ds = MultichannelClipDataset(cfg.val_npy, normalizer=norm, time_steps=cfg.time_steps, phys_channels=cfg.phys_channels)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    G = build_generator(cfg).to(device)
    D = PatchGANDiscriminator(2, 16).to(device)
    R = build_refiner(cfg)
    if R is not None:
        R = R.to(device)

    Ext = build_extension_model(cfg).to(device)

    if gain_ckpt_path and Path(gain_ckpt_path).exists():
        G.load_state_dict(ckpt_gain["model"])
        if R is not None and "refiner" in ckpt_gain and ckpt_gain["refiner"] is not None:
            R.load_state_dict(ckpt_gain["refiner"])
        print(f"  加载 GAIN 预训练: {gain_ckpt_path} (epoch={ckpt_gain.get('epoch','?')}, V_MAE={ckpt_gain.get('V_MAE','?')})")
    else:
        init_weights(G)
        init_weights(D)
        if R is not None:
            init_weights(R)
        print("  未找到 GAIN 预训练，随机初始化")

    ext_ckpt_path = Path(cfg.ext_pretrain_path) if cfg.ext_pretrain_path else None
    if ext_ckpt_path and ext_ckpt_path.exists():
        ckpt = torch.load(ext_ckpt_path, map_location=device, weights_only=False)
        Ext.load_state_dict(ckpt["model"])
        print(f"  加载扩展网络预训练: {ext_ckpt_path}")
    else:
        init_extension_weights(Ext)
        print("  扩展网络随机初始化")

    init_weights(D)

    try:
        print_model_stats(G, D, R)
    except Exception:
        pass
    n_ext = sum(p.numel() for p in Ext.parameters())
    print(f"  扩展网络参数量: {n_ext:,}")

    gp = list(G.parameters()) + (list(R.parameters()) if R is not None else [])
    optim_G = torch.optim.Adam(gp, lr=cfg.lr_g_joint, betas=cfg.betas)
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)
    optim_Ext = torch.optim.Adam(
        Ext.parameters(), lr=cfg.lr_ext_joint,
        betas=cfg.betas, weight_decay=cfg.ext_weight_decay,
    )

    sch_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_G, mode="min", patience=5, factor=0.5, min_lr=1e-7)
    sch_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_D, mode="min", patience=5, factor=0.5, min_lr=1e-7)
    sch_Ext = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_Ext, mode="min", patience=5, factor=0.5, min_lr=1e-7)

    best = float("inf")
    pat = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(G, D, R, Ext, train_loader, optim_G, optim_D, optim_Ext, cfg, device)
        comp_mae, comp_rmse, ext_mae, ext_rmse = validate(G, R, Ext, val_loader, cfg, device)

        sch_G.step(comp_mae)
        sch_D.step(comp_mae)
        sch_Ext.step(ext_mae)

        row = {
            "epoch": epoch,
            **tr,
            "comp_MAE": comp_mae, "comp_RMSE": comp_rmse,
            "ext_MAE": ext_mae, "ext_RMSE": ext_rmse,
        }
        history.append(row)

        print(f"\n[Epoch {epoch:03d}] "
              f"D={tr['D']:.4f} G={tr['G']:.4f} "
              f"D(real)={tr['D_real']:.3f} D(fake)={tr['D_fake']:.3f}")
        print(f"  Comp MAE={comp_mae:.6f} RMSE={comp_rmse:.6f} | "
              f"Ext MAE={ext_mae:.6f} RMSE={ext_rmse:.6f}")

        ckpt = {
            "model": G.state_dict(),
            "refiner": R.state_dict() if R is not None else None,
            "ext_model": Ext.state_dict(),
            "normalizer": norm.__dict__,
            "cfg": cfg.to_dict(),
            "epoch": epoch,
            "comp_MAE": comp_mae, "comp_RMSE": comp_rmse,
            "ext_MAE": ext_mae, "ext_RMSE": ext_rmse,
        }
        torch.save(ckpt, out_dir / "joint_ext_last.pth")

        # 联合衡量补全与扩展质量
        monitor = comp_mae + cfg.lambda_ext * ext_mae
        if monitor < best:
            best = monitor
            pat = 0
            torch.save(ckpt, out_dir / "joint_ext_best.pth")
            print(f"  保存最佳模型，联合 MAE 指标={best:.6f}")
        else:
            pat += 1

        if pat >= cfg.patience:
            print(f"早停：epoch={epoch}, best 联合 MAE 指标={best:.6f}")
            break

    save_history(history, out_dir)
    print(f"\n联合训练完成。最佳模型：{out_dir / 'joint_ext_best.pth'}")


if __name__ == "__main__":
    main()