# -*- coding: utf-8 -*-
"""深度补全基线共享训练器（GAIN-2D / DSA-UNet / UNet++ cGAN 通用）。

结构对应官方 GAIN（third_party/GAIN/gain.py）：
  D_loss = BCE(D(Hat_X, Hint), M)
  G_loss = adv(D(G), 1) + alpha * MSE(M*(X-G)) / mean(M)
主项目公平性协议（config.py）：
  adv_weight=0.2, recon_weight=1.0, alpha=10, hint_rate=0.30, noise U(0,noise)
  lr_g=2e-4, lr_d=5e-5, betas=(0.5,0.999), batch=8, epochs=100, patience=20, clip=5.0
人工洞：untils.data.generate_pure_structural_missing_mask(drop_prob=0.3,
valid_mask=obs*rel)——与主模型训练同策略。
"""
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import cfg, data as cdata


def set_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_infer_input(v_near, cond, obs_near, rel_near, noise_scale=0.0):
    """推理期输入（numpy -> torch）：真实缺失处加小均匀噪声（官方 GAIN 风格）。

    v_near [B,H,W], cond [B,3,H,W], obs/rel [B,H,W] -> x [B,6,H,W]
    """
    m = ((obs_near > 0.5) & (rel_near > 0.5)).astype(np.float32)
    m4 = torch.from_numpy(m[:, None])
    v = torch.from_numpy(v_near[:, None]) * m4
    if noise_scale > 0:
        z = (torch.rand_like(v) * 2.0 - 1.0) * noise_scale
        v = v + (1.0 - m4) * z
    return torch.cat([v, torch.from_numpy(cond), m4,
                      torch.from_numpy(rel_near[:, None])], dim=1)


def train_completion(G, D, train_arrays, val_arrays, out_dir, device,
                     epochs=None, patience=None, batch_size=None,
                     use_amp=True, use_checkpoint=False):
    """训练补全网络。train_arrays=(v,cnr,sw,skew,obs,rel) numpy。

    返回 G（已加载 best 权重）。
    """
    from untils.data import generate_pure_structural_missing_mask
    from untils.losses import (discriminator_loss, generate_hint,
                               generator_adv_loss, reconstruction_loss)

    set_seed(cfg.SEED)
    device = torch.device(device)
    epochs = epochs or cfg.EPOCHS
    patience = patience or cfg.PATIENCE
    batch_size = batch_size or cfg.BATCH_SIZE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = use_amp and (device.type == "cuda")
    use_checkpoint = use_checkpoint and (device.type == "cuda")

    v_t, c_t, o_t, r_t = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                          for a in train_arrays]
    v_v, c_v, o_v, r_v = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                          for a in val_arrays]
    tr_loader = DataLoader(TensorDataset(v_t, c_t, o_t, r_t),
                           batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    G, D = G.to(device), D.to(device)
    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.LR_G, betas=cfg.BETAS)
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.LR_D, betas=cfg.BETAS)
    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_G, mode="min", patience=cfg.EXT_SCHED_PATIENCE,
        factor=cfg.EXT_SCHED_FACTOR, min_lr=cfg.EXT_MIN_LR)

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    n_params = sum(p.numel() for p in G.parameters()) + sum(p.numel() for p in D.parameters())
    amp_status = "AMP(FP16)" if use_amp else "FP32"
    ckpt_status = "+GradCheckpoint" if use_checkpoint else ""
    print(f"  [deep_comp] G={sum(p.numel() for p in G.parameters()):,} "
          f"D={sum(p.numel() for p in D.parameters()):,} 参数 [{amp_status}{ckpt_status}]")

    def val_hole_mae():
        """验证：val 健康区随机挖 15% 洞，洞像素归一化 MAE（分 batch 避免大模型 OOM）。"""
        G.eval()
        total_err_sum = 0.0
        total_art_sum = 0.0
        val_batch_size = min(batch_size, 4)  # 验证时用更小的 batch

        with torch.no_grad():
            vv, cc, oo, rr = v_v.to(device), c_v.to(device), o_v.to(device), r_v.to(device)
            ve = oo * rr
            torch.manual_seed(cfg.SEED + 7)

            # 分 batch 处理以避免 DSA-UNet 等 OOM
            for i in range(0, vv.shape[0], val_batch_size):
                vv_b = vv[i:i+val_batch_size]
                cc_b = cc[i:i+val_batch_size]
                oo_b = oo[i:i+val_batch_size]
                rr_b = rr[i:i+val_batch_size]
                ve_b = ve[i:i+val_batch_size]

                art, _ = generate_pure_structural_missing_mask(
                    vv_b.shape[0], vv_b.shape[1], vv_b.shape[2],
                    device, 0.15, cfg=None, valid_mask=ve_b)
                if art.dim() == 4:
                    art = art[:, 0]
                art = art * ve_b
                m_eff = ve_b * (1.0 - art)
                v_masked = vv_b * m_eff

                with torch.amp.autocast('cuda', enabled=use_amp):
                    x = torch.cat([v_masked[:, None], cc_b, m_eff[:, None], rr_b[:, None]], dim=1)
                    if use_checkpoint:
                        from torch.utils.checkpoint import checkpoint
                        v_fake = checkpoint(G, x, use_reentrant=False)[:, 0]
                    else:
                        v_fake = G(x)[:, 0]

                err = torch.abs(v_fake - vv_b) * art
                total_err_sum += err.sum().item()
                total_art_sum += art.sum().item()

            return total_err_sum / (total_art_sum + 1e-8)

    best = float("inf")
    best_state = None
    pat = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        s = {"d": 0., "g_adv": 0., "g_rec": 0., "n": 0}
        for v, cond, obs, rel in tr_loader:
            v, cond, obs, rel = [x.to(device) for x in (v, cond, obs, rel)]
            ve = obs * rel
            art, _ = generate_pure_structural_missing_mask(
                v.shape[0], v.shape[1], v.shape[2], device, cfg.DROP_PROB,
                cfg=None, valid_mask=ve)
            if art.dim() == 4:
                art = art[:, 0]
            art = art * ve
            m_eff = ve * (1.0 - art)

            v_masked = v * m_eff
            z = torch.rand_like(v_masked) * cfg.NOISE_SCALE
            v_in = v_masked + (1.0 - m_eff) * z
            x = torch.cat([v_in[:, None], cond, m_eff[:, None], rel[:, None]], dim=1)

            with torch.amp.autocast('cuda', enabled=use_amp):
                if use_checkpoint:
                    from torch.utils.checkpoint import checkpoint
                    v_fake = checkpoint(G, x, use_reentrant=False)
                else:
                    v_fake = G(x)
                vhat = (m_eff[:, None] * v[:, None] + (1.0 - m_eff[:, None]) * v_fake)  # [B,1,H,W]
                hint = generate_hint(m_eff, cfg.HINT_RATE)[:, None]  # [B,1,H,W]

                d_loss = discriminator_loss(D(vhat.detach(), hint), m_eff[:, None], ve[:, None])

            optim_D.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            if cfg.GRAD_CLIP > 0:
                scaler.unscale_(optim_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.GRAD_CLIP)
            scaler.step(optim_D)
            scaler.update()

            with torch.amp.autocast('cuda', enabled=use_amp):
                hint2 = generate_hint(m_eff, cfg.HINT_RATE)[:, None]
                d_g = D(vhat, hint2)
                g_adv = generator_adv_loss(d_g, (1.0 - m_eff)[:, None], ve[:, None])
                g_rec = reconstruction_loss(v_fake, v, art, ve)
                gt = cfg.ADV_WEIGHT * g_adv + cfg.ALPHA * cfg.RECON_WEIGHT * g_rec

            optim_G.zero_grad(set_to_none=True)
            scaler.scale(gt).backward()
            if cfg.GRAD_CLIP > 0:
                scaler.unscale_(optim_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.GRAD_CLIP)
            scaler.step(optim_G)
            scaler.update()

            s["d"] += d_loss.item(); s["g_adv"] += g_adv.item()
            s["g_rec"] += g_rec.item(); s["n"] += 1

        vmae = val_hole_mae()
        sched_G.step(vmae)
        n = max(s["n"], 1)
        history.append({"epoch": epoch, "d_loss": s["d"] / n,
                        "g_adv": s["g_adv"] / n, "g_rec": s["g_rec"] / n,
                        "val_hole_MAE": vmae})
        print(f"  [comp {epoch:03d}] d={s['d']/n:.4f} adv={s['g_adv']/n:.4f} "
              f"rec={s['g_rec']/n:.4f} valHoleMAE={vmae:.4f}")
        if vmae < best:
            best = vmae; pat = 0
            best_state = {k: t.detach().cpu().clone() for k, t in G.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                print(f"  [comp] 早停于 epoch {epoch}")
                break

    if best_state is not None:
        G.load_state_dict(best_state)
    torch.save({"model": G.state_dict(), "val_hole_MAE": best},
               out_dir / "comp_best.pth")

    import csv as _csv
    with open(out_dir / "comp_history.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    print(f"  [comp] 训练完成 best valHoleMAE={best:.4f} "
          f"用时{time.time()-t0:.0f}s")
    return G


def make_complete_fn_from_G(G, device=None, noise_scale=0.0):
    """由训练好的 G 构造 pipeline complete_fn（冻结推理）。"""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    G = G.to(device); G.eval()

    def complete_fn(v_in, obs_near, rel_near, cond):
        out = np.empty_like(v_in)
        bs = cfg.BATCH_SIZE
        with torch.no_grad():
            for i in range(0, v_in.shape[0], bs):
                x = make_infer_input(v_in[i:i+bs], cond[i:i+bs],
                                     obs_near[i:i+bs], rel_near[i:i+bs],
                                     noise_scale=noise_scale).to(device)
                out[i:i+bs] = G(x)[:, 0].cpu().numpy()
        return out.astype(np.float32)

    return complete_fn


def arrays_for_split(arr, obs, rel, norm):
    """-> (v[N,Hn,W], cond[N,3,Hn,W], obs_n[N,Hn,W], rel_n[N,Hn,W])"""
    nd = cdata.normalize_phys(arr, norm, obs)
    ns, _ = cdata.near_slices()
    return (nd[:, ns, :, 0].astype(np.float32),
            cdata.build_near_cond(nd, obs),
            obs[:, ns].astype(np.float32),
            rel[:, ns].astype(np.float32))