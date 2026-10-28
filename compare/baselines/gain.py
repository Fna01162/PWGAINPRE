# -*- coding: utf-8 -*-
"""GAIN 基线（ICML 2018）：UNetGenerator + PatchGAN，一阶段 GAN 补全。

与其他深度基线口径一致：4 个物理通道输入（V 为补全目标，CNR/SW/Skew 为条件），
只输出/补全 V；固定扇区遮挡、归一化、观测位置保留全部统一。
"""
import numpy as np
import torch

from common import cfg
from untils.data import pad_to, crop_to
from model import UNetGenerator, PatchGANDiscriminator, init_weights
from untils.losses import (
    generate_hint,
    discriminator_loss,
    generator_adv_loss,
    reconstruction_loss,
)


def build_g():
    return UNetGenerator(
        in_channels=8,
        out_channels=1,
        base_filter=cfg.OURS_BASE_FILTER,
        use_cue=False,
        use_nafblock=False,
        use_volformer=False,
        use_axial_w=False,
    )


def _win_tensors(win, device):
    mobs = win["valid"] * (1.0 - win["art"])
    xb = np.transpose(win["x"], (3, 0, 1, 2))
    m4 = np.repeat(mobs[None], cfg.PHYS_CHANNELS, axis=0)
    xb = pad_to(xb)
    m4 = pad_to(m4)
    truth = pad_to(win["clean"][..., 0])[None, None]
    art = pad_to(win["art"])[None, None]
    valid = pad_to(win["valid"])[None, None]
    given = pad_to(mobs)[None, None]
    return (
        torch.tensor(xb[None], device=device),
        torch.tensor(m4[None], device=device),
        torch.tensor(truth, device=device),
        torch.tensor(art, device=device),
        torch.tensor(valid, device=device),
        torch.tensor(given, device=device),
    )


def _gan_forward(g, xb, mm, tv, gv):
    fake = g(xb, mm)
    return tv * gv + fake * (1 - gv), fake


def _train_gan(g, d, windows, epochs, device):
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg.LR, betas=cfg.BETAS)
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg.LR, betas=cfg.BETAS)
    n = len(windows)

    for ep in range(epochs):
        order = np.random.permutation(n)
        gtot = dtot = 0.0
        cnt = 0
        for s in range(0, n, cfg.BATCH_SIZE):
            xb_l, mm_l, tv_l, art_l, vd_l, gv_l = [], [], [], [], [], []
            for i in order[s:s + cfg.BATCH_SIZE]:
                t = _win_tensors(windows[i], device)
                xb_l.append(t[0]); mm_l.append(t[1]); tv_l.append(t[2])
                art_l.append(t[3]); vd_l.append(t[4]); gv_l.append(t[5])
            xb = torch.cat(xb_l); mm = torch.cat(mm_l)
            tv = torch.cat(tv_l); art = torch.cat(art_l)
            vd = torch.cat(vd_l); gv = torch.cat(gv_l)

            vhat, fake = _gan_forward(g, xb, mm, tv, gv)
            hint = generate_hint(gv[:, :, 0])
            dlogit = d(vhat[:, :, 0].detach(), hint)
            dloss = discriminator_loss(dlogit, gv[:, :, 0], vd[:, :, 0])
            opt_d.zero_grad()
            dloss.backward()
            opt_d.step()

            vhat, fake = _gan_forward(g, xb, mm, tv, gv)
            dlogit = d(vhat[:, :, 0], hint)
            gadv = generator_adv_loss(dlogit, art[:, :, 0], vd[:, :, 0])
            grec = reconstruction_loss(fake, tv, art, vd)
            gloss = gadv + cfg.RECON_WEIGHT * grec

            opt_g.zero_grad()
            gloss.backward()
            opt_g.step()

            gtot += gloss.item(); dtot += dloss.item(); cnt += 1
        print(f"  [gain] epoch {ep+1}/{epochs} "
              f"G={gtot/max(1,cnt):.4f} D={dtot/max(1,cnt):.4f}")


def train_gain(windows, epochs, device=cfg.DEVICE):
    g = build_g().to(device)
    d = PatchGANDiscriminator(2, 16).to(device)
    init_weights(g)
    init_weights(d)
    _train_gan(g, d, windows, epochs, device)
    g.eval()
    return {"G": g, "D": d}


@torch.no_grad()
def complete(g, win, device=cfg.DEVICE):
    t = _win_tensors(win, device)
    xb, mm, tv, gv = t[0], t[1], t[2], t[5]
    vhat, _ = _gan_forward(g, xb, mm, tv, gv)
    return crop_to(vhat[0, 0]).cpu().numpy().astype(np.float32)
