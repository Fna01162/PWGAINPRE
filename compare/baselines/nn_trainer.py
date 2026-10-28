# -*- coding: utf-8 -*-
"""深度基线的训练与推理：补丁式（BRITS/SAITS）与网格式（HESS/BiConvLSTM）。"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import cfg
from untils.data import pad_to, crop_to
from . import nn_models


# ---------------------------- 通用 ----------------------------

def make_delta(m):
    """时间间隔：m [B,T,n]，等间隔帧，缺失时间隔累加。"""
    b, t, n = m.shape
    d = torch.zeros_like(m)
    d[:, 0] = 1.0 - m[:, 0]
    for i in range(1, t):
        d[:, i] = 1.0 + (1.0 - m[:, i - 1]) * d[:, i - 1]
    return d


def patch_grid(h, w, p, stride):
    rs = list(range(0, max(1, h - p + 1), stride))
    cs = list(range(0, max(1, w - p + 1), stride))
    if rs[-1] != h - p:
        rs.append(h - p)
    if cs[-1] != w - p:
        cs.append(w - p)
    return [(r, c) for r in rs for c in cs]


def _extract_patch(win, r, c, p, multi_channel=False):
    if multi_channel:
        valid = win["valid"][:, r:r + p, c:c + p]
        art = win["art"][:, r:r + p, c:c + p]
        m = valid * (1.0 - art)
        t = win["x"].shape[0]
        c_channels = win["x"].shape[-1]

        def chan_first(a):
            return np.transpose(a, (0, 3, 1, 2)).reshape(t, -1)

        x = chan_first(win["x"][:, r:r + p, c:c + p, :])
        clean = chan_first(win["clean"][:, r:r + p, c:c + p, :])
        mfull = np.repeat(m[:, None], c_channels, axis=1).reshape(t, -1)
        artfull = np.repeat(art[:, None], c_channels, axis=1).reshape(t, -1)
        return x, mfull, clean, artfull
    else:
        x = win["x"][:, r:r + p, c:c + p, 0]
        clean = win["clean"][:, r:r + p, c:c + p, 0]
        valid = win["valid"][:, r:r + p, c:c + p]
        art = win["art"][:, r:r + p, c:c + p]
        m = valid * (1.0 - art)
        t = x.shape[0]
        return (
            x.reshape(t, -1),
            m.reshape(t, -1),
            clean.reshape(t, -1),
            art.reshape(t, -1),
        )


# ---------------------------- 补丁式 ----------------------------

def _patch_train_seqs(windows, p, multi_channel=False):
    seqs = []
    rng = np.random.default_rng(cfg.SEED)
    positions = patch_grid(cfg.ORI_H, cfg.ORI_W, p, p)
    for win in windows:
        chosen = [positions[i] for i in rng.choice(
            len(positions), size=min(cfg.PATCH_TRAIN_PER_WIN, len(positions)),
            replace=False)]
        for (r, c) in chosen:
            x, m, clean, art = _extract_patch(win, r, c, p, multi_channel=multi_channel)
            if art.sum() > 0:
                seqs.append((x, m, clean, art))
    return seqs


def train_patch(name, windows, epochs, device=cfg.DEVICE):
    p = cfg.PATCH
    n_v = p * p
    n_in = cfg.PHYS_CHANNELS * n_v
    if name == "brits":
        model = nn_models.BRITS(n_v, n_in, hidden=64)
    else:
        model = nn_models.SAITS(n_v, n_in)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, betas=getattr(cfg, 'BETAS', (0.5, 0.999)))

    seqs = _patch_train_seqs(windows, p, multi_channel=True)
    for ep in range(epochs):
        np.random.shuffle(seqs)
        tot = 0.0
        for s in range(0, len(seqs), 32):
            batch = seqs[s:s + 32]
            x = torch.tensor(np.stack([b[0] for b in batch]), device=device)
            m = torch.tensor(np.stack([b[1] for b in batch]), device=device)
            tv = torch.tensor(np.stack([b[2] for b in batch]), device=device)
            art = torch.tensor(np.stack([b[3] for b in batch]), device=device)
            pred = model(x, m, make_delta(m))
            tv_v, art_v = tv[..., :n_v], art[..., :n_v]
            loss = ((pred - tv_v) ** 2 * art_v).sum() / art_v.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  [{name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(seqs)//32):.5f}")
    model.eval()
    return model


def patch_infer(model, win, device=cfg.DEVICE):
    p, stride = cfg.PATCH, cfg.PATCH_STRIDE
    t, h, w = win["x"].shape[:3]
    acc = np.zeros((t, h, w), np.float64)
    wacc = np.zeros((t, h, w), np.float64)
    hann = np.outer(np.hanning(p), np.hanning(p)) + 1e-3

    with torch.no_grad():
        for (r, c) in patch_grid(h, w, p, stride):
            x, m, _clean, _art = _extract_patch(win, r, c, p, multi_channel=True)
            xt = torch.tensor(x[None], device=device)
            mt = torch.tensor(m[None], device=device)
            pred = model(xt, mt, make_delta(mt))[0].cpu().numpy()
            # 输出只有 V，直接还原成补丁
            v_pred = pred.reshape(t, p, p)
            acc[:, r:r + p, c:c + p] += v_pred * hann
            wacc[:, r:r + p, c:c + p] += hann

    stitched = (acc / np.maximum(wacc, 1e-8)).astype(np.float32)
    mobs = (win["valid"] * (1.0 - win["art"]))
    observed = win["x"][..., 0]
    return (mobs * observed + (1.0 - mobs) * stitched).astype(np.float32)


# ---------------------------- 网格式 ----------------------------

def _pad_seq(arr):
    return pad_to(arr)


def train_grid(name, windows, epochs, device=cfg.DEVICE):
    partial = (name == "hess")
    model = nn_models.MemoryUNet(base=16, partial=partial).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, betas=getattr(cfg, 'BETAS', (0.5, 0.999)))

    for ep in range(epochs):
        order = np.random.permutation(len(windows))
        tot = 0.0
        cnt = 0
        for s in range(0, len(windows), cfg.BATCH_SIZE):
            idxs = order[s:s + cfg.BATCH_SIZE]
            xs, ms, tvs, arts = [], [], [], []
            for i in idxs:
                win = windows[i]
                mobs = win["valid"] * (1.0 - win["art"])
                # 4 个物理通道（V 为补全目标，CNR/SW/Skew 为条件）
                xt = np.transpose(win["x"], (0, 3, 1, 2))  # [T,C,H,W]
                xs.append(_pad_seq(xt))
                ms.append(_pad_seq(mobs))
                tvs.append(_pad_seq(win["clean"][..., 0]))
                arts.append(_pad_seq(win["art"]))
            x = torch.tensor(np.stack(xs), device=device)
            m = torch.tensor(np.stack(ms), device=device)[:, :, None]
            tv = torch.tensor(np.stack(tvs), device=device)[:, :, None]
            art = torch.tensor(np.stack(arts), device=device)[:, :, None]
            pred = model(x, m)
            loss = ((pred - tv) ** 2 * art).sum() / art.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        print(f"  [{name}] epoch {ep+1}/{epochs} loss={tot/max(1,cnt):.5f}")
    model.eval()
    return model


def grid_infer(model, win, device=cfg.DEVICE):
    mobs = win["valid"] * (1.0 - win["art"])
    with torch.no_grad():
        xt = _pad_seq(np.transpose(win["x"], (0, 3, 1, 2)))
        x = torch.tensor(xt[None], device=device)
        m = torch.tensor(_pad_seq(mobs)[None], device=device)[:, :, None]
        pred = model(x, m)[0, :, 0].cpu().numpy()
    pred = crop_to(pred)
    return (mobs * win["x"][..., 0] + (1.0 - mobs) * pred).astype(np.float32)


# ---------------------------- 逐帧网格（CGAN / DSA-UNet） ----------------------------

def _frame_samples(windows):
    """把窗口拆成逐帧样本（已 pad）：x [4,H,W] / m [1,H,W] / tv [1,H,W] / art [1,H,W]。"""
    X, M, TV, ART = [], [], [], []
    for win in windows:
        mobs = win["valid"] * (1.0 - win["art"])
        xt = _pad_seq(np.transpose(win["x"], (0, 3, 1, 2)))
        mt = _pad_seq(mobs)
        tt = _pad_seq(win["clean"][..., 0])
        at = _pad_seq(win["art"])
        for i in range(xt.shape[0]):
            X.append(xt[i])
            M.append(mt[i, None])
            TV.append(tt[i, None])
            ART.append(at[i, None])
    return X, M, TV, ART


def _batch_tensors(batch, device):
    x = torch.tensor(np.stack([b[0] for b in batch]), device=device)
    m = torch.tensor(np.stack([b[1] for b in batch]), device=device)
    tv = torch.tensor(np.stack([b[2] for b in batch]), device=device)
    art = torch.tensor(np.stack([b[3] for b in batch]), device=device)
    return x, m, tv, art


def train_frame_grid(name, windows, epochs, device=cfg.DEVICE):
    X, M, TV, ART = _frame_samples(windows)
    n = len(X)
    bs = max(1, cfg.BATCH_SIZE)

    if name == "dsaunet":
        model = nn_models.DSAUNet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                               betas=getattr(cfg, "BETAS", (0.5, 0.999)))
        for ep in range(epochs):
            order = np.random.permutation(n)
            tot = 0.0
            cnt = 0
            for s in range(0, n, bs):
                x, m, tv, art = _batch_tensors(
                    [batch for batch in [(X[i], M[i], TV[i], ART[i])
                                        for i in order[s:s + bs]]], device)
                pred = model(x, m)
                loss = ((pred - tv) ** 2 * art).sum() / art.sum().clamp_min(1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                cnt += 1
            print(f"  [{name}] epoch {ep+1}/{epochs} loss={tot/max(1,cnt):.5f}")
        model.eval()
        return model

    # cgan：U-Net++ 生成器 + 条件 PatchGAN
    g = nn_models.NestedUNet().to(device)
    d = nn_models.CondPatchDisc().to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg.LR, betas=cfg.BETAS)
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg.LR, betas=cfg.BETAS)

    for ep in range(epochs):
        order = np.random.permutation(n)
        gtot = dtot = 0.0
        cnt = 0
        for s in range(0, n, bs):
            x, m, tv, art = _batch_tensors(
                [(X[i], M[i], TV[i], ART[i]) for i in order[s:s + bs]], device)
            xv = x[:, 0:1]
            fake = g(x, m)
            comp = m * xv + (1 - m) * fake

            # 判别器（LSGAN）
            d_real = d(x, tv)
            d_fake = d(x, comp.detach())
            dloss = 0.5 * (
                ((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
            opt_d.zero_grad()
            dloss.backward()
            opt_d.step()

            # 生成器
            d_fake = d(x, comp)
            gadv = ((d_fake - 1) ** 2).mean()
            l1 = ((comp - tv).abs() * art).sum() / art.sum().clamp_min(1.0)
            gloss = gadv + 100.0 * l1
            opt_g.zero_grad()
            gloss.backward()
            opt_g.step()

            gtot += gloss.item()
            dtot += dloss.item()
            cnt += 1
        print(f"  [{name}] epoch {ep+1}/{epochs} "
              f"G={gtot/max(1,cnt):.4f} D={dtot/max(1,cnt):.4f}")
    g.eval()
    return g


def frame_grid_infer(model, win, device=cfg.DEVICE):
    mobs = win["valid"] * (1.0 - win["art"])
    with torch.no_grad():
        xt = _pad_seq(np.transpose(win["x"], (0, 3, 1, 2)))
        mt = _pad_seq(mobs)
        x = torch.tensor(xt, device=device)
        m = torch.tensor(mt[:, None], device=device)
        pred = model(x, m)[:, 0].cpu().numpy()
    pred = crop_to(pred)
    return (mobs * win["x"][..., 0] + (1.0 - mobs) * pred).astype(np.float32)