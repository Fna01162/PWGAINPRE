# -*- coding: utf-8 -*-
"""统一的补全接口：训练（或加载缓存）各方法，返回补全后的 V 场。

模型权重和补全结果都缓存到 outputs，避免重复训练。
"""
import numpy as np
import torch

from common import cfg
from common.data import build_windows
from baselines import classic, nn_models, nn_trainer, gain

CLASSIC = ["zero", "linear", "knn"]
PATCH = ["brits", "saits"]
GRID_TEMPORAL = ["hess", "biconvlstm"]
GRID_FRAME = ["cgan", "dsaunet"]
GRID = GRID_TEMPORAL + GRID_FRAME
GAN = ["gain"]
ALL = CLASSIC + PATCH + GRID + GAN


def _ckpt_path(method, rate):
    return cfg.OUT_DIR / f"ckpt_{method}_{int(rate*100)}.pth"


def _cache_path(method, split, rate, limit):
    suffix = f"_{int(limit)}" if limit else ""
    return cfg.OUT_DIR / (
        f"completed_{method}_{split}_{int(rate*100)}{suffix}.npy")


def _train_model(method, train_windows, epochs, rate):
    if method in PATCH:
        model = nn_trainer.train_patch(method, train_windows, epochs)
        torch.save(model.state_dict(), _ckpt_path(method, rate))
        return model
    if method in GRID_TEMPORAL:
        model = nn_trainer.train_grid(method, train_windows, epochs)
        torch.save(model.state_dict(), _ckpt_path(method, rate))
        return model
    if method in GRID_FRAME:
        model = nn_trainer.train_frame_grid(method, train_windows, epochs)
        torch.save(model.state_dict(), _ckpt_path(method, rate))
        return model
    if method == "gain":
        bundle = gain.train_gain(train_windows, epochs)
        torch.save({"G": bundle["G"].state_dict()}, _ckpt_path(method, rate))
        return {"G": bundle["G"]}


def _load_model(method, rate, device=cfg.DEVICE):
    ckpt = torch.load(_ckpt_path(method, rate), map_location=device)
    # 多通道输入：4个物理通道 × PATCH×PATCH
    n_v = cfg.PATCH ** 2
    n_in = cfg.PHYS_CHANNELS * n_v
    if method == "brits":
        model = nn_models.BRITS(n_v, n_in, 64)
        model.load_state_dict(ckpt)
        return model.to(device).eval()
    if method == "saits":
        model = nn_models.SAITS(n_v, n_in)
        model.load_state_dict(ckpt)
        return model.to(device).eval()
    if method in GRID_TEMPORAL:
        model = nn_models.MemoryUNet(base=16, partial=(method == "hess"))
        model.load_state_dict(ckpt)
        return model.to(device).eval()
    if method == "cgan":
        model = nn_models.NestedUNet()
        model.load_state_dict(ckpt)
        return model.to(device).eval()
    if method == "dsaunet":
        model = nn_models.DSAUNet()
        model.load_state_dict(ckpt)
        return model.to(device).eval()
    if method == "gain":
        g = gain.build_g()
        g.load_state_dict(ckpt["G"])
        return {"G": g.to(device).eval()}


def _get_model(method, train_windows, epochs, rate):
    if _ckpt_path(method, rate).exists():
        return _load_model(method, rate)
    return _train_model(method, train_windows, epochs, rate)


def _infer(method, model, win):
    if method in PATCH:
        return nn_trainer.patch_infer(model, win)
    if method in GRID_TEMPORAL:
        return nn_trainer.grid_infer(model, win)
    if method in GRID_FRAME:
        return nn_trainer.frame_grid_infer(model, win)
    return gain.complete(model["G"], win)


def _classic_run(method, win):
    observed = win["valid"] * (1.0 - win["art"])
    if method == "zero":
        return classic.zero_fill(win["x"][..., 0])
    if method == "linear":
        return classic.linear_interp(win["x"][..., 0], observed)
    return classic.knn(win["x"][..., 0], observed)


def completed_v(method, split, rate=cfg.MAIN_RATE,
                 epochs=cfg.EPOCHS, limit=None, train_limit=None,
                 force=False):
    """返回补全后的 V [N,Tin,H,W]（归一化）。"""
    cache = _cache_path(method, split, rate, limit)
    if cache.exists() and not force:
        return np.load(cache)

    windows = build_windows(split, rate, limit)
    if method in CLASSIC:
        comp = [_classic_run(method, w) for w in windows]
    else:
        tl = train_limit if train_limit is not None else limit
        train_windows = build_windows("train", rate, tl)
        model = _get_model(method, train_windows, epochs, rate)
        comp = [_infer(method, model, w) for w in windows]

    arr = np.stack(comp).astype(np.float32)
    np.save(cache, arr)
    return arr