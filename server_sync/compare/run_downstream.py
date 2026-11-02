# -*- coding: utf-8 -*-
"""表 4-Y：下游预测对比。

各补全方法（分离训练）的结果统一送入 ConvLSTM 预测器，比较未来帧预测指标。
用法：
    python run_downstream.py
    python run_downstream.py --methods zero,linear --limit 20 --epochs 20
"""
import argparse
import csv

import numpy as np
import torch

from common import cfg
from common.data import build_windows, future_norm, inverse_v
from common import metrics
from baselines import api, nn_models
from untils.data import pad_to, crop_to

DEFAULT_METHODS = [
    "zero", "linear", "gain", "saits", "hess", "biconvlstm",
    "cgan", "dsaunet",
]
DISPLAY = {
    "cgan": "CGAN (U-Net++, TGRS'23)",
    "dsaunet": "DSA-UNet (RS'23)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--rate", type=float, default=cfg.MAIN_RATE)
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--split", default="test")
    return p.parse_args()


def _predictor_input(win, comp_v):
    comp4 = win["x"].copy()
    comp4[..., 0] = comp_v
    arr = np.transpose(comp4, (3, 0, 1, 2))
    return pad_to(arr)


def _future(win):
    yt = pad_to(future_norm(win))[None]
    yv = pad_to(win["y_valid"])[None]
    return yt, yv


def train_pred(train_windows, comp_train, epochs, device=cfg.DEVICE):
    p = nn_models.build_predictor().to(device)
    opt = torch.optim.Adam(p.parameters(), lr=cfg.LR)
    n = len(train_windows)

    for ep in range(epochs):
        order = np.random.permutation(n)
        tot = 0.0
        cnt = 0
        for s in range(0, n, cfg.BATCH_SIZE):
            x_l, yt_l, yv_l = [], [], []
            for i in order[s:s + cfg.BATCH_SIZE]:
                x_l.append(_predictor_input(train_windows[i], comp_train[i]))
                yt, yv = _future(train_windows[i])
                yt_l.append(yt)
                yv_l.append(yv)
            x = torch.tensor(np.stack(x_l), device=device).permute(0, 2, 1, 3, 4)
            yt = torch.tensor(np.stack(yt_l), device=device)[:, :, None]
            yv = torch.tensor(np.stack(yv_l), device=device)[:, :, None]

            pred = p(x)
            loss = (((pred - yt) ** 2) * yv).sum() / yv.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        print(f"  [convlstm] epoch {ep+1}/{epochs} loss={tot/max(1,cnt):.5f}")
    p.eval()
    return p


@torch.no_grad()
def predict_pred(p, win, comp_v, device=cfg.DEVICE):
    x = torch.tensor(_predictor_input(win, comp_v)[None], device=device)
    x = x.permute(0, 2, 1, 3, 4)
    pred = p(x)[0, :, 0]
    return crop_to(pred).cpu().numpy()


def main():
    args = parse_args()
    cfg.set_seed()
    methods = [m.strip() for m in args.methods.split(",")]

    test_windows = build_windows(args.split, args.rate, args.limit)
    yt_raw = np.stack([w["y"][..., 0] for w in test_windows])
    yv = np.stack([w["y_valid"] for w in test_windows])

    rows = []
    for method in methods:
        print(f"== 下游方法: {method} ==")
        comp_train = api.completed_v(
            method, "train", rate=args.rate, epochs=args.epochs,
            limit=args.train_limit, train_limit=args.train_limit)
        comp_test = api.completed_v(
            method, args.split, rate=args.rate, epochs=args.epochs,
            limit=args.limit, train_limit=args.train_limit)
        train_windows = build_windows(
            "train", args.rate,
            args.train_limit if args.train_limit is not None else args.limit)
        p = train_pred(train_windows, comp_train, args.epochs)
        pred = np.stack([
            predict_pred(p, w, comp_test[i])
            for i, w in enumerate(test_windows)])

        pred_flat = pred.reshape(-1, *pred.shape[-2:])
        yt_flat = yt_raw.reshape(-1, *yt_raw.shape[-2:])
        yv_flat = yv.reshape(-1, *yv.shape[-2:])
        pred_raw = inverse_v(pred_flat)
        m = metrics.all_metrics(pred_raw, yt_flat, yv_flat)
        rows.append({
            "补全方法": DISPLAY.get(method, method),
            "下游预测器": "ConvLSTM",
            "遮挡率": args.rate,
            "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "SSIM": round(m["SSIM"], 4),
        })
        print(f"   MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} "
              f"SSIM={m['SSIM']:.4f}")

    out = cfg.OUT_DIR / f"table4Y_downstream_{args.split}_{int(args.rate*100)}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已保存：{out}")


if __name__ == "__main__":
    main()
