# -*- coding: utf-8 -*-
"""Ours 真实模型评估（只推理，不训练）。

加载主程序训练好的联合 checkpoint（joint_best.pth），用 compare 的固定扇区
mask、同样的物理单位，产出与基线同一口径的两行：
    表4-X：Ours 遮挡区补全 MAE/RMSE/SSIM
    表4-Y：Ours 未来帧（1步）预测 MAE/RMSE/SSIM

掩码（window_sector）、数据窗口、物理单位换算与基线完全一致，可直接对比。

用法：
    python run_ours_eval.py                 # 全量 test，30% 扇区
    python run_ours_eval.py --limit 10      # 冒烟
    python run_ours_eval.py --rate 0.5
    python run_ours_eval.py --no-append     # 只写独立 CSV，不追加到基线表
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

# 主程序模块（config/model/untils）在父目录，加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config
from train import (
    generate_contextual_cue, build_contextual_fusion, make_training_cue,
)
from common import cfg as ccfg
from common import metrics
from common.masks import window_sector, window_seed, window_structural
from untils.data import (
    SlidingWindowDataset, build_generator_inputs, crop_to,
)
from untils.refiner import apply_refiner
from model import UNetGenerator
from predictor import WindFieldPredictor
from test_joint import build_refiner_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rate", type=float, default=ccfg.MAIN_RATE)
    p.add_argument("--split", default="test")
    p.add_argument("--no-append", action="store_true",
                   help="不追加到基线表，只写独立 CSV")
    p.add_argument("--chain", default="D",
                   help="链路模式：C=补全固定 / D=联合训练（默认 D）")
    p.add_argument("--label", default=None,
                   help="追加到表里的方法名（默认按链路自动命名）")
    return p.parse_args()


def to_physical(v_norm, norm):
    """归一化 V（[-1,1]）→ 物理单位（m/s），返回 numpy。"""
    if isinstance(v_norm, torch.Tensor):
        v_norm = v_norm.detach().cpu().numpy()
    lo, hi = float(norm.lo[0]), float(norm.hi[0])
    return (v_norm + 1.0) * 0.5 * (hi - lo) + lo


def main():
    args = parse_args()
    ccfg.set_seed()
    cfg = Config(chain_mode=args.chain)
    device = torch.device(cfg.device)
    label = args.label or ("Ours (补全固定)" if args.chain.upper() == "C"
                           else "Ours (联合训练)")

    model_path = f"{cfg.output_dir}/joint_best.pth"
    print(f"加载真实模型（链路 {cfg.chain_mode}）：{model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    norm = ckpt["normalizer"]

    # G：8 通道（4 物理 + 4 掩码），输出 1 通道 V
    G = UNetGenerator(cfg.phys_channels * 2, 1, cfg.base_filter,
                      use_cue=True, use_nafblock=True,
                      use_axial_w=True).to(device)
    # P：按多步（5 步）构建以匹配权重，评估只取第 1 步
    P = WindFieldPredictor(cfg.phys_channels + 2, cfg.hidden_channels,
                           cfg.num_layers, 3,
                           max(cfg.multi_step_predictions)).to(device)
    G.load_state_dict(ckpt["model"])
    P.load_state_dict(ckpt["predictor"])
    G.eval(); P.eval()
    R, rphys, rlev = build_refiner_from_checkpoint(ckpt, cfg, device)

    ds = SlidingWindowDataset(cfg.test_npy, normalizer=norm,
                              input_steps=cfg.input_steps, pred_steps=1,
                              phys_channels=cfg.phys_channels)
    if args.limit is not None:
        ds.length = min(ds.length, args.limit)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size,
                                        shuffle=False, num_workers=0)

    # 累积（物理单位）
    comp = {"ae": 0.0, "se": 0.0, "ssim": []}
    pred = {"ae": 0.0, "se": 0.0, "ssim": []}
    ncomp = npred = 0
    win_idx = 0

    with torch.no_grad():
        for hx, hm, hr, fx, fm, fr in loader:
            hx, hm, hr = hx.to(device), hm.to(device), hr.to(device)
            fx, fm, fr = fx.to(device), fm.to(device), fr.to(device)
            B = hx.shape[0]; Tin = hx.shape[2]
            target_t = min(int(ccfg.TARGET_FRAME), Tin - 1)

            # 结构楔掩码（与基线同一掩码、同一种子；只遮 target 帧）
            art_l = []
            for b in range(B):
                vw = (hm * hr)[b, 0].cpu().numpy()
                art_l.append(window_structural(
                    vw, args.rate,
                    window_seed(args.split, args.rate, win_idx + b),
                    target_t))
            win_idx += B
            art = torch.tensor(np.stack(art_l), device=device)   # [B,Tin,H,W]
            art5 = art[:, None].repeat(1, cfg.phys_channels, 1, 1, 1)

            valid_h = hm * hr
            # ---- 只对 target 帧打人工结构缺失（train.py validate 口径）----
            given_all = hm * (1 - art5)
            xin = hx.clone()
            xin[:, :, target_t:target_t + 1] = (
                hx[:, :, target_t:target_t + 1]
                * given_all[:, :, target_t:target_t + 1])
            min_ = hm.clone()
            min_[:, :, target_t:target_t + 1] = (
                given_all[:, :, target_t:target_t + 1])
            z = torch.zeros_like(xin)
            cue, em = make_training_cue(xin, min_, hr, hm, target_t)
            if not cfg.use_cue:
                cue = torch.zeros(B, 1, *hx.shape[-2:], device=device,
                                  dtype=hx.dtype)
            xb = build_contextual_fusion(xin, em, cue, z, 0.)

            # ---- G 整帧前向（与预训练 time_steps=10 一致）----
            vcomp = G(xb, em)  # [B, 1, Tin, H, W]

            # ---- Refiner 只 refine target 帧（与官方 test_joint 一致）----
            if R is not None:
                v_t = vcomp[:, :, target_t]                # [B,1,H,W] 4D
                x_t = hx[:, :, target_t:target_t + 1]      # [B,4,1,H,W] 5D
                g_t = hm[:, :, target_t]                   # [B,4,H,W]
                r_t = hr[:, :, target_t]                   # [B,4,H,W]
                v_t2 = valid_h[:, :, target_t]             # [B,4,H,W]
                v_refined = apply_refiner(
                    R, v_t, x_t, g_t, r_t, v_t2, cfg, device)
                vcomp[:, :, target_t:target_t + 1] = v_refined.unsqueeze(2)

            # ===== 表4-X：补全（物理单位）=====
            cp = to_physical(crop_to(vcomp)[:, 0], norm)
            ct = to_physical(crop_to(hx[:, 0:1])[:, 0], norm)
            cmask = crop_to(art).cpu().numpy()
            cp = cp.reshape(-1, *cp.shape[-2:])
            ct = ct.reshape(-1, *ct.shape[-2:])
            cmask = cmask.reshape(-1, *cmask.shape[-2:])
            m = cmask > 0.5
            comp["ae"] += np.abs(cp - ct)[m].sum()
            comp["se"] += ((cp - ct) ** 2)[m].sum()
            comp["ssim"].append(metrics.all_metrics(cp, ct, cmask)["SSIM"])
            ncomp += m.sum()

            # ===== 表4-Y：未来 1 步预测（端到端）=====
            feats = hx[:, 1:cfg.phys_channels]
            pin = torch.cat([vcomp, feats, hm[:, 0:1], hr[:, 0:1]], dim=1)
            out = P(pin.permute(0, 2, 1, 3, 4))[:, 0:1]

            pp = to_physical(crop_to(out)[:, 0, 0], norm)
            tt = to_physical(crop_to(fx[:, 0:1])[:, 0, 0], norm)
            vv = crop_to((fm * fr)[:, 0:1])[:, 0, 0].cpu().numpy()
            mv = vv > 0.5
            pred["ae"] += np.abs(pp - tt)[mv].sum()
            pred["se"] += ((pp - tt) ** 2)[mv].sum()
            pred["ssim"].append(metrics.all_metrics(pp, tt, vv)["SSIM"])
            npred += mv.sum()

    def finish(acc, n):
        s = [v for v in acc["ssim"] if not np.isnan(v)]
        return {
            "MAE": acc["ae"] / max(n, 1),
            "RMSE": np.sqrt(acc["se"] / max(n, 1)),
            "SSIM": float(np.mean(s)) if s else float("nan"),
        }

    crow = finish(comp, ncomp)
    prow = finish(pred, npred)

    print("\n" + "=" * 60)
    print(f"Ours（{int(args.rate*100)}% 固定扇区，{win_idx} 窗口）")
    print("=" * 60)
    print(f"表4-X 补全：MAE={crow['MAE']:.4f} RMSE={crow['RMSE']:.4f} "
          f"SSIM={crow['SSIM']:.4f}")
    print(f"表4-Y 预测：MAE={prow['MAE']:.4f} RMSE={prow['RMSE']:.4f} "
          f"SSIM={prow['SSIM']:.4f}")

    # 独立 CSV（始终写）
    standalone = ccfg.OUT_DIR / \
        f"ours_real_{args.split}_{int(args.rate*100)}.csv"
    with open(standalone, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["表", "MAE", "RMSE", "SSIM"])
        w.writerow(["表4-X 补全", round(crow["MAE"], 4),
                    round(crow["RMSE"], 4), round(crow["SSIM"], 4)])
        w.writerow(["表4-Y 预测", round(prow["MAE"], 4),
                    round(prow["RMSE"], 4), round(prow["SSIM"], 4)])
    print(f"\n独立结果：{standalone}")

    # 追加到基线表（若存在）
    if not args.no_append:
        _append_rows(args, crow, prow, label)


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []
    return rows, list(fieldnames)


def _append_rows(args, crow, prow, label):
    rate = int(args.rate * 100)
    t4x = ccfg.OUT_DIR / f"table4X_completion_{args.split}_{rate}.csv"
    t4y = ccfg.OUT_DIR / f"table4Y_downstream_{args.split}_{rate}.csv"

    if t4x.exists():
        rows, fn = _read_csv(t4x)
        rows = [r for r in rows if r.get("补全方法") != label]
        rows.append({
            "补全方法": label, "遮挡率": args.rate,
            "MAE": round(crow["MAE"], 4),
            "RMSE": round(crow["RMSE"], 4),
            "SSIM": round(crow["SSIM"], 4),
        })
        with open(t4x, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader(); w.writerows(rows)
        print(f"已追加到：{t4x}")
    else:
        print(f"未找到基线表 {t4x}，跳过追加（独立 CSV 已写）")

    if t4y.exists():
        rows, fn = _read_csv(t4y)
        rows = [r for r in rows if r.get("补全方法") != label]
        row = {
            "补全方法": label, "遮挡率": args.rate,
            "MAE": round(prow["MAE"], 4),
            "RMSE": round(prow["RMSE"], 4),
            "SSIM": round(prow["SSIM"], 4),
        }
        if "下游预测器" in fn:
            row["下游预测器"] = "ConvLSTM"
        rows.append(row)
        with open(t4y, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader(); w.writerows(rows)
        print(f"已追加到：{t4y}")
    else:
        print(f"未找到基线表 {t4y}，跳过追加（独立 CSV 已写）")


if __name__ == "__main__":
    main()