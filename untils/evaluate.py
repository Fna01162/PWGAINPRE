
# -*- coding: utf-8 -*-
"""V 风速补全评估与可视化。"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from untils.data import make_obs_rel, ORI_H, ORI_W, BAD_RAYS
from test import impute_sequence


TARGET_CHANNEL = 0


def make_holes(data, mode, rng, ratio=0.15, target_idx=0, time_steps=1):
    """在原本可靠的 V 观测区域生成人工缺失。"""
    obs, rel = make_obs_rel(data, 4)
    H, W = ORI_H, ORI_W

    left = target_idx
    right = time_steps - 1 - target_idx

    if left < data.shape[0] and right < data.shape[0]:
        valid_frames = np.arange(left, data.shape[0] - right)
    else:
        valid_frames = np.arange(data.shape[0])

    obs_sufficient = np.where(obs.sum(axis=(1, 2)) > H * W * 0.5)[0]
    candidates = np.intersect1d(obs_sufficient, valid_frames)

    if len(candidates) == 0:
        candidates = obs_sufficient
    if len(candidates) == 0:
        candidates = np.arange(data.shape[0])

    frame_idx = int(rng.choice(candidates))
    hole = np.zeros((H, W), dtype=bool)
    total = H * W
    n = max(1, int(total * ratio))

    if mode == "random":
        hole = rng.random((H, W)) < ratio
    elif mode == "sector":
        width = min(max(1, n // H), W - 1)
        ws = rng.integers(0, W - width + 1)
        hole[:, ws:ws + width] = True
    elif mode == "height":
        height = min(max(1, n // W), H - 1)
        hs = rng.integers(0, H - height + 1)
        hole[hs:hs + height, :] = True
    elif mode == "rect":
        height = min(max(5, H // 3), H - 1)
        width = min(max(1, n // height), W - 1)
        hs = rng.integers(0, H - height + 1)
        ws = rng.integers(0, W - width + 1)
        hole[hs:hs + height, ws:ws + width] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    hole &= rel[frame_idx].astype(bool)

    masked = data.copy()
    masked[frame_idx, hole, TARGET_CHANNEL] = 0.0
    masked[frame_idx, hole, 7] = 0.0

    return masked, hole, frame_idx


def metrics(pred_v, true_v, hole):
    """计算 V 缺失区域的 MAE、RMSE、Bias 和 Corr。"""
    pred = np.asarray(pred_v)[hole]
    true = np.asarray(true_v)[hole]

    valid = np.isfinite(pred) & np.isfinite(true)
    pred = pred[valid]
    true = true[valid]

    if len(pred) == 0:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}

    diff = pred - true
    result = {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
        "corr": np.nan
    }

    if len(pred) > 1 and np.std(pred) > 0 and np.std(true) > 0:
        result["corr"] = float(np.corrcoef(pred, true)[0, 1])

    return result


def evaluate_holes(model, data, normalizer, cfg, device,
                   modes=("random", "sector", "height", "rect"),
                   n_trials=5, seed=0):
    """评价不同人工缺失模式下的 V 补全效果。"""
    rng = np.random.default_rng(seed)
    results = {mode: [] for mode in modes}

    for mode in modes:
        for _ in range(n_trials):
            masked, hole, frame_idx = make_holes(
                data,
                mode,
                rng,
                ratio=0.15,
                target_idx=cfg.get("target_idx", 0),
                time_steps=cfg.get("time_steps", 1)
            )

            out = impute_sequence(
                model,
                masked,
                normalizer,
                cfg,
                device,
                deterministic=True
            )

            result = metrics(
                out[..., TARGET_CHANNEL],
                data[..., TARGET_CHANNEL],
                hole
            )
            results[mode].append(result)

        mae = np.array([x["mae"] for x in results[mode]])
        rmse = np.array([x["rmse"] for x in results[mode]])
        bias = np.array([x["bias"] for x in results[mode]])
        corr = np.array([x["corr"] for x in results[mode]])

        print(f"\n=== 缺失模式: {mode} ===")
        print(f"V MAE  = {np.nanmean(mae):.4f} ± {np.nanstd(mae):.4f}")
        print(f"V RMSE = {np.nanmean(rmse):.4f} ± {np.nanstd(rmse):.4f}")
        print(f"V Bias = {np.nanmean(bias):.4f} ± {np.nanstd(bias):.4f}")
        print(f"V Corr = {np.nanmean(corr):.4f} ± {np.nanstd(corr):.4f}")

    return results


def visualize_single_frame(data, masked, out, hole, frame_idx,
                           save_path=None, vmin=None, vmax=None,
                           cmap="RdBu_r"):
    """V 单帧原始、缺失、补全和误差可视化。"""
    orig = data[frame_idx, :, :, TARGET_CHANNEL]
    masked_v = masked[frame_idx, :, :, TARGET_CHANNEL]
    pred = out[frame_idx, :, :, TARGET_CHANNEL]

    error = np.full_like(orig, np.nan, dtype=np.float32)
    error[hole] = pred[hole] - orig[hole]

    if vmin is None:
        vmin = float(np.nanpercentile(orig, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(orig, 99))

    if np.any(np.isfinite(error)):
        emax = float(np.nanmax(np.abs(error))) + 1e-6
    else:
        emax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    items = [
        (orig, "Original V", vmin, vmax),
        (masked_v, "Masked V", vmin, vmax),
        (pred, "Completed V", vmin, vmax),
        (error, "Error V", -emax, emax)
    ]

    for ax, (arr, title, lo, hi) in zip(axes, items):
        im = ax.imshow(
            arr,
            aspect="auto",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            origin="lower"
        )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.05)

    fig.suptitle(f"Frame {frame_idx} | Wind Speed V")
    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def visualize_multiframe(data, out, frames, save_path, cmap="RdBu_r"):
    """V 多帧原始结果与补全结果对比。"""
    frames = list(frames)
    if not frames:
        return

    v = data[frames, :, :, TARGET_CHANNEL]
    vmin = float(np.nanpercentile(v, 1))
    vmax = float(np.nanpercentile(v, 99))

    fig, axes = plt.subplots(
        2,
        len(frames),
        figsize=(5 * len(frames), 10),
        squeeze=False
    )

    for j, frame_idx in enumerate(frames):
        axes[0, j].imshow(
            data[frame_idx, :, :, TARGET_CHANNEL],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower"
        )
        axes[0, j].set_title(f"Original t={frame_idx}")
        axes[0, j].axis("off")

        axes[1, j].imshow(
            out[frame_idx, :, :, TARGET_CHANNEL],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower"
        )
        axes[1, j].set_title(f"Completed t={frame_idx}")
        axes[1, j].axis("off")

    fig.suptitle("Wind Speed V Temporal Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_azimuth_spectrum_v(data, out, save_path, num_frames=10):
    """V 方位角频谱对比。"""
    W = ORI_W
    freqs = np.fft.rfftfreq(W)
    n = min(num_frames, data.shape[0])

    original_spectrum = []
    completed_spectrum = []

    for k in range(n):
        original = np.nan_to_num(
            data[k, :, :, TARGET_CHANNEL],
            nan=0.0
        )
        completed = np.nan_to_num(
            out[k, :, :, TARGET_CHANNEL],
            nan=0.0
        )

        original_spectrum.append(
            np.abs(np.fft.rfft(original, axis=1)).mean(axis=0)
        )
        completed_spectrum.append(
            np.abs(np.fft.rfft(completed, axis=1)).mean(axis=0)
        )

    if not original_spectrum:
        return

    original_spectrum = np.mean(original_spectrum, axis=0)
    completed_spectrum = np.mean(completed_spectrum, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        freqs[1:],
        original_spectrum[1:],
        label="Original V"
    )
    ax.plot(
        freqs[1:],
        completed_spectrum[1:],
        label="Completed V"
    )
    ax.set_xlabel("Azimuth frequency")
    ax.set_ylabel("Amplitude")
    ax.set_title("V Azimuth Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def check_9ch_3panel(original_npy_path, completed_npy_path,
                      bad_rays=None, mid_idx=None):
    """显示原始 V、Observation Mask 和补全 V。"""
    if bad_rays is None:
        bad_rays = BAD_RAYS

    orig_data = np.load(original_npy_path)
    comp_data = np.load(completed_npy_path)

    if orig_data.ndim != 4 or orig_data.shape[-1] < 9:
        raise ValueError(
            f"原始文件形状应为 (T,H,W,9)，实际为 {orig_data.shape}"
        )

    if comp_data.shape != orig_data.shape:
        raise ValueError(
            f"补全结果与原始数据形状不一致："
            f"{orig_data.shape} vs {comp_data.shape}"
        )

    if mid_idx is None:
        mid_idx = orig_data.shape[0] // 2

    if not 0 <= mid_idx < orig_data.shape[0]:
        raise IndexError(f"mid_idx={mid_idx} 超出范围")

    raw_v = orig_data[mid_idx, :, :, TARGET_CHANNEL]
    mask_val = orig_data[mid_idx, :, :, 7]
    completed_v = comp_data[mid_idx, :, :, TARGET_CHANNEL]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        constrained_layout=True
    )

    im0 = axes[0].imshow(
        raw_v,
        aspect="auto",
        cmap="turbo",
        origin="lower"
    )
    axes[0].set_title(f"Original V | Frame {mid_idx}")
    fig.colorbar(im0, ax=axes[0], label="V")

    im1 = axes[1].imshow(
        mask_val,
        aspect="auto",
        cmap="gray_r",
        origin="lower",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Observation Mask\n1=Observed, 0=Missing")

    for ray in bad_rays:
        if 0 <= ray < ORI_W:
            axes[1].axvline(
                ray,
                linestyle="--",
                linewidth=1.5
            )

    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        completed_v,
        aspect="auto",
        cmap="turbo",
        origin="lower"
    )
    axes[2].set_title("Completed V")
    fig.colorbar(im2, ax=axes[2], label="V")

    output_path = Path("check_v_3panel.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"三联图已保存：{output_path.resolve()}")

    plt.close(fig)


def main():
    test_path = "../dataset/split_32/test.npy"
    complete_path = "../outputs_single_v/test_completed.npy"

    check_9ch_3panel(
        test_path,
        complete_path,
        bad_rays=BAD_RAYS
    )


if __name__ == "__main__":
    main()