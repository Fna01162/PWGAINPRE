# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyart
from scipy import ndimage
from pathlib import Path

from untils.data import _pure_structural_mask, BAD_RAYS
from config import Config

cfg = Config()
DROP_PROB = cfg.drop_prob

DATA_PATH = cfg.input_npy
GAIN_OUT = str(Path(cfg.output_dir) / (Path(cfg.input_npy).stem + "_completed_gain.npy"))
EXT_OUT = str(Path(cfg.output_dir) / (Path(cfg.input_npy).stem + "_completed_ext.npy"))

FIELD_CH = 0
MASK_CH = 7
T_IDX = 1

ORI_H_FULL, ORI_W = 85, 125
NEAR_END = cfg.near_end_gate
FAR_END = cfg.far_end_gate
ORI_H = FAR_END

AZ_DEG = np.linspace(0, 360, ORI_W, endpoint=False)
RANGE_M = 660.0 + 30.0 * np.arange(ORI_H)
ELEV_DEG = 3.0

SAVE_PATH = f"ext_ppi_compare_T{T_IDX}_drop_prob{DROP_PROB:.1f}.png"


def make_radar(field, az, ranges):
    n_rays, n_gates = field.shape
    return pyart.core.Radar(
        scan_type="ppi",
        time={"units": "seconds", "data": np.arange(n_rays, dtype="float32")},
        _range={"units": "meters", "data": ranges.astype("float32")},
        latitude={"data": np.array([0.0])},
        longitude={"data": np.array([0.0])},
        altitude={"data": np.array([0.0])},
        sweep_number={"data": np.array([0], dtype="int32")},
        sweep_mode={"data": np.array(["ppi"], dtype=object)},
        sweep_start_ray_index={"data": np.array([0], dtype="int32")},
        sweep_end_ray_index={"data": np.array([n_rays - 1], dtype="int32")},
        azimuth={"data": az.astype("float32")},
        elevation={"data": np.full(n_rays, ELEV_DEG, dtype="float32")},
        fixed_angle={"data": np.array([ELEV_DEG], dtype="float32")},
        fields={"velocity": {
            "data": np.ma.masked_invalid(field.astype("float32")),
            "long_name": "radial_velocity",
            "units": "m/s",
            "standard_name": "radial_velocity"
        }},
        metadata={}
    )


def plot_one(ax, field, title, vlim, cmap="jet", vmin=None, vmax=None,
             cbar_label="Vr [m/s]", max_range_km=None):
    ranges = RANGE_M - RANGE_M[0]
    radar = make_radar(field, AZ_DEG, ranges)
    display = pyart.graph.RadarDisplay(radar)
    display.plot_ppi(
        "velocity", sweep=0, ax=ax,
        vmin=-vlim if vmin is None else vmin,
        vmax=vlim if vmax is None else vmax,
        cmap=cmap, title=title, colorbar_label=cbar_label,
        rasterized=True
    )
    km = max_range_km if max_range_km is not None else ranges[-1] / 1000.0
    display.set_limits(xlim=[-km, km], ylim=[-km, km])


def draw_artificial_outlines(ax, artificial_mask):
    labeled, n = ndimage.label(artificial_mask)
    ranges_km = (RANGE_M - RANGE_M[0]) / 1000.0
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        g_min, g_max = ys.min(), ys.max()
        a_min, a_max = xs.min(), xs.max()
        r1, r2 = ranges_km[g_min], ranges_km[g_max]
        th1, th2 = np.deg2rad(AZ_DEG[a_min]), np.deg2rad(AZ_DEG[a_max])
        corners = [
            (r1 * np.sin(th1), r1 * np.cos(th1)),
            (r2 * np.sin(th1), r2 * np.cos(th1)),
            (r2 * np.sin(th2), r2 * np.cos(th2)),
            (r1 * np.sin(th2), r1 * np.cos(th2)),
        ]
        ax.add_patch(plt.Polygon(corners, fill=False, edgecolor="red",
                                 linewidth=1.5, linestyle="--"))


def draw_range_ring(ax, range_km, color="green", linestyle="--", linewidth=1.5):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(range_km * np.sin(theta), range_km * np.cos(theta),
            linestyle=linestyle, color=color, linewidth=linewidth)


def mark_bad_rays(ax):
    r = (RANGE_M[-1] - RANGE_M[0]) / 1000.0
    for k in BAD_RAYS:
        if 0 <= k < ORI_W:
            a = np.deg2rad(AZ_DEG[k])
            ax.plot([0, r * np.sin(a)], [0, r * np.cos(a)],
                    "--", linewidth=0.8, alpha=0.7)


def main():
    data = np.load(DATA_PATH)
    print("original:", data.shape)

    frame = data[T_IDX, :ORI_H, :ORI_W, FIELD_CH]
    mask = data[T_IDX, :ORI_H, :ORI_W, MASK_CH]

    obs_slice = data[T_IDX:T_IDX + 1, :ORI_H, :ORI_W, 7]
    rel_slice = data[T_IDX:T_IDX + 1, :ORI_H, :ORI_W, 8]

    allowed = (obs_slice[0, :, :] > 0.5) & (rel_slice[0, :, :] > 0.5)

    np.random.seed(42 + T_IDX)
    art_2d = _pure_structural_mask(allowed, DROP_PROB)

    observed = frame.copy()
    observed[mask < 0.5] = np.nan
    for ray in BAD_RAYS:
        if 0 <= ray < ORI_W:
            observed[:, ray] = np.nan
    observed[art_2d > 0.5] = np.nan

    combined_mask = mask * (1.0 - art_2d)
    for ray in BAD_RAYS:
        if 0 <= ray < ORI_W:
            combined_mask[:, ray] = 0

    gain_v = None
    if Path(GAIN_OUT).exists():
        gain_arr = np.load(GAIN_OUT)
        print("loaded:", GAIN_OUT, gain_arr.shape)
        if gain_arr.ndim == 4:
            gain_v = gain_arr[T_IDX, :ORI_H, :ORI_W, FIELD_CH]
        else:
            gain_v = gain_arr[T_IDX, :ORI_H, :ORI_W]
    else:
        print("not found:", GAIN_OUT)

    ext_v = None
    if Path(EXT_OUT).exists():
        ext_arr = np.load(EXT_OUT)
        print("loaded:", EXT_OUT, ext_arr.shape)
        if ext_arr.ndim == 4:
            ext_v = ext_arr[T_IDX, :ORI_H, :ORI_W, FIELD_CH]
        else:
            ext_v = ext_arr[T_IDX, :ORI_H, :ORI_W]
    else:
        print("not found:", EXT_OUT)

    gain_refiner_v = None
    if gain_v is not None:
        gain_refiner_v = gain_v.copy()
        gain_refiner_v[NEAR_END:, :] = np.nan

    observed_v = observed.copy()
    observed_v[NEAR_END:, :] = np.nan

    panels = [("Observed", observed_v)]
    if gain_refiner_v is not None:
        panels.append(("GAIN + Refiner", gain_refiner_v))
    if ext_v is not None:
        panels.append(("GAIN + Extension", ext_v))
    panels.append(("Mask (white=valid, black=missing)", combined_mask))

    valid = frame[np.isfinite(frame) & (mask > 0.5)]
    vlim = float(np.percentile(np.abs(valid), 99))

    near_km = (RANGE_M[NEAR_END] - RANGE_M[0]) / 1000.0
    far_km = (RANGE_M[min(FAR_END, ORI_H) - 1] - RANGE_M[0]) / 1000.0

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.8))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, field) in zip(axes, panels):
        if "Mask" in title:
            plot_one(ax, field.T, title, vlim=1.0, cmap="gray",
                     vmin=0, vmax=1, cbar_label="")
        elif "Observed" in title:
            plot_one(ax, field.T, title, vlim, max_range_km=near_km)
        elif "Refiner" in title:
            plot_one(ax, field.T, title, vlim, max_range_km=near_km)
        elif "Extension" in title:
            plot_one(ax, field.T, title, vlim, max_range_km=far_km)
        else:
            plot_one(ax, field.T, title, vlim)
        mark_bad_rays(ax)
        if "Observed" in title:
            draw_artificial_outlines(ax, art_2d)
        if "Refiner" in title:
            draw_range_ring(ax, near_km, color="green")
        if "Extension" in title:
            draw_range_ring(ax, near_km, color="green")
            draw_range_ring(ax, far_km, color="red")

    fig.suptitle(
        f"Wind Lidar PPI Completion + Range Extension | t={T_IDX} | V | "
        f"{ORI_H}x{ORI_W} | near 0~{NEAR_END-1} ({near_km:.1f}km) | "
        f"ext {NEAR_END}~{FAR_END-1} ({near_km:.1f}~{far_km:.1f}km) | "
        f"|Vr| P99={vlim:.1f} m/s",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[ok] saved:", SAVE_PATH)


if __name__ == "__main__":
    main()