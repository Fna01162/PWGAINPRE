# -*- coding: utf-8 -*-
"""
preprocess_SL74_generalization.py

将 PIANO 公开数据集 (SL74 Doppler wind lidar PPI 扫描) 预处理为
已训练补全模型的兼容格式，用于零样本泛化性验证。

输入:  dataset/level1_SL74/level1/SL74_l1/YYYYMMDD/SL74_ppi3_l1_*.nc
输出:  dataset/SL74_generalization/test.npy   shape=(T, 85, 125, 9)

通道映射 (与训练数据 9 通道格式对齐):
  Ch0  V        径向速度 (m/s) — 模型监督目标
  Ch1  CNR      载噪比 (dB) — 从 intensity 换算: 10*log10(intensity - 1)
  Ch2  SW       谱宽 (m/s) — 公开数据集无此变量, 置 0
  Ch3  Skew     偏度 — 公开数据集无此变量, 置 0
  Ch4  Range    归一化距离 — 与 Refiner build_geometry_maps 一致
  Ch5  sin(az)  方位角正弦 — 与 Refiner build_geometry_maps 一致
  Ch6  cos(az)  方位角余弦 — 与 Refiner build_geometry_maps 一致
  Ch7  obs_mask 观测掩码 (1=有观测)
  Ch8  rel_mask 可靠掩码 (1=可靠)

关键处理步骤:
  1. 从 nc 文件中筛选 ppi3 扫描模式
  2. 分离单个 PPI 扫描 (CSM 模式下每帧含 16~33 次重复扫描)
  3. 将 ~90° 方位角扇区数据映射到 360° 模型网格 (未覆盖区域标记为缺失)
  4. 将 167×18m 距离门重采样到 85×30m 模型网格 (超出范围标记为缺失)
  5. 从 intensity 计算 CNR, 根据信号阈值确定有效观测

用法:
  python preprocess_SL74_generalization.py
  python preprocess_SL74_generalization.py --max-scans-per-file 3 --intensity-threshold 1.01
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ORI_H = 85
ORI_W = 125
RANGE_START_M = 660.0
RANGE_STEP_M = 30.0
N_CHANNELS = 9
PHYS_CHANNELS = 4

TARGET_RANGES = RANGE_START_M + RANGE_STEP_M * np.arange(ORI_H)
TARGET_AZ_DEG = np.linspace(0.0, 360.0, ORI_W, endpoint=False)
TARGET_AZ_RAD = TARGET_AZ_DEG * np.pi / 180.0

RANGE_NORM = TARGET_RANGES / TARGET_RANGES[-1]
SIN_AZ = np.sin(TARGET_AZ_RAD).astype(np.float32)
COS_AZ = np.cos(TARGET_AZ_RAD).astype(np.float32)

DEFAULT_INTENSITY_THRESHOLD = 1.0


def separate_ppi_scans(azimuth):
    n = len(azimuth)
    boundaries = [0]
    for i in range(1, n):
        if azimuth[i] < azimuth[i - 1] - 1.0:
            boundaries.append(i)
    boundaries.append(n)

    scans = []
    for k in range(len(boundaries) - 1):
        start, end = boundaries[k], boundaries[k + 1]
        if end - start >= 10:
            scans.append((start, end))
    return scans


def regrid_one_scan(rv, intensity, az, gate_centers,
                    intensity_threshold=DEFAULT_INTENSITY_THRESHOLD):
    frame = np.zeros((ORI_H, ORI_W, N_CHANNELS), dtype=np.float32)

    frame[:, :, 4] = RANGE_NORM[:, None]
    frame[:, :, 5] = SIN_AZ[None, :]
    frame[:, :, 6] = COS_AZ[None, :]

    az_min, az_max = float(az.min()), float(az.max())
    az_span = az_max - az_min

    if az_span < 10.0:
        return None

    az_tol = max(0.5, (TARGET_AZ_DEG[1] - TARGET_AZ_DEG[0]) * 0.5)
    valid_az_mask = (TARGET_AZ_DEG >= az_min - az_tol) & (TARGET_AZ_DEG <= az_max + az_tol)

    if not valid_az_mask.any():
        return None

    range_min, range_max = float(gate_centers[0]), float(gate_centers[-1])
    valid_range_mask = (TARGET_RANGES >= range_min) & (TARGET_RANGES <= range_max)

    if not valid_range_mask.any():
        return None

    sort_idx = np.argsort(az)
    az_sorted = az[sort_idx]
    rv_sorted = rv[:, sort_idx]
    intensity_sorted = intensity[:, sort_idx]

    target_r = TARGET_RANGES[valid_range_mask]
    target_a = TARGET_AZ_DEG[valid_az_mask]

    R, A = np.meshgrid(target_r, target_a, indexing='ij')
    points = np.column_stack([R.ravel(), A.ravel()])

    try:
        rv_interp = RegularGridInterpolator(
            (gate_centers, az_sorted), rv_sorted,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        rv_regrid = rv_interp(points).reshape(len(target_r), len(target_a))

        intensity_interp = RegularGridInterpolator(
            (gate_centers, az_sorted), intensity_sorted,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        intensity_regrid = intensity_interp(points).reshape(len(target_r), len(target_a))
    except Exception:
        return None

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_linear = np.maximum(intensity_regrid - 1.0, 1e-10)
        cnr_db = 10.0 * np.log10(snr_linear)

    valid = (np.isfinite(rv_regrid)
             & np.isfinite(intensity_regrid)
             & (intensity_regrid > intensity_threshold))

    if valid.sum() < 10:
        return None

    range_indices = np.where(valid_range_mask)[0]
    az_indices = np.where(valid_az_mask)[0]

    Ri, Ai = np.meshgrid(range_indices, az_indices, indexing='ij')
    ri_flat = Ri.ravel()
    ai_flat = Ai.ravel()
    valid_flat = valid.ravel()

    idx_valid = valid_flat.nonzero()[0]
    ri_v = ri_flat[idx_valid]
    ai_v = ai_flat[idx_valid]

    frame[ri_v, ai_v, 0] = rv_regrid.ravel()[idx_valid]
    frame[ri_v, ai_v, 1] = cnr_db.ravel()[idx_valid]
    frame[ri_v, ai_v, 7] = 1.0
    frame[ri_v, ai_v, 8] = 1.0

    return frame


def process_one_nc(nc_path, intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
                   max_scans_per_file=None):
    try:
        ds = nc.Dataset(str(nc_path))
    except Exception as e:
        print(f"  无法打开 {nc_path}: {e}")
        return []

    try:
        rv = ds.variables['radial_velocity'][:]
        intensity = ds.variables['intensity'][:]
        az = ds.variables['azimuth'][:]
        gate_centers = ds.variables['gate_centers'][:]
    except Exception as e:
        print(f"  无法读取变量 {nc_path}: {e}")
        ds.close()
        return []

    ds.close()

    scans = separate_ppi_scans(az)

    if max_scans_per_file is not None:
        scans = scans[:max_scans_per_file]

    frames = []
    for start, end in scans:
        scan_az = az[start:end]
        scan_rv = rv[:, start:end]
        scan_intensity = intensity[:, start:end]

        frame = regrid_one_scan(
            scan_rv, scan_intensity, scan_az, gate_centers,
            intensity_threshold=intensity_threshold
        )

        if frame is not None:
            frames.append(frame)

    return frames


def main():
    parser = argparse.ArgumentParser(
        description='PIANO SL74 PPI 数据预处理 (泛化性验证)')
    parser.add_argument('--src', type=str,
                        default=r'dataset/level1_SL74/level1/SL74_l1',
                        help='源数据目录')
    parser.add_argument('--out', type=str,
                        default=r'dataset/SL74_generalization/test.npy',
                        help='输出文件路径')
    parser.add_argument('--intensity-threshold', type=float,
                        default=DEFAULT_INTENSITY_THRESHOLD,
                        help='信号强度阈值 (intensity > threshold 视为有信号)')
    parser.add_argument('--max-scans-per-file', type=int, default=1,
                        help='每个 nc 文件最多提取的 PPI 扫描数 (0=全部)')
    parser.add_argument('--min-obs-rate', type=float, default=0.02,
                        help='最低观测率阈值 (低于此值的帧被丢弃)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='最多保留的帧数 (0=全部)')
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_scans = None if args.max_scans_per_file == 0 else args.max_scans_per_file

    nc_files = sorted([
        f for d in sorted(src_dir.iterdir()) if d.is_dir()
        for f in d.glob('SL74_ppi3_*.nc')
    ])

    print(f"找到 {len(nc_files)} 个 PPI 文件")
    print(f"信号强度阈值: {args.intensity_threshold}")
    print(f"每个文件最多提取: {max_scans or '全部'} 扫描")
    print(f"最低观测率: {args.min_obs_rate}")

    all_frames = []
    skipped_obs = 0
    skipped_err = 0

    for i, nc_path in enumerate(nc_files):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  处理 {i + 1}/{len(nc_files)}: {nc_path.parent.name}/{nc_path.name}")

        try:
            frames = process_one_nc(
                nc_path,
                intensity_threshold=args.intensity_threshold,
                max_scans_per_file=max_scans
            )
        except Exception as e:
            print(f"  错误 {nc_path}: {e}")
            skipped_err += 1
            continue

        for frame in frames:
            obs_rate = frame[:, :, 7].mean()
            if obs_rate >= args.min_obs_rate:
                all_frames.append(frame)
            else:
                skipped_obs += 1

    print(f"\n有效帧: {len(all_frames)}")
    print(f"跳过 (观测率不足): {skipped_obs}")
    print(f"跳过 (处理错误): {skipped_err}")

    if not all_frames:
        print("没有有效帧, 退出")
        return

    if args.max_frames > 0 and len(all_frames) > args.max_frames:
        indices = np.linspace(0, len(all_frames) - 1, args.max_frames, dtype=int)
        all_frames = [all_frames[i] for i in indices]
        print(f"均匀采样 {args.max_frames} 帧")

    data = np.stack(all_frames, axis=0).astype(np.float32)

    print(f"\n输出 shape: {data.shape}")
    print(f"输出 dtype: {data.dtype}")
    print(f"输出大小: {data.nbytes / 1024**2:.1f} MB")

    obs_rate = data[:, :, :, 7].mean()
    print(f"总观测率: {obs_rate:.4f}")

    ch_names = ['V', 'CNR', 'SW', 'Skew', 'Range', 'sin_az', 'cos_az',
                'obs_mask', 'rel_mask']
    for c, name in enumerate(ch_names):
        v = data[:, :, :, c]
        finite = v[np.isfinite(v)]
        if len(finite) > 0:
            print(f"  Ch{c} ({name:8s}): "
                  f"min={finite.min():10.4f}, max={finite.max():10.4f}, "
                  f"mean={finite.mean():10.4f}")

    np.save(out_path, data)
    print(f"\n已保存到: {out_path}")

    print("\n--- 验证: make_obs_rel 兼容性 ---")
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from untils.data import make_obs_rel
    obs, rel = make_obs_rel(data, PHYS_CHANNELS)
    print(f"make_obs_rel: obs_rate={obs.mean():.4f}, rel_rate={rel.mean():.4f}")
    print(f"obs & rel 一致率: {(obs == rel).mean():.4f}")

    print("\n--- 验证: Normalizer 兼容性 ---")
    from untils.data import Normalizer
    norm = Normalizer("minmax", 99.5, PHYS_CHANNELS).fit(data, obs)
    for c in range(PHYS_CHANNELS):
        print(f"  通道{c} ({ch_names[c]}): lo={norm.lo[c]:.4f}, hi={norm.hi[c]:.4f}")

    print("\n预处理完成! 使用方法:")
    print(f"  1. 在 config.py 中设置: self.input_npy = r'{out_path}'")
    print(f"  2. 运行推理: python main.py test")
    print(f"  3. (可选) 归一化重拟合: self.normalizer_refit_npy = r'{out_path}'")


if __name__ == "__main__":
    main()