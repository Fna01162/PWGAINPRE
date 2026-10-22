# split_32km.py
# -*- coding: utf-8 -*-
"""将 _3.4km 源数据裁到 3.2km (H=85)，按时间顺序划分为 train/val/test，输出到 dataset/split_32/。

裁剪：range gate 从近到远 (660m 起，gate=30m)，H=90→85 保留前 85 行，
最远 660 + 30*84 = 3180m ≈ 3.2km。

划分规则（与 dataset/split_34/ 一致）：
- train: 20230401 + 20230405 沿时间轴拼接
- val:   20230410 的前 30 帧
- test:  20230410 的剩余帧
"""

from pathlib import Path
import numpy as np


CROP_H = 85


def inspect(arr: np.ndarray, name: str) -> None:
    mask = arr[..., 7] == 1
    rel = arr[..., 8] == 1
    v = arr[..., 0][mask]
    v = v[np.isfinite(v)]
    print(f"[{name}] shape={arr.shape} 观测率={mask.mean():.3f} "
          f"可靠率={(mask & rel).mean():.3f} V[min={v.min():.1f},max={v.max():.1f}]")


def main() -> None:
    ROOT = Path(__file__).resolve().parent.parent
    src = ROOT / "dataset"
    out = ROOT / "dataset" / "split_32"
    out.mkdir(parents=True, exist_ok=True)

    train = np.concatenate([
        np.load(src / "20230401_data_channals_9_3.4km.npy")[:, :CROP_H, :, :],
        np.load(src / "20230405_data_channals_9_3.4km.npy")[:, :CROP_H, :, :],
    ], axis=0)

    val_raw = np.load(src / "20230410_data_channals_9_3.4km.npy")[:, :CROP_H, :, :]
    val_frames = 30
    val, test = val_raw[:val_frames], val_raw[val_frames:]

    inspect(train, "train")
    inspect(val, "val")
    inspect(test, "test")

    for name, arr in [("train", train), ("val", val), ("test", test)]:
        np.save(out / f"{name}.npy", arr)
        print(f"已保存 {out / name}.npy  {arr.shape}")


if __name__ == "__main__":
    main()