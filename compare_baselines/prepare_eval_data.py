# -*- coding: utf-8 -*-
"""生成设定 A 人工洞 mask（4 形态 × 15%/30%）并缓存到 eval_masks/。

用法（项目根执行）：
    python -m compare_baselines.prepare_eval_data
    python compare_baselines/prepare_eval_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_baselines.common import cfg, data as cdata, holes  # noqa: E402


def main():
    print("[prepare_eval_data] 加载 test 集 ...")
    arr, obs, rel = cdata.load_split("test")
    print(f"  test: T={arr.shape[0]} H={arr.shape[1]} W={arr.shape[2]}")
    masks = holes.load_or_generate_masks(obs, rel, force=True)
    for (shape, ratio), m in sorted(masks.items()):
        n = int(m.sum())
        print(f"  {shape:8s} {ratio:.0%}: {n:7d} 像素/全集 "
              f"({n / m.size:.1%} of near)")
    print(f"[完成] 缓存: {cfg.EVAL_MASKS_DIR / 'test_holes.npz'}")


if __name__ == "__main__":
    main()
