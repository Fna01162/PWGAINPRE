# -*- coding: utf-8 -*-
"""基线 3：无补全直接外推（表2 第一行）。

近门 V 缺失处保持 0（不补全），直接用 [v, CNR, SW, Skew, obs, rel]
训练 CNNRangeExtension 外推远门。表1 无此行（近门补全 MAE = "-"）。

用法（项目根）：
    python compare_baselines/no_completion/run_no_completion.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # compare_baselines

from compare_baselines.common import cfg, data as cdata, ext_trainer, pipeline  # noqa: E402


def identity_complete(v_in, obs_near, rel_near, cond):
    """不做任何补全（pipeline 的 oracle 保留逻辑会自动保持 obs 处原值）。"""
    return np.zeros_like(v_in)


def main():
    t0 = time.time()
    meta = pipeline.run_standard_pipeline(
        "no_completion", identity_complete,
        run_meta_extra={"note": "no near-gate completion; table1 N/A"})
    # 表1 占位：无补全没有近门补全指标
    out_dir = cfg.method_out_dir("no_completion")
    metrics_path = out_dir / "table1_no_completion.csv"
    if not metrics_path.exists():
        from compare_baselines.common import metrics
        metrics.write_csv(metrics_path, [
            {"shape": "-", "ratio": "-", "MAE": "-", "RMSE": "-",
             "Bias": "-", "Corr": "-", "n": "-"},
        ])
    print(f"总用时 {time.time() - t0:.1f}s  {meta}")


if __name__ == "__main__":
    main()
