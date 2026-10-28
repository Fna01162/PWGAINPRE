# -*- coding: utf-8 -*-
import os
import sys

# 将 compare_baselines_pred 目录添加到路径，以便导入 common 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_PRED_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(BASELINES_PRED_DIR)

from common.sota_trainer import train_openstl_baseline

if __name__ == "__main__":
    # 使用 30% (0.3) 的极端缺失率工况进行抗压测试
    train_openstl_baseline(model_name="SimVP", missing_rate=0.3)