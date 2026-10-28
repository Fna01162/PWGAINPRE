# -*- coding: utf-8 -*-
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_PRED_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(BASELINES_PRED_DIR)

from common.sota_trainer import train_openstl_baseline

if __name__ == "__main__":
    # 注意：Earthformer 较吃显存，如果 OOM，请在 config 中调小 batch_size
    train_openstl_baseline(model_name="Earthformer", missing_rate=0.3)