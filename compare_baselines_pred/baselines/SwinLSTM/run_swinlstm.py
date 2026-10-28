# -*- coding: utf-8 -*-
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_PRED_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(BASELINES_PRED_DIR)

from common.sota_trainer import train_openstl_baseline

if __name__ == "__main__":
    train_openstl_baseline(model_name="SwinLSTM", missing_rate=0.3)