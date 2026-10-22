# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_joint import SlidingWindowDataset
from untils.data import Normalizer, make_obs_rel
import numpy as np
from .cfg import TRAIN_NPY
def load_normalizer():
    normalizer = Normalizer("minmax", 99.5, 4)
    arr = np.load(TRAIN_NPY)
    obs, _ = make_obs_rel(arr, 4)
    return normalizer.fit(arr, obs)
