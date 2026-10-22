# -*- coding: utf-8 -*-
"""对比实验共享包。

必须最先执行：Windows 下 torch 与 numpy/matplotlib 的 OMP 冲突防护。
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
