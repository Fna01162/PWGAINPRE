# main.py
# -*- coding: utf-8 -*-
"""入口：python main.py <mode>（不带参数默认 train）

模式说明：
    train      : 原始 GAIN 补全网络训练
    test       : 原始 GAIN 补全推理
    joint      : GAIN + ConvLSTM 联合训练
    pred_test  : 联合模型预测推理
    ext_train  : 径向量程扩展网络独立训练
    joint_ext  : GAIN + 径向量程扩展 联合微调
    ext_test   : GAIN + 径向量程扩展 推理
"""

import sys


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        from train import main as train_main
        train_main()
    elif mode == "test":
        from test import main as test_main
        test_main()
    elif mode == "joint":
        from train_joint import main as joint_main
        joint_main()
    elif mode == "pred_test":
        from test import pred_test_main
        pred_test_main()
    elif mode == "ext_train":
        from train_extension import main as ext_train_main
        ext_train_main()
    elif mode == "joint_ext":
        from train_joint_ext import main as joint_ext_main
        joint_ext_main()
    elif mode == "ext_test":
        from test_ext import main as ext_test_main
        ext_test_main()
    else:
        print(f"未知模式: {mode}（可用 train / test / joint / pred_test / ext_train / joint_ext / ext_test）")
        sys.exit(1)


if __name__ == "__main__":
    main()