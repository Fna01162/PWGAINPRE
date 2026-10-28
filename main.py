# main.py
# -*- coding: utf-8 -*-
"""入口：python main.py <mode>（不带参数默认 train）

模式说明：
    train      : 原始 GAIN 补全网络训练
    test       : 原始 GAIN 补全推理
    joint      : GAIN + ConvLSTM 联合训练（python main.py joint <A|B|C|D>）
    joint_test : 联合模型测试评估（python main.py joint_test <A|B|C|D> [--oracle|--no-oracle]）
"""

import sys

AVAILABLE_MODES = ("train", "test", "joint", "joint_test")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    # 链路模式统一由 Config 管理，命令行只做透传
    chain_mode = sys.argv[2] if len(sys.argv) > 2 else None
    extra_args = sys.argv[3:]

    if mode == "train":
        from train import main as train_main
        train_main()
    elif mode == "test":
        from test import test_main
        test_main()
    elif mode == "joint":
        from train_joint import train_joint
        train_joint(chain_mode=chain_mode)
    elif mode == "joint_test":
        from test_joint import test_joint
        test_joint(
            chain_mode=chain_mode,
            force_oracle="--oracle" in extra_args,
            disable_oracle="--no-oracle" in extra_args
        )
    else:
        print(f"未知模式: {mode}（可用 {' / '.join(AVAILABLE_MODES)}）")
        sys.exit(1)


if __name__ == "__main__":
    main()