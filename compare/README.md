# 补全对比实验（compare）

围绕"补全质量决定预测上限"这一主线，在同一套固定扇区遮挡下比较各补全方法，
并把补全结果统一送入 ConvLSTM 预测器，评估对下游预测的影响。

## 目录结构

```
compare/
├── common/
│   ├── cfg.py        # 路径、窗口、遮挡率、训练超参
│   ├── masks.py      # 扇区遮挡掩码生成（连续方位射线，缓存）
│   ├── data.py       # 数据加载、归一化、(输入窗口, 未来帧)
│   └── metrics.py    # MAE / RMSE / SSIM（带掩码）
├── baselines/
│   ├── classic.py    # 零填充、方位向线性插值、空间 KNN
│   ├── nn_models.py  # BRITS、SAITS、MemoryUNet(HESS/BiConvLSTM)、NestedUNet(CGAN)、DSAUNet、下游ConvLSTM
│   ├── nn_trainer.py # 深度基线的训练与推理（补丁式 / 时序网格 / 逐帧网格）
│   ├── gain.py       # GAIN 基线（ICML 2018）：UNetGenerator+PatchGAN
│   └── api.py        # 统一补全接口，权重与结果都缓存
├── run_completion.py # 表 4-X：补全质量
├── run_downstream.py # 表 4-Y：下游预测
└── outputs/          # 掩码、权重、补全结果、CSV
```

## 方法对应关系

| 方法名 | 含义 | 处理方式 |
| --- | --- | --- |
| `zero` | 不补全，零填充 | 逐帧 |
| `linear` | 方位向线性插值（距离向兜底） | 逐帧 |
| `knn` | 空间 KNN（方位向循环） | 逐帧 |
| `brits` | BRITS（NeurIPS 2018） | 局部补丁（16×16） |
| `saits` | SAITS（ESWA 2023） | 局部补丁（16×16） |
| `hess` | HESS 雷达填补（2025） | 整幅网格，偏置卷积+记忆 |
| `biconvlstm` | BiConvLSTM-UNet（AMT 2026） | 整幅网格，普通卷积+BiConvLSTM |
| `cgan` | CGAN 波束遮挡矫正（TGRS 2023） | 逐帧，U-Net++ 生成器+条件 PatchGAN |
| `dsaunet` | DSA-UNet（Remote Sensing 2023） | 逐帧，空洞卷积+自注意力 UNet |
| `gain` | GAIN（ICML 2018） | 整幅网格，GAN |

说明：**本框架只包含对比基线，不含 Ours**。Ours 的补全/联合结果直接用主程序
（PWGAIN_PRE）训练好的真实模型，不在这里复现。

BRITS、SAITS 原本面向少量变量的时序数据，直接展平整幅网格（10625 维）
会导致维度爆炸，因此采用局部补丁变体（论文中表述为"传统/通用时序方法在全局高维
网格上破坏空间拓扑，故采用局部变体"）；补丁内若整窗完全缺失，由空间 KNN 兜底。

**输入/输出对齐**：所有深度方法（BRITS/SAITS/HESS/BiConvLSTM/CGAN/DSA-UNet/GAIN）
都接收相同的 4 个物理通道（V 为补全目标，CNR/SW/Skew 为条件），且**只输出/补全 V**；
掩码、扇区、归一化、观测位置保留口径全部一致。经典方法（zero/linear/knn）是单变量
插值，天然只用 V，保持标准做法。

## 运行

先跑小规模冒烟测试，确认能跑通：

```powershell
python run_completion.py --methods zero,linear,knn --limit 20
python run_downstream.py --methods zero,linear --limit 20 --epochs 10
```

再跑全量（CPU 很慢，建议按方法逐个跑）：

```powershell
python run_completion.py
python run_downstream.py
```

常用参数：

- `--methods`：逗号分隔的方法名，`all` 表示全部
- `--rate`：遮挡率，默认 0.30；缺失率扫描用 0.10 / 0.50
- `--epochs`：深度方法训练轮数，默认 60
- `--limit`：只取前 N 个窗口，用于快速验证
- `--split`：评估的数据划分，默认 test

## 指标口径

- 补全质量（表 4-X）：在人工遮挡位置（art）上计算 MAE / RMSE（物理单位，m/s）
  与 SSIM。
- 下游预测（表 4-Y）：在未来帧的真实有效位置（y_valid）上计算 MAE / RMSE / SSIM。
- 不使用 CSI：连续风速无天然二分类阈值；若确需加入，必须注明阈值（如 |V|>10 m/s）。

## 缓存与重算

- 掩码、模型权重、补全结果都缓存在 `outputs/`，改了配置想重跑时删除对应文件，
  或在 `completed_v(..., force=True)` 强制重算。
