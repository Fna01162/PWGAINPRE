# DSA-UNet 基线

论文：Gong, A.; Chen, H.; Ni, G. *Improving the Completion of Weather Radar Missing
Data with Deep Learning*. Remote Sens. 2023, 15, 4568.
官方代码：https://github.com/THUGAF/Radar-Completion（已克隆到 `../third_party/Radar-Completion/`）

## 结构（照抄官方 models/dsa_unet.py，~18.17M 参数）

- in_conv 1×1 (6→32)
- 4× 下采样 DoubleConv: 32→64→128→256→512
- 4× 空洞卷积 (512ch, dilation 1/2/4/8)
- 4× 上采样 DoubleDeconv (ConvT): 1024→256→128→64→32，前两个上采样输出接
  SelfAttention (channel/8, gamma 残差)
- out 1×1 + Sigmoid

## 与官方的公平性改动（方案 3.1 节要求）

1. 所有 3×3 卷积改为 W 维 `CircularWConv2d`（官方用 ±20 列显式方位 padding；
   circular 等价消除 0°/360° 接缝伪影且不改变张量宽度）
2. 输入 6 通道（V×m_eff+噪声 / CNR / SW / Skew / m_eff / rel）、输出 1 通道
   （官方为多仰角反射率 6ch / 单目标反射率 1ch）
3. 训练协议不沿用官方 weighted-L1，而是与 GAIN/UNet++ 完全一致
   （`common/deep_comp.py`：GAIN 式 adv+alpha*MSE、lr_g=2e-4 / lr_d=5e-5、
   batch=8、epochs=100、patience=20）——方案要求深度基线间协议统一

## 运行（项目根）

```bash
python compare_baselines/dsa_unet/run_dsa.py          # 全量（GPU 建议）
python compare_baselines/dsa_unet/run_dsa.py --smoke  # 2 帧自检
```

输出：`compare_baselines/outputs/dsa_unet/`
