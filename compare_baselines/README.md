# compare_baselines — 对比实验

按 `PWGAIN_PRE 对比实验实施方案.md` 实施。每个方法独立目录、互不干扰；
统一走 `common/` 共享协议（公平性：控制变量法——所有方法冻结补全器后接
**同一个** CNNRangeExtension，`in_ch=6, base_filter=32, far_gates=10, 0.183M`）。

## 快速开始（项目根执行，Windows 记得 `--num-workers` 不适用——本目录脚本固定 0）

```bash
# 1) 生成设定 A 洞 mask（缓存 eval_masks/）
python compare_baselines/prepare_eval_data.py

# 2) 传统基线（CPU 可全量跑）
python compare_baselines/linear_interp/run_linear.py
python compare_baselines/spatial_knn/run_knn.py
python compare_baselines/no_completion/run_no_completion.py

# 3) 深度基线（训练补全网络 + 扩展；GPU 建议）
python compare_baselines/official_gain/run_gain.py
python compare_baselines/unetpp_cgan/run_unetpp.py
python compare_baselines/dsa_unet/run_dsa.py

# 4) Ours（引用已有 checkpoint，不重训 G）
python compare_baselines/ours/run_ours.py --mode sep
python compare_baselines/ours/run_ours.py --mode joint

# 5) 汇总表1/表2
python compare_baselines/eval_all.py
```

每个 runner 都有 `--smoke`（2 帧 + 1 epoch 自检，CPU 秒级）。

## 评估设定

- **设定 A（表1 补全）**：test 健康区（obs∧rel）挖 4 形态
  （scatter 随机散点 / sector 方位扇区 / strip 径向条带 / block 矩形块）
  × 15%/30% 洞，洞像素物理空间 MAE/RMSE/Bias/Corr。洞 mask 缓存于
  `eval_masks/test_holes.npz`（seed=1234 确定性）。
- **设定 B（表2 扩展）**：真实带洞输入（缺失=0 + mask 通道；传统方法跳过
  缺失点）。近门补全 MAE = 表1 平均（真实缺失无真值，方案约定）；
  远门逐门物理 MAE + R_eff（τ=1.0 m/s，gate 68=2.70km 起）。

## 公平性协议（全部硬编码于 common/cfg.py，注明出处）

| 项 | 值 | 来源 |
|---|---|---|
| 归一化 | 主 ckpt G_best.pth 内嵌 normalizer，禁止重拟合 | 方案 |
| 补全训练 | lr_g=2e-4, lr_d=5e-5, betas=(0.5,0.999), adv=0.2, recon=1.0, alpha=10, drop_prob=0.3, hint=0.30, batch=8, epochs=100, patience=20, clip=5.0, seed=42 | config.py |
| 扩展训练 | Adam 2e-4 wd=0, ReduceLROnPlateau(5,0.5,min1e-6), clip=5.0, batch=8, epochs=100, patience=20, seed=42 | train_extension.py |
| 洞生成 | untils.data.generate_pure_structural_missing_mask（与主模型同策略） | 方案 |

## 输出约定

```
outputs/<method>/
  table1_<method>.csv   # 设定 A：shape×ratio → MAE/RMSE/Bias/Corr/n
  table2_<method>.csv   # 设定 B：逐门 MAE/RMSE/Bias/Corr + overall + R_eff_km
  ext_best.pth          # CNNRangeExtension（传统/深度/ours_sep）
  ext_history.csv / comp_history.csv / run_meta.json
outputs/summary_table{1,2}.csv   # eval_all.py 汇总
```

## 结构

```
common/            共享：cfg(协议硬编码) data(normalizer/组装) holes(洞mask)
                   metrics(MAE/R_eff) ext_trainer(CNN扩展训练) pipeline(标准流程)
                   deep_comp(深度补全训练器，GAIN式损失)
linear_interp/     方位周期(unwrap+interp1d) → 径向线性 → 最近邻兜底
spatial_knn/       d=Δr+λ·r·Δθ_wrap，k=8 逆距加权
no_completion/     缺失=0 直接外推（表2 第一行）
official_gain/     jsyoon0823/GAIN 的 2D 卷积改造（third_party/GAIN 参考）
unetpp_cgan/       Zhou嵌套UNet++ 生成器 + 本项目 PatchGANDiscriminator
dsa_unet/          DSA-UNet（Gong 2023，circular-W 改造，18.17M 参数）
ours/              G_best / joint_ext_best 冻结评估
third_party/       GAIN + UNetPlusPlus + Radar-Completion 只读参考
```

## 深度基线协议统一

GAIN、DSA-UNet、U-Net++ cGAN 三者共用 `common/deep_comp.py` 训练器
（GAIN 式 adv+alpha*MSE，lr_g=2e-4 / lr_d=5e-5，batch=8，epochs=100，patience=20），
保证补全质量对比只反映网络结构差异。DSA-UNet 所有 3×3 卷积 W 维 circular padding。

## 已知边界

- 本目录新增代码不修改主项目任何文件；import 主项目仅限
  untils.data/losses/circular_pad、model.PatchGANDiscriminator、
  extension_model.CNNRangeExtension、test/test_ext 的加载函数（Ours）。
- 训练/评估默认 CUDA 可用则用，`--device cpu` 可强制。
