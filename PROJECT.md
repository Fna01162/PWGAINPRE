# PWGAIN_PRE 项目说明

面向不了解本项目的读者（含 AI 助手），完整描述该项目的任务、数据、模型、训练、测试、评估与可视化。

> **当前主线：补全 + 预测**（GAIN 径向风速补全 → ConvLSTM 时间序列预测）。
> 代码为单模块 PyTorch 研究项目，历史项目名含"PRE"即 prediction。

---

## 1. 任务概述

**主线（正在进行）**

- **补全**：PPI 扫描中 V（径向风速）存在真实缺失（如坏射线、低信噪比区），用条件 GAIN（GAN + 多通道物理量条件）补全缺失区；已有观测区保持不变。
- **细化**：两阶段残差 Refiner，利用 Haar 小波特征与 PPI 物理一致性约束，对 GAIN 输出的未知区做残差修正（ΔV = Gate · ΔV_raw，限幅 ±0.30）。
- **预测**：历史 10 帧（`input_steps=10`）的多通道 PPI 序列 → 未来 1 帧（`pred_steps=1`）V 风速场，ConvLSTM 编解码器（`predictor.py`）。训练入口 `main.py joint`（`train_joint.py`），推理入口 `main.py pred_test`（`test.py:pred_test_main`）。

**遗留（暂停，代码保留、非主线）**

- **量程扩展**：近距离（门 0~67）补全结果预测远距离（门 68~77）V。相关文件 `extension_model.py` / `train_extension.py` / `train_joint_ext.py` / `test_ext.py` / `plot_ext.py` 仍可运行，但不再作为主线推进；`config.py` 里扩展相关字段仍在，且会在末尾覆盖 `output_dir` / `model_path`（见 §11 注意事项）。

> ⚠️ **预测链路当前处于"代码已写、待修复"状态**：`train_joint.py` 与 `test.py:pred_test_main` 存在多处必现阻塞错误（`Config()` 无法实例化、形状/签名不匹配等）。完整问题清单与最小修复见桌面报告《代码审查报告_预测链路_20261022.md》，项目内摘要见 §15。

唯一入口：

```powershell
python main.py <mode>   # mode ∈ train / test / joint / pred_test / ext_train / joint_ext / ext_test；无参数默认 train
```

所有开关和路径集中在 `config.py` 的 `Config()`，**没有命令行参数**。

---

## 2. 运行环境（重要）

| 项 | 值 |
|---|---|
| 解释器 | `D:\software\Anaconda\python.exe`（Anaconda base，Python 3.13） |
| torch | 2.12.1+**cpu**，`torch.cuda.is_available()` = False → 设备恒为 CPU，训练/推理很慢 |
| 已装依赖 | torch / numpy / scipy / matplotlib / tqdm / pywavelets / sklearn / pandas / pytest |
| 未装 | `pyart`（`plot_ext.py` 会 ImportError）、`netCDF4`/`xarray`（仅 SL74 数据检查脚本用） |
| **OMP 坑** | 若 `import torch` 先于 numpy/matplotlib，进程直接崩 `OMP: Error #15`。新脚本须先 `import numpy`，或在 import torch 前设 `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"`（`smoke_test.py:8` 是范例）。现有模块因 matplotlib 先加载而无事。 |
| 版本控制 | **无 git**，改动无法回滚，改文件前先确认必要性 |

---

## 3. 目录结构

```
PWGAIN_PRE/
├── main.py                  # 唯一 CLI 入口（7 种 mode）
├── config.py                # Config 类：全部超参/路径；device 恒 cpu
├── model.py                 # UNetGenerator（GAIN 生成器）、PatchGANDiscriminator、init_weights
├── train.py                 # GAIN+Refiner 训练（mode=train）——补全第一阶段
├── test.py                  # 补全推理（mode=test）+ 预测推理（mode=pred_test）、load_models/load_joint_models
├── predictor.py             # 【主线】ConvLSTM 编解码器（预测网络）
├── train_joint.py           # 【主线】GAIN + ConvLSTM 联合训练（mode=joint）
├── extension_model.py       # 遗留：CNNRangeExtension / PPINAFRangeExtension、build_extension_model
├── train_extension.py       # 遗留：量程扩展独立训练（mode=ext_train）
├── train_joint_ext.py       # 遗留：GAIN+Refiner+扩展联合微调（mode=joint_ext）
├── test_ext.py              # 遗留：补全+扩展推理（mode=ext_test）
├── plot_ext.py              # 遗留：扩展结果 PPI 出图（需要 pyart，当前不可运行）
├── preprocess_SL74_generalization.py  # SL74 公开数据 → (T,85,125,9) npy
├── level_SL74.py / _inspect_ppi.py / _check_channels.py  # 一次性数据检查脚本
├── untils/                  # 注意拼写就是 untils，非 utils
│   ├── data.py              # 数据集（MultichannelClipDataset / SlidingWindowDataset）、Normalizer、掩码生成、pad/crop、尺寸常量
│   ├── losses.py            # GAIN/扩展全部损失 + 预测损失（masked_prediction_loss / compute_prediction_metrics）
│   ├── losses_refine.py     # compute_refiner_losses（Refiner 三项损失聚合）
│   ├── refiner.py           # ResidualRefiner、build_refiner_input、apply_refiner、merge_v_final
│   ├── wavelet.py           # Haar DWT（方位向周期 pad）、wavelet_loss、wavelet_features
│   ├── physics.py           # physics_consistency_loss（径向/方位梯度一致性）
│   ├── NAF_Block.py         # NAFBlock3D（通道线性注意力 + SimplifiedChannelAttention）
│   ├── axial_attn.py        # AxialWAttention3D（方位向 1D 周期自注意力）
│   ├── circular_pad.py      # CircularWConv2d/3d（W 方向周期卷积）
│   ├── volformer.py         # VolSelfAttention3D（use_volformer=False，实验用）
│   ├── evaluate.py          # 人工洞评估、指标、可视化（含 make_holes/metrics）
│   ├── split_32km.py        # 3.4km 源数据裁到 3.2km 并划分 train/val/test
│   └── exp_utils.py         # print_model_stats 等
├── dataset/
│   ├── split_32/{train,val,test}.npy       # 主数据集（见 §5）
│   ├── SL74_generalization/test.npy        # 跨域泛化数据
│   └── level1_SL74/...                     # SL74 原始 netCDF（PIANO 公开数据）
├── outputs_single_v/outputs_*/             # 训练输出：ckpt、history csv、png、npy
├── compare_baselines/                      # 补全对比实验（传统 + 深度基线，已完成）
├── compare_baselines_pred/                 # 【主线】预测对比实验（当前仅 common/cfg.py 骨架）
├── docx/ docx_work/                        # 与模型无关（Word 文档处理脚本），可忽略
└── AGENTS.md                               # 给 AI 的环境/运行速记（本文件是其扩展）
```

---

## 5. 数据集与数据格式

### 5.1 npy 数据格式

所有数据集均为 `float32`，形状 `[T, H=85, W=125, C=9]`：

- `T`：时间帧（约 5 分钟/帧）
- `H=85`：距离门，从 **660 m** 起、门距 **30 m**，最远 660+30×84 = 3180 m ≈ 3.2 km
- `W=125`：方位角射线，0~360° 均匀分布（周期性）
- `C=9` 通道：

| 通道 | 含义 |
|---|---|
| 0 | **V** 径向风速（唯一补全/监督目标） |
| 1 | CNR 信噪比 |
| 2 | SW 谱宽 |
| 3 | Skew 谱偏度 |
| 4 | Range 归一化距离 |
| 5 / 6 | sin(方位角) / cos(方位角) |
| 7 | obs_mask：1=有观测 |
| 8 | rel_mask：1=可靠 |

只有前 4 个通道（`PHYS_CHANNELS=4`：V/CNR/SW/Skew）作为网络的物理量输入；CNR/SW/Skew 是**条件输入**，不被预测。

### 5.2 特殊标记

- **坏射线** `BAD_RAYS=[73,74,75]`（`untils/data.py:14`）：固定三条方位射线，obs=0 但物理值存 0.0；`make_obs_rel` 以通道 7/8 掩码为准避免误判。推理时 `set_bad_rays`/`apply_bad_rays` 会把这三列从掩码中剔除，作为"已知真实缺失"让 GAIN 补全；Refiner 默认**不改写**坏射线（`config.protect_bad_rays_refiner=True`，`test.py:143`），防止无监督外推破坏 GAIN 输出。训练结束后还有 `fixed_ray_surrogate`（`train.py:167`）用邻近观测当替身真值评估坏射线补全。

### 5.3 数据划分（`untils/split_32km.py`）

源数据 `dataset/*_data_channals_9_3.4km.npy`（90 门）**不在仓库中**，裁前 85 行得 3.2 km 版本：

- train：20230401 + 20230405 沿时间拼接 → 326 帧
- val：20230410 前 30 帧
- test：20230410 剩余 89 帧

→ `dataset/split_32/{train,val,test}.npy`。**切分脚本不可重跑**（源文件缺失）。

> ⚠️ **对预测任务的两个已知数据问题**（详见 §15 M6/M7）：
> 1. train 由两天数据**直接沿时间轴 concat**，滑窗会在拼接处生成"4/01 末帧 → 4/05 首帧"的假序列，污染训练。
> 2. val 与 test 来自**同一天相邻时段**，首个 test 样本与最后一个 val 样本时间相邻，预测指标偏乐观。
>
> 补全任务受此影响较小（逐帧独立），预测任务（依赖时间连续性）必须重视。

### 5.4 跨域泛化数据（SL74）

- 原始：`dataset/level1_SL74/level1/SL74_l1/<日期>/SL74_ppi3_l1_*.nc`（PIANO 公开 SL74 激光雷达）。
- `preprocess_SL74_generalization.py` 把 nc 转成 `(T,85,125,9)` npy → `dataset/SL74_generalization/test.npy`；SL74 没有 SW/Skew，对应通道置 0。
- 零样本跨域推理：`config.normalizer_refit_npy` 指向目标域 npy 时，`test.py` 会用该数据**重新拟合归一化参数**（训练域参数不适用）。

---

## 6. 数据处理流水线（`untils/data.py`）

1. **观测/可靠掩码** `make_obs_rel(arr)`：由通道 7/8 与前 4 通道有限性共同决定 obs/rel。
2. **归一化** `Normalizer`：minmax + 99.5 百分位截断 → V 等物理量映射到 [-1,1]；只在观测点上拟合。训练时在 train.npy 上拟合并**内嵌进 checkpoint**；推理默认从 ckpt 还原。
3. **Pad** `pad_to`：85×125 → **91×128**（`PAD_TOP=3, PAD_BOTTOM=3, PAD_LEFT=1, PAD_RIGHT=2`）。距离向 replicate，**方位向周期拼接**（0°/360° 首尾相接）；`crop_to` 裁回 85×125。
4. **补全数据集** `MultichannelClipDataset`：返回 `(x, m, r, v)`，形状均 `[C=4, T, 91, 128]`；缺失位置 x 置 0；m/r/v 为掩码广播。
5. **预测数据集** `SlidingWindowDataset`（`untils/data.py:431`，**通道优先**）：滑窗返回 `(history_x, history_m, history_r, future_x, future_m, future_r)`，每个都是 `[C=4, T, H, W]`（history 的 T=`input_steps=10`，future 的 T=`pred_steps=1`）。
   - ⚠️ `train_joint.py` 内另有一套**时间优先** `[B,T,C,H,W]` 的滑窗实现，与这里相反 → 混用是当前预测链路最主要的崩溃来源（§15 S2/S7）。
6. **人工缺失**（训练核心）`generate_pure_structural_missing_mask`：在可靠观测区随机挖"纯结构"洞——近场全径向楔形（`n_wedge = drop_prob*10` 条射线）+ 远场（径向后 30%~70% 起）楔形（`n_far_wedge = drop_prob*25` 条射线），`drop_prob=0.3`。只在 obs∧rel 区域挖洞。旧版多模式掩码 `generate_structural_missing_mask`（方位块/径向带/矩形/混合）保留兼容但非当前默认。
7. **Contextual cue**（`train.generate_contextual_cue`）：对目标帧每个位置，找时间上最近的有效 V 帧当"上下文提示"填入缺失区。
8. **上下文融合** `build_contextual_fusion`：V 通道 = 观测值 ⊕ cue ⊕ `noise_scale·z`（缺失处），其余物理通道 = 观测 ⊕ `noise_scale·z`；`use_cue=True`。
9. **Hint** `generate_hint`：GAIN 训练技巧，在观测区按 `hint_rate=0.3` 随机保留/置 0.5 的提示掩码喂给判别器。

---

## 7. 模型架构

### 7.1 GAIN 生成器 `UNetGenerator`（`model.py`）

- 输入 `x_bar [B,4,T,91,128]`（上下文融合的 4 物理通道）+ 掩码 `m [B,4,T,91,128]`，拼接为 **8 通道**。
- 3 级 Encoder（enc1 非 NAF 的 CircularWConv 块；enc2/enc3 为 NAFBlock3D）+ Bottleneck + 3 级上采样 Decoder（skip concat）。
- Bottleneck = `BottleneckWithAxial`：1×1 对齐后交替 `NAFBlock3D` 与 `AxialWAttention3D`（方位向 1D 周期自注意力，4 heads，1 对）。
- **所有空间卷积在 W（方位）方向做周期 padding**（`CircularWConv3d`），距离向不做周期。
- `base_filter=48`；输出 `tanh`，**只输出 V 一通道** `[B,1,T,91,128]`。
- 可选组件由 Config 开关：`use_nafblock=True`、`use_axial_w=True`、`use_volformer=False`、`use_wavelet_attn/use_fits/use_sed_disc`（遗留开关）。

### 7.2 判别器 `PatchGANDiscriminator`

- 输入 `v [B,1,H,W]` + `hint [B,1,H,W]` → 2 通道 PatchGAN（4 层 stride-2/3 卷积 + GroupNorm），CircularWConv2d。
- 训练中实例化用 `PatchGANDiscriminator(2, 16)`（`train.py:208`）。
- 目标是**观测/缺失逐 patch 判别**（不是真假图判别）：label = 观测掩码下采样后做 0.9/0.1 label smoothing。

### 7.3 残差细化器 `ResidualRefiner`（`untils/refiner.py`）

- 输入 **13 通道**：V_gain、CNR、SW、Skew、Range、sinθ、cosθ、GivenMask、RelMask、Haar 小波 4 分量（LL/LH/HL/HH，由 V_gain 在线分解，方位向周期 pad）。
- 轻量 U 结构：enc1/enc2 → NAFBlock3D bottleneck → 上采样 → concat → 输出 2 通道（ΔV_raw + Gate logits）；输出层**零初始化**，训练初期 ΔV≈0 不破坏 GAIN。
- `ΔV = sigmoid(Gate) · ΔV_raw`，再 `clamp_delta` 限幅 ±`refiner_max_delta=0.30`。
- `merge_v_final`：已知可靠区保持 V_gain，未知区 = V_gain + ΔV。
- `build_refiner_input` 动态算通道数：`C + 5 + (3L+1)`（L=`n_wavelet_levels=1` → 13）。

### 7.4 量程扩展网络（`extension_model.py`，遗留非主线）

- 任务：近门 0~67 的多通道补全结果 → 远门 68~77 的 V（10 门 × 125 方位）。
- 输入 6 通道 = 4 物理量 + obs 掩码 + rel 掩码（`assemble_extension_input`）。
- `CNNRangeExtension`（默认，`ext_model_type="cnn"`）：全场编码器（3 级 ConvBlock）+ 边界编码器（对最后 8 个近门做 `(8,1)` 卷积并广播到远区）+ 特征融合解码头。
- `PPINAFRangeExtension`（`"ppinaf"`）：加 NAFBlock、多尺度下采样/跨尺度融合的改进版。

### 7.5 ConvLSTM 预测器（`predictor.py`）——主线

- 结构：`WindFieldPredictor` = 空间编码器（ConvLSTM 前降采样）→ `ConvLSTMCell` 堆叠（`hidden_channels=64`，`num_layers=2`）→ 空间解码器，输出 `tanh`。
- 输入 `[B, Tin=10, 6, H, W]`，6 通道 = **补全后的 V(1) + CNR/SW/Skew(3) + obs 掩码(1) + rel 掩码(1)**。
- 输出 `[B, Tout=1, 1, H, W]`，即未来一帧的 V。
- **已知设计缺口**：内部全部用 `nn.Conv2d`（零 padding），没有像 GAIN/Refiner 那样对方位向（W）做周期卷积 → 0°/360° 接缝处是硬边界（§15 M4）。
- 训练入口 `train_joint.py` 已存在（旧版文档记为"缺失"，已过时），但当前无法直接跑通（§15）。

---

## 8. 损失函数

### 8.1 GAIN 主损失（`train.py:122`，权重来自 Config）

| 项 | 函数（`untils/losses.py`） | 权重 | 说明 |
|---|---|---|---|
| 对抗 | `generator_adv_loss` | `adv_weight=0.2` | 仅在**人工缺失区**让判别器判"真" |
| 重建 | `reconstruction_loss` | `alpha×recon_weight = 10×1` | 人工缺失区 masked MSE |
| TV 平滑 | `tv_loss` | 0 | 缺失区空间 TV（当前关闭） |
| 时间一致 | `temporal_consistency_loss` | 0 | T>1 时启用（当前 time_steps=1，恒 0） |
| 首尾连续 | ring：首末列差绝对值 | 0.01 | 方位角 0°/360° 接缝 |
| 方位 TV | `azimuth_tv_loss` | 0.02 | 方位向一阶周期差分（含 wrap） |
| 方位曲率 | `azimuth_curvature_loss` | 0 | 二阶差分（当前关闭） |
| Refiner 重建 | `compute_refiner_losses` 内 | `lambda_refine=1.0` | v_final 在缺失区对真值 MSE |
| 小波 | `wavelet_loss` | `lambda_wavelet=0.5` | 缺失区 Haar 四子带 masked L1（各 0.25），支持多层+深度衰减 |
| 物理一致 | `physics_consistency_loss` | `lambda_physics=0.1` | 对**非可靠有效区**约束径向/方位梯度与可靠观测上下文一致（方位用周期差分；gradient_weight=0 关闭径向+方位联合项） |

判别器损失 `discriminator_loss`：观测区=0.9、缺失区=0.1 的 label smoothing BCE。

Refiner 各项由 `untils/losses_refine.py:compute_refiner_losses` 一次性计算并返回 `v_final, g_refine, g_wave, g_phy, delta, gate`；训练循环里 `gt += λ_refine·GR + λ_wave·GW + λ_phy·GP`，G 与 R 共用一个 Adam（`optim_G` 同时含两者参数）。

### 8.2 扩展损失（`untils/losses.py`，遗留）

- `extension_mae_loss`：远门预测对真值的 masked MAE（`far_valid = obs∧rel`）。
- `compute_extension_metrics`：同 mask 下 MAE/RMSE。

### 8.3 预测损失（主线）

- `masked_prediction_loss`（`untils/losses.py:190`）：缺失区 masked **MAE**，是当前联合训练的预测主损失。
- `masked_prediction_mse_loss`（L219）：同结构的 MSE 版本（备选）。
- `compute_prediction_metrics`（L248）：MAE/RMSE/Bias/Corr。
- 联合训练总损失 = **GAIN 全套损失 + `lambda_pred × 预测 MAE`**（`train_joint.py:379`）。
  注意当前实现把 `lambda_pred` 直接乘进 loss 再反传，导致预测网络有效学习率被缩小约 10 倍（§15 M9）。

---

## 9. 训练流程

### 9.1 `train`（GAIN + Refiner，`train.py`）

1. 拟合 Normalizer（train.npy）→ 构建 Dataset/DataLoader（batch=8）。
2. 建 G（UNetGenerator）+ D（PatchGAN 2→16）+ R（若 `use_refiner=True`），`init_weights` 正态(0, 0.02)。
3. 每个 batch：先 D 步（vhat=真值⊕生成拼接 + hint），再 G+R 步（对抗+重建+正则+Refiner 三项）。
4. 优化器：Adam，`lr_g=2e-4 / lr_d=5e-5`，betas=(0.5,0.999)，`grad_clip=5`；`ReduceLROnPlateau(patience=5, factor=0.5)` 按 val MAE 调。
5. 验证 `validate_artificial`：val 集上重新挖人工洞，算缺失区 V MAE/RMSE；早停 `patience=20`，seed=42 确定性模式。
6. 每轮存 `G_last.pth`，最佳存 `G_best.pth`（含 `model`/`refiner`/`normalizer`/`cfg` 全量状态）；每 50 轮存 `G_epoch_*.pth`。
7. 结束后跑 `fixed_ray_surrogate` 评估 73-75 坏射线，并 `save_history` 输出 csv + 4 张曲线 png。

### 9.2 `joint`（补全 + 预测联合训练，`train_joint.py`）——当前主线

1. 加载 GAIN 预训练（`cfg.gain_pretrain_path` → `G_best.pth`，含 normalizer）；路径不存在时**静默随机初始化**（无 else 分支，§15 M8），务必核对启动日志。
2. 用 `SlidingWindowDataset`（`untils/data.py:431`）构造 10 帧历史 → 1 帧未来的样本。
3. 每 batch：先按 GAIN 方式训练补全（人工洞 + 判别器 + Refiner 损失），再把**补全后的序列**送入 ConvLSTM 预测下一帧，叠加 `lambda_pred × 预测 MAE`。
4. 优化器：G(+R) `lr_g_joint=1e-5`（已定义但当前未被用于联合微调），预测网络 `lr_p=2e-4`；`grad_clip=5`；早停 `patience=cfg.patience`。
5. 验证 `validate`：在 val 滑窗上重新挖洞 → 补全 → 预测 → 统计 V MAE/RMSE 与预测 MAE/RMSE。
6. 输出目录自拼为 `outputs_joint_{exp_name}`，最佳存 `joint_best.pth`（应含 `model`/`predictor`/`refiner`/`normalizer`/`cfg`/指标）。
7. **当前状态**：代码骨架完整但无法跑通（8 个必崩点 + 2 个设计缺陷，见 §15）。

### 9.3 `ext_train`（量程扩展独立训练，`train_extension.py`）——遗留

- `RangeExtensionDataset`：每帧取近门 6 通道为输入、远门 68~77 的 V 为监督。
- 损失 = `extension_mae_loss`；输出目录按 `ext_exp_name`（如 `ext_cnn_near68_far78`）自拼，最佳存 `ext_best.pth`。

### 9.4 `joint_ext`（补全+扩展联合微调，`train_joint_ext.py`）——遗留

1. 加载 GAIN 预训练（`cfg.gain_pretrain_path`）与扩展预训练（`cfg.ext_pretrain_path`）；**路径不存在时静默随机初始化**（只打印一行，务必核对启动日志）。
2. 每个 PPI：GAIN 逐帧补全 → 近门 completed V + 条件 → 扩展网络 → 远门预测；联合损失 = GAIN 全套损失 + `lambda_ext=0.1 × 扩展 MAE`。
3. 优化器拆分：G(+R) `lr_g_joint=1e-5`，扩展 `lr_ext_joint=1e-4`。
4. 输出目录自拼（如 `joint_ext_cnn_near68_far78_le0.1`），存 `joint_ext_best.pth` / `joint_ext_last.pth`、`training_history.csv`、`loss_G_D.png`、`extension_metrics.png`、`completion_metrics.png` 等。

### 9.5 分阶段训练顺序

**主线（补全 + 预测）：**

```
train (GAIN+Refiner 补全) → joint (ConvLSTM 预测) → pred_test (推理)
```

**遗留（量程扩展）：**

```
train → ext_train (扩展) → joint_ext (联合微调) → ext_test (推理)
```

> `pred_test` 需要 `joint_best.pth`；`ext_test` 需要 `joint_ext_best.pth`，两者产物不同，勿混用。

### 9.6 当前不可直接运行的模式

- `joint` / `pred_test`：`Config()` 实例化即崩（`config.py` 属性顺序错误，§15 S1），修好 S1 后还有 7 个硬崩溃点。
- 能正常 import 的模块：`train` / `test` / `test_ext` / `train_joint_ext` / `train_extension` / `train_joint`（import 通过 ≠ 能运行）。

---

## 10. 测试、评估与可视化

### 10.1 补全推理 `test`（`test.py`）

- `load_models(model_path)`：从 checkpoint 的 `cfg` dict 还原网络结构（改 `config.py` 不影响已训模型），加载 G（及 Refiner，通道数按 `use_wavelet` 取 13/9）。
- `impute_sequence`：逐帧构造 cue→融合→G→（可选）Refiner（含坏射线保护）→ `crop_to` → 反归一化；观测区回填原值，只替换缺失区。
- 输出 `{stem}_completed_epoch100_patience20.npy`（另存 `_gain.npy`、`_gain_raw.npy`），写入 `config.output_dir`。
- `normalizer_refit_npy` 非空 → 打印"[零样本跨域模式]"并重拟合归一化。

### 10.2 预测推理 `pred_test`（`test.py:pred_test_main`）——主线

- 入口 `python main.py pred_test`；模型由 `load_joint_models(path)` 加载，读取 ckpt 的 `model`（G）、`predictor`（P）、`refiner`、`normalizer`、`cfg`。
- 流程：test 集滑窗 → 历史 10 帧逐帧补全（G，可选 Refiner）→ `assemble_predictor_input` 拼 6 通道序列 → ConvLSTM 预测下一帧 → 反归一化 → 与真值比较。
- 输出：`prediction_results.npz`（预测、真值、掩码）+ 指标 npy/日志。
- **当前状态**：`Config()` 崩溃 + `assemble_predictor_input` 布局错配 + ckpt key 不匹配，见 §15 S1/S7/S8。

### 10.3 扩展推理 `ext_test`（`test_ext.py`）——遗留

- `load_joint_ext_models`：加载 G / R / Ext 三网络（ckpt 含 `ext_model` 键）。
- 流程：补全 → 取近门 completed V + obs/rel/CNR/SW/Skew 条件 → 扩展网络 → 远门 68~77 预测；远门有真实观测处回填真值后算指标。
- 输出 `test_completed_ext.npy`、`test_completed_gain.npy`、`test_ext_metrics.npy`、对比图 png（近/远分界虚线标注）。

### 10.4 评估指标与工具（`untils/evaluate.py`）

- `make_holes`：在可靠区人工挖 15% 的洞，4 种模式（random / sector 方位扇区 / height 径向带 / rect 矩形）。
- `metrics`：洞内 MAE / RMSE / Bias / Corr；`evaluate_holes` 每模式 5 次试验打印均值±标准差。
- 可视化：`visualize_single_frame`（原始/掩膜/补全/误差四联）、`visualize_multiframe`、`plot_azimuth_spectrum_v`（方位频谱对比）、`check_9ch_3panel`（原始 V / 掩码 / 补全三联，标注坏射线）。

### 10.5 训练出图

`save_history`：`training_history.csv` + `loss_G_D.png` + `D_real_fake.png` + `wind_speed_metrics.png`（V MAE/RMSE）+ `missing_rate.png`（人工/真实/总缺失率，参考线 6%）。

### 10.6 对比实验

- `compare_baselines/`：**补全**对比实验（传统插值 / UNet++ cGAN / DSA-UNet / 本方法），已完成并通过冒烟测试；入口 `eval_all.py`、`prepare_eval_data.py`、`run_all.bat`。
- `compare_baselines_pred/`：**预测**对比实验，目前只有 `common/cfg.py` 骨架，定义了 `persistence` / `linear_extrap` / `optical_flow` / `convlstm` / `ours` 等方法名，尚未实现 runner。

### 10.7 PPI 极坐标出图 `plot_ext.py`（遗留）

用 pyart 构造 Radar 对象画真实 PPI 对比图；**pyart 未安装，当前必 ImportError**。根目录的 `real_ppi_compare_*.png` 是历史出图产物。

---

## 11. 关键配置速查（`config.py` 当前值）

```
数据:  dataset/split_32/{train,val,test}.npy；input_npy=test.npy
人工缺失 drop_prob=0.3；hint_rate=0.30；noise_scale=0.10
模型:  base_filter=48；use_nafblock/use_axial_w=True；axial heads=4, pairs=1
Refiner: use_refiner/wavelet/physics=True；base_filter=16；max_delta=0.30
         lambda_refine/wavelet/physics = 1.0 / 0.5 / 0.1
         protect_bad_rays_refiner=True
GAIN:  alpha=10；adv=0.2；ring=0.01；aztv=0.02；其余正则=0
训练:  batch=8；epochs=100；patience=20；lr_g=2e-4；lr_d=5e-5；seed=42
归一化: minmax + 99.5% 截断；normalizer_refit_npy=test.npy（跨域重拟合）
扩展:  near_end_gate=68；far_end_gate=78；type=cnn；base_filter=32
       lambda_ext=0.1；lr_g_joint=1e-5；lr_ext_joint=1e-4
预训练: gain_pretrain_path=outputs_PPNAF_AZI_34km_.../G_best.pth（存在）
        ext_pretrain_path=outputs_ext_cnn_near68_far78/ext_best.pth（当前缺失）
推理:  model_path=outputs_single_v/outputs_joint_ext_cnn_near68_far78_le0.1/joint_ext_best.pth
       output_dir=outputs_single_v/outputs_joint_ext_cnn_near68_far78_le0.1
```

**注意两套命名**：`config.output_dir` 是推理输出；各训练脚本按 `exp_name` / `joint_ext_{type}_near{n}_far{f}_le{le}` 自拼训练目录，两者不一致时别混淆。`exp_name` 指向历史目录 `joint_PPNAF_AZI_32km_...`。

---

## 12. 已训练产物（`outputs_single_v/`）

| 目录 | 内容 |
|---|---|
| `outputs_PPNAF_AZI_34km_CIR_aztv0.02_h0.3_bot_wavelet_phy_0.1` | GAIN 预训练（G_best/G_last/G_epoch_050、曲线 png、test_completed*） |
| `outputs_joint_PPNAF_AZI_32km_...` | 联合（G+ConvLSTM 时代）历史 checkpoint |
| `outputs_joint_ext_cnn_near68_far78_le0.1` | **当前主用**：joint_ext_best/last.pth、test_completed_ext.npy、test_ext_metrics.npy、对比图 |
| `outputs_SL74_generalization` | SL74 跨域推理结果（completed_ext / completed_gain / gain_raw、对比图） |

---

## 13. 快速自检（改代码后必跑）

```powershell
# 1) 模块可导入
python -c "import train, test, test_ext, train_joint_ext, train_extension; print('ok')"
# 2) checkpoint 完整
python -c "from config import Config; from test import load_models; load_models(Config().model_path,'cpu'); print('ckpt ok')"
```

**不要跑 `pytest`**：会收集 `smoke_test.py` 并因 train_joint 缺失报 collection error。全量推理/训练很贵（CPU + 89 个 PPI，`python main.py test` 约写 32MB npy），自检通过后再考虑真跑。

---

## 14. 设计要点备忘

1. **方位周期性贯穿全链路**：卷积（CircularWConv）、注意力（AxialWAttention）、小波（周期 pad）、损失（wrap 差分）、数据 pad，全部把 W 当环面处理；距离向一律非周期。
2. **只有缺失区被监督/被改写**：GAIN 的 adv/rec、Refiner 的 merge、推理回填，都保证可靠观测区原样保留。
3. **Refiner 是纯残差 + 限幅 + 零初始化**：设计上保证"最差也只在 GAIN 基础上小幅修正"；坏射线保护是显式开关。
4. **checkpoint 自描述**：cfg dict + normalizer 状态随权重保存，推理与训练解耦；跨域靠重拟合归一化而非微调。
5. **人工缺失模拟真实缺失形态**（近场整射线楔形 + 远场楔形），而非均匀随机像素，使训练分布贴近坏射线/衰减区。
