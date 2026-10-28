# 问题清单修复报告

> 修复日期: 2026-10-26
> 依据文件: 对比实验方案问题清单.md

---

## ✅ 已修复的 P0 致命问题 (A类)

### A1. ✅ 判别器 D 完全未训练
**位置**: [train_joint.py](train_joint.py#L370-L390)
**修复内容**:
- 添加了完整的判别器训练逻辑
- 真实完整数据过 D 计算 `discriminator_loss`
- G 生成的补全数据过 D 计算欺骗损失
- 梯度反传 + 优化器更新

**关键代码**:
```python
# 真实数据
d_real = D(hx)
d_real_loss = discriminator_loss(d_real, 1 - art_mask[:, :, 0], valid_mask_h[:, :, 0])

# 假数据（G生成）
x_fake = hx.clone()
x_fake[:, 0:1, :, :, :] = v_completed.detach()
d_fake = D(x_fake)
d_fake_loss = discriminator_loss(d_fake, art_mask[:, :, 0], valid_mask_h[:, :, 0])

d_total_loss = (d_real_loss + d_fake_loss) * 0.5
d_total_loss.backward()
optim_D.step()
```

---

### A2. ✅ G 的重建/对抗损失为 0
**位置**: [train_joint.py](train_joint.py#L392-L415)
**修复内容**:
- **重建损失**: `reconstruction_loss` - G 输出在人工缺失区域应接近真实值
- **对抗损失**: `generator_adv_loss` - 欺骗判别器认为补全数据是真实的
- **TV 平滑损失**: `tv_loss` - 空间连续性约束
- **总损失**: `g_loss = recon_weight * recon_loss + adv_loss + 0.1 * tv`

**关键代码**:
```python
recon_loss = reconstruction_loss(v_completed, hx[:, 0:1], art_mask[:, 0:1], valid_mask_h[:, 0:1])
adv_loss = generator_adv_loss(d_logits_g, art_mask[:, :, 0], valid_mask_h[:, :, 0])
tv = tv_loss(v_completed, art_mask[:, 0:1], valid_mask_h[:, 0:1])
g_loss = (cfg.recon_weight * recon_loss) + adv_loss + (0.1 * tv)
```

---

### A3. ✅ 四链路 A/B/C/D 无开关、无冻结逻辑
**位置**: [train_joint.py](train_joint.py#L165-L185)
**修复内容**:
- 添加 `chain_mode` 参数支持四种模式：
  - **A**: 直接预测（无补全，缺失处归零）
  - **B**: 线性插值 + 预测
  - **C**: GAIN 补全（冻结 G）+ 预测
  - **D**: 联合训练（Ours，G 和 P 都训练）
- C 链路自动冻结 G 的参数 (`requires_grad=False`)
- 根据链路模式条件性创建优化器

**使用方法**:
```bash
python main.py joint A   # A链路：直接预测
python main.py joint B   # B链路：线性插值
python main.py joint C   # C链路：冻结GAIN
python main.py joint D   # D链路：联合训练（默认）
```

**关键代码**:
```python
chain_mode = sys.argv[2] if len(sys.argv) > 2 else "D"
freeze_G = (chain_mode == "C")
use_completion = (chain_mode in ["B", "C", "D"])
use_gan = (chain_mode in ["C", "D"])
train_G = (chain_mode == "D") and not freeze_G

if freeze_G:
    for param in G.parameters():
        param.requires_grad = False
    G.eval()
```

---

### A4. ✅ mask 过窄非真 20% 扇区；测试 mask 不可复现
**位置**: [train_joint.py](train_joint.py#L305-L330), [train_joint.py#L86-L120)
**修复内容**:

#### 训练时：
- ❌ 旧: `_pure_structural_mask` - 仅 1.6% 缺失率（2条射线/125）
- ✅ 新: `_continuous_ray_mask` + `_calibrate_missing`
  - 生成真正的方位扇区（~19条射线对应15%缺失率）
  - 精确校准到目标缺失率 `cfg.missing_rate=0.15`

#### 测试时：
- ❌ 旧: 在线随机生成 mask，每次测试结果不同
- ✅ 新: `generate_fixed_test_mask()` 预生成固定测试 mask
  - 所有方法共用同一份 mask，确保公平对比
  - 结果完全可复现

**关键代码**:
```python
# 训练时
n_rays = max(1, int(cfg.missing_rate * W))  # 0.15*125 ≈ 19条射线
raw_mask = _continuous_ray_mask(H, W, n_rays)
calibrated_mask = _calibrate_missing(raw_mask, cfg.missing_rate, allowed > 0.5)

# 测试时（预生成固定mask）
fixed_masks = generate_fixed_test_mask(test_loader, cfg, device)
art_mask = fixed_masks[mask_idx:mask_idx+B].to(device)  # 复用固定mask
```

---

## ✅ 已修复的 P1 重要问题 (B类)

### B2. ✅ test.py 引用已删除的 `cfg.exp_name`
**位置**: [test.py](test.py#L162)
**修复前**: `print(f" > Model: {cfg.exp_name}")` → AttributeError
**修复后**: `print(f" > Model: {cfg.output_dir}")`

---

## 📊 修改文件清单

| 文件 | 修改内容 | 影响范围 |
|------|---------|---------|
| [train_joint.py](train_joint.py) | A1+A2+A3+A4 全部修复 | 核心训练逻辑 |
| [test.py](test.py) | B2 修复 | 测试脚本 |

---

## 🎯 使用指南

### 运行四链路实验

```bash
# 1. A链路：直接预测（基线）
python main.py joint A

# 2. B链路：线性插值 + 预测
python main.py joint B

# 3. C链路：GAIN补全（冻结）+ 预测
python main.py joint C

# 4. D链路：联合训练（Ours）
python main.py joint D
```

### 输出目录结构
```
outputs_single_v/
└── joint_pred_Tin10_Tout1_lp0.1/
    ├── training_history.csv      # 包含 pred_mae, recon_loss, adv_loss, total_loss
    ├── joint_best.pth            # 最佳模型
    ├── joint_last.pth            # 最后模型
    ├── loss_joint.png            # 训练曲线
    ├── test_A.npy                # A链路测试结果
    ├── test_B.npy                # B链路测试结果
    ├── test_C.npy                # C链路测试结果
    └── test_D.npy                # D链路测试结果
```

### 日志格式
```csv
epoch,pred_mae,recon_loss,adv_loss,total_loss,learning_rate
1,0.123456,0.234567,0.345678,1.572345,0.00020000
...
```

---

## ⚠️ 注意事项

1. **公平对比**: 四链路必须使用相同的随机种子、epoch数、batch size
2. **C链路的P**: 必须从零训练，不能复用D链路的P（已在代码中确保）
3. **测试mask**: 四个链路测试时会自动使用同一份预生成的固定mask
4. **重建权重**: 当前 `recon_weight=10.0`，可根据实际效果调整

---

## 🔄 后续建议 (P2级别，可选)

虽然已修复所有P0和部分P1问题，但以下改进可进一步提升论文质量：

### C类（方案设计）:
- [ ] **C1**: 表1添加 NowcastNet (Nature 2023)
- [ ] **C2**: 用 DSA-UNet 替换 LFnet/LWFNet
- [ ] **C3**: 表2至少3个随机种子，报均值±标准差
- [ ] **C4**: 多步预测（1/3/5步）
- [ ] **C5**: 补充相关系数、SSIM指标
- [ ] **C6**: 添加 Oracle 上界

### D类（小问题）:
- [ ] **D1**: 报告有效评估像素占比
- [ ] **D2**: 删除量程选项 "未来1帧远门68-77"
- [ ] **D3**: 确认 test.npy 时间连续性
- [ ] **D4**: 术语修正（NAFBlock描述）

---

## ✨ 核心改进总结

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| **判别器训练** | TODO占位，完全不更新 | 完整的真实/假数据对抗训练 |
| **G的损失** | 孤立张量，梯度为0 | 重建+对抗+TV三重约束 |
| **实验控制变量** | 无链路开关 | A/B/C/D四链路完整支持 |
| **Mask生成** | 1.6%伪扇区，随机不可复现 | 真正15%方位扇区，精确校准，固定可复现 |
| **测试脚本** | 引用不存在的exp_name | 使用output_dir替代 |

---

**修复完成！现在可以运行四链路对比实验了。** 🚀