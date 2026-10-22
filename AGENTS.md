# AGENTS.md

单模块 PyTorch 研究代码：雷达 PPI 风场 GAIN 补全 + ConvLSTM 预测 + 径向量程扩展。
非 monorepo，非 git 仓库，无 requirements/pyproject、无 CI、无 README。

## 环境（先读）

- 解释器：`D:\software\Anaconda\python.exe`（Anaconda base，Python 3.13），torch 2.12.1+**cpu**，
  `torch.cuda.is_available()` = False → `config.py` 的 `device` 恒为 cpu，一切训练/推理都在 CPU 上跑，很慢。
- 已装依赖：torch / numpy / scipy / matplotlib / tqdm / pywavelets / sklearn / pandas / pytest。
  **未装 `pyart`** → `plot_real.py` 一定 ImportError。
- **OMP 坑**：脚本里若把 `import torch` 放在最前（先于 numpy/matplotlib），进程直接崩 `OMP: Error #15`。
  解法：先 `import numpy`，或在 import torch 前设 `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"`（`smoke_test.py:8` 就是这么做的）。
  现有模块都因为 matplotlib/numpy 先加载而没事，只有新写的脚本会踩。
- 终端里中文输出可能乱码，是控制台编码问题，不是代码 bug。

## 运行入口

- 唯一 CLI：`python main.py <mode>`，mode ∈ `train / test / joint / pred_test / ext_train / joint_ext / ext_test`（无参数 = train）。
- **没有命令行参数**：所有开关和路径都在 `config.py` 的 `Config()`。换数据集/换 checkpoint 要改
  `input_npy` / `model_path` / `output_dir`。
- 已损坏的模式（缺文件，别浪费时间排查）：
  - `train_joint.py` 在仓库里不存在 → `main.py joint`、`main.py pred_test`、`smoke_test.py`
    全部 `ModuleNotFoundError`。`train_joint_epoch` / `validate_joint` / `assemble_predictor_input` 在整个仓库无定义。
  - `plot_real.py` 需要未安装的 pyart。
  - 另外 `cfg.ext_pretrain_path` 指向的 `outputs_single_v/outputs_ext_cnn_near68_far78/ext_best.pth` 不存在
    （该目录整体缺失）→ 见下方“分阶段训练”。
- 核心模式 `train` / `test` / `ext_train` / `joint_ext` / `ext_test` 模块均可正常 import，现有 checkpoint 路径存在。

## 验证 / 测试

- 没有单元测试。**不要跑 `pytest`**：它会收集 `smoke_test.py` 并因 train_joint 缺失报 collection error（rc=2）。
- 改代码后的快速自检：
  ```powershell
  python -c "import train, test, test_ext, train_joint_ext, train_extension; print('ok')"
  python -c "from config import Config; from test import load_models; load_models(Config().model_path,'cpu'); print('ckpt ok')"
  ```
- 全量推理/训练很贵（CPU + 89 个 PPI），确认自检通过后再考虑真跑；`python main.py test` 会往 `output_dir`
  写约 32MB 的 npy。

## 架构速记

- 数据：`dataset/split_32/{train,val,test}.npy` = (326/30/89, 85, 125, 9) float32；
  85 行是把 3.4km 数据裁前 85 行得到的 3.2km 版本（`untils/split_32km.py`）。
  重建数据需要仓库里没有的 `dataset/*_data_channals_9_3.4km.npy`，**别试图重跑切分脚本**。
  另有 `dataset/era5_split/`（520, 61, 61, 9）用于跨域实验。
- 尺寸与常量集中在 `untils/data.py`：`ORI_H,W = 85,125` → `PAD = 91,128`；
  `PHYS_CHANNELS=4`（V / CNR / SW / Skew，只有 V 是监督目标，其余作条件输入）；
  `BAD_RAYS=[73,74,75]`（坏射线，推理时默认不被 Refiner 改写，见 `config.protect_bad_rays_refiner`）。
- 模型分布：`model.py` UNetGenerator(8 in / 1 out) + PatchGAN；`untils/refiner.py` 二阶段残差 Refiner
  （`use_refiner/use_wavelet/use_physics` 三个开关）；`predictor.py` ConvLSTM；
  `extension_model.py` 径向量程扩展（近 0~67 门 → 远 68~77 门）。
- **checkpoint 内嵌 cfg**：`test.load_models()` 用 `ckpt["cfg"]` 还原网络结构、超参和归一化统计量。
  改 `config.py` 不会影响已训模型的结构；只有设 `normalizer_refit_npy` 才会重新拟合归一化。
- 输出目录约定 `outputs_single_v/outputs_{exp_name}`，但各训练脚本自己拼名字
  （如 `train_joint_ext` 用 `joint_ext_{type}_near{n}_far{f}_le{le}`），与 `config.output_dir` 是两套东西，别混。
  已训练权重都在 `outputs_single_v/` 下，直接可用。
- **分阶段训练**，顺序有依赖：`train`（GAIN+Refiner）→ `ext_train`（量程扩展）→ `joint_ext`（联合微调）。
  `joint_ext` 读 `cfg.gain_pretrain_path` / `cfg.ext_pretrain_path`，路径不存在时**静默回退为随机初始化**
  （只打印一行“随机初始化”）→ 当前 `ext_best.pth` 缺失，跑 joint_ext 前先跑 `ext_train`，并核对启动日志。
- `cfg.output_dir` 指向 `outputs_joint_ext_cnn_near68_far78_le0.1`，而 `cfg.exp_name` 指向另一个历史目录；
  `plot_real.py` 读的是 `outputs_{exp_name}/test_completed_epoch100_patience20*.npy`，改 config 后两者要对齐。

## 其它

- 根目录的 `real_ppi_compare_*.png` 是 `plot_real.py` 的出图（根目录也是输出目录）。
- `docx/` 是 opencode docx skill 的本地副本，`docx_work/` 是处理 Word 文档的中间产物，与模型代码无关。
- 没有版本控制：改动无法回滚，改文件前先确认必要性。
