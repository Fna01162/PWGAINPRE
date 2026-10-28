# -*- coding: utf-8 -*-
"""
批量运行四链路 + Oracle 对比实验脚本

用法:
    python run_all_chains.py

功能:
    - 自动运行 A/B/C/D 四个链路的训练和测试
    - 运行 Oracle 上界评估
    - 汇总所有结果到 CSV 文件
    - 生成对比表格

作者: Auto-fix based on 问题清单
日期: 2026-10-26
"""

import os
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime


def run_chain(chain_mode, description):
    """运行单个链路的训练和测试"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting Chain {chain_mode}: {description}")
    print(f"{'='*70}\n")

    # 训练（链路模式由 Config 统一决定输出目录）
    result = subprocess.run(["python", "main.py", "joint", chain_mode],
                            capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Chain {chain_mode} training FAILED!")
        return None

    # 测试（与训练使用同一个 chain_mode，因此指向同一个输出目录）
    result = subprocess.run(["python", "main.py", "joint_test", chain_mode],
                            capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Chain {chain_mode} testing FAILED!")
        return None

    print(f"✅ Chain {chain_mode} completed successfully!")
    return True


def collect_results(output_base_dir):
    """收集所有链路的测试结果"""

    chain_modes = ["A", "B", "C", "D", "D_oracle"]
    chain_names = {
        "A": "Direct Prediction (Baseline)",
        "B": "Linear Interpolation",
        "C": "Frozen GAIN",
        "D": "Joint Training (Ours)",
        "D_oracle": "Oracle Upper Bound"
    }

    all_results = []

    for mode in chain_modes:
        # 输出目录按链路模式命名（由 Config 统一生成），Oracle 结果存放在对应链路的目录中
        base_chain = mode.replace("_oracle", "")
        possible_dirs = [
            os.path.join(output_base_dir, f"joint_pred_Tin10_Tout1_lp0.1_{base_chain}"),
            os.path.join(output_base_dir, "joint_pred_Tin10_Tout1_lp0.1"),
        ]

        result_file = None
        for dir_path in possible_dirs:
            test_file = os.path.join(dir_path, f"test_{mode}.npy")
            if os.path.exists(test_file):
                result_file = test_file
                break

        if result_file is None:
            print(f"⚠️  Warning: Result file for chain {mode} not found")
            continue

        try:
            data = np.load(result_file, allow_pickle=True).item()

            # 提取各步长的指标
            for step_key in sorted([k for k in data.keys() if k.startswith('step')]):
                step_num = step_key.replace('step', '')
                metrics = data[step_key]

                all_results.append({
                    'Chain': mode,
                    'Chain_Name': chain_names.get(mode, mode),
                    'Prediction_Step': int(step_num),
                    'Time_Min': int(step_num) * 5,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'Correlation': metrics['corr'],
                    'SSIM': metrics['ssim'],
                    'Valid_Pixel_Ratio': metrics['valid_ratio']
                })

        except Exception as e:
            print(f"⚠️  Error reading {result_file}: {e}")
            continue

    return all_results


def generate_comparison_table(results, output_path):
    """生成对比表格并保存为 CSV 和 Markdown"""

    if not results:
        print("❌ No results to generate table!")
        return

    df = pd.DataFrame(results)

    # 保存为 CSV
    csv_path = os.path.join(output_path, "comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV saved to: {csv_path}")

    # 生成 Markdown 表格
    md_path = os.path.join(output_path, "comparison_table.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 四链路对比实验结果\n\n")
        f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 按预测步长分组生成表格
        for pred_step in sorted(df['Prediction_Step'].unique()):
            step_df = df[df['Prediction_Step'] == pred_step]
            time_min = int(pred_step) * 5

            f.write(f"## 预测步长: {pred_step} 步 ({time_min} 分钟)\n\n")
            f.write("| 链路 | 方法 | MAE ↓ | RMSE ↓ | Corr ↑ | SSIM ↑ | 有效像素% |\n")
            f.write("|------|------|-------|--------|--------|--------|----------|\n")

            for _, row in step_df.iterrows():
                f.write(f"| {row['Chain']} | {row['Chain_Name']} "
                       f"| {row['MAE']:.6f} | {row['RMSE']:.6f} "
                       f"| {row['Correlation']:.4f} | {row['SSIM']:.4f} "
                       f"| {row['Valid_Pixel_Ratio']*100:.2f}% |\n")

            f.write("\n")

        # 添加分析说明
        f.write("## 分析与结论\n\n")
        f.write("### 关键发现:\n\n")
        f.write("1. **联合训练增益**: D行 vs A/B/C行的性能差异体现补全赋能预测的效果\n")
        f.write("2. **Oracle上界距离**: D_oracle 与 D 的差距反映缺失数据造成的理论上限损失\n")
        f.write("3. **多步预测衰减**: 观察不同时间尺度的性能变化趋势\n\n")

        f.write("### 符号说明:\n")
        f.write("- ↑ 越高越好\n")
        f.write("- ↓ 越低越好\n\n")

    print(f"📊 Markdown table saved to: {md_path}")

    # 打印摘要
    print(f"\n{'='*70}")
    print(f" 📈 实验结果摘要")
    print(f"{'='*70}")

    for pred_step in sorted(df['Prediction_Step'].unique()):
        step_df = df[df['Prediction_Step'] == pred_step]
        time_min = int(pred_step) * 5
        print(f"\n⏱️  预测 {pred_step} 步 ({time_min} 分钟):")
        print(f"{'-'*60}")

        for _, row in step_df.iterrows():
            marker = "👑" if row['Chain'] == "D" else "🔮" if 'oracle' in str(row['Chain']).lower() else "  "
            print(f"{marker} [{row['Chain']}] {row['Chain_Name'][:30]:<30s} MAE={row['MAE']:.6f} Corr={row['Correlation']:.4f}")

    print(f"\n{'='*70}\n")


def main():
    """主函数：运行所有链路并汇总结果"""

    print("\n" + "="*70)
    print(" 🔬 四链路 + Oracle 对比实验批处理脚本")
    print("="*70)
    print(f" ⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # 定义四个链路
    chains = [
        ("A", "直接预测（基线）"),
        ("B", "线性插值 + 预测"),
        ("C", "GAIN补全（冻结）+ 预测"),
        ("D", "联合训练（Ours）✨"),
    ]

    # 依次运行每个链路
    success_count = 0
    for mode, desc in chains:
        if run_chain(mode, desc):
            success_count += 1
        print(f"\n✨ Progress: {success_count}/{len(chains)} chains completed\n")

    # 注意：D 链路的 Oracle 上界评估由 test_joint.py 默认一并执行

    # 收集和汇总结果
    print(f"\n{'='*70}")
    print(f" 📊 收集实验结果...")
    print(f"{'='*70}\n")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_single_v")
    results = collect_results(output_dir)

    if results:
        # 生成对比表格
        generate_comparison_table(results, output_dir)

        print(f"\n🎉 所有实验完成！")
        print(f"   成功完成链路数: {success_count}/{len(chains)}")
        print(f"   结果文件:")
        print(f"     - comparison_results.csv")
        print(f"     - comparison_table.md")
    else:
        print("⚠️  未找到任何实验结果，请检查训练是否成功完成")

    print(f"\n⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()