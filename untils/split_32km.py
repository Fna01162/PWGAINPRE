# -*- coding: utf-8 -*-
import os
import numpy as np


def split_32km_data():
    """
    [最终修复]: 时空断层采样隔离 & 空间裁剪 & 科学数据集划分
    """
    data_dir = "D:/windnet/GAIN/PWGAIN_PRE/dataset/raw_days"
    out_dir = "D:/windnet/GAIN/PWGAIN_PRE/dataset/split_32"
    os.makedirs(out_dir, exist_ok=True)

    day_files = [
        "20230401_data_channals_9_3.4km.npy",
        "20230405_data_channals_9_3.4km.npy",
        "20230410_data_channals_9_3.4km.npy"
    ]

    GAP_FRAMES = 11  # 隔离带长度: input_steps(10) + pred_steps(1)
    TARGET_H = 85  # 强制目标高度为 85

    # 1. 拼接所有文件，并在跨天处插入隔离带
    all_stitched = []
    for i, f in enumerate(day_files):
        file_path = os.path.join(data_dir, f)
        if not os.path.exists(file_path):
            print(f"[Warning] 找不到文件: {file_path}")
            continue
        day_data = np.load(file_path)

        # 裁剪 H 到 85
        current_h = day_data.shape[1]
        if current_h > TARGET_H:
            day_data = day_data[:, :TARGET_H, :, :]

        all_stitched.append(day_data)

        # 在非最后一天的末尾插入 GAP
        if i < len(day_files) - 1:
            gap = np.full((GAP_FRAMES,) + day_data.shape[1:], np.nan, dtype=np.float32)
            all_stitched.append(gap)

    if len(all_stitched) == 0:
        print("没有加载到任何数据！")
        return

    full_data = np.concatenate(all_stitched, axis=0)
    total_frames = len(full_data)
    print(f"合并后的总数据帧数 (含隔离带): {total_frames}")

    # 2. 按照 60% Train, 20% Val, 20% Test 进行时间轴硬切分
    # 为了防止集合之间互相污染，我们在切分点也必须留出空白 (GAP_FRAMES)

    train_end = int(total_frames * 0.6)
    val_end = int(total_frames * 0.8)

    # 切分训练集
    train_data = full_data[:train_end]

    # 切分验证集 (跳过 GAP_FRAMES 以隔离训练集)
    val_start = train_end + GAP_FRAMES
    val_data = full_data[val_start: val_end]

    # 切分测试集 (跳过 GAP_FRAMES 以隔离验证集)
    test_start = val_end + GAP_FRAMES
    test_data = full_data[test_start:]

    # 3. 保存
    if len(train_data) > 0:
        np.save(os.path.join(out_dir, "train.npy"), train_data)
        print(f"Saved train.npy, shape: {train_data.shape}")
    if len(val_data) > 0:
        np.save(os.path.join(out_dir, "val.npy"), val_data)
        print(f"Saved val.npy, shape: {val_data.shape}")
    if len(test_data) > 0:
        np.save(os.path.join(out_dir, "test.npy"), test_data)
        print(f"Saved test.npy, shape: {test_data.shape}")


if __name__ == "__main__":
    split_32km_data()