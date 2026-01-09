import numpy as np
import torch
import os
import tqdm

# 全局类别映射
mapping = { 0 :'airplane'  ,
            1 :'automobile',
            2 :'bird' ,
            3 :'cat'   ,
            4 :'deer'  ,
            5 :'dog'    ,
            6 :'frog'   ,
            7 :'horse'       ,
            8 :'ship'      ,
            9 :'truck'     }

def read_aedat_events(file_path):
    """
    用 Python 读取并解析 .aedat 文件的函数，功能等同于 dat2mat.m。
    *** 已更新：增加了版本检测功能，可以处理 6字节/事件 和 8字节/事件 两种格式。 ***
    """
    with open(file_path, 'rb') as f:
        # --- 1. 改进了文件头解析和版本检测 ---
        version = None
        data_start_pos = 0
        while True:
            # 记录当前位置，这可能是二进制数据的起点
            data_start_pos = f.tell()
            line_bytes = f.readline()
            
            # 如果读到文件末尾或者行不是以'#'开头，则文件头结束
            if not line_bytes or not line_bytes.startswith(b'#'):
                break
            
            # 检查版本字符串
            if b'#!AER-DAT' in line_bytes:
                try:
                    # 从 "b'#!AER-DAT2.0'" 中提取出 2.0
                    version_str = line_bytes.split(b'#!AER-DAT')[1].strip()
                    version = float(version_str)
                except (IndexError, ValueError):
                    # 无法解析版本号，将使用默认值
                    pass

        # --- 2. 根据检测到的版本动态设置数据类型 ---
        if version is not None and version >= 2.0:
            # AEDAT 2.0: 8 字节/事件 (4字节地址 + 4字节时间戳)
            event_dtype = np.dtype([('addr', '>u4'), ('ts', '>u4')])
        else:
            # AEDAT 1.0 或未知版本: 6 字节/事件 (2字节地址 + 4字节时间戳)
            event_dtype = np.dtype([('addr', '>u2'), ('ts', '>u4')])

        # --- 3. 读取二进制数据 ---
        # 将文件指针移回到二进制数据的真正起点
        f.seek(data_start_pos)
        binary_data = f.read()
        
        # 使用正确的 dtype 来解析数据，这可以解决之前的 ValueError
        events = np.frombuffer(binary_data, dtype=event_dtype)
        
        addrs = events['addr']
        
        # 位掩码和位移值对于 DVS128 保持不变
        x_mask = 0x00FE
        y_mask = 0x7F00
        pol_mask = 0x0001
        x_shift = 1
        y_shift = 8

        x = (addrs & x_mask) >> x_shift
        y = (addrs & y_mask) >> y_shift
        x_flipped = 127 - x
        polarity = (addrs & pol_mask)

        num_events = len(addrs)
        output_matrix = np.zeros((num_events, 6), dtype=np.int32)
        
        output_matrix[:, 3] = y
        output_matrix[:, 4] = x_flipped
        output_matrix[:, 5] = polarity

        return output_matrix

def events_to_frames(events_matrix, T=10):
    """
    将事件矩阵转换为帧序列。
    """
    frames = np.zeros((T, 2, 128, 128), dtype=np.float32)
    num_events = events_matrix.shape[0]
    if num_events == 0:
        return frames

    for i in range(T):
        r1 = i * (num_events // T)
        r2 = (i + 1) * (num_events // T)
        
        current_events = events_matrix[r1:r2]
        if len(current_events) == 0:
            continue
            
        y_coords = current_events[:, 3]
        x_coords = current_events[:, 4]
        channels = current_events[:, 5]

        np.add.at(frames[i], (channels, y_coords, x_coords), 1)

    for i in range(T):
        frame_max = np.max(frames[i])
        if frame_max > 0:
            frames[i] /= frame_max
            
    return frames

def create_pt_files(T: int = 4, split: float = 0.9):
    """
    主函数： orchestrates the entire preprocessing pipeline.
    """
    raw_data_dir = os.path.expanduser('~/datasets/cifar10dvs_aug/raw_events')
    output_dir = os.path.expanduser('~/datasets/cifar10dvs_aug/')
    train_dir = os.path.join(output_dir, f'T{T}', 'train')
    test_dir = os.path.join(output_dir, f'T{T}', 'test')

    # 安全创建目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 软链接
    os.symlink(train_dir, os.path.join(output_dir, 'train'))
    os.symlink(test_dir, os.path.join(output_dir, 'test'))

    print(f"Checking for raw data in: {os.path.abspath(raw_data_dir)}")
    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw data directory not found at '{raw_data_dir}'.")
        print("Please ensure the raw DVS-CIFAR10 dataset is in that folder, or modify the 'raw_data_dir' variable in the script.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = []
    for class_id, class_name in mapping.items():
        class_path = os.path.join(raw_data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Directory for class '{class_name}' not found. Skipping.")
            continue
        for i in range(1000):
            file_path = os.path.join(class_path, f'cifar10_{class_name}_{i}.aedat')
            if os.path.exists(file_path):
                all_files.append({'path': file_path, 'label': class_id})

    train_files = []
    test_files = []
    for class_id in mapping.keys():
        class_files = [f for f in all_files if f['label'] == class_id]
        split_idx = int(len(class_files) * split)
        train_files.extend(class_files[:split_idx])
        test_files.extend(class_files[split_idx:])
        
    print(f"Found {len(all_files)} total files.")
    print(f"Processing {len(train_files)} files for training set...")
    for i, file_info in enumerate(tqdm.tqdm(train_files, desc="Training Data")):
        event_matrix = read_aedat_events(file_info['path'])
        frames = events_to_frames(event_matrix, T=T)
        output_path = os.path.join(train_dir, f'{i}.pt')
        torch.save([torch.from_numpy(frames), torch.tensor([file_info['label']])], output_path)

    print(f"Processing {len(test_files)} files for testing set...")
    for i, file_info in enumerate(tqdm.tqdm(test_files, desc="Testing Data")):
        event_matrix = read_aedat_events(file_info['path'])
        frames = events_to_frames(event_matrix, T=T)
        output_path = os.path.join(test_dir, f'{i}.pt')
        torch.save([torch.from_numpy(frames), torch.tensor([file_info['label']])], output_path)

    print('\nPreprocessing complete!')
    print(f"Processed data saved in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    create_pt_files(T=2, split=0.9)