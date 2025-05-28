import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

# 组件通道范围定义
CHANNEL_MAP = {
    'motor': list(range(0, 9)),
    'gearbox': list(range(9, 15)),
    'leftaxlebox': list(range(15, 18)),
    'rightaxlebox': list(range(18, 21))
}

# 标签编码表（手动定义，匹配你的数据结构）
LABEL_MAP = {
    'M': {'M0': 0, 'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4},
    'G': {'G0': 0, 'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'G5': 5, 'G6': 6, 'G7': 7, 'G8': 8},
    'LA': {'LA0': 0, 'LA1': 1, 'LA2': 2, 'LA3': 3, 'LA4': 4},
    'RA': {'RA0': 0, 'RA1': 1, 'RA2': 2, 'RA3': 3, 'RA4': 4}
}


def parse_label_from_dirname(dirname):
    """从文件夹名中解析多标签 [M, G, LA, RA]"""
    parts = dirname.split('_')
    m = LABEL_MAP['M'][parts[0]]
    g = LABEL_MAP['G'][parts[1]]
    la = LABEL_MAP['LA'][parts[2]]
    ra = LABEL_MAP['RA'][parts[3]]
    return [m, g, la, ra]


class PHMDataset(Dataset):
    def __init__(self, root_dir, seq_len=1024, stride=512, components=("motor", "gearbox", "leftaxlebox", "rightaxlebox"),
                 channels=None, target_idx=None):
        """
        root_dir: PHM数据集的根目录（包含多个Mx_Gx_LAx_RAx子文件夹）
        seq_len: 滑动窗口长度
        stride: 滑动窗口步长
        components: 选用的组件（默认使用全部）
        channels: 指定使用哪些通道（默认None表示自动从components推导）
        target_idx: 指定预测目标（可选）
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.stride = stride
        self.target_idx = target_idx

        self.samples = []  # 保存所有滑窗样本

        # 计算通道索引
        if channels is None:
            self.channels = []
            for c in components:
                self.channels.extend(CHANNEL_MAP[c])
        else:
            self.channels = channels

        self._load_all_samples()

    def _load_all_samples(self):
        for cond_folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, cond_folder)
            if not os.path.isdir(folder_path):
                continue

            label = parse_label_from_dirname(cond_folder)

            for sample_dir in os.listdir(folder_path):
                sample_path = os.path.join(folder_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue

                # 读取所有数据文件并拼接
                full_data = []
                for fname in ['data_motor.csv', 'data_gearbox.csv', 'data_leftaxlebox.csv', 'data_rightaxlebox.csv']:
                    fpath = os.path.join(sample_path, fname)
                    if os.path.exists(fpath):
                        df = pd.read_csv(fpath)
                        full_data.append(df.values)
                if len(full_data) != 4:
                    continue

                data = np.concatenate(full_data, axis=1)  # [T, 21]
                data = data[:, self.channels]  # 只保留选中的通道

                # 滑动窗口切分
                total_len = data.shape[0]
                for start in range(0, total_len - self.seq_len + 1, self.stride):
                    seq_x = data[start: start + self.seq_len]  # [seq_len, C]
                    if self.target_idx is not None:
                        target = data[start: start + self.seq_len, self.target_idx]  # [seq_len]
                    else:
                        target = np.zeros(self.seq_len)  # 默认使用零目标
                    self.samples.append((seq_x, label, target, cond_folder + '/' + sample_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq_x, label, target, name = self.samples[index]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(target, dtype=torch.float32), name