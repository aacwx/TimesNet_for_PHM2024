import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from data_provider.data_loader import Dataset_PHM


import torch

def collate_phm_fn(batch):
    data = torch.stack([b["data"] for b in batch], dim=0)  # [B, 1024, 21]
    label = torch.stack([torch.tensor(b["label"], dtype=torch.float32) for b in batch], dim=0)  # [B, 20]
    name = [b["name"] for b in batch]  # 字符串列表
    return {"data": data, "label": label, "name": name}



def test_dataset_phm(args):
    print("Loading PHM Dataset...")

    dataset = Dataset_PHM(
        root_dir=args.root_path,
        seq_len=args.seq_len,
        stride=args.stride
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_phm_fn
    )

    print(f"Total samples: {len(dataset)}")
    for batch in dataloader:
        print(f"\nSample batch:")
        print(f"Data shape: {batch['data'].shape}")     # (B, seq_len, 21)
        print(f"Label shape: {batch['label'].shape}")   # (B, 20)
        print(f"Name list: {batch['name']}")            # list of file paths

        # Optional visualization
        sample_idx = 0
        ch = 0  # channel index
        plt.plot(batch['data'][sample_idx, :, ch].numpy())
        plt.title(f"Channel {ch} of Sample {sample_idx}")
        plt.xlabel("Time step")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

        break  # just one batch for test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../dataset/phm', help='PHM data root directory')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length (window size)')
    parser.add_argument('--stride', type=int, default=256, help='Stride for sliding window')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test_dataset_phm(args)