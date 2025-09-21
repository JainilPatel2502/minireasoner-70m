import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ShardedDataset(Dataset):
    def __init__(self, data_dir, seq_len=1024):
        self.seq_len = seq_len
        # sort files so we always go train_0000.bin → train_0001.bin → ...
        self.files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith(".bin")]

        # memory-map all shards into one long array view
        self.data = [np.memmap(f, dtype=np.uint32, mode="r") for f in self.files]
        self.data = np.concatenate(self.data)

        # how many full samples fit
        self.num_samples = len(self.data) // (self.seq_len + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Return one sequential sample"""
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = self.data[start:end]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_sequential_dataloader(data_dir, batch_size=2, seq_len=1024, num_workers=0):
    dataset = ShardedDataset(data_dir, seq_len)
    # sequential only → shuffle=False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


if __name__ == "__main__":
    data_dir = "."  # path with train_0000.bin ... train_0059.bin
    loader = get_sequential_dataloader(data_dir, batch_size=4, seq_len=1024)

    for xb, yb in loader:
        print("X batch:", xb.shape)  # (B, 1024)
        print("Y batch:", yb.shape)  # (B, 1024)
        break

