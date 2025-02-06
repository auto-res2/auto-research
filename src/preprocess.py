import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, input_dim=768):
        self.data = torch.randn(num_samples, seq_len, input_dim)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(batch_size=32):
    train_dataset = SyntheticDataset()
    val_dataset = SyntheticDataset(num_samples=200)
    test_dataset = SyntheticDataset(num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
