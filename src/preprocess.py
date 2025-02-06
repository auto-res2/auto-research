import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int, d1: int):
        self.data = torch.randn(num_samples, d1)
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticDataset(1000, config['model']['d1'])
    test_dataset = SyntheticDataset(200, config['model']['d1'])
    
    train_loader = DataLoader(train_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset,
                           batch_size=config['training']['batch_size'],
                           shuffle=False)
    
    return train_loader, test_loader
