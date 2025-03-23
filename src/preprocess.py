#!/usr/bin/env python

import torch
import numpy as np
from torch.utils.data import Dataset
import random

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class PCFGDataset(Dataset):
    def __init__(self, num_samples=1000, rule="aNbN", ood=False):
        """
        Args:
          num_samples: number of sequences in the dataset.
          rule: string identifier for the rule. Supported examples:
                "aNbN": exactly n a's followed by n b's.
                "aNbM": a's followed by b's but allowed variation.
                "aNbNaN": a's-b's-a's.
          ood: if True (or rule-specific conditions) produce out-of-distribution examples.
        """
        self.samples = []
        self.rule = rule
        self.ood = ood
        for _ in range(num_samples):
            n = np.random.randint(1, 10)
            if rule == "aNbN":
                # For in-distribution, generate exactly n a's followed by n b's
                if not ood:
                    seq = "a" * n + "b" * n
                else:
                    # Out-of-distribution: add an extra symbol ('x') in the middle
                    seq = "a" * n + "x" + "b" * n
            elif rule == "aNbM":
                # a's followed by b's but allow different numbers (simulate domain shift)
                if not ood:
                    seq = "a" * n + "b" * n
                else:
                    seq = "a" * n + "b" * (n+np.random.randint(0,2))
            elif rule == "aNbNaN":
                # Example: a^n b^n a^n, no OOD modifications implemented here
                seq = "a" * n + "b" * n + "a" * n
            else:
                # Default fallback
                seq = "a" * n + "b" * n
            self.samples.append(seq)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Convert characters to indices using a simplistic encoding ('a'->1, 'b'->2, etc.)
        seq = self.samples[idx]
        token_ids = [ord(c) - 96 for c in seq]  # 'a'->1, 'b'->2, 'x'->24, etc.
        return torch.tensor(token_ids, dtype=torch.long)

def pad_sequence(sequences, batch_first=True, padding_value=0):
    """
    Custom implementation of pad_sequence for batching sequences of different lengths.
    
    Args:
        sequences: List of tensors to pad
        batch_first: If True, returns tensor of shape (batch, seq_len, *), else (seq_len, batch, *)
        padding_value: Value to pad with
        
    Returns:
        Padded tensor containing all sequences
    """
    max_len = max([s.size(0) for s in sequences])
    batch_size = len(sequences)
    
    padded = torch.full((batch_size, max_len), padding_value, dtype=sequences[0].dtype)
    
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        padded[i, :length] = tensor
        
    if not batch_first:
        padded = padded.transpose(0, 1)
        
    return padded

def create_dataloaders(batch_size=32, num_train=5000, num_val=1000, rule="aNbN", include_ood=True):
    """
    Create train, validation, and optionally OOD dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        num_train: Number of training samples
        num_val: Number of validation samples
        rule: Rule to use for dataset generation
        include_ood: Whether to include OOD dataloader
        
    Returns:
        Dictionary containing dataloaders
    """
    from torch.utils.data import DataLoader
    
    train_dataset = PCFGDataset(num_samples=num_train, rule=rule, ood=False)
    val_dataset = PCFGDataset(num_samples=num_val, rule=rule, ood=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: pad_sequence([item for item in batch], batch_first=True, padding_value=0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=lambda batch: pad_sequence([item for item in batch], batch_first=True, padding_value=0)
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    if include_ood:
        ood_dataset = PCFGDataset(num_samples=num_val, rule=rule, ood=True)
        ood_loader = DataLoader(
            ood_dataset, 
            batch_size=batch_size,
            collate_fn=lambda batch: pad_sequence([item for item in batch], batch_first=True, padding_value=0)
        )
        loaders['ood'] = ood_loader
    
    return loaders

if __name__ == "__main__":
    # Test the dataset
    dataset = PCFGDataset(num_samples=5, rule="aNbN", ood=False)
    print("Dataset samples:")
    for i in range(len(dataset)):
        print(f"Sample {i}: {dataset[i]}")
    
    # Test dataloader creation
    loaders = create_dataloaders(batch_size=2, num_train=10, num_val=5)
    print("\nDataloader test:")
    for batch in loaders['train']:
        print(f"Batch shape: {batch.shape}")
        break
