from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple, Dict

def prepare_cifar10(data_dir: str) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Prepare CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader

def prepare_ptb(data_dir: str) -> Dict[str, torch.utils.data.DataLoader]:
    """Prepare PTB dataset."""
    # Implementation for PTB dataset preprocessing
    # Will be implemented in a future PR to keep the initial changes focused
    # and easier to review
    raise NotImplementedError("PTB dataset preparation will be implemented in a future PR")
