# src/utils/data.py
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset

def add_gaussian_noise(img, sigma):
    """
    Add Gaussian noise to an image tensor.
    """
    noise = torch.randn_like(img) * sigma
    return img + noise

class NoisyCIFAR10(Dataset):
    """
    A wrapper dataset that adds Gaussian noise to CIFAR10 images.
    """
    def __init__(self, root='./data', train=True, download=True, noise_level=0.1):
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.noise_level = noise_level
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        img = self.transform(img)
        noisy_img = add_gaussian_noise(img, self.noise_level)
        return noisy_img, target

def get_dataloaders(batch_size=64, noise_level=0.1):
    """
    Get train and test dataloaders for CIFAR10 with added noise.
    """
    train_dataset = NoisyCIFAR10(root='./data', train=True, download=True, noise_level=noise_level)
    test_dataset = NoisyCIFAR10(root='./data', train=False, download=True, noise_level=noise_level)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
