"""
Data preprocessing module for TCPGS experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_dataset(dataset_name="CIFAR10", batch_size=128, train=True):
    """Load and return the specified dataset with appropriate transforms."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
        
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return data_loader


def add_gaussian_noise(images, std=0.1):
    """Add Gaussian noise to images."""
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)


def add_salt_pepper_noise(images, amount=0.05, salt_vs_pepper=0.5):
    """Add salt and pepper noise to images."""
    images_np = images.cpu().numpy()
    noisy_images = []
    
    for img in images_np:
        out = img.copy()
        # Compute number of pixels for salt and pepper noise
        num_pixels = img.size
        num_salt = int(np.ceil(amount * num_pixels * salt_vs_pepper))
        num_pepper = int(np.ceil(amount * num_pixels * (1.0 - salt_vs_pepper)))
        
        # Salt noise (set to 1)
        coords = [np.random.randint(0, i, num_salt) for i in img.shape]
        out[coords[0], coords[1], :] = 1
        
        # Pepper noise (set to 0)
        coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
        out[coords[0], coords[1], :] = 0
        
        noisy_images.append(out)
        
    return torch.tensor(np.array(noisy_images)).to(images.device, dtype=images.dtype)


def get_initial_noise(num_samples=64, image_size=(3, 32, 32), device=torch.device("cpu")):
    """Generate initial noise for diffusion models."""
    return torch.randn(num_samples, *image_size, device=device)
