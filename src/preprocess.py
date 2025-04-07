"""
Data preprocessing for Cov-Purify++ experiments.
"""
import torch
import numpy as np
import random
from torchvision import datasets, transforms

def load_cifar10(batch_size=16, num_workers=4, image_size=224):
    """
    Load CIFAR-10 dataset with preprocessing for the experiments.
    
    Args:
        batch_size (int): Batch size for the dataloader
        num_workers (int): Number of workers for the dataloader
        image_size (int): Size to resize images to (for compatibility with models like ResNet)
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=transform, 
        download=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def setup_device(use_cuda=True):
    """
    Set up the device (CPU or CUDA) for the experiments.
    
    Args:
        use_cuda (bool): Whether to use CUDA if available
        
    Returns:
        torch.device: The device to use
    """
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    if device.type == "cuda":
        cuda_device_count = torch.cuda.device_count()
        print(f"Using CUDA with {cuda_device_count} device(s)")
        for i in range(cuda_device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - VRAM: {props.total_memory / (1024**3):.2f} GB")
    else:
        print("Using CPU")
    
    return device

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
