import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def get_cifar10_data():
    """Get CIFAR-10 dataset with standard preprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform)
    
    return trainset, testset

def get_dataloaders(batch_size=128):
    """Create DataLoaders for training and testing."""
    trainset, testset = get_cifar10_data()
    
    trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size,
                          shuffle=False, num_workers=2)
    
    return trainloader, testloader
