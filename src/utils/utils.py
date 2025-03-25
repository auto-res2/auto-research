import os
import random
import numpy as np
import torch
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device(gpu_id=0):
    """Get device (CPU or GPU)."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    else:
        return torch.device('cpu')

def get_optimizer(model, config):
    """Get optimizer based on configuration."""
    if config['training']['optimizer'].lower() == 'adam':
        return optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'].lower() == 'sgd':
        return optim.SGD(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")

def get_scheduler(optimizer, config, num_epochs):
    """Get learning rate scheduler based on configuration."""
    if config['training']['scheduler'].lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif config['training']['scheduler'].lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif config['training']['scheduler'].lower() == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {config['training']['scheduler']}")

def get_dataloader(config):
    """
    Get data loaders for dataset specified in config.
    
    Returns:
        train_loader, test_loader
    """
    if config['dataset']['name'].lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.CIFAR10(
            root=config['dataset']['data_dir'],
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=config['dataset']['data_dir'],
            train=False,
            download=True,
            transform=transform
        )
    elif config['dataset']['name'].lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.CIFAR100(
            root=config['dataset']['data_dir'],
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.CIFAR100(
            root=config['dataset']['data_dir'],
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']['name']}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers']
    )
    
    return train_loader, test_loader

def save_model(model, epoch, save_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {load_path} (epoch {epoch})")
    return epoch

def save_images(images, save_path, nrow=8):
    """Save a batch of images as a grid."""
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
    # Convert from [C, H, W] to [H, W, C] for matplotlib
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Images saved to {save_path}")

def plot_training_progress(twist_losses, sid_losses, consistency_losses, save_dir, timestamp):
    """Plot training progress and save to file."""
    plt.figure(figsize=(12, 8))
    
    # Plot main losses
    plt.subplot(2, 1, 1)
    plt.plot(twist_losses, label='TwiST Loss')
    plt.plot(sid_losses, label='SID Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot consistency loss
    plt.subplot(2, 1, 2)
    plt.plot(consistency_losses, label='Consistency Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Consistency Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_progress_{timestamp}.png"))
    plt.close()
    print(f"Training progress plot saved to {os.path.join(save_dir, f'training_progress_{timestamp}.png')}")
