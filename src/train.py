"""
Training module for Score-Aligned Step Distillation experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from src.utils.diffusion import compute_kl_loss, compute_score_loss
import matplotlib.pyplot as plt


class DiffusionModel(nn.Module):
    """
    Simple CNN-based diffusion model with learnable or fixed schedule.
    """
    def __init__(self, learnable_schedule=True, num_steps=10, in_channels=3, 
                 hidden_channels=64, image_size=32):
        """
        Initialize the diffusion model.
        
        Args:
            learnable_schedule: Whether to use a learnable schedule
            num_steps: Number of diffusion steps
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
            image_size: Size of input images
        """
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        self.image_size = image_size
        
        # Convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        
        # Output head
        self.head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        
        # Schedule parameter
        if learnable_schedule:
            # Initialize a learnable schedule parameter linearly spaced from 1.0 to 0.1
            self.schedule = nn.Parameter(torch.linspace(1.0, 0.1, steps=num_steps))
        else:
            # Fixed schedule registered as a buffer
            self.register_buffer('schedule', torch.linspace(1.0, 0.1, steps=num_steps))
    
    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            t: Time step indices [B]
        
        Returns:
            torch.Tensor: Model output
        """
        # Get current noise level from schedule
        if isinstance(t, int):
            current_noise = self.schedule[t]
        else:
            # Handle batched timesteps
            current_noise = self.schedule[t.long()]
            # Reshape for broadcasting
            current_noise = current_noise.view(-1, 1, 1, 1)
        
        # Pass through backbone and head
        features = self.backbone(x)
        out = self.head(features) * current_noise
        
        return out


def teacher_score_estimate(x):
    """
    Dummy teacher score estimation function for Score Distillation.
    In a real implementation, this would be a more complex model.
    
    Args:
        x: Input tensor
    
    Returns:
        torch.Tensor: Estimated score
    """
    return x * 0.5


def train_model(model, dataloader, optimizer, lambda_value=0.5, num_epochs=10, 
                device='cuda', save_dir='./models', experiment_name='experiment'):
    """
    Train the diffusion model with SASD.
    
    Args:
        model: DiffusionModel to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        lambda_value: Weight for the score loss term in the dual loss
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints
        experiment_name: Name of the experiment
    
    Returns:
        dict: Training results containing loss history and final schedule
    """
    model.to(device)
    loss_history = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        
        start_time = time.time()
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # For demonstration, sample a random timestep
            t = torch.randint(0, model.num_steps, (images.size(0),), device=device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(images, t)
            
            # KL loss between output and original image
            loss_kl = compute_kl_loss(output, images)
            
            # Score loss using teacher model
            teacher_score = teacher_score_estimate(images)
            loss_score = compute_score_loss(output, teacher_score)
            
            # Combined loss with lambda weight
            total_loss = loss_kl + lambda_value * loss_score
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {total_loss.item():.4f}, Time: {time.time() - start_time:.2f}s")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} complete. Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"{experiment_name}_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'schedule': model.schedule.detach().cpu().numpy(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Record final schedule
    final_schedule = model.schedule.detach().cpu().numpy()
    print(f"Final schedule values: {final_schedule}")
    
    return {
        'loss_history': loss_history,
        'final_schedule': final_schedule
    }


def experiment1(config, device='cuda'):
    """
    Experiment 1: Ablation Study on the Dual-Loss Objective
    
    Args:
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        dict: Experiment results
    """
    from src.preprocess import prepare_data
    
    print("\n=== Experiment 1: Ablation Study on the Dual-Loss Objective ===")
    
    # Get experiment configuration
    exp_config = config['EXPERIMENT_CONFIG']['experiment1']
    lambda_values = exp_config.get('lambda_values', [0.0, 0.1, 0.5, 1.0])
    num_epochs = exp_config.get('num_epochs', 10)
    batch_size = exp_config.get('batch_size', 64)
    
    # Prepare data
    _, train_loader = prepare_data(config, 'cifar10', train=True)
    
    results = {}  # will hold loss histories per lambda
    
    for lam in lambda_values:
        print(f"\n--- Training with λ = {lam} ---")
        
        # Create a new model instance with learnable schedule
        model = DiffusionModel(learnable_schedule=True, num_steps=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train the model
        result = train_model(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            lambda_value=lam,
            num_epochs=num_epochs,
            device=device,
            save_dir='./models',
            experiment_name=f'experiment1_lambda{lam}'
        )
        
        results[f'lambda_{lam}'] = result
    
    return results


def experiment2(config, device='cuda'):
    """
    Experiment 2: Comparison of Learnable Schedule Versus Fixed Schedule
    
    Args:
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        dict: Experiment results
    """
    from src.preprocess import prepare_data
    
    print("\n=== Experiment 2: Learnable Schedule Versus Fixed Schedule ===")
    
    # Get experiment configuration
    exp_config = config['EXPERIMENT_CONFIG']['experiment2']
    num_epochs = exp_config.get('num_epochs', 10)
    batch_size = exp_config.get('batch_size', 64)
    
    # Prepare data
    _, train_loader = prepare_data(config, 'cifar10', train=True)
    
    configurations = {"learnable": True, "fixed": False}
    results = {}
    
    for config_name, learnable_flag in configurations.items():
        print(f"\n--- Training with {config_name} schedule ---")
        
        # Create model with learnable or fixed schedule
        model = DiffusionModel(learnable_schedule=learnable_flag, num_steps=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train the model
        result = train_model(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            lambda_value=0.5,  # Fixed lambda for this experiment
            num_epochs=num_epochs,
            device=device,
            save_dir='./models',
            experiment_name=f'experiment2_{config_name}'
        )
        
        results[config_name] = result
    
    return results


def experiment3(config, device='cuda'):
    """
    Experiment 3: Step Efficiency and Robustness Across Datasets
    
    Args:
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        dict: Experiment results
    """
    from src.preprocess import prepare_data
    
    print("\n=== Experiment 3: Step Efficiency and Robustness Across Datasets ===")
    
    # Get experiment configuration
    exp_config = config['EXPERIMENT_CONFIG']['experiment3']
    num_epochs = exp_config.get('num_epochs', 5)
    batch_size = exp_config.get('batch_size', 64)
    step_configs = exp_config.get('step_configs', [5, 10, 25])
    datasets_to_run = exp_config.get('datasets', ["cifar10", "celeba"])
    
    results = {}
    
    for dataset_name in datasets_to_run:
        print(f"\n--- Processing dataset: {dataset_name} ---")
        
        # Prepare data
        _, loader = prepare_data(config, dataset_name, train=True)
        
        # Set image size and channels based on dataset
        if dataset_name.lower() == 'cifar10':
            image_size = 32
            in_channels = 3
        elif dataset_name.lower() == 'celeba':
            image_size = 64
            in_channels = 3
        
        for steps in step_configs:
            print(f"\nTraining SASD model with {steps} diffusion steps on {dataset_name}")
            
            # Create model with specified number of steps
            model = DiffusionModel(
                learnable_schedule=True,
                num_steps=steps,
                in_channels=in_channels,
                image_size=image_size
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Train the model
            result = train_model(
                model=model,
                dataloader=loader,
                optimizer=optimizer,
                lambda_value=0.5,  # Fixed lambda for this experiment
                num_epochs=num_epochs,
                device=device,
                save_dir='./models',
                experiment_name=f'experiment3_{dataset_name}_steps{steps}'
            )
            
            key = f"{dataset_name}_steps{steps}"
            results[key] = result
    
    return results


def run_test_experiments(config, device='cuda'):
    """
    Run quick tests for each experiment to verify functionality.
    
    Args:
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        dict: Test results
    """
    print("\n#############################################")
    print("Running quick tests for each experiment...")
    print("#############################################")
    
    # Use test configuration
    test_config = config.copy()
    test_config['EXPERIMENT_CONFIG'] = {
        'experiment1': {
            'lambda_values': [0.0, 0.1],
            'num_epochs': 1,
            'batch_size': 32,
        },
        'experiment2': {
            'num_epochs': 1,
            'batch_size': 32,
        },
        'experiment3': {
            'num_epochs': 1,
            'batch_size': 32,
            'step_configs': [5, 10],
            'datasets': ['cifar10'],
        },
    }
    
    # Run quick tests for each experiment
    results = {}
    
    # Test Experiment 1
    print("\n[TEST] Experiment 1 (Ablation Study)")
    results['experiment1'] = experiment1(test_config, device)
    
    # Test Experiment 2
    print("\n[TEST] Experiment 2 (Learnable vs Fixed Schedule)")
    results['experiment2'] = experiment2(test_config, device)
    
    # Test Experiment 3
    print("\n[TEST] Experiment 3 (Step Efficiency and Robustness)")
    results['experiment3'] = experiment3(test_config, device)
    
    print("\nAll tests completed!")
    
    return results
