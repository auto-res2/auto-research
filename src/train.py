"""
Model training module for A2Diff experiments.
"""
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DummyDiffusionModel:
    """
    Dummy diffusion model that simulates sampling by sleeping a little time per step
    and returning a random dummy tensor.
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def sample(self, schedule, batch_size):
        """
        Simulate sampling by sleeping a little time per step and returning random tensors.
        
        Args:
            schedule: Sampling schedule (list of steps)
            batch_size: Number of samples to generate
            
        Returns:
            samples: Generated samples
            total_steps: Total number of steps taken
        """
        total_steps = sum(schedule)
        # Simulate computation: sleep 0.005 sec per step for demonstration
        time.sleep(0.005 * total_steps)
        # Generate a dummy sample tensor; e.g., images of shape (batch_size, 3, 32, 32)
        samples = torch.randn(batch_size, 3, 32, 32, device=self.device)
        return samples, total_steps

class SeverityEstimator(nn.Module):
    """
    A simple convolutional network that estimates the severity of the noise level
    in the input image.
    """
    def __init__(self):
        super(SeverityEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass through the severity estimator.
        
        Args:
            x: Input image
            
        Returns:
            severity: Estimated severity score
        """
        return self.net(x).squeeze()

def init_models():
    """
    Initialize the models for the experiment.
    
    Returns:
        diffusion_model: Dummy diffusion model
        severity_estimator: Severity estimator network
    """
    # Initialize diffusion model
    diffusion_model = DummyDiffusionModel()
    
    # Initialize severity estimator
    severity_estimator = SeverityEstimator()
    if torch.cuda.is_available():
        severity_estimator = severity_estimator.cuda()
    severity_estimator.eval()
    
    return diffusion_model, severity_estimator

def get_schedules(severity_score, config):
    """
    Get the different schedules for the experiment.
    
    Args:
        severity_score: Estimated severity score
        config: Experiment configuration
        
    Returns:
        fixed_schedule: Fixed schedule
        adapted_schedule: Adapted schedule
        fixed_extra_schedule: Fixed schedule with extra steps
    """
    # Fixed schedule from AYS
    base_schedule = config['model']['base_schedule']
    
    # Adapted schedule based on severity
    if severity_score > config['model']['severity_threshold']:
        adapted_schedule = [t+1 for t in base_schedule]
        print(f"Adaptive schedule chosen (severity={severity_score:.3f}): {adapted_schedule}")
    else:
        adapted_schedule = base_schedule
        print(f"Adaptive schedule chosen (severity={severity_score:.3f}): {adapted_schedule}")
    
    # Fixed schedule with extra steps to match the adapted budget
    fixed_extra_schedule = [t+1 for t in base_schedule]
    
    return base_schedule, adapted_schedule, fixed_extra_schedule

def random_dynamic_schedule(base_schedule):
    """
    Generate a random dynamic schedule.
    
    Args:
        base_schedule: Base schedule
        
    Returns:
        schedule: Random dynamic schedule
    """
    rand_score = np.random.uniform(0, 1)
    if rand_score > 0.5:
        schedule = [t+1 for t in base_schedule]
        print(f"Random adaptation (score={rand_score:.3f}): {schedule}")
    else:
        schedule = base_schedule
        print(f"Random adaptation (score={rand_score:.3f}): {base_schedule}")
    
    return schedule
