"""
Training module for TCPGS method and baseline diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDiffusionModel(nn.Module):
    """
    Baseline diffusion model that performs simple denoising.
    
    This model serves as a baseline for comparison with the TCPGS method.
    """
    def __init__(self):
        super(BaseDiffusionModel, self).__init__()
        # Simple convolutional architecture for denoising
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        """Forward pass for denoising."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # Keep output in [0, 1] range
        return x
    
    def denoise_step(self, images, step, total_steps):
        """Perform a single denoising step in the diffusion process."""
        return self.forward(images)
    
    def denoise_with_grad(self, images):
        """
        Denoise images and compute gradients with respect to input.
        
        Returns:
            Tuple of (denoised_images, gradients)
        """
        images.requires_grad_()
        out = self.forward(images)
        loss = out.mean()
        grad = torch.autograd.grad(loss, images, create_graph=False)[0]
        return out, grad


class TCPGSDiffusionModel(nn.Module):
    """
    Tweedie-Consistent Projection Gradient Sampler (TCPGS) diffusion model.
    
    This model extends the baseline diffusion model by incorporating:
    1. Gradient estimation through projection
    2. Tweedie-based consistency correction
    """
    def __init__(self, use_consistency=True, consistency_weight=0.05):
        super(TCPGSDiffusionModel, self).__init__()
        self.use_consistency = use_consistency
        self.consistency_weight = consistency_weight
        
        # Convolutional layers for denoising
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # Additional layers for Tweedie correction
        if use_consistency:
            self.tweedie_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass with optional Tweedie consistency correction.
        """
        # Base denoising
        features = F.relu(self.conv1(x))
        features = F.relu(self.conv2(features))
        denoised = torch.sigmoid(self.conv3(features))
        
        # Apply Tweedie-based correction if enabled
        if self.use_consistency:
            # Tweedie correction based on input and denoised output
            tweedie_correction = torch.sigmoid(
                self.tweedie_conv(x)
            ) * self.consistency_weight
            
            # Apply correction
            denoised = torch.clamp(denoised + tweedie_correction, 0, 1)
            
        return denoised
    
    def denoise_step(self, images, step, total_steps):
        """
        Perform a single denoising step with Tweedie correction.
        
        The step parameter allows for potential step-dependent behavior.
        """
        # Adjust consistency weight based on step progress
        if self.use_consistency:
            # Gradually reduce consistency influence as steps progress
            step_factor = 1.0 - (step / total_steps)
            temp_weight = self.consistency_weight * step_factor
            old_weight = self.consistency_weight
            
            # Temporarily change weight for this step
            self.consistency_weight = temp_weight
            denoised = self.forward(images)
            
            # Restore original weight
            self.consistency_weight = old_weight
            return denoised
        else:
            return self.forward(images)
    
    def denoise_with_grad(self, images):
        """
        Denoise images and compute gradients with Tweedie correction.
        
        Returns:
            Tuple of (denoised_images, gradients)
        """
        images.requires_grad_()
        out = self.forward(images)
        
        # Compute loss and gradients
        loss = out.mean()
        grad = torch.autograd.grad(loss, images, create_graph=False)[0]
        
        return out, grad


def train_model(model, train_loader, epochs=1, device=torch.device("cuda")):
    """
    Train the diffusion model (simplified for this implementation).
    
    Note: In a real-world scenario, this would involve more complex
    training with proper loss functions specific to diffusion models.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Add noise to create noisy targets
            noisy_data = add_gaussian_noise(data, std=0.1).to(device)
            
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_epoch_loss:.6f}")
    
    return model


# Import here to avoid circular imports
from src.preprocess import add_gaussian_noise
