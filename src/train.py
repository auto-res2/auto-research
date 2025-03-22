"""Training models for CSTD experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from config import cstd_config as cfg
from src.preprocess import implant_trigger, add_gaussian_noise, get_trigger_patch

class Denoiser(nn.Module):
    """Simple CNN-based denoiser."""
    def __init__(self):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class IterativeRefiner(nn.Module):
    """Network for iterative refinement of trigger estimation."""
    def __init__(self, num_steps=5, threshold=1e-3):
        super(IterativeRefiner, self).__init__()
        self.num_steps = num_steps
        self.threshold = threshold
        # A simple scoring network
        self.score_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Initialize trigger estimate as zeros
        trigger_est = torch.zeros_like(x)
        history = []
        for step in range(self.num_steps):
            delta = self.score_net(x - trigger_est)
            new_est = trigger_est + delta
            history.append(new_est)
            # Check convergence with L2 difference
            if torch.mean((new_est - trigger_est)**2) < self.threshold:
                print(f"Converged at step: {step+1}")
                trigger_est = new_est
                break
            trigger_est = new_est
        return trigger_est, history

class FullDiffusionModel(nn.Module):
    """Placeholder for the full diffusion defense model."""
    def __init__(self):
        super(FullDiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

class DistilledGenerator(nn.Module):
    """Lighter network for the distilled generator."""
    def __init__(self):
        super(DistilledGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

def consistency_loss(out1, out2):
    """Calculate consistency loss between two outputs."""
    return torch.mean((out1 - out2) ** 2)

def double_pass_denoise(model, image, sigma1, sigma2):
    """Perform double pass denoising with two noise levels."""
    # First pass: with noise level sigma1
    noisy_input_1 = add_gaussian_noise(image, sigma1)
    out1 = model(noisy_input_1)
    # Second pass: further refine with a different noise level sigma2
    noisy_input_2 = add_gaussian_noise(out1, sigma2)
    out2 = model(noisy_input_2)
    return out1, out2

def train_denoiser(train_loader, device, epochs=10, sigma1=0.1, sigma2=0.05, consistency_weight=0.5):
    """Train denoiser model for Experiment 1: Ambient-Consistent Trigger Estimation."""
    print("\n=== Experiment 1: Ambient-Consistent Trigger Estimation Under Noisy Conditions ===")
    model = Denoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.EXP1_LEARNING_RATE)
    criterion = nn.MSELoss()

    # Create trigger and mask (the shape is (1,3,32,32) for CIFAR images)
    trigger, mask = get_trigger_patch((1, 3, 32, 32), patch_size=cfg.TRIGGER_PATCH_SIZE)
    trigger = trigger.to(device)
    mask = mask.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        for images, _ in train_loader:
            images = images.to(device)
            # Create an attacked image by implanting trigger
            images_attacked = implant_trigger(images.clone(), trigger, mask, ratio=cfg.TRIGGER_IMPLANT_RATIO)

            optimizer.zero_grad()
            out1, out2 = double_pass_denoise(model, images_attacked, sigma1, sigma2)
            rec_loss = criterion(out2, images)  # target is the clean image
            cons_loss = consistency_loss(out1, out2)
            loss = rec_loss + consistency_weight * cons_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

            # For a quick test, run only one batch
            if cfg.TEST_MODE:
                break

        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1}/{epochs} | Average Loss: {running_loss/batch_count:.4f} | Time: {epoch_time:.2f}s")
        
    # Evaluation on a test batch
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(train_loader))
        images = images.to(device)
        images_attacked = implant_trigger(images.clone(), trigger, mask, ratio=cfg.TRIGGER_IMPLANT_RATIO)
        _, out2 = double_pass_denoise(model, images_attacked, sigma1, sigma2)
        mse_error = criterion(out2, images).item()
    print(f"Reconstruction MSE on test batch: {mse_error:.6f}")
    
    return model, trigger, mask

def train_refiner(train_loader, device, epochs=10, num_steps=10, threshold=1e-4):
    """Train refiner for Experiment 2: Sequential Score-Based Trigger Refinement."""
    print("\n=== Experiment 2: Sequential Score-Based Trigger Refinement and Adaptivity ===")
    refiner = IterativeRefiner(num_steps=num_steps, threshold=threshold).to(device)
    optimizer = optim.Adam(refiner.parameters(), lr=cfg.EXP2_LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Use the same trigger as in Experiment 1
    trigger, mask = get_trigger_patch((1, 3, 32, 32), patch_size=cfg.TRIGGER_PATCH_SIZE)
    trigger = trigger.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        for images, _ in train_loader:
            images = images.to(device)
            images_attacked = implant_trigger(images.clone(), trigger, mask, ratio=cfg.TRIGGER_IMPLANT_RATIO)
            
            optimizer.zero_grad()
            refined_trigger, history = refiner(images_attacked)
            loss = criterion(refined_trigger, trigger.expand_as(refined_trigger))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

            # For testing, run only one batch
            if cfg.TEST_MODE:
                break
                
        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1}/{epochs} | Loss: {running_loss/batch_count:.4f} | Time: {epoch_time:.2f}s")
    
    return refiner, trigger, mask

def train_distilled_model(train_loader, device, epochs=10):
    """Train distilled model for Experiment 3: Fast Defense Distillation."""
    print("\n=== Experiment 3: Efficiency Gains from Fast Defense Distillation ===")
    full_model = FullDiffusionModel().to(device)
    distilled_model = DistilledGenerator().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(distilled_model.parameters(), lr=cfg.EXP3_LEARNING_RATE)

    for epoch in range(epochs):
        full_model.eval()
        distilled_model.train()
        running_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        for images, _ in train_loader:
            images = images.to(device)
            
            # Get full model output as target
            with torch.no_grad():
                target = full_model(images)
                
            optimizer.zero_grad()
            output = distilled_model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            
            # For testing, run only one batch
            if cfg.TEST_MODE:
                break
                
        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1}/{epochs} | Loss: {running_loss/batch_count:.4f} | Time: {epoch_time:.2f}s")
    
    return full_model, distilled_model
