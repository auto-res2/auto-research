"""
Training module for SPCDD MRI Super-Resolution.

This module handles the training of diffusion models and progressive distillation.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from src.preprocess import AnatomyExtractor
from src.utils.metrics import calculate_psnr, calculate_ssim

class DiffusionModel(nn.Module):
    """Conditional diffusion model that can use an anatomical prior."""
    
    def __init__(self, use_anatomy_prior=False, channels=[1, 64, 64, 1]):
        """
        Initialize the diffusion model.
        
        Args:
            use_anatomy_prior: Whether to use the anatomical prior as input
            channels: List of channel dimensions [input, hidden1, hidden2, output]
        """
        super(DiffusionModel, self).__init__()
        self.use_anatomy_prior = use_anatomy_prior
        
        self.input_channels = channels[0] + (1 if use_anatomy_prior else 0)
        
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            
            *[nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.ReLU()
            ) for i in range(1, len(channels)-2)],
            
            nn.Conv2d(channels[-2], channels[-1], kernel_size=3, padding=1)
        )
    
    def forward(self, x, anatomy_prior=None):
        """
        Forward pass through the diffusion model.
        
        Args:
            x: Input image (1.5T MRI)
            anatomy_prior: Anatomical prior (optional)
            
        Returns:
            Generated 7T-like image
        """
        if self.use_anatomy_prior and (anatomy_prior is not None):
            x = torch.cat((x, anatomy_prior), dim=1)
        return self.network(x)

class IntensityModulationModule(nn.Module):
    """Module for modulating intensities in the diffusion model."""
    
    def __init__(self, channels=[1, 32, 32, 1]):
        """
        Initialize the intensity modulation module.
        
        Args:
            channels: List of channel dimensions [input, hidden1, hidden2, output]
        """
        super(IntensityModulationModule, self).__init__()
        
        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1))
            if i < len(channels)-2:  # No activation after last layer
                layers.append(nn.ReLU())
                
        self.intensity_net = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the intensity modulation module."""
        modulation = self.intensity_net(x)
        return modulation

class DiffusionModelWithIntensity(nn.Module):
    """Diffusion model that can use an intensity modulation pathway."""
    
    def __init__(self, use_intensity_modulation=False, channels=[1, 64, 64, 1]):
        """
        Initialize the diffusion model with intensity modulation.
        
        Args:
            use_intensity_modulation: Whether to use intensity modulation
            channels: List of channel dimensions
        """
        super(DiffusionModelWithIntensity, self).__init__()
        self.use_intensity_modulation = use_intensity_modulation
        
        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1))
            if i < len(channels)-2:  # No activation after last layer
                layers.append(nn.ReLU())
                
        self.main_branch = nn.Sequential(*layers)
        
        if self.use_intensity_modulation:
            self.intensity_module = IntensityModulationModule(channels=[channels[0], 32, 32, 1])
    
    def forward(self, x):
        """Forward pass through the diffusion model with intensity modulation."""
        main_output = self.main_branch(x)
        
        if self.use_intensity_modulation:
            intensity_mod = self.intensity_module(x)
            output = main_output + intensity_mod  # Fusion by addition
        else:
            output = main_output
            
        return output

class TeacherModel(nn.Module):
    """Teacher model incorporating both anatomical extraction and intensity modulation."""
    
    def __init__(self, channels=[2, 64, 64, 1]):
        """
        Initialize the teacher model.
        
        Args:
            channels: List of channel dimensions
        """
        super(TeacherModel, self).__init__()
        self.anatomy_extractor = AnatomyExtractor()
        self.intensity_module = IntensityModulationModule()
        
        main_layers = []
        input_ch = channels[0]  # Input + anatomy prior
        
        for i in range(1, len(channels)):
            main_layers.append(nn.Conv2d(input_ch, channels[i], kernel_size=3, padding=1))
            if i < len(channels)-1:  # No activation after last layer
                main_layers.append(nn.ReLU())
            input_ch = channels[i]
            
        self.main_branch = nn.Sequential(*main_layers)
    
    def forward(self, x):
        """
        Forward pass through the teacher model.
        
        Returns:
            output: Final output
            main_output: Intermediate output (for distillation)
        """
        anatomy_prior = self.anatomy_extractor(x)
        x_cat = torch.cat((x, anatomy_prior), dim=1)
        main_output = self.main_branch(x_cat)
        intensity_mod = self.intensity_module(x)
        output = main_output + intensity_mod
        
        return output, main_output

class StudentModel(nn.Module):
    """Compact student model for distillation."""
    
    def __init__(self, channels=[1, 32, 32, 1]):
        """
        Initialize the student model.
        
        Args:
            channels: List of channel dimensions
        """
        super(StudentModel, self).__init__()
        
        main_layers = []
        for i in range(len(channels)-1):
            main_layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1))
            if i < len(channels)-2:  # No activation after last layer
                main_layers.append(nn.ReLU())
                
        self.main_branch = nn.Sequential(*main_layers)
    
    def forward(self, x):
        """
        Forward pass through the student model.
        
        Returns:
            output: Final output
            intermediate: Intermediate output (for distillation)
        """
        output = self.main_branch(x)
        
        return output, output

def gradient_loss(output, target):
    """Calculate gradient loss to capture local intensity differences."""
    grad_x_output = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
    grad_y_output = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
    
    grad_x_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    grad_y_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    
    loss = torch.mean(torch.abs(grad_x_output - grad_x_target)) + \
           torch.mean(torch.abs(grad_y_output - grad_y_target))
    
    return loss

def distillation_loss(student_out, teacher_out, student_feat, teacher_feat, alpha=0.5, beta=0.5):
    """Calculate combined distillation loss."""
    loss_out = nn.L1Loss()(student_out, teacher_out)
    loss_feat = nn.MSELoss()(student_feat, teacher_feat)
    return alpha * loss_out + beta * loss_feat

def train_ablation_model(config, train_loader, val_loader, use_anatomy_prior=True):
    """
    Train a model for the ablation study (with or without anatomical prior).
    
    Args:
        config: Configuration dictionary
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        use_anatomy_prior: Whether to use anatomical prior
        
    Returns:
        model: Trained model
        extractor: Trained anatomy extractor (if used)
        metrics: Dictionary of training and validation metrics
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    model = DiffusionModel(use_anatomy_prior=use_anatomy_prior, 
                          channels=config.diffusion_channels).to(device)
    
    extractor = None
    if use_anatomy_prior:
        extractor = AnatomyExtractor(
            in_channels=1, 
            hidden_channels=config.anatomy_extractor_channels[1]
        ).to(device)
        optimizer = optim.Adam(
            list(model.parameters()) + list(extractor.parameters()), 
            lr=config.diffusion_lr
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.diffusion_lr)
        
    criterion = nn.L1Loss()
    
    metrics = {
        'train_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }
    
    for epoch in range(config.num_epochs['ablation']):
        model.train()
        if extractor is not None:
            extractor.train()
            
        train_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs['ablation']}")):
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            optimizer.zero_grad()
            
            if extractor is not None:
                prior = extractor(img_15T)
                output = model(img_15T, anatomy_prior=prior)
            else:
                output = model(img_15T, anatomy_prior=None)
            
            loss = criterion(output, target_7T)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        model.eval()
        if extractor is not None:
            extractor.eval()
            
        val_psnr, val_ssim = evaluate_model(model, extractor, val_loader, device)
        metrics['val_psnr'].append(val_psnr)
        metrics['val_ssim'].append(val_ssim)
        
        print(f"Epoch {epoch+1}/{config.num_epochs['ablation']}: Train Loss = {avg_train_loss:.4f}, Val PSNR = {val_psnr:.2f}, Val SSIM = {val_ssim:.4f}")
    
    return model, extractor, metrics

def train_intensity_model(config, train_loader, val_loader, use_intensity_modulation=True):
    """
    Train a model with or without intensity modulation.
    
    Args:
        config: Configuration dictionary
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        use_intensity_modulation: Whether to use intensity modulation
        
    Returns:
        model: Trained model
        metrics: Dictionary of training and validation metrics
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    model = DiffusionModelWithIntensity(
        use_intensity_modulation=use_intensity_modulation,
        channels=config.diffusion_channels
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.diffusion_lr)
    criterion = nn.L1Loss()
    
    metrics = {
        'train_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }
    
    for epoch in range(config.num_epochs['intensity']):
        model.train()
        train_loss = 0.0
        
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs['intensity']}")):
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            optimizer.zero_grad()
            output = model(img_15T)
            
            l1_loss = criterion(output, target_7T)
            grad_loss_val = gradient_loss(output, target_7T)
            loss = l1_loss + config.gradient_loss_weight * grad_loss_val
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        model.eval()
        val_psnr, val_ssim = evaluate_intensity_model(model, val_loader, device)
        metrics['val_psnr'].append(val_psnr)
        metrics['val_ssim'].append(val_ssim)
        
        print(f"Epoch {epoch+1}/{config.num_epochs['intensity']}: Train Loss = {avg_train_loss:.4f}, Val PSNR = {val_psnr:.2f}, Val SSIM = {val_ssim:.4f}")
    
    return model, metrics

def train_distillation(config, train_loader, val_loader):
    """
    Train a student model using distillation from a teacher model.
    
    Args:
        config: Configuration dictionary
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        
    Returns:
        student: Trained student model
        metrics: Dictionary of training and validation metrics
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    teacher = TeacherModel(channels=config.teacher_channels).to(device)
    student = StudentModel(channels=config.student_channels).to(device)
    
    teacher.eval()
    
    optimizer = optim.Adam(student.parameters(), lr=config.student_lr)
    
    metrics = {
        'train_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }
    
    for epoch in range(config.num_epochs['distillation']):
        student.train()
        train_loss = 0.0
        
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs['distillation']}")):
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_output, teacher_features = teacher(img_15T)
            
            student_output, student_features = student(img_15T)
            
            loss = distillation_loss(
                student_output, teacher_output, 
                student_features, teacher_features,
                alpha=config.distillation_alpha,
                beta=config.distillation_beta
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        student.eval()
        val_psnr, val_ssim = evaluate_student_model(student, val_loader, device)
        metrics['val_psnr'].append(val_psnr)
        metrics['val_ssim'].append(val_ssim)
        
        print(f"Epoch {epoch+1}/{config.num_epochs['distillation']}: Train Loss = {avg_train_loss:.4f}, Val PSNR = {val_psnr:.2f}, Val SSIM = {val_ssim:.4f}")
    
    return student, metrics

def evaluate_model(model, extractor, val_loader, device):
    """
    Evaluate a model on the validation set.
    
    Args:
        model: The model to evaluate
        extractor: The anatomy extractor (if used)
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        avg_psnr: Average PSNR on validation set
        avg_ssim: Average SSIM on validation set
    """
    model.eval()
    if extractor is not None:
        extractor.eval()
        
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for data in val_loader:
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            if extractor is not None:
                prior = extractor(img_15T)
                output = model(img_15T, anatomy_prior=prior)
            else:
                output = model(img_15T, anatomy_prior=None)
            
            output_np = output.cpu().numpy()
            target_np = target_7T.cpu().numpy()
            
            for i in range(output_np.shape[0]):
                psnr = calculate_psnr(output_np[i, 0], target_np[i, 0])
                ssim = calculate_ssim(output_np[i, 0], target_np[i, 0])
                psnr_values.append(psnr)
                ssim_values.append(ssim)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return avg_psnr, avg_ssim

def evaluate_intensity_model(model, val_loader, device):
    """
    Evaluate an intensity modulation model on the validation set.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        avg_psnr: Average PSNR on validation set
        avg_ssim: Average SSIM on validation set
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for data in val_loader:
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            output = model(img_15T)
            
            output_np = output.cpu().numpy()
            target_np = target_7T.cpu().numpy()
            
            for i in range(output_np.shape[0]):
                psnr = calculate_psnr(output_np[i, 0], target_np[i, 0])
                ssim = calculate_ssim(output_np[i, 0], target_np[i, 0])
                psnr_values.append(psnr)
                ssim_values.append(ssim)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return avg_psnr, avg_ssim

def evaluate_student_model(model, val_loader, device):
    """
    Evaluate a student model on the validation set.
    
    Args:
        model: The student model to evaluate
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        avg_psnr: Average PSNR on validation set
        avg_ssim: Average SSIM on validation set
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for data in val_loader:
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            output, _ = model(img_15T)
            
            output_np = output.cpu().numpy()
            target_np = target_7T.cpu().numpy()
            
            for i in range(output_np.shape[0]):
                psnr = calculate_psnr(output_np[i, 0], target_np[i, 0])
                ssim = calculate_ssim(output_np[i, 0], target_np[i, 0])
                psnr_values.append(psnr)
                ssim_values.append(ssim)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return avg_psnr, avg_ssim

def measure_inference(model, sample_input, device, iterations=10):
    """
    Measure inference speed and memory usage of a model.
    
    Args:
        model: The model to measure
        sample_input: Sample input tensor
        device: Device to run measurement on
        iterations: Number of iterations to average over
        
    Returns:
        avg_time: Average inference time
        memory_used: Memory used during inference
    """
    model.eval()
    sample_input = sample_input.to(device)
    
    with torch.no_grad():
        _ = model(sample_input)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(sample_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    
    memory_used = 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
        memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    return avg_time, memory_used
