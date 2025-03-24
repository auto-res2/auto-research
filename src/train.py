#!/usr/bin/env python3
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import itertools
from torch.utils.tensorboard import SummaryWriter

# Set fixed random seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Dummy implementation of a FID calculation function
def calculate_dummy_fid(images, dataset='cifar10'):
    """
    Dummy implementation of FID calculation for demonstration purposes.
    In a real implementation, use a proper FID calculator.
    
    Args:
        images (numpy.ndarray): Batch of images
        dataset (str): Name of the dataset
        
    Returns:
        float: Dummy FID score
    """
    # Here we simply return a random FID value (to be replaced by a true implementation)
    return np.random.uniform(10.0, 50.0)

# Generator network
class Generator(nn.Module):
    """
    Generator network that maps from a latent vector to an image.
    """
    def __init__(self, latent_dim=100, image_channels=3, image_size=32):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        
        # Calculate feature map dimensions
        self.initial_size = image_size // 4
        self.initial_features = 256
        
        # Projection and reshape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.initial_features * self.initial_size * self.initial_size),
            nn.ReLU(inplace=True)
        )
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(self.initial_features, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, image_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z (torch.Tensor): Latent code tensor
            
        Returns:
            torch.Tensor: Generated image
        """
        x = self.fc(z)
        x = x.view(-1, self.initial_features, self.initial_size, self.initial_size)
        x = self.conv_blocks(x)
        return x

# Score network for ABSD with Bayesian uncertainty weighting
class ScoreNetworkABSD(nn.Module):
    """
    Score network for ABSD with Bayesian uncertainty weighting.
    """
    def __init__(self, image_channels=3, config=None):
        super(ScoreNetworkABSD, self).__init__()
        self.config = config if config is not None else {
            'uncertainty_factor': 1.0, 
            'sde_stepsize': 0.005
        }
        
        # Score estimation network
        self.score_net = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, image_channels, kernel_size=3, padding=1)
        )
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, image_channels, kernel_size=3, padding=1),
            nn.Softplus()  # Ensures positive uncertainty values
        )
    
    def teacher_score(self, images):
        """Simulates a teacher model's score function."""
        return self.score_net(images)
    
    def generator_score(self, images):
        """Generates score for synthetic images."""
        return self.score_net(images)
    
    def compute_uncertainty(self, images):
        """Computes uncertainty weights for the Bayesian update."""
        base_uncertainty = self.uncertainty_net(images)
        return base_uncertainty * self.config['uncertainty_factor']
    
    def compute_absd_loss(self, generator, real_images, device):
        """
        Compute the ABSD loss with Bayesian uncertainty weighting.
        
        Args:
            generator (nn.Module): Generator model
            real_images (torch.Tensor): Batch of real images
            device (torch.device): Device to run computation on
            
        Returns:
            torch.Tensor: Computed loss value
        """
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        synthetic_images = generator(z)
        
        # Get scores from teacher and generator
        teacher = self.teacher_score(real_images)
        gen_score = self.generator_score(synthetic_images)
        
        # Compute uncertainty weights
        uncertainty = self.compute_uncertainty(real_images)
        
        # Bayesian weighted loss
        loss = ((teacher - gen_score) * uncertainty).pow(2).mean()
        return loss

# Score network for SiD (baseline comparison)
class ScoreNetworkSiD(nn.Module):
    """Score network for SiD (baseline comparison)."""
    def __init__(self, image_channels=3):
        super(ScoreNetworkSiD, self).__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, image_channels, kernel_size=3, padding=1)
        )
    
    def teacher_score(self, images):
        """Simulates a teacher model's score function."""
        return self.score_net(images)
    
    def generator_score(self, images):
        """Generates score for synthetic images."""
        return self.score_net(images)
    
    def compute_sid_loss(self, generator, real_images, device):
        """
        Compute the SiD loss.
        
        Args:
            generator (nn.Module): Generator model
            real_images (torch.Tensor): Batch of real images
            device (torch.device): Device to run computation on
            
        Returns:
            torch.Tensor: Computed loss value
        """
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        synthetic_images = generator(z)
        
        # Get scores from teacher and generator
        teacher = self.teacher_score(real_images)
        gen_score = self.generator_score(synthetic_images)
        
        # Standard SiD loss without uncertainty weighting
        loss = (teacher - gen_score).pow(2).mean()
        return loss

# A simplified version of ABSD for ablation studies
class ScoreNetworkABSD_Control(nn.Module):
    """A simplified version of ABSD for ablation studies."""
    def __init__(self, image_channels=3):
        super(ScoreNetworkABSD_Control, self).__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, image_channels, kernel_size=3, padding=1)
        )
    
    def teacher_score(self, images):
        """Simulates a teacher model's score function."""
        return self.score_net(images)
    
    def generator_score(self, images):
        """Generates score for synthetic images."""
        return self.score_net(images)
    
    def compute_absd_loss(self, generator, real_images, device):
        """
        Compute the ABSD loss without uncertainty weighting (control version).
        
        Args:
            generator (nn.Module): Generator model
            real_images (torch.Tensor): Batch of real images
            device (torch.device): Device to run computation on
            
        Returns:
            torch.Tensor: Computed loss value
        """
        batch_size = real_images.size(0)
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        synthetic_images = generator(z)
        
        # Get scores from teacher and generator
        teacher = self.teacher_score(real_images)
        gen_score = self.generator_score(synthetic_images)
        
        # Control version without uncertainty weighting
        loss = (teacher - gen_score).pow(2).mean()
        return loss

def train_experiment(experiment_type, dataloader, config, device, log_dir='./logs'):
    """
    Train models for the specified experiment type.
    
    Args:
        experiment_type (str): Type of experiment to run ('performance', 'ablation', or 'sensitivity')
        dataloader (DataLoader): DataLoader for training data
        config (dict): Configuration parameters for the experiment
        device (torch.device): Device to run training on
        log_dir (str): Directory for TensorBoard logs
        
    Returns:
        dict: Results of the experiment
    """
    set_seeds(config.get('seed', 42))
    
    if experiment_type == 'performance':
        return train_experiment_performance(dataloader, config, device, log_dir)
    elif experiment_type == 'ablation':
        return train_experiment_ablation(dataloader, config, device, log_dir)
    elif experiment_type == 'sensitivity':
        return train_experiment_sensitivity(dataloader, config, device, log_dir)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

def train_experiment_performance(dataloader, config, device, log_dir='./logs'):
    """
    Train models for the performance benchmarking experiment.
    
    Args:
        dataloader (DataLoader): DataLoader for training data
        config (dict): Configuration parameters for the experiment
        device (torch.device): Device to run training on
        log_dir (str): Directory for TensorBoard logs
        
    Returns:
        dict: Results of the experiment
    """
    print("\n=== Experiment 1: Performance Benchmarking ===")
    
    # Configure experiment parameters
    num_epochs = config.get('num_epochs', 2)
    latent_dim = config.get('latent_dim', 100)
    image_channels = config.get('image_channels', 3)
    image_size = config.get('image_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    
    # Initialize networks for ABSD
    generator_absd = Generator(latent_dim, image_channels, image_size).to(device)
    score_net_absd = ScoreNetworkABSD(image_channels, config).to(device)
    optimizer_absd = optim.Adam(
        list(generator_absd.parameters()) + list(score_net_absd.parameters()), 
        lr=learning_rate
    )
    
    # Initialize networks for SiD (baseline)
    generator_sid = Generator(latent_dim, image_channels, image_size).to(device)
    score_net_sid = ScoreNetworkSiD(image_channels).to(device)
    optimizer_sid = optim.Adam(
        list(generator_sid.parameters()) + list(score_net_sid.parameters()), 
        lr=learning_rate
    )
    
    # TensorBoard writers
    writer_absd = SummaryWriter(log_dir=os.path.join(log_dir, 'ABSD'))
    writer_sid = SummaryWriter(log_dir=os.path.join(log_dir, 'SiD'))
    
    # Train and evaluate for each epoch
    performance_results = {'ABSD': [], 'SiD': []}
    
    for epoch in range(1, num_epochs + 1):
        # Training loop for ABSD
        generator_absd.train()
        score_net_absd.train()
        epoch_loss_absd = 0.0
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer_absd.zero_grad()
            loss = score_net_absd.compute_absd_loss(generator_absd, images, device)
            loss.backward()
            optimizer_absd.step()
            epoch_loss_absd += loss.item()
        
        end_time = time.time()
        avg_loss_absd = epoch_loss_absd / len(dataloader)
        iter_time_absd = (end_time - start_time) / len(dataloader)
        peak_memory_absd = torch.cuda.max_memory_allocated(device) / (1024*1024) if torch.cuda.is_available() else 0.0
        
        # Training loop for SiD
        generator_sid.train()
        score_net_sid.train()
        epoch_loss_sid = 0.0
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer_sid.zero_grad()
            loss = score_net_sid.compute_sid_loss(generator_sid, images, device)
            loss.backward()
            optimizer_sid.step()
            epoch_loss_sid += loss.item()
        
        end_time = time.time()
        avg_loss_sid = epoch_loss_sid / len(dataloader)
        iter_time_sid = (end_time - start_time) / len(dataloader)
        peak_memory_sid = torch.cuda.max_memory_allocated(device) / (1024*1024) if torch.cuda.is_available() else 0.0
        
        # Generate images for FID evaluation
        generator_absd.eval()
        generator_sid.eval()
        with torch.no_grad():
            z = torch.randn(64, latent_dim, device=device)
            gen_images_absd = generator_absd(z).cpu()
            gen_images_sid = generator_sid(z).cpu()
        
        # Calculate FID (in a real implementation, use a proper FID calculator)
        fid_absd = calculate_dummy_fid(gen_images_absd.numpy())
        fid_sid = calculate_dummy_fid(gen_images_sid.numpy())
        
        # Log metrics
        writer_absd.add_scalar('Loss/train', avg_loss_absd, epoch)
        writer_absd.add_scalar('Time/iter_ms', iter_time_absd * 1000, epoch)
        writer_absd.add_scalar('Memory/peak_MB', peak_memory_absd, epoch)
        writer_absd.add_scalar('FID', fid_absd, epoch)
        
        writer_sid.add_scalar('Loss/train', avg_loss_sid, epoch)
        writer_sid.add_scalar('Time/iter_ms', iter_time_sid * 1000, epoch)
        writer_sid.add_scalar('Memory/peak_MB', peak_memory_sid, epoch)
        writer_sid.add_scalar('FID', fid_sid, epoch)
        
        # Store results
        performance_results['ABSD'].append({
            'epoch': epoch,
            'loss': avg_loss_absd,
            'iter_time_ms': iter_time_absd * 1000,
            'peak_memory_MB': peak_memory_absd,
            'fid': fid_absd
        })
        
        performance_results['SiD'].append({
            'epoch': epoch,
            'loss': avg_loss_sid,
            'iter_time_ms': iter_time_sid * 1000,
            'peak_memory_MB': peak_memory_sid,
            'fid': fid_sid
        })
        
        print(f"Epoch {epoch}: ABSD Loss {avg_loss_absd:.4f}, iter_time {(iter_time_absd*1000):.2f}ms, peak_mem {peak_memory_absd:.2f}MB, FID {fid_absd:.2f}")
        print(f"Epoch {epoch}: SiD  Loss {avg_loss_sid:.4f}, iter_time {(iter_time_sid*1000):.2f}ms, peak_mem {peak_memory_sid:.2f}MB, FID {fid_sid:.2f}")
    
    # Save models
    os.makedirs('./models', exist_ok=True)
    torch.save(generator_absd.state_dict(), os.path.join('./models', 'generator_absd.pth'))
    torch.save(generator_sid.state_dict(), os.path.join('./models', 'generator_sid.pth'))
    
    return performance_results

def train_experiment_ablation(dataloader, config, device, log_dir='./logs'):
    """
    Train models for the ablation study experiment.
    
    Args:
        dataloader (DataLoader): DataLoader for training data
        config (dict): Configuration parameters for the experiment
        device (torch.device): Device to run training on
        log_dir (str): Directory for TensorBoard logs
        
    Returns:
        dict: Results of the experiment
    """
    print("\n=== Experiment 2: Ablation Study ===")
    
    # Configure experiment parameters
    num_epochs = config.get('num_epochs', 2)
    latent_dim = config.get('latent_dim', 100)
    image_channels = config.get('image_channels', 3)
    image_size = config.get('image_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    
    # Initialize models for ablation study
    generator_model = Generator(latent_dim, image_channels, image_size).to(device)
    score_net_full = ScoreNetworkABSD(image_channels, config).to(device)
    score_net_control = ScoreNetworkABSD_Control(image_channels).to(device)
    
    optimizer_full = optim.Adam(
        list(generator_model.parameters()) + list(score_net_full.parameters()), 
        lr=learning_rate
    )
    optimizer_control = optim.Adam(
        list(generator_model.parameters()) + list(score_net_control.parameters()), 
        lr=learning_rate
    )
    
    # TensorBoard writers
    writer_full = SummaryWriter(log_dir=os.path.join(log_dir, 'ABSD_Full'))
    writer_control = SummaryWriter(log_dir=os.path.join(log_dir, 'ABSD_Control'))
    
    # Train and evaluate for each epoch
    ablation_results = {'Full_ABSD': [], 'Control': []}
    
    for epoch in range(1, num_epochs + 1):
        # Training metrics
        running_loss_full = 0.0
        running_loss_control = 0.0
        grad_norm_epoch_full = []
        grad_norm_epoch_control = []
        
        # Training loop
        for images, _ in dataloader:
            images = images.to(device)
            
            # Full ABSD variant
            optimizer_full.zero_grad()
            loss_full = score_net_full.compute_absd_loss(generator_model, images, device)
            loss_full.backward()
            
            # Compute gradient norm for full ABSD
            total_norm_full = 0.0
            for p in list(generator_model.parameters()) + list(score_net_full.parameters()):
                if p.grad is not None:
                    total_norm_full += p.grad.data.norm(2).item() ** 2
            total_norm_full = total_norm_full ** 0.5
            grad_norm_epoch_full.append(total_norm_full)
            
            optimizer_full.step()
            running_loss_full += loss_full.item()
            
            # Control variant
            optimizer_control.zero_grad()
            loss_control = score_net_control.compute_absd_loss(generator_model, images, device)
            loss_control.backward()
            
            # Compute gradient norm for control
            total_norm_control = 0.0
            for p in list(generator_model.parameters()) + list(score_net_control.parameters()):
                if p.grad is not None:
                    total_norm_control += p.grad.data.norm(2).item() ** 2
            total_norm_control = total_norm_control ** 0.5
            grad_norm_epoch_control.append(total_norm_control)
            
            optimizer_control.step()
            running_loss_control += loss_control.item()
        
        # Calculate average metrics
        avg_loss_full = running_loss_full / len(dataloader)
        avg_loss_control = running_loss_control / len(dataloader)
        avg_grad_norm_full = np.mean(grad_norm_epoch_full)
        avg_grad_norm_control = np.mean(grad_norm_epoch_control)
        
        # Log metrics
        writer_full.add_scalar('Loss/train', avg_loss_full, epoch)
        writer_full.add_scalar('Gradient/norm', avg_grad_norm_full, epoch)
        
        writer_control.add_scalar('Loss/train', avg_loss_control, epoch)
        writer_control.add_scalar('Gradient/norm', avg_grad_norm_control, epoch)
        
        # Store results
        ablation_results['Full_ABSD'].append({
            'epoch': epoch,
            'loss': avg_loss_full,
            'grad_norm': avg_grad_norm_full
        })
        
        ablation_results['Control'].append({
            'epoch': epoch,
            'loss': avg_loss_control,
            'grad_norm': avg_grad_norm_control
        })
        
        print(f"Epoch {epoch}: Full ABSD Loss {avg_loss_full:.4f}, Avg Grad Norm {avg_grad_norm_full:.4f}")
        print(f"Epoch {epoch}: Control Loss   {avg_loss_control:.4f}, Avg Grad Norm {avg_grad_norm_control:.4f}")
    
    # Save models
    os.makedirs('./models', exist_ok=True)
    torch.save(generator_model.state_dict(), os.path.join('./models', 'generator_ablation.pth'))
    torch.save(score_net_full.state_dict(), os.path.join('./models', 'score_net_full.pth'))
    torch.save(score_net_control.state_dict(), os.path.join('./models', 'score_net_control.pth'))
    
    return ablation_results

def train_experiment_sensitivity(dataloader, config, device, log_dir='./logs'):
    """
    Train models for the sensitivity analysis experiment.
    
    Args:
        dataloader (DataLoader): DataLoader for training data
        config (dict): Configuration parameters for the experiment
        device (torch.device): Device to run training on
        log_dir (str): Directory for TensorBoard logs
        
    Returns:
        dict: Results of the experiment
    """
    print("\n=== Experiment 3: Sensitivity Analysis ===")
    
    # Configure experiment parameters
    num_epochs = config.get('num_epochs', 2)
    latent_dim = config.get('latent_dim', 100)
    image_channels = config.get('image_channels', 3)
    image_size = config.get('image_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    
    # Grid search parameters
    uncertainty_factors = config.get('uncertainty_factors', [0.1, 0.5, 1.0, 2.0])
    sde_stepsizes = config.get('sde_stepsizes', [0.001, 0.005, 0.01])
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'Sensitivity'))
    
    # Results dictionary
    sensitivity_results = {}
    
    # Grid search over hyperparameters
    for uf in uncertainty_factors:
        for ss in sde_stepsizes:
            config_trial = {
                'uncertainty_factor': uf, 
                'sde_stepsize': ss
            }
            trial_key = f'uf_{uf}_ss_{ss}'
            
            print(f"\n>>> Running sensitivity trial with uncertainty_factor={uf}, sde_stepsize={ss}")
            
            # Initialize models for this trial
            generator_model = Generator(latent_dim, image_channels, image_size).to(device)
            score_net = ScoreNetworkABSD(image_channels, config_trial).to(device)
            
            optimizer = optim.Adam(
                list(generator_model.parameters()) + list(score_net.parameters()), 
                lr=learning_rate
            )
            
            # Store metrics for this trial
            trial_metrics = []
            
            # Train for specified number of epochs
            for epoch in range(1, num_epochs + 1):
                # Training loop
                running_loss = 0.0
                
                for images, _ in dataloader:
                    images = images.to(device)
                    
                    optimizer.zero_grad()
                    loss = score_net.compute_absd_loss(generator_model, images, device)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                # Calculate average loss
                avg_loss = running_loss / len(dataloader)
                
                # Generate images for evaluation
                generator_model.eval()
                with torch.no_grad():
                    z = torch.randn(64, latent_dim, device=device)
                    gen_images = generator_model(z).cpu()
                
                # Calculate FID (dummy implementation)
                fid_val = calculate_dummy_fid(gen_images.numpy())
                
                # Log metrics
                writer.add_scalar(f'Loss/{trial_key}', avg_loss, epoch)
                writer.add_scalar(f'FID/{trial_key}', fid_val, epoch)
                
                # Store results
                trial_metrics.append({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'fid': fid_val
                })
                
                print(f"   Epoch {epoch}: Loss {avg_loss:.4f}, FID {fid_val:.2f}")
                
                # Switch back to training mode
                generator_model.train()
            
            # Save model for this trial
            os.makedirs('./models', exist_ok=True)
            torch.save(generator_model.state_dict(), os.path.join('./models', f'generator_{trial_key}.pth'))
            torch.save(score_net.state_dict(), os.path.join('./models', f'score_net_{trial_key}.pth'))
            
            # Store trial results
            sensitivity_results[trial_key] = trial_metrics
    
    # Save the sensitivity analysis results to a JSON file
    os.makedirs('./logs', exist_ok=True)
    with open(os.path.join('./logs', 'sensitivity_results.json'), 'w') as f:
        json.dump(sensitivity_results, f, indent=2)
    
    print("\nSensitivity analysis complete. Results saved to 'logs/sensitivity_results.json'.")
    
    return sensitivity_results

def test_experiments(config, device):
    """
    Run a quick test of all experiments to verify code execution.
    
    Args:
        config (dict): Configuration parameters for the experiments
        device (torch.device): Device to run tests on
        
    Returns:
        bool: True if all tests pass
    """
    print("\n======= Starting Test Experiments =======")
    
    # Override config for quick testing
    test_config = config.copy()
    test_config['num_epochs'] = 1
    test_config['uncertainty_factors'] = [0.1, 1.0]  # Reduced set for testing
    test_config['sde_stepsizes'] = [0.001, 0.01]     # Reduced set for testing
    
    # Import preprocessing module
    from preprocess import preprocess_data
    
    # Get a small dataloader for testing
    test_config['batch_size'] = 8
    dataloader = preprocess_data(test_config)
    
    # Test each experiment type
    try:
        print("Testing performance experiment...")
        train_experiment('performance', dataloader, test_config, device)
        
        print("Testing ablation experiment...")
        train_experiment('ablation', dataloader, test_config, device)
        
        print("Testing sensitivity experiment...")
        train_experiment('sensitivity', dataloader, test_config, device)
        
        print("Test experiments finished successfully.")
        return True
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    # Default configuration
    config = {
        'num_epochs': 2,
        'batch_size': 64,
        'image_size': 32,
        'latent_dim': 100,
        'image_channels': 3,
        'learning_rate': 1e-4,
        'uncertainty_factor': 1.0,
        'sde_stepsize': 0.005,
        'uncertainty_factors': [0.1, 0.5, 1.0, 2.0],
        'sde_stepsizes': [0.001, 0.005, 0.01],
        'data_dir': './data',
        'seed': 42
    }
    
    # Run test experiments
    test_experiments(config, device)
