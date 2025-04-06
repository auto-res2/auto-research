"""
Training module for QD²P experiment.

Implements the model classes and training functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SimpleDiffusion(nn.Module):
    """
    A simple diffusion model with a single linear transformation plus additive noise.
    """
    def __init__(self, latent_dim):
        super(SimpleDiffusion, self).__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to have normalized outputs."""
        nn.init.orthogonal_(self.linear.weight)
    
    def forward(self, x, noise_level):
        """
        Apply one step of the diffusion process.
        
        Args:
            x: Input latent tensor
            noise_level: Amount of noise to add
            
        Returns:
            Diffused latent tensor
        """
        noise = noise_level * torch.randn_like(x)
        return self.linear(x) + noise
    
    def save(self, path):
        """Save model weights to file."""
        Path(path).parent.mkdir(exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model weights from file."""
        self.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")


class QProbe(nn.Module):
    """
    A Q-probe module for scoring candidate latent vectors.
    """
    def __init__(self, latent_dim):
        super(QProbe, self).__init__()
        self.linear = nn.Linear(latent_dim, 1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to small values."""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        Score a latent vector.
        
        Args:
            x: Input latent tensor
            
        Returns:
            Quality score (scalar)
        """
        return self.linear(x)
    
    def save(self, path):
        """Save model weights to file."""
        Path(path).parent.mkdir(exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Q-Probe saved to {path}")
    
    def load(self, path):
        """Load model weights from file."""
        self.load_state_dict(torch.load(path))
        logger.info(f"Q-Probe loaded from {path}")


def train_models(config, data, device="cuda"):
    """
    Train the diffusion model and Q-probe.
    
    Args:
        config: Configuration parameters
        data: Dictionary containing training data
        device: Device to train on
        
    Returns:
        Trained models
    """
    logger.info("Training models...")
    start_time = time.time()
    
    diffusion_model = SimpleDiffusion(config.LATENT_DIM).to(device)
    q_probe = QProbe(config.LATENT_DIM).to(device)
    
    optimizer = optim.Adam(q_probe.parameters(), lr=0.01)
    
    latent = data["latent"]
    ideal = data["ideal"]
    
    for step in range(config.DIFFUSION_STEPS):
        noise_level = 1.0 / (step + 1)
        
        candidates = []
        for _ in range(config.CANDIDATE_COUNT):
            candidate = diffusion_model(latent, noise_level)
            candidates.append(candidate)
        candidates_tensor = torch.stack(candidates)
        
        quality_metrics = []
        for candidate in candidates:
            quality = quality_metric(candidate, ideal.expand_as(candidate))
            quality_metrics.append(quality)
        
        target_q_values = torch.stack(quality_metrics).unsqueeze(-1)
        
        q_values = q_probe(candidates_tensor.reshape(-1, config.LATENT_DIM)).reshape(
            config.CANDIDATE_COUNT, -1, 1
        )
        
        batch_size = q_values.size(1)
        
        target_q_values_expanded = target_q_values.view(config.CANDIDATE_COUNT, 1, 1).expand(-1, batch_size, -1)
        
        loss = F.mse_loss(q_values, target_q_values_expanded)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        latent = candidates_tensor[0].detach()  # Use first candidate for simplicity
        
        logger.info(f"Step {step+1}/{config.DIFFUSION_STEPS}, Loss: {loss.item():.6f}")
    
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    diffusion_model.save("models/diffusion_model.pt")
    q_probe.save("models/q_probe.pt")
    
    return {
        "diffusion_model": diffusion_model,
        "q_probe": q_probe,
        "train_time": train_time
    }


def quality_metric(output, target):
    """
    Compute quality metric (cosine similarity) between output and target.
    
    Args:
        output: Output tensor
        target: Target tensor
        
    Returns:
        Cosine similarity (higher is better)
    """
    output_norm = F.normalize(output, dim=1)
    target_norm = F.normalize(target, dim=1)
    similarity = (output_norm * target_norm).sum(dim=1).mean()
    return similarity


if __name__ == "__main__":
    from preprocess import create_synthetic_data, set_seed
    set_seed(42)
    
    class Config:
        LATENT_DIM = 8
        DIFFUSION_STEPS = 3
        CANDIDATE_COUNT = 3
    
    data = create_synthetic_data(Config.LATENT_DIM, 4, device="cpu")
    train_models(Config, data, device="cpu")
