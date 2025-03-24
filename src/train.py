"""
Training script for DPC-3D model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import logging
from tqdm import tqdm
import time
from pathlib import Path

from src.utils.models import DPC3D
from src.utils.profiling import PerformanceTracker, track_gpu_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    """Dataset for molecular data"""
    
    def __init__(self, data, device="cuda"):
        """
        Initialize dataset
        
        Args:
            data: List of processed molecule data
            device: Device to load tensors to
        """
        self.data = data
        self.device = device
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """Get a single data item"""
        mol_data = self.data[idx]
        
        # Get token IDs for language model
        token_ids = torch.tensor(mol_data['token_ids'], dtype=torch.long)
        
        # Get atom types
        atom_types = torch.tensor(mol_data['atom_types'], dtype=torch.long)
        
        # Get conformer coordinates (use first conformer)
        coords = torch.tensor(mol_data['conformers'][0], dtype=torch.float)
        
        # Get attention mask
        mask = mol_data['mask']
        
        # Get number of atoms
        num_atoms = mol_data['num_atoms']
        
        return {
            'token_ids': token_ids,
            'atom_types': atom_types,
            'coords': coords,
            'mask': mask,
            'num_atoms': num_atoms,
            'smiles': mol_data['smiles']
        }
        
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        # Stack tensors
        token_ids = torch.stack([item['token_ids'] for item in batch])
        atom_types = torch.stack([item['atom_types'] for item in batch])
        coords = torch.stack([item['coords'] for item in batch])
        
        # Stack masks
        masks = torch.stack([item['mask'] for item in batch])
        
        # Get number of atoms
        num_atoms = [item['num_atoms'] for item in batch]
        
        # Get SMILES
        smiles = [item['smiles'] for item in batch]
        
        return {
            'token_ids': token_ids.to(self.device),
            'atom_types': atom_types.to(self.device),
            'coords': coords.to(self.device),
            'mask': masks.to(self.device),
            'num_atoms': num_atoms,
            'smiles': smiles
        }

class DiffusionTrainer:
    """Trainer for DPC-3D model"""
    
    def __init__(self, config, model=None):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            model: Pre-initialized model (optional)
        """
        self.config = config
        self.device = torch.device(config["device"])
        
        # Create model if not provided
        if model is None:
            self.model = DPC3D(config).to(self.device)
        else:
            self.model = model.to(self.device)
            
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["epochs"]
        )
        
        # Create loss function
        self.criterion = nn.MSELoss()
        
        # Create directories
        self.checkpoint_dir = Path("models")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize best validation loss
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Performance tracking
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        
    def _add_noise(self, x, t):
        """
        Add noise to coordinates based on diffusion timestep
        
        Args:
            x: Clean coordinates [batch_size, num_atoms, 3]
            t: Diffusion timesteps [batch_size]
            
        Returns:
            noisy_x: Noisy coordinates
            noise: Added noise
        """
        # Create noise
        noise = torch.randn_like(x)
        
        # Create noise schedule
        beta_min = 0.1
        beta_max = 20.0
        
        if self.config["diffusion"]["beta_schedule"] == "linear":
            # Linear schedule
            beta_t = beta_min + t * (beta_max - beta_min)
        elif self.config["diffusion"]["beta_schedule"] == "cosine":
            # Cosine schedule
            s = 0.008
            alpha_t = torch.cos((t + s) / (1 + s) * torch.tensor(np.pi / 2))
            beta_t = 1 - alpha_t * alpha_t
        else:
            # Default to linear
            beta_t = beta_min + t * (beta_max - beta_min)
            
        # Reshape for broadcasting
        beta_t = beta_t.view(-1, 1, 1)
        
        # Add noise
        noisy_x = x + torch.sqrt(beta_t) * noise
        
        return noisy_x, noise
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            # Get data
            token_ids = batch['token_ids']
            atom_types = batch['atom_types']
            coords = batch['coords']
            mask = batch['mask']
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Sample random timesteps
            batch_size = token_ids.shape[0]
            t = torch.rand(batch_size, device=self.device)
            
            # Add noise to coordinates
            noisy_coords, noise = self._add_noise(coords, t)
            
            # Get initial prompt from language model
            prompt = self.model.encode_molecule(token_ids)
            
            # Predict noise
            noise_pred, _, _ = self.model.diffusion_step(
                noisy_coords, atom_types, t, prompt, mask
            )
            
            # Calculate loss
            loss = self.criterion(noise_pred, noise)
            
            # Backpropagate
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config["gradient_clip"]
            )
            
            # Update weights
            self.optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss
        
    def validate(self, val_loader):
        """
        Validate model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get data
                token_ids = batch['token_ids']
                atom_types = batch['atom_types']
                coords = batch['coords']
                mask = batch['mask']
                
                # Sample random timesteps
                batch_size = token_ids.shape[0]
                t = torch.rand(batch_size, device=self.device)
                
                # Add noise to coordinates
                noisy_coords, noise = self._add_noise(coords, t)
                
                # Get initial prompt from language model
                prompt = self.model.encode_molecule(token_ids)
                
                # Predict noise
                noise_pred, _, _ = self.model.diffusion_step(
                    noisy_coords, atom_types, t, prompt, mask
                )
                
                # Calculate loss
                loss = self.criterion(noise_pred, noise)
                
                # Update total loss
                total_loss += loss.item()
                
        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss
        
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        if epoch % self.config["checkpoint_interval"] == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            epoch: Epoch of the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get epoch and loss
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        logger.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
        
        return epoch
        
    def train(self, train_loader, val_loader, start_epoch=0):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            start_epoch: Starting epoch (for resuming training)
            
        Returns:
            model: Trained model
        """
        logger.info("Starting training")
        
        # Create performance tracker
        tracker = PerformanceTracker()
        
        # Training loop
        for epoch in range(start_epoch, self.config["epochs"]):
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # Track epoch time
            epoch_start = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Log results
            logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Check for early stopping
            if self.patience_counter >= self.config["early_stopping_patience"]:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
                
            # Log GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = track_gpu_memory()
                logger.info(f"GPU memory usage: {gpu_memory:.2f} MB")
                
        # Return trained model
        return self.model

# Model variants for ablation study
class DPC3D_Full(nn.Module):
    """Full DPC-3D model with dynamic prompt tuning and Bayesian adaptation"""
    
    def __init__(self, config):
        super().__init__()
        # Create a deep copy of the config to avoid modifying the original
        config_copy = {}
        for key, value in config.items():
            if isinstance(value, dict):
                config_copy[key] = dict(value)
            else:
                config_copy[key] = value
                
        # Ensure required config sections exist
        if "prompt_tuning" not in config_copy:
            config_copy["prompt_tuning"] = {"use_prompt_tuning": True}
        if "bayesian" not in config_copy:
            config_copy["bayesian"] = {"use_bayesian": True}
                
        # Use full DPC-3D model
        self.model = DPC3D(config_copy)
        
    def forward(self, x, atom_types, t, prompt=None, mask=None):
        # Generate prompt if not provided
        if prompt is None:
            # Dummy prompt
            batch_size = x.shape[0]
            prompt = torch.randn(batch_size, self.model.config["lm"]["embed_dim"], 
                                device=x.device)
                                
        # Run diffusion step
        noise_pred, updated_prompt, uncertainty = self.model.diffusion_step(
            x, atom_types, t, prompt, mask
        )
        
        return noise_pred, updated_prompt, uncertainty

class DPC3D_Static(nn.Module):
    """DPC-3D model with static prompt (no dynamic tuning)"""
    
    def __init__(self, config):
        super().__init__()
        # Create a deep copy of the config to avoid modifying the original
        config_copy = {}
        for key, value in config.items():
            if isinstance(value, dict):
                config_copy[key] = dict(value)
            else:
                config_copy[key] = value
        
        # Disable prompt tuning
        if "prompt_tuning" in config_copy:
            config_copy["prompt_tuning"]["use_prompt_tuning"] = False
        
        # Create model with modified config
        self.model = DPC3D(config_copy)
        
    def forward(self, x, atom_types, t, prompt=None, mask=None):
        # Generate prompt if not provided
        if prompt is None:
            # Dummy prompt
            batch_size = x.shape[0]
            prompt = torch.randn(batch_size, self.model.config["lm"]["embed_dim"], 
                                device=x.device)
                                
        # Run diffusion step (prompt will not be updated)
        noise_pred, prompt, uncertainty = self.model.diffusion_step(
            x, atom_types, t, prompt, mask
        )
        
        return noise_pred, prompt, uncertainty

class DPC3D_NoBayesian(nn.Module):
    """DPC-3D model without Bayesian adaptation"""
    
    def __init__(self, config):
        super().__init__()
        # Create a deep copy of the config to avoid modifying the original
        config_copy = {}
        for key, value in config.items():
            if isinstance(value, dict):
                config_copy[key] = dict(value)
            else:
                config_copy[key] = value
        
        # Disable Bayesian adaptation
        if "bayesian" in config_copy:
            config_copy["bayesian"]["use_bayesian"] = False
        
        # Create model with modified config
        self.model = DPC3D(config_copy)
        
    def forward(self, x, atom_types, t, prompt=None, mask=None):
        # Generate prompt if not provided
        if prompt is None:
            # Dummy prompt
            batch_size = x.shape[0]
            prompt = torch.randn(batch_size, self.model.config["lm"]["embed_dim"], 
                                device=x.device)
                                
        # Run diffusion step (uncertainty will be constant)
        noise_pred, updated_prompt, uncertainty = self.model.diffusion_step(
            x, atom_types, t, prompt, mask
        )
        
        return noise_pred, updated_prompt, uncertainty

def measure_variant(model_variant, sample_input, atom_types, max_steps=20):
    """
    Run the model variant for a maximum number of diffusion steps
    
    Args:
        model_variant: Model variant to test
        sample_input: Input coordinates [batch_size, num_atoms, 3]
        atom_types: Atom type indices [batch_size, num_atoms]
        max_steps: Maximum number of steps to run
        
    Returns:
        n_steps: Number of steps taken
        elapsed: Elapsed time
    """
    model_variant.eval()
    
    # Create timestep
    batch_size = sample_input.shape[0]
    device = sample_input.device
    
    # Start timer
    start_time = time.time()
    
    # Run steps
    n_steps = 0
    x = sample_input.clone()
    prompt = None
    
    with torch.no_grad():
        while n_steps < max_steps:
            # Create random timestep
            t = torch.rand(batch_size, device=device)
            
            # Run diffusion step
            noise_pred, prompt, uncertainty = model_variant(x, atom_types, t, prompt)
            
            # Update coordinates
            x = x - 0.1 * noise_pred
            
            n_steps += 1
            
            # Dummy convergence check (in practice, use chemical validity)
            if torch.norm(x) < 0.1:
                break
                
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    return n_steps, elapsed

def load_datasets(config):
    """
    Load datasets for training
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Set device
    device = torch.device(config["device"])
    
    # Load processed data
    processed_dir = Path(config["processed_data_dir"])
    
    # Load training data
    train_path = processed_dir / "train_dataset.pkl"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    logger.info(f"Loaded {len(train_data)} training samples")
    
    # Load validation data
    val_path = processed_dir / "val_dataset.pkl"
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    logger.info(f"Loaded {len(val_data)} validation samples")
    
    # Create datasets
    train_dataset = MolecularDataset(train_data, device=device)
    val_dataset = MolecularDataset(val_data, device=device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=val_dataset.collate_fn
    )
    
    return train_loader, val_loader

def train_model(config):
    """
    Main function to train DPC-3D model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        trainer: Trained model trainer
    """
    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Load datasets
    train_loader, val_loader = load_datasets(config)
    
    # Create trainer
    trainer = DiffusionTrainer(config)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    return trainer

def run_ablation_study(config):
    """
    Run ablation study on model variants
    
    Args:
        config: Configuration dictionary
        
    Returns:
        results: Dictionary with ablation study results
    """
    # Set device
    device = torch.device(config["device"])
    
    # Create model variants
    model_full = DPC3D_Full(config).to(device)
    model_static = DPC3D_Static(config).to(device)
    model_no_bayesian = DPC3D_NoBayesian(config).to(device)
    
    # Create sample input
    batch_size = 4
    num_atoms = config["max_atoms"]
    sample_input = torch.randn(batch_size, num_atoms, 3, device=device)
    atom_types = torch.randint(0, 10, (batch_size, num_atoms), device=device)
    
    # Measure performance
    steps_full, time_full = measure_variant(model_full, sample_input, atom_types)
    steps_static, time_static = measure_variant(model_static, sample_input, atom_types)
    steps_no_bayesian, time_no_bayesian = measure_variant(model_no_bayesian, sample_input, atom_types)
    
    # Log results
    logger.info("Ablation study results:")
    logger.info(f"Full DPC-3D: steps = {steps_full}, time = {time_full:.4f}s")
    logger.info(f"Static Prompt: steps = {steps_static}, time = {time_static:.4f}s")
    logger.info(f"No Bayesian Adaptation: steps = {steps_no_bayesian}, time = {time_no_bayesian:.4f}s")
    
    # Return results
    results = {
        "full": {"steps": steps_full, "time": time_full},
        "static": {"steps": steps_static, "time": time_static},
        "no_bayesian": {"steps": steps_no_bayesian, "time": time_no_bayesian}
    }
    
    return results

if __name__ == "__main__":
    # Import config when run as script
    from config.dpc3d_config import (
        DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, EXPERIMENT_CONFIG
    )
    
    # Combine configs
    config = {
        **DATA_CONFIG, 
        **MODEL_CONFIG, 
        **TRAIN_CONFIG, 
        **EXPERIMENT_CONFIG
    }
    
    # Run ablation study
    run_ablation_study(config)
