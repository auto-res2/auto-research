#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
import os
from preprocess import PCFGDataset, pad_sequence

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class TransformerDecoderModel(pl.LightningModule):
    def __init__(self, vocab_size=26, dim_model=64, num_heads=4, num_layers=2, add_memory=False, mem_reg_weight=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["add_memory"])  # save configuration for logging
        self.embedding = nn.Embedding(vocab_size, dim_model)
        # Simple learnable positional encoding; alternatively, use fixed encoding schemes
        self.positional_encoding = nn.Parameter(torch.rand(500, dim_model), requires_grad=True)
        
        # Define a basic Transformer decoder using PyTorch's built-in modules.
        encoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(dim_model, vocab_size)
        self.add_memory = add_memory
        self.mem_reg_weight = mem_reg_weight
        
        # If using dynamic memory, add a learnable memory bank (10 entries)
        if self.add_memory:
            self.memory_bank = nn.Parameter(torch.randn(10, dim_model))
    
    def forward(self, x):
        # x: (batch, seq_length)
        batch, seq_len = x.shape
        emb = self.embedding(x) + self.positional_encoding[:seq_len]
        # Use same embeddings as query and key/value for toy example
        decoder_input = emb.transpose(0, 1)  # (seq_len, batch, dim_model)
        out = self.decoder(decoder_input, decoder_input)
        out = out.transpose(0, 1)  # (batch, seq_len, dim_model)
        logits = self.fc_out(out)
        
        # Memory consistency loss if dynamic memory enabled
        mem_loss = 0.0
        if self.add_memory:
            # Compute mean representation per sample and encourage similarity with memory bank
            avg_rep = out.mean(dim=1)  # (batch, dim_model)
            
            # Calculate similarity between each sample representation and memory entries
            sim = torch.matmul(avg_rep, self.memory_bank.t())  # (batch, mem_entries)
            
            # For each sample, find the most similar memory entry
            max_sim_values, max_sim_indices = torch.max(sim, dim=1)
            
            # Memory consistency loss: encourage high similarity with best matching memory entry
            mem_loss = -max_sim_values.mean()
            
            # Optional: Update memory bank with moving average of representations
            # This would be more complex in a real implementation with proper gating
            # Not implemented here to keep the example simple
        
        return logits, mem_loss
    
    def training_step(self, batch, batch_idx):
        logits, mem_loss = self.forward(batch)
        # Standard cross-entropy loss for next token prediction
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))
        
        # Add memory consistency loss if using dynamic memory
        if self.add_memory:
            loss = loss + self.mem_reg_weight * mem_loss
            self.log("mem_loss", mem_loss, prog_bar=True)
            
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, mem_loss = self.forward(batch)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        logits, mem_loss = self.forward(batch)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class ContinualLearningModel(pl.LightningModule):
    def __init__(self, model):
        """
        Wrapper for continual learning phase. Only adapts the provided model (e.g., DMI model)
        with a smaller learning rate and possibly gradient clipping.
        """
        super().__init__()
        self.model = model
        
    def training_step(self, batch, batch_idx):
        logits, mem_loss = self.model(batch)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))
        loss = loss + self.model.mem_reg_weight * mem_loss
        self.log("continual_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Smaller learning rate for adaptation phase
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

def train_model(model, train_loader, val_loader=None, max_epochs=10, gpus=1):
    """
    Train a model using PyTorch Lightning.
    
    Args:
        model: The model to train (TransformerDecoderModel)
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        max_epochs: Maximum number of training epochs
        gpus: Number of GPUs to use (0 for CPU)
        
    Returns:
        Trained model
    """
    # Configure trainer
    trainer_kwargs = {
        'max_epochs': max_epochs,
        'logger': True,
        'enable_checkpointing': True,
    }
    
    # Set up GPU acceleration if available
    if gpus > 0 and torch.cuda.is_available():
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = gpus
    else:
        trainer_kwargs['accelerator'] = 'cpu'
    
    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train the model
    if val_loader:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    
    return model

def save_model(model, path):
    """
    Save a trained model to disk.
    
    Args:
        model: The model to save
        path: Path to save the model to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, **kwargs):
    """
    Load a model from disk.
    
    Args:
        model_class: The class of the model to load
        path: Path to load the model from
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Loaded model
    """
    # Create model instance
    model = model_class(**kwargs)
    
    # Load state dict
    model.load_state_dict(torch.load(path))
    
    return model

def experiment_rule_extrapolation(max_epochs=2, batch_size=32, save_models=True):
    """
    Run the synthetic rule extrapolation experiment.
    
    Args:
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        save_models: Whether to save the trained models
        
    Returns:
        Dictionary with experiment results
    """
    print("\n=== Experiment 1: Synthetic Rule Extrapolation ===")
    
    # Prepare in-distribution training data (aNbN rule)
    train_dataset = PCFGDataset(num_samples=5000, rule="aNbN", ood=False)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: pad_sequence([item for item in batch], batch_first=True, padding_value=0)
    )
    
    # Prepare validation dataset
    val_dataset = PCFGDataset(num_samples=1000, rule="aNbN", ood=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=lambda batch: pad_sequence([item for item in batch], batch_first=True, padding_value=0)
    )
    
    # Prepare OOD dataset (introducing an extra symbol)
    ood_dataset = PCFGDataset(num_samples=1000, rule="aNbN", ood=True)
    ood_loader = DataLoader(
        ood_dataset, 
        batch_size=batch_size,
        collate_fn=lambda batch: pad_sequence([item for item in batch], batch_first=True, padding_value=0)
    )

    # Initialize models
    print("Initializing baseline and DMI models...")
    baseline_model = TransformerDecoderModel(add_memory=False)
    dmi_model = TransformerDecoderModel(add_memory=True, mem_reg_weight=0.1)
    
    # Train models
    print("Training baseline model...")
    baseline_model = train_model(baseline_model, train_loader, val_loader, max_epochs=max_epochs)
    
    print("Training DMI model...")
    dmi_model = train_model(dmi_model, train_loader, val_loader, max_epochs=max_epochs)
    
    # Save models if requested
    if save_models:
        os.makedirs("models", exist_ok=True)
        save_model(baseline_model, "models/baseline_model.pt")
        save_model(dmi_model, "models/dmi_model.pt")
    
    # Evaluate on OOD data
    baseline_model.eval()
    dmi_model.eval()
    
    baseline_ood_loss = 0.0
    dmi_ood_loss = 0.0
    
    with torch.no_grad():
        for batch in ood_loader:
            baseline_logits, _ = baseline_model(batch)
            baseline_loss = F.cross_entropy(baseline_logits.view(-1, baseline_logits.size(-1)), batch.view(-1))
            baseline_ood_loss += baseline_loss.item()
            
            dmi_logits, _ = dmi_model(batch)
            dmi_loss = F.cross_entropy(dmi_logits.view(-1, dmi_logits.size(-1)), batch.view(-1))
            dmi_ood_loss += dmi_loss.item()
    
    baseline_ood_loss /= len(ood_loader)
    dmi_ood_loss /= len(ood_loader)
    
    print(f"Baseline OOD Loss: {baseline_ood_loss:.4f}")
    print(f"DMI OOD Loss: {dmi_ood_loss:.4f}\n")
    
    return {
        'baseline_ood_loss': baseline_ood_loss,
        'dmi_ood_loss': dmi_ood_loss,
        'baseline_model': baseline_model,
        'dmi_model': dmi_model
    }

if __name__ == "__main__":
    # Run the experiment with a small number of epochs for testing
    results = experiment_rule_extrapolation(max_epochs=2, batch_size=32)
