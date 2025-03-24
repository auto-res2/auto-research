"""
Model implementations for the DPC-3D method
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class SelfAttention(nn.Module):
    """Self-attention layer for transformer models"""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        # Linear projections
        q = self.query(x).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final projection
        output = self.proj(context)
        
        return output
        
class FeedForward(nn.Module):
    """Feedforward layer for transformer models"""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    """Transformer layer with self-attention and feedforward network"""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class MolecularLanguageModel(nn.Module):
    """1D Transformer model for molecular language modeling"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.position_embedding = nn.Parameter(torch.zeros(1, config["max_seq_len"], config["embed_dim"]))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                config["embed_dim"], 
                config["num_heads"], 
                config["dropout"]
            ) for _ in range(config["num_layers"])
        ])
        
        # Output layer
        self.fc_out = nn.Linear(config["embed_dim"], config["vocab_size"])
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Token and position embeddings
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding[:, :seq_len, :]
        
        # Combined embeddings
        x = token_embed + pos_embed
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output logits
        logits = self.fc_out(x)
        
        return logits
    
    def get_embeddings(self, x, mask=None):
        """Extract embeddings only without producing logits"""
        batch_size, seq_len = x.shape
        
        # Token and position embeddings
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding[:, :seq_len, :]
        
        # Combined embeddings
        x = token_embed + pos_embed
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class DiffusionModel(nn.Module):
    """3D Diffusion model for molecular conformer generation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config["hidden_dim"]
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Condition embedding (for prompt)
        self.cond_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Initial projection for atom coordinates
        self.coord_embed = nn.Linear(3, self.hidden_dim)
        
        # Type embedding for different atom types
        self.type_embed = nn.Embedding(10, self.hidden_dim)  # Assuming max 10 atom types
        
        # Transformer layers for processing
        self.layers = nn.ModuleList([
            TransformerLayer(
                self.hidden_dim, 
                config["num_heads"], 
                config["dropout"]
            ) for _ in range(config["num_layers"])
        ])
        
        # Output layer
        self.fc_out = nn.Linear(self.hidden_dim, 3)  # Predict 3D coordinates
        
    def forward(self, x, atom_types, t, cond=None, mask=None):
        """
        Args:
            x: Noisy coordinates [batch_size, num_atoms, 3]
            atom_types: Atom type indices [batch_size, num_atoms]
            t: Diffusion timesteps [batch_size]
            cond: Conditional embedding from LM [batch_size, hidden_dim]
            mask: Attention mask [batch_size, num_atoms, num_atoms]
        """
        batch_size, num_atoms, _ = x.shape
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [batch_size, hidden_dim]
        
        # Embed coordinates
        x_emb = self.coord_embed(x)  # [batch_size, num_atoms, hidden_dim]
        
        # Add atom type embeddings
        type_emb = self.type_embed(atom_types)  # [batch_size, num_atoms, hidden_dim]
        x_emb = x_emb + type_emb
        
        # Add time embedding to each atom
        t_emb = t_emb.unsqueeze(1).expand(-1, num_atoms, -1)
        x_emb = x_emb + t_emb
        
        # Add conditional embedding if provided
        if cond is not None:
            cond_emb = self.cond_embed(cond)  # [batch_size, hidden_dim]
            cond_emb = cond_emb.unsqueeze(1).expand(-1, num_atoms, -1)
            x_emb = x_emb + cond_emb
        
        # Apply transformer layers
        for layer in self.layers:
            x_emb = layer(x_emb, mask)
        
        # Predict noise (or coordinates)
        pred = self.fc_out(x_emb)  # [batch_size, num_atoms, 3]
        
        return pred

class DynamicPromptTuning(nn.Module):
    """Dynamic prompt tuning module for adapting LM embeddings"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get("prompt_dim", 256)
        
        # Prompt update network
        self.update_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Projection to ensure prompt stays in valid embedding space
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, prompt, current_state):
        """
        Update the prompt based on current diffusion state
        
        Args:
            prompt: Current prompt embedding [batch_size, hidden_dim]
            current_state: Current state from diffusion [batch_size, hidden_dim]
            
        Returns:
            Updated prompt [batch_size, hidden_dim]
        """
        # Concatenate prompt and current state
        combined = torch.cat([prompt, current_state], dim=-1)
        
        # Compute update
        update = self.update_net(combined)
        
        # Update prompt
        prompt_lr = self.config.get("prompt_lr", 1e-4)
        updated_prompt = prompt + update * prompt_lr
        
        # Project to valid embedding space
        updated_prompt = self.proj(updated_prompt)
        
        return updated_prompt

class BayesianAdaptation(nn.Module):
    """Bayesian adaptation module for uncertainty estimation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get("prompt_dim", 256)
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state):
        """
        Estimate uncertainty in current state
        
        Args:
            state: Current state from diffusion [batch_size, hidden_dim]
            
        Returns:
            uncertainty: Uncertainty score [batch_size, 1]
        """
        uncertainty = self.uncertainty_net(state)
        return uncertainty
        
    def sample(self, state, n_samples=10):
        """
        Generate multiple samples to estimate uncertainty
        
        Args:
            state: Current state from diffusion [batch_size, hidden_dim]
            n_samples: Number of samples to generate
            
        Returns:
            samples: Multiple uncertainty estimates
            mean: Mean uncertainty
            std: Standard deviation of uncertainty
        """
        samples = []
        batch_size = state.shape[0]
        
        # Generate samples with dropout enabled
        for _ in range(n_samples):
            self.train()  # Enable dropout
            samples.append(self.forward(state))
        
        # Stack samples
        samples = torch.stack(samples, dim=1)  # [batch_size, n_samples, 1]
        
        # Compute statistics
        mean = samples.mean(dim=1)
        std = samples.std(dim=1)
        
        self.eval()  # Restore eval mode
        
        return samples, mean, std

class DPC3D(nn.Module):
    """Dynamic Prompt-Coupled 3D Molecular Diffusion (DPC-3D) model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create component models
        self.lm = MolecularLanguageModel(config["lm"])
        self.diffusion = DiffusionModel(config["diffusion"])
        
        # Dynamic prompt tuning module
        if config["prompt_tuning"].get("use_prompt_tuning", True):
            self.prompt_tuner = DynamicPromptTuning(config["prompt_tuning"])
        else:
            self.prompt_tuner = None
            
        # Bayesian adaptation module
        if config["bayesian"].get("use_bayesian", True):
            self.bayesian = BayesianAdaptation(config["bayesian"])
        else:
            self.bayesian = None
            
        # Current prompt memory
        self.register_buffer("current_prompt", None, persistent=False)
            
    def encode_molecule(self, mol_seq):
        """Encode molecule sequence to latent representation"""
        # Get embeddings from language model
        embeddings = self.lm.get_embeddings(mol_seq)
        
        # Use CLS token or average pooling for sequence-level embedding
        pooled_emb = embeddings.mean(dim=1)  # [batch_size, hidden_dim]
        
        return pooled_emb
    
    def update_prompt(self, prompt, current_state):
        """Update prompt based on current diffusion state"""
        if self.prompt_tuner is None:
            return prompt
            
        updated_prompt = self.prompt_tuner(prompt, current_state)
        return updated_prompt
        
    def estimate_uncertainty(self, state):
        """Estimate uncertainty in current diffusion state"""
        if self.bayesian is None:
            return torch.ones(state.shape[0], 1, device=state.device) * 0.5
            
        # Get uncertainty estimate
        n_samples = self.config["bayesian"].get("n_samples", 10)
        _, mean_uncertainty, _ = self.bayesian.sample(state, n_samples=n_samples)
        return mean_uncertainty
        
    def diffusion_step(self, x, atom_types, t, prompt, mask=None):
        """Single diffusion step with prompt and uncertainty estimation"""
        # Run diffusion model
        noise_pred = self.diffusion(x, atom_types, t, prompt, mask)
        
        # Get current state representation for prompt update
        # Using mean across atoms for simplicity
        current_state = noise_pred.mean(dim=1)  # [batch_size, 3]
        
        # Project to hidden dimension
        current_state = self.diffusion.coord_embed(current_state)  # [batch_size, hidden_dim]
        
        # Estimate uncertainty
        uncertainty = self.estimate_uncertainty(current_state)
        
        # Update prompt if needed based on uncertainty and timestep
        update_interval = self.config["prompt_tuning"].get("update_interval", 10)
        if t % update_interval == 0:
            # Scale update by uncertainty
            uncertainty_threshold = self.config["bayesian"].get("uncertainty_threshold", 0.5)
            update_scale = torch.where(
                uncertainty > uncertainty_threshold,
                torch.ones_like(uncertainty),  # High uncertainty: full update
                uncertainty / uncertainty_threshold  # Low uncertainty: scaled update
            )
            
            # Apply update with scaling
            updated_prompt = self.update_prompt(prompt, current_state)
            prompt = prompt + (updated_prompt - prompt) * update_scale
            
        return noise_pred, prompt, uncertainty
        
    def sample(self, mol_seq, num_atoms, atom_types, mask=None, device="cuda"):
        """
        Sample 3D conformers using the DPC-3D model
        
        Args:
            mol_seq: Molecule sequence tokens [batch_size, seq_len]
            num_atoms: Number of atoms in each molecule [batch_size]
            atom_types: Atom type indices [batch_size, max_atoms]
            mask: Attention mask [batch_size, max_atoms, max_atoms]
            device: Device to run sampling on
            
        Returns:
            coords: Generated 3D coordinates [batch_size, max_atoms, 3]
            prompt_trajectory: Trajectory of prompts during sampling
            uncertainty_trajectory: Trajectory of uncertainties during sampling
        """
        batch_size = mol_seq.shape[0]
        max_atoms = atom_types.shape[1]
        
        # Get initial prompt from language model
        prompt = self.encode_molecule(mol_seq)
        
        # Initialize with random noise
        x = torch.randn(batch_size, max_atoms, 3, device=device)
        
        # Create timestep sequence for sampling
        timesteps = list(range(self.config["diffusion"]["sampling_steps"]))[::-1]
        
        # Track trajectories
        prompt_trajectory = [prompt.detach().cpu()]
        uncertainty_trajectory = []
        
        # Sampling loop
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            # Convert timestep to tensor
            t_tensor = torch.ones(batch_size, device=device) * t / self.config["diffusion"]["sampling_steps"]
            
            # Run diffusion step with dynamic prompt tuning
            with torch.no_grad():
                noise_pred, prompt, uncertainty = self.diffusion_step(
                    x, atom_types, t_tensor, prompt, mask
                )
            
            # Update coordinates based on predicted noise
            alpha = 0.9  # Denoising strength
            x = x - alpha * noise_pred
            
            # Track trajectories
            prompt_trajectory.append(prompt.detach().cpu())
            uncertainty_trajectory.append(uncertainty.detach().cpu())
            
        return x, prompt_trajectory, uncertainty_trajectory
