import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

# DynamicTokenizer class for adaptive tokenization
class DynamicTokenizer(nn.Module):
    def __init__(self, in_channels=3, token_dim=64):
        super(DynamicTokenizer, self).__init__()
        # CNN layers for generating complexity maps
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # complexity map channel
        # Token projection layer
        self.token_proj = nn.Conv2d(in_channels, token_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        feat = F.relu(self.conv1(x))
        # Generate a complexity map in [0,1] – higher values where there is more detail
        complexity_map = torch.sigmoid(self.conv2(feat))
        # Project original image into token space
        tokens = self.token_proj(x)
        # Weight token features by complexity
        weighted_tokens = tokens * complexity_map
        # Adaptive pooling to get fixed number of tokens
        pooled_tokens = F.adaptive_avg_pool2d(weighted_tokens, (8, 8))  # 64 tokens total
        B, token_dim, H_tok, W_tok = pooled_tokens.shape
        # Flatten tokens: [B, token_count, token_dim]
        tokens_flat = pooled_tokens.view(B, token_dim, H_tok * W_tok).permute(0, 2, 1)
        return tokens_flat, complexity_map

# Fixed step SDE module
class FixedStepSDE(nn.Module):
    def __init__(self, token_dim, dt=0.1):
        super(FixedStepSDE, self).__init__()
        self.dt = dt
        self.update_net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim)
        )

    def forward(self, tokens):
        # tokens: [B, N, token_dim]
        noise = torch.randn_like(tokens)
        update = self.update_net(tokens)
        new_tokens = tokens + self.dt * update + torch.sqrt(torch.tensor(self.dt)) * noise
        return new_tokens

# Adaptive SDE module
class AdaptiveSDE(nn.Module):
    def __init__(self, token_dim, base_dt=0.1):
        super(AdaptiveSDE, self).__init__()
        self.base_dt = base_dt
        self.update_net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim)
        )
        # Small network to compute per-token uncertainty
        self.uncertainty_net = nn.Sequential(
            nn.Linear(token_dim, 1),
            nn.Sigmoid()  # uncertainty in [0,1]
        )

    def forward(self, tokens):
        # tokens: [B, N, token_dim]
        uncertainty = self.uncertainty_net(tokens) # shape [B, N, 1]
        # Adapt step size: higher uncertainty gets larger dt
        dt = self.base_dt * (1.0 + uncertainty)
        update = self.update_net(tokens)
        noise = torch.randn_like(tokens)
        # Reduce noise in low-uncertainty regions (using inverse scaling)
        noise_modulation = 1.0 / (1.0 + uncertainty)
        # sqrt(dt) must be computed per token
        sqrt_dt = torch.sqrt(dt)
        new_tokens = tokens + dt * update + sqrt_dt * noise * noise_modulation
        return new_tokens, uncertainty

# Cross-token attention module
class CrossTokenAttention(nn.Module):
    def __init__(self, token_dim, num_heads=4):
        super(CrossTokenAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, tokens):
        # tokens: [B, N, token_dim]
        # Use tokens as query, key, value
        attn_output, attn_weights = self.attention(tokens, tokens, tokens)
        # Residual connection
        tokens = tokens + attn_output
        return tokens, attn_weights

# Complete AT-BFN Pipeline
class ATBFNPipeline(nn.Module):
    def __init__(self, config, token_dim=64, use_attention=True):
        super(ATBFNPipeline, self).__init__()
        self.config = config
        self.use_attention = use_attention
        self.token_dim = token_dim
        
        # Tokenizer
        self.tokenizer = DynamicTokenizer(
            in_channels=config.get('in_channels', 3), 
            token_dim=token_dim
        )
        
        # Token evolution - use adaptive or fixed SDE based on config
        if config.get('use_adaptive_sde', True):
            self.token_evolution = AdaptiveSDE(
                token_dim, 
                base_dt=config.get('base_dt', 0.1)
            )
        else:
            self.token_evolution = FixedStepSDE(
                token_dim, 
                dt=config.get('base_dt', 0.1)
            )
            
        # Attention module if enabled
        if use_attention:
            self.attention_module = CrossTokenAttention(token_dim, num_heads=4)
            
        # Reconstruction module
        self.reconstruction = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 32 * 32),  # reconstruct image with shape (3,32,32)
            nn.Tanh()
        )

    def forward(self, x, num_steps=None):
        # If num_steps not provided, use config value
        if num_steps is None:
            num_steps = self.config.get('quick_test_iterations', 3) if self.config.get('quick_test', False) else self.config.get('num_evolution_steps', 5)
        
        # Check if input is already tokens or an image
        if len(x.shape) == 3:  # [B, N, token_dim] - already tokens
            tokens = x
            complexity_map = torch.ones((x.shape[0], 1, 8, 8), device=x.device)  # Dummy complexity map
        else:  # [B, C, H, W] - image input
            tokens, complexity_map = self.tokenizer(x)
        
        # Evolution steps
        uncertainties = []
        attn_weights_history = []
        
        for i in range(num_steps):
            # Apply token evolution
            if isinstance(self.token_evolution, AdaptiveSDE):
                tokens, uncertainty = self.token_evolution(tokens)
                uncertainties.append(uncertainty.mean().item())
                print(f"Iteration {i+1} -- avg uncertainty: {uncertainty.mean().item():.4f}")
            else:
                tokens = self.token_evolution(tokens)
                
            # Apply attention if enabled
            if self.use_attention:
                tokens, attn_weights = self.attention_module(tokens)
                attn_weights_history.append(attn_weights)
                print(f"Iteration {i+1} -- avg attention weight: {attn_weights.mean().item():.4f}")
        
        # Reconstruct image from tokens
        tokens_avg = tokens.mean(dim=1)
        img_flat = self.reconstruction(tokens_avg)
        reconstructed = img_flat.view(-1, 3, 32, 32)
        
        return {
            'tokens': tokens,
            'complexity_map': complexity_map,
            'reconstructed': reconstructed,
            'uncertainties': uncertainties,
            'attention_weights': attn_weights_history
        }

def train_epoch(model, dataloader, optimizer, config, device='cuda'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Limit iterations if in quick test mode
    max_iters = 10 if config.get('quick_test', False) else len(dataloader)
    
    for i, (imgs, _) in enumerate(dataloader):
        if i >= max_iters:
            break
            
        imgs = imgs.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(imgs)
        reconstructed = output['reconstructed']
        
        # Simple reconstruction loss (MSE)
        loss = F.mse_loss(reconstructed, F.interpolate(imgs, size=(32, 32)))
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i+1) % 10 == 0 or i == 0:
            print(f"Batch {i+1}/{max_iters}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / max_iters
    elapsed = time.time() - start_time
    print(f"Epoch completed in {elapsed:.2f} seconds. Average loss: {avg_loss:.4f}")
    
    return avg_loss

def train_model(model, train_loader, config, device='cuda'):
    """Train the model"""
    # Set device
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 0.0001)
    )
    
    # Number of epochs (reduced for quick test)
    num_epochs = 2 if config.get('quick_test', False) else config.get('num_epochs', 10)
    
    print(f"Starting training for {num_epochs} epochs")
    losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        loss = train_epoch(model, train_loader, optimizer, config, device)
        losses.append(loss)
        
    print(f"Training completed with final loss: {losses[-1]:.4f}")
    return losses
