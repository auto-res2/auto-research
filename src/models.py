"""
Model implementations for SAC-Seg experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union, Any


class BaseModel(nn.Module):
    """
    Base model implementing the CAT-Seg framework without seed-based optimizations.
    This serves as the baseline for comparison with SAC-Seg.
    """
    def __init__(
        self,
        input_channels: int = 3,
        embedding_dim: int = 64,
        num_classes: int = 16,
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize the BaseModel.
        
        Args:
            input_channels: Number of input image channels
            embedding_dim: Dimension of the embeddings
            num_classes: Number of segmentation classes
            image_size: Input image size (height, width)
        """
        super(BaseModel, self).__init__()
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embedding_dim, kernel_size=1)
        )
        
        self.text_encoder = nn.Embedding(num_classes, embedding_dim)
        
        num_heads = min(4, max(1, embedding_dim // 16 * 4))
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.class_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.image_size = image_size
    
    def _compute_cost_volume(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dense cost volume between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings (B, C, H, W)
            text_embeddings: Text embeddings (num_classes, C)
            
        Returns:
            Cost volume tensor (B, num_classes, H, W)
        """
        b, c, h, w = image_embeddings.shape
        
        if c != self.embedding_dim:
            raise ValueError(f"was expecting embedding dimension of {self.embedding_dim}, but got {c}")
            
        image_embeddings_flat = image_embeddings.view(b, c, h * w)
        
        image_embeddings_flat = image_embeddings_flat.permute(0, 2, 1)
        
        image_embeddings_flat = F.normalize(image_embeddings_flat, dim=2)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        cost_volume = torch.matmul(image_embeddings_flat, text_embeddings.t())
        
        cost_volume = cost_volume.permute(0, 2, 1).view(b, -1, h, w)
        
        return cost_volume
    
    def _aggregate_costs(self, cost_volume: torch.Tensor) -> torch.Tensor:
        """
        Aggregate cost volume using transformer blocks.
        
        Args:
            cost_volume: Cost volume tensor (B, num_classes, H, W)
            
        Returns:
            Aggregated cost volume (B, num_classes, H, W)
        """
        b, c, h, w = cost_volume.shape
        
        cost_seq = cost_volume.permute(0, 2, 3, 1).view(b, h * w, c)
        
        spatial_features = self.spatial_transformer(cost_seq)
        
        class_input = spatial_features.view(b * h * w, c)
        
        class_features = self.class_transformer(class_input.unsqueeze(1)).squeeze(1)
        
        aggregated_cost = class_features.view(b, h, w, c).permute(0, 3, 1, 2)
        
        return aggregated_cost
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        image_embeddings = self.image_encoder(x)
        
        text_embeddings = self.text_encoder.weight  # (num_classes, embedding_dim)
        
        cost_volume = self._compute_cost_volume(image_embeddings, text_embeddings)
        
        aggregated_cost = self._aggregate_costs(cost_volume)
        
        logits = self.decoder(aggregated_cost)
        
        return logits


class SACSeg(BaseModel):
    """
    SAC-Seg model implementing Seed-Augmented Cost Aggregation for Segmentation.
    This extends the BaseModel with seed-based gradient approximation mechanisms.
    """
    def __init__(
        self,
        input_channels: int = 3,
        embedding_dim: int = 64,
        num_classes: int = 16,
        image_size: Tuple[int, int] = (512, 512),
        seed_config: Optional[Dict] = None
    ):
        """
        Initialize the SAC-Seg model.
        
        Args:
            input_channels: Number of input image channels
            embedding_dim: Dimension of the embeddings
            num_classes: Number of segmentation classes
            image_size: Input image size (height, width)
            seed_config: Configuration for seed-based methods
        """
        super(SACSeg, self).__init__(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            image_size=image_size
        )
        
        self.seed_config = seed_config or {
            'num_seeds': 10,
            'seed_prob_threshold': 0.5,
            'perturbation_scale': 0.01
        }
        
        self.seed_prob_predictor = nn.Sequential(
            nn.Conv2d(embedding_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _generate_seed_map(
        self,
        image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a seed map for sampling the cost volume.
        
        Args:
            image_embeddings: Image embeddings (B, C, H, W)
            
        Returns:
            Tuple of (seed_indices, seed_probs) tensors
        """
        b, c, h, w = image_embeddings.shape
        
        seed_probs = self.seed_prob_predictor(image_embeddings)  # (B, 1, H, W)
        
        seed_probs_flat = seed_probs.view(b, -1)  # (B, H*W)
        
        num_seeds = min(self.seed_config['num_seeds'], h * w)
        
        seed_indices = []
        for i in range(b):
            valid_probs = seed_probs_flat[i] > self.seed_config['seed_prob_threshold']
            
            if valid_probs.sum() == 0:
                indices = torch.randperm(h * w, device=image_embeddings.device)[:num_seeds]
            else:
                probs = seed_probs_flat[i].clone()
                probs[~valid_probs] = 0.0
                probs = probs / probs.sum()
                
                indices = torch.multinomial(probs, num_samples=min(num_seeds, valid_probs.sum()), replacement=False)
            
            seed_indices.append(indices)
            
        seed_indices = torch.stack(seed_indices)  # (B, num_seeds)
        
        return seed_indices, seed_probs
    
    def _compute_seed_guided_cost_volume(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        seed_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute seed-guided cost volume between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings (B, C, H, W)
            text_embeddings: Text embeddings (num_classes, C)
            seed_indices: Indices of seeds to sample (B, num_seeds)
            
        Returns:
            Seed-guided cost volume tensor (B, num_classes, H, W)
        """
        b, c, h, w = image_embeddings.shape
        num_classes = text_embeddings.shape[0]
        
        image_embeddings_flat = image_embeddings.view(b, c, h * w)
        
        image_embeddings_flat = image_embeddings_flat.permute(0, 2, 1)
        
        image_embeddings_flat = F.normalize(image_embeddings_flat, dim=2)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        cost_volume = torch.zeros(b, num_classes, h * w, device=image_embeddings.device)
        
        for i in range(b):
            seed_embeddings = image_embeddings_flat[i, seed_indices[i]]  # (num_seeds, C)
            
            seed_costs = torch.matmul(seed_embeddings, text_embeddings.t())  # (num_seeds, num_classes)
            
            for j, idx in enumerate(seed_indices[i]):
                cost_volume[i, :, idx] = seed_costs[j]
        
        cost_volume = cost_volume.view(b, num_classes, h, w)
        
        return cost_volume
    
    def _zeroth_order_gradient_approximation(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        seed_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform zeroth-order gradient approximation for fine-tuning.
        
        Args:
            image_embeddings: Image embeddings (B, C, H, W)
            text_embeddings: Text embeddings (num_classes, C)
            seed_indices: Indices of seeds to sample (B, num_seeds)
            
        Returns:
            Gradient-approximated cost volume (B, num_classes, H, W)
        """
        b, c, h, w = image_embeddings.shape
        num_classes = text_embeddings.shape[0]
        perturbation_scale = self.seed_config['perturbation_scale']
        
        image_embeddings_flat = image_embeddings.view(b, c, h * w)
        
        grad_cost_volume = torch.zeros(b, num_classes, h * w, device=image_embeddings.device)
        
        for i in range(b):
            for j, idx in enumerate(seed_indices[i]):
                seed_embedding = image_embeddings_flat[i, :, idx].clone()  # (C,)
                
                perturbation = torch.randn_like(seed_embedding) * perturbation_scale
                pos_embedding = F.normalize(seed_embedding + perturbation, dim=0)
                neg_embedding = F.normalize(seed_embedding - perturbation, dim=0)
                
                pos_cost = torch.matmul(pos_embedding, F.normalize(text_embeddings, dim=1).t())  # (num_classes,)
                neg_cost = torch.matmul(neg_embedding, F.normalize(text_embeddings, dim=1).t())  # (num_classes,)
                
                grad_cost = (pos_cost - neg_cost) / (2 * perturbation_scale)
                
                grad_cost_volume[i, :, idx] = grad_cost
        
        grad_cost_volume = grad_cost_volume.view(b, num_classes, h, w)
        
        return grad_cost_volume
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with seed-guided cost aggregation.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        image_embeddings = self.image_encoder(x)
        
        text_embeddings = self.text_encoder.weight  # (num_classes, embedding_dim)
        
        seed_indices, seed_probs = self._generate_seed_map(image_embeddings)
        
        cost_volume = self._compute_seed_guided_cost_volume(
            image_embeddings, text_embeddings, seed_indices
        )
        
        if self.training:
            grad_cost_volume = self._zeroth_order_gradient_approximation(
                image_embeddings, text_embeddings, seed_indices
            )
            cost_volume = cost_volume + grad_cost_volume
        
        aggregated_cost = self._aggregate_costs(cost_volume)
        
        logits = self.decoder(aggregated_cost)
        
        return logits
