import torch
import numpy as np
from typing import Tuple

def generate_synthetic_data(d1: int, d2: int, rank: int, missing_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    true_matrix = torch.randn(d1, rank) @ torch.randn(rank, d2)
    mask = torch.rand(d1, d2) > missing_ratio
    observed_matrix = true_matrix * mask
    return observed_matrix, mask

def prepare_data_batch(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    num_samples = data.size(0)
    indices = torch.randperm(num_samples)[:batch_size]
    return data[indices]
