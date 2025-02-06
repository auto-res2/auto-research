import torch
import torch.nn as nn
from typing import Tuple

class LearnableGatedPooling(nn.Module):
    def __init__(self, input_dim: int, seq_len: int):
        super(LearnableGatedPooling, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.gate_linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weighted_x = x * self.weights
        gate_values = torch.sigmoid(self.gate_linear(x)).squeeze(2)
        gated_x = weighted_x * gate_values.unsqueeze(2)
        pooled_vector = torch.mean(gated_x, dim=1)
        return pooled_vector, gate_values
