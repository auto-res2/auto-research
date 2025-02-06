import torch
import torch.nn as nn
from typing import Tuple

class MemoryModule(nn.Module):
    def __init__(self, d1: int, device: str = 'cpu'):
        super().__init__()
        self.d1 = d1
        self.device = device
        
        self.key_network = nn.Sequential(
            nn.Linear(d1, d1 * 2),
            nn.ReLU(),
            nn.Linear(d1 * 2, d1)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(d1, d1 * 2),
            nn.ReLU(),
            nn.Linear(d1 * 2, d1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key_network(x)
        values = self.value_network(x)
        attention = torch.matmul(keys, keys.transpose(-2, -1))
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, values)
        return context
