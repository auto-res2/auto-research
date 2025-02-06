import torch
import torch.nn as nn
from typing import List
from .memory_module import MemoryModule
from .lrmc_net import LRMCNet

class HybridOptimizer(nn.Module):
    def __init__(self, d1: int, d2: int, r: int, alpha: float, p: int,
                 max_iter: int, zeta0: List[float], eta0: List[float],
                 device: str = 'cpu'):
        super().__init__()
        self.memory_module = MemoryModule(d1, device)
        self.optimization_net = LRMCNet(d1, d2, r, alpha, p, max_iter,
                                      zeta0, eta0, device)
        
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        context_information = self.memory_module(input_data)
        optimized_result = self.optimization_net(context_information)
        return optimized_result
