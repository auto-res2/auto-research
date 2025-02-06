import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class TitansMemory(nn.Module):
    def __init__(self, input_dim: int, memory_dim: int):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_dim, input_dim))
        self.query_proj = nn.Linear(input_dim, memory_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        attention = torch.softmax(query @ self.memory, dim=-1)
        return attention @ self.memory

class LRMCModule(nn.Module):
    def __init__(self, d1: int, d2: int, rank: int, max_iter: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d1, rank) / np.sqrt(rank))
        self.V = nn.Parameter(torch.randn(rank, d2) / np.sqrt(rank))
        self.max_iter = max_iter
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for _ in range(self.max_iter):
            pred = self.U @ self.V
            error = mask * (x - pred)
            self.U.data += error @ self.V.t()
            self.V.data += self.U.t() @ error
        return self.U @ self.V

class IntegratedMemoryLRMC(nn.Module):
    def __init__(self, d1: int, d2: int, rank: int, max_iter: int):
        super().__init__()
        self.memory = TitansMemory(d1, rank)
        self.completion = LRMCModule(d1, d2, rank, max_iter)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        memory_enhanced = self.memory(x)
        completed = self.completion(memory_enhanced, mask)
        return completed

def train_model(model: nn.Module, data: torch.Tensor, mask: torch.Tensor, 
                num_epochs: int, lr: float = 1e-3) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data, mask)
        loss = criterion(output * mask, data * mask)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    return model
