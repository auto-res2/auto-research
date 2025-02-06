import torch
import torch.nn as nn
from typing import List

class LRMCNet(nn.Module):
    def __init__(self, d1: int, d2: int, r: int, alpha: float, p: int,
                 max_iter: int, zeta0: List[float], eta0: List[float],
                 device: str = 'cpu'):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.r = r
        self.alpha = alpha
        self.p = p
        self.max_iter = max_iter
        self.zeta0 = nn.Parameter(torch.tensor(zeta0))
        self.eta0 = nn.Parameter(torch.tensor(eta0))
        self.device = device
        
        self.projection_layer = nn.Sequential(
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, r)
        )
    
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.requires_grad_(True)
            loss = 0.5 * torch.norm(x, p=self.p) ** 2
            grad = torch.autograd.grad(loss, x)[0]
        return grad
    
    def _project_rank(self, x: torch.Tensor, r: int) -> torch.Tensor:
        U, S, V = torch.svd(x)
        S = torch.cat([S[:r], torch.zeros_like(S[r:])])
        return torch.matmul(torch.matmul(U, torch.diag(S)), V.t())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        projected = self.projection_layer(x)
        
        for i in range(self.max_iter):
            grad = self._compute_gradient(projected)
            projected = projected - self.zeta0[i] * grad
            projected = self._project_rank(projected, self.r)
        
        return projected
