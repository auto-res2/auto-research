from typing import List, Optional
import torch
from torch.optim import Optimizer

class HybridOptimizer(Optimizer):
    """Hybrid optimizer combining AggMo's momentum with MADGRAD's adaptive updates."""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: List[float] = [0.0, 0.9, 0.99],
        momentum: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0
    ):
        defaults = dict(lr=lr, betas=betas, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, betas, momentum, eps, weight_decay = (
                group['lr'],
                group['betas'],
                group['momentum'],
                group['eps'],
                group['weight_decay']
            )
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                param_state = self.state[p]
                
                # Initialize state
                if len(param_state) == 0:
                    param_state['momentum_buffers'] = {
                        beta: torch.zeros_like(p.data)
                        for beta in betas
                    }
                    param_state['dual_avg_buffer'] = torch.zeros_like(p.data)
                
                # Update momentum buffers
                avg_updates = torch.zeros_like(p.data)
                for beta in betas:
                    buf = param_state['momentum_buffers'][beta]
                    buf.mul_(beta).add_(d_p)
                    avg_updates.add_(buf)
                
                # Update dual averaging buffer
                dual_avg_buffer = param_state['dual_avg_buffer']
                dual_avg_buffer.add_(d_p)
                
                # Apply updates
                p.data.sub_(
                    lr * avg_updates / len(betas) + dual_avg_buffer * eps
                )
        
        return loss
