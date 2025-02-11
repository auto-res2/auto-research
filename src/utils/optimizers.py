import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple
from torch import Tensor

class HybridOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.0, 0.9, 0.99], momentum=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super(HybridOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            momentum = group['momentum']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffers'] = {beta: torch.zeros_like(p) for beta in betas}
                    state['dual_avg_buffer'] = torch.zeros_like(p)

                state['step'] += 1

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Calculate aggregated momentum updates
                avg_updates = torch.zeros_like(p)
                for beta in betas:
                    buf = state['momentum_buffers'][beta]
                    buf.mul_(beta).add_(grad)
                    avg_updates.add_(buf)
                avg_updates.div_(len(betas))

                # Update dual averaging buffer
                dual_avg = state['dual_avg_buffer']
                dual_avg.add_(grad)

                # Combined update step
                p.add_(avg_updates.mul(-lr))
                p.add_(dual_avg.mul(-eps))

        return loss
