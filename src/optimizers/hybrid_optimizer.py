import torch
from torch.optim import Optimizer

class HybridOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.0, 0.9, 0.99], momentum=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
            
        defaults = dict(lr=lr, betas=betas, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super(HybridOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
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
                
                if len(param_state) == 0:
                    param_state['momentum_buffers'] = {
                        beta: torch.zeros_like(p.data)
                        for beta in betas
                    }
                    param_state['dual_avg_buffer'] = torch.zeros_like(p.data)
                
                avg_updates = torch.zeros_like(p.data)
                
                for beta in betas:
                    buf = param_state['momentum_buffers'][beta]
                    buf.mul_(beta).add_(d_p)
                    avg_updates.add_(buf)
                
                dual_avg_buffer = param_state['dual_avg_buffer']
                dual_avg_buffer.add_(d_p)
                
                p.data.sub_(lr * avg_updates / len(betas) + dual_avg_buffer * eps)
        
        return loss
