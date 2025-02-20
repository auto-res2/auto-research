import torch
from torch.optim import Optimizer

class AggMoOptimizer(Optimizer):
    def __init__(self, params, lr=0.1, betas=[0.0, 0.9, 0.99], weight_decay=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(AggMoOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            betas = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['momentum_buffers'] = [torch.zeros_like(p.data) for _ in betas]
                
                for i, beta in enumerate(betas):
                    buf = param_state['momentum_buffers'][i]
                    buf.mul_(beta).add_(d_p)
                
                update = torch.zeros_like(p.data)
                for buf in param_state['momentum_buffers']:
                    update.add_(buf)
                update.div_(len(betas))
                
                p.data.add_(update, alpha=-lr)

        return loss
