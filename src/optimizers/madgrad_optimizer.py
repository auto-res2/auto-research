import torch
from torch.optim import Optimizer

class MADGRADOptimizer(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0, eps=1e-6):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super(MADGRADOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data)
                    state['x0'] = p.data.clone()
                    state['momentum'] = torch.zeros_like(p.data)

                s = state['s']
                x0 = state['x0']
                momentum = state['momentum']
                state['step'] += 1
                step = state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update momentum
                momentum.mul_(group['momentum']).add_(grad)
                
                # Update s (accumulator)
                s.add_(momentum.abs())
                
                # Compute update
                denom = s.sqrt().add_(group['eps'])
                update = momentum / denom
                
                # Update parameters
                p.data = x0.sub(update.mul(group['lr'] * torch.sqrt(torch.tensor(step))))

        return loss
