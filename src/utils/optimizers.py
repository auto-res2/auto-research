import torch
import torch.optim as optim

class ACMOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9, curvature_influence=0.1):
        # The momentum_buffer variable will be stored in state, so we do not initialize it in defaults.
        defaults = dict(lr=lr, beta=beta, curvature_influence=curvature_influence)
        super(ACMOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        # Only single parameter group is assumed in this simple implementation.
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            curvature_influence = group['curvature_influence']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    # For the very first update, simply store the current gradient.
                    state['momentum_buffer'] = torch.clone(grad).detach()
                    p.data.add_(-lr * grad)
                else:
                    buf = state['momentum_buffer']
                    # Estimate curvature simply as the absolute difference between current gradient and previous momentum.
                    curvature_est = (grad - buf).abs()
                    # Compute an adaptive per-component learning rate.
                    adaptive_lr = lr / (1.0 + curvature_influence * curvature_est)
                    # Update the momentum buffer with exponential moving average.
                    buf.mul_(beta).add_(grad, alpha=1 - beta)
                    # Update parameters using (elementwise) adaptive learning rate and momentum buffer.
                    p.data.add_(-adaptive_lr * buf)
        
        return loss
