"""Implementation of the Adaptive Curvature Momentum (ACM) optimizer."""

import torch
from torch.optim import Optimizer


class ACMOptimizer(Optimizer):
    """Adaptive Curvature Momentum (ACM) optimizer.
    
    This optimizer utilizes local quadratic approximations to adaptively
    adjust the update direction and scale based on curvature information.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): momentum factor (default: 0.9)
        curvature_influence (float, optional): factor controlling the influence
            of curvature on learning rate adjustment (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, lr=1e-3, beta=0.9, curvature_influence=0.1, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, curvature_influence=curvature_influence,
                       weight_decay=weight_decay)
        super(ACMOptimizer, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            curvature_influence = group['curvature_influence']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                # Apply weight decay if specified
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                
                # Get momentum buffer and previous gradient
                momentum_buffer = state['momentum_buffer']
                prev_grad = state['prev_grad']
                
                # Estimate curvature as the absolute difference between current and previous gradient
                if state['step'] > 1:
                    curvature_est = (grad - prev_grad).abs()
                    
                   # Compute adaptive learning rate based on curvature
                    # Convert to scalar learning rates for each parameter element
                    adaptive_lr = lr / (1.0 + curvature_influence * curvature_est)
                else:
                    adaptive_lr = lr
                
                # Update momentum buffer
                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                
                # Update parameters using adaptive learning rate and momentum
                # Use element-wise multiplication for tensor learning rates
                p.data.add_(momentum_buffer.mul(-adaptive_lr))
                
                # Store current gradient for next iteration
                state['prev_grad'].copy_(grad)
                
        return loss
