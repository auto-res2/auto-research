"""Implementation of the Adaptive Curvature Momentum (ACM) optimizer."""

import torch
from torch.optim import Optimizer


class ACMOptimizer(Optimizer):
    """Adaptive Curvature Momentum (ACM) optimizer.
    
    This optimizer utilizes local quadratic approximations to adaptively
    adjust the update direction and scale. It maintains momentum based on
    past gradients and adjusts the learning rate based on curvature.
    
    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float, optional): Base learning rate (default: 0.01)
        beta (float, optional): Momentum factor (default: 0.9)
        curvature_influence (float, optional): Factor controlling the influence
            of curvature on learning rate adjustment (default: 0.1)
    """
    
    def __init__(self, params, lr=0.01, beta=0.9, curvature_influence=0.1, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= curvature_influence:
            raise ValueError(f"Invalid curvature influence: {curvature_influence}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, beta=beta, 
                       curvature_influence=curvature_influence,
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
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                    
                state = self.state[p]
                
                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                
                momentum_buffer = state['momentum_buffer']
                prev_grad = state['prev_grad']
                
                # Calculate curvature estimate as the difference between current and previous gradients
                if state['step'] > 1:
                    curvature_est = (grad - prev_grad).abs()
                    # Compute adaptive learning rate based on curvature
                    adaptive_lr = lr / (1.0 + curvature_influence * curvature_est)
                else:
                    adaptive_lr = lr * torch.ones_like(grad)
                    
                # Update momentum buffer
                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                
                # Update parameters
                p.data.add_(-adaptive_lr * momentum_buffer)
                
                # Store current gradient for next iteration
                state['prev_grad'].copy_(grad)
                
        return loss
