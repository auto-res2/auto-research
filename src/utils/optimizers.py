"""
Optimizers module for the auto-research project.
Contains implementation of the Adaptive Curvature Momentum (ACM) optimizer.
"""

import torch
from torch.optim.optimizer import Optimizer, required

class ACM(Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer combines momentum with adaptive learning rate scaling based on
    curvature approximations. It utilizes local quadratic approximations to adaptively
    adjust the update direction and scale.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: required)
        beta (float, optional): coefficient for momentum (default: 0.9)
        curvature_scale (float, optional): scaling factor for curvature adaptation (default: 1.0)
        eps (float, optional): term added to the denominator for numerical stability (default: 1e-8)
    """
    def __init__(self, params, lr=required, beta=0.9, curvature_scale=1.0, eps=1e-8):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0 or beta >= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if curvature_scale < 0.0:
            raise ValueError(f"Invalid curvature scale: {curvature_scale}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(lr=lr, beta=beta, curvature_scale=curvature_scale, eps=eps)
        super(ACM, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """
        Performs a single optimization step and also logs surrogate curvature statistics.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        
        Returns:
            loss: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            cs = group['curvature_scale']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['grad_square_avg'] = torch.zeros_like(p.data)
                
                # Update exponential moving average of squared gradients (surrogate curvature metric)
                state['grad_square_avg'] = beta * state['grad_square_avg'] + (1 - beta) * grad.pow(2)
                
                # Compute the adaptive scaling factor (inverse sqrt of smoothed curvature)
                adaptive_lr = lr / (torch.sqrt(state['grad_square_avg']) * cs + eps)
                
                # Update momentum buffer
                state['momentum_buffer'] = beta * state['momentum_buffer'] + grad
                
                # Parameter update: momentum + adaptive learning rate
                p.data.addcmul_(value=-1, tensor1=adaptive_lr, tensor2=state['momentum_buffer'])
                
        return loss
