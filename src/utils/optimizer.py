"""
Implementation of the Adaptive Curvature Momentum (ACM) optimizer.

The ACM optimizer utilizes local quadratic approximations to adaptively adjust
the update direction and scale based on curvature information.
"""

import torch
from torch.optim.optimizer import Optimizer


class ACM(Optimizer):
    """
    Adaptive Curvature Momentum (ACM) optimizer.
    
    The adaptive rule adjusts learning rate based on curvature:
        adaptive_lr = lr/(1 + beta * curvature_estimate)
    where the curvature is computed as a moving average of absolute gradients.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.01)
        beta (float, optional): curvature influence factor (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=0.01, beta=0.1, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(ACM, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
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
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                # Apply weight decay if needed
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)
                
                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['curvature'] = torch.zeros_like(grad)
                    state['prev_grad'] = torch.zeros_like(grad)
                
                # Store previous gradient for curvature estimation
                if 'prev_grad' in state:
                    prev_grad = state['prev_grad']
                    # Estimate curvature using gradient change rate
                    curvature_update = (grad - prev_grad).abs()
                    state['prev_grad'] = grad.clone()
                else:
                    curvature_update = grad.abs()
                    state['prev_grad'] = grad.clone()
                
                # Update curvature estimate using moving average
                state['curvature'] = 0.9 * state['curvature'] + 0.1 * curvature_update
                
                # Compute adaptive learning rate for this parameter tensor
                adaptive_lr = lr / (1.0 + beta * state['curvature'])
                
                # Update parameters
                p.data.add_(-adaptive_lr * grad)
                
        return loss
