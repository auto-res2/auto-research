"""
Optimizers module containing the implementation of the Adaptive Curvature Momentum (ACM) optimizer.
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class ACM(Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer utilizes local quadratic approximations to adaptively adjust the update direction and scale.
    It combines Adam-style adaptability with curvature-aware updates for faster convergence.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.001)
        beta (float, optional): coefficient for momentum term (default: 0.9)
        beta2 (float, optional): coefficient for second moment estimation (default: 0.999)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adaptive_weight_decay (bool, optional): whether to use adaptive weight decay (default: True)
    """
    
    def __init__(self, params, lr=0.001, beta=0.9, beta2=0.999, eps=1e-8, 
                 weight_decay=0, adaptive_weight_decay=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, beta=beta, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, adaptive_weight_decay=adaptive_weight_decay)
        super(ACM, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        
        Returns:
            loss: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            beta = group['beta']
            beta2 = group['beta2']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            adaptive_weight_decay = group['adaptive_weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(p.data)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                prev_grad = state['prev_grad']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Calculate gradient change rate (delta_g)
                delta_grad = grad - prev_grad
                state['prev_grad'] = grad.clone()
                
                # Update biased first moment estimate (momentum)
                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias correction
                bias_correction1 = 1 - beta ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute curvature estimate
                # We use the norm of the gradient as a simple curvature estimate
                curvature = grad.norm() + eps
                
                # Adaptive learning rate based on curvature
                # η_t = α/(1+β⋅Curvature(g_t))
                adaptive_lr = lr / (1 + beta * curvature)
                
                # Store adaptive learning rate for logging
                state['adaptive_lr'] = adaptive_lr.item() if isinstance(adaptive_lr, torch.Tensor) else adaptive_lr
                
                # Compute adaptive weight decay if enabled
                if adaptive_weight_decay and weight_decay > 0:
                    # Increase weight decay when curvature is high
                    adaptive_wd = weight_decay * (1 + curvature)
                    p.data.mul_(1 - adaptive_lr * adaptive_wd)
                
                # Compute the update
                # Include both the momentum term and the delta_grad term
                update = exp_avg / bias_correction1 + beta * delta_grad
                
                # Apply the update
                p.data.add_(update, alpha=-adaptive_lr)
                
        return loss


class ACMNumpy:
    """
    NumPy implementation of the Adaptive Curvature Momentum (ACM) optimizer.
    This is used for synthetic function optimization experiments.
    
    Args:
        lr (float, optional): learning rate (default: 0.01)
        beta (float, optional): coefficient for momentum term (default: 0.9)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """
    
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.prev_grad = None
        self.step_count = 0
        
    def update(self, x, grad):
        """
        Update parameters using ACM.
        
        Args:
            x: Current parameter values
            grad: Gradient of the loss with respect to x
            
        Returns:
            x_new: Updated parameter values
            adaptive_lr: The adaptive learning rate used for this update
        """
        import numpy as np
        
        self.step_count += 1
        
        # Initialize previous gradient if this is the first step
        if self.prev_grad is None or self.prev_grad.shape != grad.shape:
            self.prev_grad = np.zeros_like(grad)
        
        # Calculate gradient change rate (delta_g)
        delta_grad = grad - self.prev_grad
        self.prev_grad = grad.copy()
        
        # Compute curvature estimate (using gradient norm)
        curvature = np.linalg.norm(grad) + self.eps
        
        # Adaptive learning rate based on curvature
        # η_t = α/(1+β⋅Curvature(g_t))
        adaptive_lr = self.lr / (1 + self.beta * curvature)
        
        # Compute the update (including both gradient and delta_grad)
        update = grad + self.beta * delta_grad
        
        # Apply the update
        x_new = x - adaptive_lr * update
        
        return x_new, adaptive_lr
