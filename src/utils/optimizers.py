#!/usr/bin/env python3
"""
Implementation of the Adaptive Curvature Momentum (ACM) optimizer.

This optimizer combines momentum-based updates with curvature-aware learning rate
adjustments to improve convergence in deep learning models.
"""

import torch
from torch.optim.optimizer import Optimizer


class ACM(Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer.
    
    This optimizer dynamically adjusts the learning rate based on the local curvature
    of the loss landscape, taking larger steps in flat regions and smaller steps in
    sharp valleys.
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Base learning rate (default: 1e-3)
        beta (float, optional): Curvature influence factor (default: 0.9)
        weight_decay (float, optional): Weight decay factor (default: 0)
        eps (float, optional): Small constant for numerical stability (default: 1e-8)
        use_gradient_history (bool, optional): Whether to use gradient history for curvature
                                              estimation (default: False)
    """
    
    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0, eps=1e-8, use_gradient_history=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0 or beta >= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, eps=eps, 
                        use_gradient_history=use_gradient_history)
        super(ACM, self).__init__(params, defaults)
        
        # Initialize gradient history for each parameter group if enabled
        if use_gradient_history:
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p]['prev_grad'] = torch.zeros_like(p.data)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                                         and returns the loss.
        
        Returns:
            loss: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state if this is the first step
                if len(state) == 0 and not group['use_gradient_history']:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                # Update step count
                state['step'] = state.get('step', 0) + 1
                
                # Estimate curvature
                if group['use_gradient_history'] and 'prev_grad' in state:
                    # Estimate curvature using gradient change rate
                    grad_diff = grad - state['prev_grad']
                    curvature = torch.abs(grad_diff) / (torch.abs(grad) + group['eps'])
                    # Update previous gradient
                    state['prev_grad'] = grad.clone()
                else:
                    # Simpler curvature estimate using squared gradients
                    curvature = grad.pow(2)
                
                # Apply weight decay if set
                if group['weight_decay'] != 0:
                    # Adaptive regularization: stronger when curvature is high
                    adaptive_decay = group['weight_decay'] * (1 + curvature.mean())
                    grad = grad.add(p.data, alpha=adaptive_decay)
                
                # Compute effective learning rate based on curvature
                effective_lr = group['lr'] / (1 + group['beta'] * curvature.mean())
                
                # Update momentum buffer if using momentum
                if 'momentum_buffer' in state:
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(group['beta']).add_(grad)
                    update = momentum_buffer
                else:
                    update = grad
                
                # Update parameter using the curvature-adjusted learning rate
                p.data.add_(update, alpha=-effective_lr)
        
        return loss
