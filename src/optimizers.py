"""
Adaptive Curvature Momentum (ACM) Optimizer Implementation

This module implements the ACM optimizer which utilizes local quadratic approximations
to adaptively adjust the update direction and scale based on curvature information.
"""

import torch
from torch.optim import Optimizer
import math


class ACMOptimizer(Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer combines Adam-style adaptability with curvature-aware updates.
    It adapts step sizes dynamically, taking larger steps in flat regions and 
    smaller steps in sharp valleys of the loss landscape.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        curvature_coef (float, optional): coefficient for curvature influence (default: 0.1)
        adaptive_regularization (bool, optional): whether to use adaptive regularization (default: True)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, curvature_coef=0.1, adaptive_regularization=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= curvature_coef:
            raise ValueError(f"Invalid curvature coefficient: {curvature_coef}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, curvature_coef=curvature_coef,
                        adaptive_regularization=adaptive_regularization)
        super(ACMOptimizer, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(ACMOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('adaptive_regularization', True)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ACMOptimizer does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient for curvature estimation
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Curvature estimate
                    state['curvature'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad, curvature = state['prev_grad'], state['curvature']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Compute curvature estimate based on gradient change
                if state['step'] > 1:
                    grad_diff = grad - prev_grad
                    curvature = beta2 * curvature + (1 - beta2) * torch.abs(grad_diff)
                
                # Store current gradient for next iteration
                prev_grad.copy_(grad)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    if group['adaptive_regularization']:
                        # Strengthen regularization in high curvature regions
                        curvature_factor = 1.0 + group['curvature_coef'] * curvature
                        effective_weight_decay = group['weight_decay'] * curvature_factor
                        p.mul_(1 - group['lr'] * effective_weight_decay)
                    else:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Compute step size with curvature adjustment
                # Use mean curvature for scalar adjustment to avoid tensor-to-scalar conversion issues
                mean_curvature = curvature.mean().item() if curvature.numel() > 1 else curvature.item()
                curvature_factor = 1.0 + group['curvature_coef'] * mean_curvature
                
                # Update parameters using standard Adam-style update with scalar learning rate
                p.addcdiv_(exp_avg, denom, value=-group['lr'] / curvature_factor * bias_correction1)
                
                # Store curvature for next iteration
                state['curvature'] = curvature
        
        return loss


class AdaBelief(Optimizer):
    """
    Implementation of AdaBelief optimizer
    
    Paper: "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
    https://arxiv.org/abs/2010.07468
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')
                
                amsgrad = group['amsgrad']
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update the variance estimate with the belief in observed gradients
                grad_diff = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
