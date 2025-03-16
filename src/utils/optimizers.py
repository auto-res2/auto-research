import math
import torch
import torch.optim as optim

class ACM(optim.Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer combines Adam-style adaptability with curvature-aware updates.
    It dynamically adjusts the learning rate based on the local curvature of the loss landscape.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients for computing running averages
            of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator for numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        curvature_coeff (float, optional): coefficient for curvature estimation (default: 1e-2)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, curvature_coeff=1e-2):
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
        if not 0.0 <= curvature_coeff:
            raise ValueError(f"Invalid curvature coefficient: {curvature_coeff}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, curvature_coeff=curvature_coeff)
        super(ACM, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(ACM, self).__setstate__(state)
    
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
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ACM does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous gradient for curvature estimation
                    state['prev_grad'] = torch.zeros_like(p.data)
                    # Curvature estimate
                    state['curvature'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad, curvature = state['prev_grad'], state['curvature']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Compute curvature estimate using gradient change
                if state['step'] > 1:
                    grad_change = grad - prev_grad
                    # Update curvature estimate (approximation of Hessian diagonal)
                    curvature.mul_(beta2).addcmul_(grad_change, grad_change, value=1 - beta2)
                
                # Store current gradient for next iteration
                prev_grad.copy_(grad)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    # Adaptive regularization based on curvature
                    if state['step'] > 1:
                        # Stronger regularization in high curvature regions
                        reg_strength = group['weight_decay'] * (1.0 + torch.sqrt(curvature))
                        # Apply element-wise weight decay with curvature-based regularization
                        p.data.add_(p.data * (-group['lr'] * reg_strength))
                    else:
                        # Apply standard weight decay
                        p.data.add_(p.data * (-group['lr'] * group['weight_decay']))
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Calculate base step size (scalar)
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply update
                if state['step'] > 1:
                    # Apply curvature-aware adjustment
                    curvature_factor = 1.0 / (1.0 + group['curvature_coeff'] * torch.sqrt(curvature + group['eps']))
                    update = -step_size * exp_avg / denom
                    p.data.add_(update * curvature_factor)
                else:
                    # Update without curvature factor
                    update = -step_size * exp_avg / denom
                    p.data.add_(update)
                
        return loss


class ACM_NoCurvature(optim.Optimizer):
    """
    Ablated version of ACM without curvature-based scaling.
    This is similar to Adam but with the specific implementation details of ACM.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
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
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ACM_NoCurvature, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(ACM_NoCurvature, self).__setstate__(state)
    
    def step(self, closure=None):
        """Performs a single optimization step without curvature adjustment."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ACM_NoCurvature does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate (without curvature adjustment)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Calculate step size (scalar)
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                update = -step_size * exp_avg / denom
                p.data.add_(update)
                
        return loss


class ACM_NoRegularization(optim.Optimizer):
    """
    Ablated version of ACM without adaptive regularization.
    This version uses curvature only for scaling the learning rate.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, curvature_coeff=1e-2):
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
        if not 0.0 <= curvature_coeff:
            raise ValueError(f"Invalid curvature coefficient: {curvature_coeff}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, curvature_coeff=curvature_coeff)
        super(ACM_NoRegularization, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(ACM_NoRegularization, self).__setstate__(state)
    
    def step(self, closure=None):
        """Performs a single optimization step with curvature scaling but without adaptive regularization."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ACM_NoRegularization does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous gradient for curvature estimation
                    state['prev_grad'] = torch.zeros_like(p.data)
                    # Curvature estimate
                    state['curvature'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad, curvature = state['prev_grad'], state['curvature']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Compute curvature estimate using gradient change
                if state['step'] > 1:
                    grad_change = grad - prev_grad
                    # Update curvature estimate (approximation of Hessian diagonal)
                    curvature.mul_(beta2).addcmul_(grad_change, grad_change, value=1 - beta2)
                
                # Store current gradient for next iteration
                prev_grad.copy_(grad)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Apply standard weight decay (not adaptive)
                if group['weight_decay'] > 0:
                    p.data.add_(p.data * (-group['lr'] * group['weight_decay']))
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Calculate base step size (scalar)
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply update
                if state['step'] > 1:
                    # Apply curvature-aware adjustment
                    curvature_factor = 1.0 / (1.0 + group['curvature_coeff'] * torch.sqrt(curvature + group['eps']))
                    update = -step_size * exp_avg / denom
                    p.data.add_(update * curvature_factor)
                else:
                    # Update without curvature factor
                    update = -step_size * exp_avg / denom
                    p.data.add_(update)
                
        return loss
