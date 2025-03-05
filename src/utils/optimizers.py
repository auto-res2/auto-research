import torch
from torch.optim import Optimizer

class ACMOptimizer(Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer combines momentum-based updates with curvature-aware learning rate adaptation.
    It estimates local curvature using the difference between current and previous gradients,
    and adjusts the learning rate accordingly.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): base learning rate (default: 0.01)
        beta (float, optional): momentum factor (default: 0.9)
        curvature_influence (float, optional): factor controlling the influence of curvature on learning rate (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adaptive_regularization (bool, optional): whether to use adaptive regularization (default: True)
    """
    def __init__(self, params, lr=0.01, beta=0.9, curvature_influence=0.1, 
                 weight_decay=0, adaptive_regularization=True):
        defaults = dict(lr=lr, beta=beta, curvature_influence=curvature_influence,
                        weight_decay=weight_decay, adaptive_regularization=adaptive_regularization)
        super(ACMOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            curvature_influence = group['curvature_influence']
            weight_decay = group['weight_decay']
            adaptive_regularization = group['adaptive_regularization']
            
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
                    # Initialize momentum buffer
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    # For the first step, we don't have previous gradients
                    state['prev_grad'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                momentum_buffer = state['momentum_buffer']
                prev_grad = state['prev_grad']
                
                # Estimate curvature as the absolute difference between current and previous gradients
                curvature_est = (grad - prev_grad).abs()
                
                # Store current gradient for next iteration
                state['prev_grad'] = grad.clone()
                
                # Compute adaptive learning rate based on curvature
                # We need to apply the adaptive learning rate element-wise
                adaptive_lr = lr / (1.0 + curvature_influence * curvature_est)
                
                # Apply adaptive regularization if enabled
                if adaptive_regularization and weight_decay > 0:
                    # Increase regularization in high curvature regions
                    adaptive_decay = weight_decay * (1.0 + curvature_est.mean().item())
                    grad = grad.add(p.data, alpha=adaptive_decay - weight_decay)
                
                # Update momentum buffer with exponential moving average
                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                
                # Update parameters using adaptive learning rate and momentum buffer
                # We need to multiply the momentum buffer by the adaptive learning rate element-wise
                # and then subtract it from the parameters
                p.data.add_(-adaptive_lr * momentum_buffer)
        
        return loss
