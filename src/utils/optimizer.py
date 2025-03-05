import torch
import torch.optim as optim

class ACMOptimizer(optim.Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer combines Adam-style adaptability with curvature-aware updates.
    It maintains a momentum term based on past gradients and uses second-order
    information to dynamically adjust the learning rate for each direction.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): base learning rate (default: 0.01)
        beta (float, optional): momentum coefficient (default: 0.9)
        curvature_influence (float, optional): coefficient for curvature adjustment (default: 0.1)
        weight_decay (float, optional): weight decay coefficient (default: 0)
    """
    def __init__(self, params, lr=0.01, beta=0.9, curvature_influence=0.1, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, curvature_influence=curvature_influence, weight_decay=weight_decay)
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
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if weight_decay != 0:
                    # Apply weight decay
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize state if this is the first step
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                momentum_buffer = state['momentum_buffer']
                prev_grad = state['prev_grad']
                
                # Estimate curvature as the absolute difference between current gradient and previous gradient
                if state['step'] > 1:
                    curvature_est = (grad - prev_grad).abs()
                    
                    # Compute adaptive per-parameter learning rate based on curvature
                    # η_t = α/(1+β⋅Curvature(g_t))
                    adaptive_lr = lr / (1.0 + curvature_influence * curvature_est)
                    
                    # Update momentum buffer with exponential moving average
                    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                    
                    # Update parameters using adaptive learning rate and momentum buffer
                    p.data.add_(-adaptive_lr * momentum_buffer)
                else:
                    # For the first step, just use standard momentum update
                    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                    p.data.add_(-lr * momentum_buffer)
                
                # Store current gradient for next iteration's curvature calculation
                state['prev_grad'].copy_(grad)
        
        return loss
