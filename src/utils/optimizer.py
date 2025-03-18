import torch
import torch.optim as optim

class ACMOptimizer(torch.optim.Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer utilizes local quadratic approximations to adaptively adjust
    the update direction and scale based on curvature information.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): curvature influence factor (default: 0.9)
        weight_decay (float, optional): weight decay factor (default: 0)
    """
    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
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
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(p.data)
                
                # Get parameters
                lr = group['lr']
                beta = group['beta']
                weight_decay = group['weight_decay']
                
                # Increment step
                state['step'] += 1
                
                # Apply weight decay if specified
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Calculate gradient change (curvature approximation)
                grad_diff = grad - state['prev_grad']
                
                # Calculate curvature-aware learning rate
                # Use a small epsilon to avoid division by zero
                epsilon = 1e-8
                curvature_term = torch.abs(grad_diff).mean()
                adaptive_lr = lr / (1.0 + beta * curvature_term + epsilon)
                
                # Update parameters
                p.data.add_(grad, alpha=-adaptive_lr)
                
                # Store current gradient for next iteration
                state['prev_grad'] = grad.clone()
        
        return loss
