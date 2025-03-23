import torch
import torch.optim as optim

class ACMOptimizer(optim.Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer.
    
    This optimizer utilizes local quadratic approximations to adaptively
    adjust the update direction and scale. It combines Adam-style adaptability 
    with curvature-aware updates.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): base learning rate (default: 0.01)
        beta (float, optional): momentum coefficient (default: 0.9)
        curvature_influence (float, optional): controls influence of curvature (default: 0.1)
    """
    def __init__(self, params, lr=0.01, beta=0.9, curvature_influence=0.1):
        defaults = dict(lr=lr, beta=beta, curvature_influence=curvature_influence)
        super(ACMOptimizer, self).__init__(params, defaults)
        
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
            curvature_influence = group['curvature_influence']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state if needed
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                    
                momentum_buffer = state['momentum_buffer']
                prev_grad = state['prev_grad']
                
                # Estimate curvature as gradient change
                curvature_est = (grad - prev_grad).abs()
                
                # Compute adaptive learning rate based on curvature
                adaptive_lr = lr / (1.0 + curvature_influence * curvature_est)
                
                # Update momentum buffer with exponential moving average
                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                
                # Update parameters using adaptive learning rate and momentum
                p.data.add_(-adaptive_lr * momentum_buffer)
                
                # Store current gradient for next iteration
                prev_grad.copy_(grad)
                
        return loss
