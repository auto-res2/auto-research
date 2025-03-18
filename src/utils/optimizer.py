import torch
import torch.optim as optim

class ACMOptimizer(optim.Optimizer):
    """
    Adaptive Curvature Momentum (ACM) Optimizer
    
    This optimizer utilizes local quadratic approximations to adaptively adjust
    the update direction and scale based on curvature information.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): momentum factor (default: 0.9)
        curvature_influence (float, optional): factor controlling curvature adaptation (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=1e-3, beta=0.9, curvature_influence=0.1, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, curvature_influence=curvature_influence, weight_decay=weight_decay)
        super(ACMOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
        
        Returns:
            loss (Tensor, optional): loss value if closure is provided
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
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                momentum = state['momentum_buffer']
                prev_grad = state['prev_grad']
                state['step'] += 1

                # Compute gradient change as a curvature proxy
                grad_change = grad.sub(prev_grad)
                curvature_scale = 1.0 / (1.0 + curvature_influence * grad_change.norm())
    
                momentum.mul_(beta).add_(grad, alpha=curvature_scale)
                state['prev_grad'] = grad.clone()
                p.data.add_(momentum, alpha=-lr)
        return loss
