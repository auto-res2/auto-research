import math
import torch
from torch.optim import Optimizer
   
class HybridOptimizer(Optimizer):
    """
    HybridOptimizer: A combination of AggMo's aggregated momentum with MADGRAD's adaptive updates.
    
    This optimizer synergistically combines the advantages of AggMo's aggregated momentum with 
    MADGRAD's adaptive updates. This is achieved by integrating the multiple momentum terms of 
    AggMo to adjust the parameters within MADGRAD's dual averaging scheme based on cumulative 
    gradient history.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1e-2).
        betas (list): List of momentum factors for multiple momentum buffers (default: [0.0, 0.9, 0.99]).
        momentum (float): Additional momentum factor for dual averaging (default: 0.9).
        eps (float): Term added to the denominator for numerical stability (default: 1e-6).
        weight_decay (float): Weight decay, i.e. a L2 penalty (default: 0).
    """
    def __init__(self, params, lr=1e-2, betas=[0.0, 0.9, 0.99], momentum=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super(HybridOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, betas, momentum, eps, weight_decay = (
                group['lr'], 
                group['betas'], 
                group['momentum'], 
                group['eps'], 
                group['weight_decay']
            )
            
            # Step counter for adaptive learning rate
            if 'k' not in self.state:
                self.state['k'] = torch.tensor([0], dtype=torch.long)
            k = self.state['k'].item()
            
            # Calculate adaptive learning rate factor
            lamb = lr * math.pow(k + 1, 0.5)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Apply weight decay if specified
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # Initialize state if needed
                if 'momentum_buffers' not in param_state:
                    param_state['momentum_buffers'] = {beta: torch.zeros_like(p.data) for beta in betas}
                if 'dual_avg_buffer' not in param_state:
                    param_state['dual_avg_buffer'] = torch.zeros_like(p.data)
                if 'grad_sum_sq' not in param_state:
                    param_state['grad_sum_sq'] = torch.zeros_like(p.data)
                
                # Get state buffers
                momentum_buffers = param_state['momentum_buffers']
                dual_avg_buffer = param_state['dual_avg_buffer']
                grad_sum_sq = param_state['grad_sum_sq']
                
                # Update AggMo momentum buffers
                avg_updates = torch.zeros_like(p.data)
                for beta in betas:
                    buf = momentum_buffers[beta]
                    buf.mul_(beta).add_(d_p)
                    avg_updates.add_(buf)
                avg_updates.div_(len(betas))  # Average the momentum buffers
                
                # Update MADGRAD-style dual averaging
                grad_sum_sq.addcmul_(d_p, d_p, value=lamb)
                rms = grad_sum_sq.pow(1/3).add_(eps)
                
                # Handle numerical stability
                if eps == 0:
                    rms[rms == 0] = float('inf')
                
                # Update dual average buffer
                dual_avg_buffer.add_(d_p, alpha=lamb)
                
                # Combine AggMo momentum with MADGRAD update
                p.data.addcdiv_(dual_avg_buffer, rms, value=-1)
                p.data.add_(avg_updates, alpha=-lr)
            
            # Increment step counter
            self.state['k'] += 1
            
        return loss
