"""
Baseline optimizers for comparison with ACM.
"""

import numpy as np


class SGDOptimizer_NP:
    """
    NumPy implementation of SGD optimizer.
    
    Args:
        lr (float, optional): learning rate (default: 0.01)
        momentum (float, optional): momentum factor (default: 0.0)
    """
    
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None
    
    def update(self, x, grad):
        """
        Update parameters using SGD.
        
        Args:
            x: Current parameter values
            grad: Gradient of the loss with respect to x
            
        Returns:
            x_new: Updated parameter values
            lr: The learning rate used for this update
        """
        if self.velocity is None or self.velocity.shape != grad.shape:
            self.velocity = np.zeros_like(grad)
        
        self.velocity = self.momentum * self.velocity - self.lr * grad
        x_new = x + self.velocity
        
        return x_new, self.lr


class AdamOptimizer_NP:
    """
    NumPy implementation of Adam optimizer.
    
    Args:
        lr (float, optional): learning rate (default: 0.01)
        beta1 (float, optional): coefficient for first moment estimation (default: 0.9)
        beta2 (float, optional): coefficient for second moment estimation (default: 0.999)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, x, grad):
        """
        Update parameters using Adam.
        
        Args:
            x: Current parameter values
            grad: Gradient of the loss with respect to x
            
        Returns:
            x_new: Updated parameter values
            lr: The learning rate used for this update
        """
        self.t += 1
        
        if self.m is None or self.m.shape != grad.shape:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        x_new = x - update
        
        return x_new, self.lr


class AdaBeliefOptimizer_NP:
    """
    NumPy implementation of AdaBelief optimizer.
    
    Args:
        lr (float, optional): learning rate (default: 0.01)
        beta1 (float, optional): coefficient for first moment estimation (default: 0.9)
        beta2 (float, optional): coefficient for second moment estimation (default: 0.999)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.s = None
        self.t = 0
    
    def update(self, x, grad):
        """
        Update parameters using AdaBelief.
        
        Args:
            x: Current parameter values
            grad: Gradient of the loss with respect to x
            
        Returns:
            x_new: Updated parameter values
            lr: The learning rate used for this update
        """
        self.t += 1
        
        if self.m is None or self.m.shape != grad.shape:
            self.m = np.zeros_like(grad)
            self.s = np.zeros_like(grad)
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update the variance estimate
        grad_residual = grad - self.m
        self.s = self.beta2 * self.s + (1 - self.beta2) * (grad_residual**2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        s_hat = self.s / (1 - self.beta2**self.t)
        
        # Update parameters
        update = self.lr * m_hat / (np.sqrt(s_hat) + self.eps)
        x_new = x - update
        
        return x_new, self.lr
