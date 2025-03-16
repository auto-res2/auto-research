"""
Synthetic functions for optimization experiments.
"""

import numpy as np


def quadratic_func(x, A, b):
    """
    Quadratic function: f(x) = 0.5 * x^T A x + b^T x
    
    Args:
        x: Input vector
        A: Matrix for quadratic term
        b: Vector for linear term
        
    Returns:
        f: Function value
        grad: Gradient
    """
    f = 0.5 * x.T @ A @ x + b.T @ x
    grad = A @ x + b
    return f, grad


def rosenbrock_func(x, a=1.0, b_param=100.0):
    """
    Rosenbrock function: f(x,y) = (a - x)^2 + b*(y - x^2)^2
    
    Args:
        x: Input vector [x, y]
        a: Parameter (default: 1.0)
        b_param: Parameter (default: 100.0)
        
    Returns:
        f: Function value
        grad: Gradient
    """
    x0, x1 = x[0], x[1]
    f = (a - x0)**2 + b_param * (x1 - x0**2)**2
    grad = np.zeros_like(x)
    grad[0] = -2*(a - x0) - 4*b_param*x0*(x1 - x0**2)
    grad[1] = 2*b_param*(x1 - x0**2)
    return f, grad


def rastrigin_func(x, A=10):
    """
    Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    
    Args:
        x: Input vector
        A: Parameter (default: 10)
        
    Returns:
        f: Function value
        grad: Gradient
    """
    n = len(x)
    f = A*n + np.sum(x**2 - A * np.cos(2*np.pi*x))
    grad = 2*x + 2*np.pi*A*np.sin(2*np.pi*x)
    return f, grad


def create_random_psd_matrix(dim, seed=None):
    """
    Create a random positive semi-definite matrix.
    
    Args:
        dim (int): Dimension of the matrix
        seed (int, optional): Random seed
        
    Returns:
        A: Positive semi-definite matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    Q = np.random.randn(dim, dim)
    A = Q.T @ Q + np.eye(dim)  # Ensure positive definiteness
    
    return A


def create_random_vector(dim, seed=None):
    """
    Create a random vector.
    
    Args:
        dim (int): Dimension of the vector
        seed (int, optional): Random seed
        
    Returns:
        b: Random vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    b = np.random.randn(dim)
    
    return b
