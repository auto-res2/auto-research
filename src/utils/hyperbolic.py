import torch
import torch.nn.functional as F

def exp_map(x, c=1.0):
    """
    Map from tangent space to hyperbolic space (exponential map).
    
    Args:
        x: Points in the tangent space
        c: Curvature of the hyperbolic space (default: 1.0)
        
    Returns:
        Points mapped to hyperbolic space
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    sqrt_c = torch.sqrt(c_tensor)
    return torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm)

def log_map(x, c=1.0):
    """
    Map from hyperbolic space to tangent space (logarithmic map).
    
    Args:
        x: Points in hyperbolic space
        c: Curvature of the hyperbolic space (default: 1.0)
        
    Returns:
        Points mapped to tangent space
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8, max=1.0 - 1e-5)
    
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    sqrt_c = torch.sqrt(c_tensor)
    return x * torch.atanh(sqrt_c * norm) / (sqrt_c * norm)

def hyperbolic_distance(x, y, c=1.0):
    """
    Compute the distance between points in hyperbolic space.
    
    Args:
        x, y: Points in hyperbolic space
        c: Curvature of the hyperbolic space (default: 1.0)
        
    Returns:
        Hyperbolic distance between x and y
    """
    x_t = log_map(x, c)
    y_t = log_map(y, c)
    
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    sqrt_c = torch.sqrt(c_tensor)
    return 2.0 * torch.atanh(sqrt_c * torch.norm(x_t - y_t, dim=-1))
