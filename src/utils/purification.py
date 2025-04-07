"""
Purification utilities for PurifyCov++ experiments.
"""

import torch

def purify_diffusion(image, timesteps, method='cov', covariance_net=None):
    """
    Purify the image using a reverse diffusion process.
    
    Args:
        image: Input image to purify
        timesteps: Number of diffusion steps
        method: 'cov' for PurifyCov++ (adaptive covariance) or 'fixed' for Purify++ (fixed covariance)
        covariance_net: Covariance prediction network (required if method='cov')
        
    Returns:
        purified: Purified image
    """
    purified = image.clone()
    for t in range(timesteps):
        if method == 'cov' and (covariance_net is not None):
            sigma_t = covariance_net(purified, t)
        else:
            sigma_t = torch.ones_like(purified) * 0.1
            
        noise = torch.randn_like(purified) * sigma_t
        purified = purified - 0.1 * purified + noise
        
    return purified
