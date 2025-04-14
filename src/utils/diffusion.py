"""
Diffusion process utilities for the Progressive Brightness Distillation Diffusion experiment.
"""

import torch
from config.pbd_diffusion_config import DIFFUSION_STEPS

def reverse_diffusion(noisy_img, init_module, progressive_module, teacher_model, steps=DIFFUSION_STEPS):
    """
    Simulated reverse diffusion process that applies brightness correction.
    
    Args:
        noisy_img: Input noisy image tensor of shape [B, C, H, W]
        init_module: Initial brightness correction module
        progressive_module: Progressive refinement module
        teacher_model: Teacher model for brightness correction guidance
        steps: Number of refinement steps
        
    Returns:
        list: Output tensors at each step of the diffusion process
    """
    x = noisy_img
    outputs = []
    
    x = init_module(x)
    outputs.append(x.clone())
    
    for t in range(steps):
        teacher_correction = teacher_model(x)  # simulate teacher output for brightness correction
        x = progressive_module(x, teacher_correction)
        outputs.append(x.clone())
    return outputs

def run_pipeline(image, init_module, progressive_module, teacher_model, variant='full'):
    """
    Run a specific variant of the brightness correction pipeline.
    
    Args:
        image: Input image tensor of shape [B, C, H, W]
        init_module: Initial brightness correction module
        progressive_module: Progressive refinement module
        teacher_model: Teacher model for brightness correction guidance
        variant: Pipeline variant ('full', 't1_only', or 'progressive_only')
        
    Returns:
        torch.Tensor: Corrected image
    """
    if variant == 't1_only':
        corrected = init_module(image)
    elif variant == 'progressive_only':
        corrected = image  # start with the raw image without t1 correction
        for _ in range(DIFFUSION_STEPS):
            teacher_corr = teacher_model(corrected)
            corrected = progressive_module(corrected, teacher_corr)
    else:  # full pipeline: initial correction followed by progressive refinement.
        corrected = init_module(image)
        for _ in range(DIFFUSION_STEPS):
            teacher_corr = teacher_model(corrected)
            corrected = progressive_module(corrected, teacher_corr)
    return corrected
