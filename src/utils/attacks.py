"""
Adversarial attack utilities for PurifyCov++ experiments.
"""

import torch
import torch.nn as nn

def generate_fgsm_examples(model, images, labels, epsilon=0.03):
    """
    Generate adversarial examples using the Fast Gradient Sign Method.
    
    Args:
        model: The target model
        images: Input images
        labels: True labels
        epsilon: Perturbation magnitude
        
    Returns:
        adv_images: Adversarial examples
    """
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images.detach()
