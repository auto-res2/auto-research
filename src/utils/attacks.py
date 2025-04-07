"""
Utility functions for generating adversarial examples.
"""
import torch
import foolbox as fb

def generate_adversarial_examples(model, x, y, attack_type='pgd', epsilon=0.03):
    """
    Generate adversarial examples using specified attack type.
    
    Args:
        model (torch.nn.Module): Model to attack
        x (torch.Tensor): Clean input images
        y (torch.Tensor): Ground truth labels
        attack_type (str): Type of attack ('pgd', 'bpda_eot', or 'black_box')
        epsilon (float): Perturbation size
        
    Returns:
        torch.Tensor: Adversarial examples
    """
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=dict())
    
    if attack_type == 'pgd':
        attack = fb.attacks.LinfPGD()
    elif attack_type == 'bpda_eot':
        attack = fb.attacks.LinfBasicIterativeAttack()
    else:  # black_box
        attack = fb.attacks.LinfSPSAAttack()
    
    _, adv_images, _ = attack(fmodel, x, y, epsilons=epsilon)
    
    return adv_images
