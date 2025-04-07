
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, noise_level):
        noise = torch.randn_like(x) * noise_level * 0.1
        return x - noise

    def dual_stage_denoise(self, x, noise_level):
        x1 = self.forward(x, noise_level)
        x2 = self.forward(x1, noise_level * 0.95)
        return x2

class DummyClassifier(nn.Module):
    def __init__(self):
        super(DummyClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.net(x)

def consistency_loss(output1, output2):
    return ((output1 - output2) ** 2).mean()

def run_purification(x, noise_level, diffusion_model, variant='base'):
    """
    Apply one of three purification variants to batch x.
      variant == 'base': basic purification using diffusion_model.forward,
      variant == 'dual': dual-stage denoising using diffusion_model.dual_stage_denoise,
      variant == 'cedp': full CEDP with dual-stage denoising and consistency-based averaging.
    """
    if variant == 'base':
        purified = diffusion_model.forward(x, noise_level)
    elif variant == 'dual':
        purified = diffusion_model.dual_stage_denoise(x, noise_level)
    elif variant == 'cedp':
        purified_dual = diffusion_model.dual_stage_denoise(x, noise_level)
        purified_adjacent = diffusion_model.dual_stage_denoise(x, noise_level * 1.05)
        loss_cons = consistency_loss(purified_dual, purified_adjacent)
        purified = (purified_dual + purified_adjacent) / 2.0
        print("Consistency loss (CEDP variant): {:.6f}".format(loss_cons.item()))
    else:
        raise ValueError("Unknown variant specified: {}".format(variant))
    return purified

def run_adaptive_purification(x, diffusion_model, initial_noise_level=0.3, iterations=5, threshold=0.01):
    noise_level = initial_noise_level
    x_current = x
    noise_levels_record = [noise_level]
    for i in range(iterations):
        purified_dual = diffusion_model.dual_stage_denoise(x_current, noise_level)
        purified_adjacent = diffusion_model.dual_stage_denoise(x_current, noise_level * 1.05)
        loss_cons = consistency_loss(purified_dual, purified_adjacent)
        if loss_cons.item() > threshold:
            noise_level *= 0.9
        noise_levels_record.append(noise_level)
        x_current = purified_dual  # update output for next iteration
        print("Iteration {}: noise_level = {:.4f}, consistency loss = {:.6f}".format(
            i + 1, noise_level, loss_cons.item()))
    return x_current, noise_levels_record

def run_fixed_purification(x, diffusion_model, noise_level=0.3, iterations=5):
    x_current = x
    for i in range(iterations):
        x_current = diffusion_model.dual_stage_denoise(x_current, noise_level)
        print("Fixed purification iteration {} complete.".format(i + 1))
    return x_current
