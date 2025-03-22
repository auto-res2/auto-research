"""
Utility functions for diffusion models and SASD implementation.
"""
import torch
import torch.nn as nn
import numpy as np


def linear_beta_schedule(beta_start, beta_end, num_steps):
    """
    Linear schedule for noise levels in diffusion process.
    """
    return torch.linspace(beta_start, beta_end, num_steps)


def get_alphas_from_betas(betas):
    """
    Convert betas to alphas for the diffusion process.
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod


def extract(a, t, x_shape):
    """
    Extract specific indices from a tensor based on timestep t.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def compute_kl_loss(pred, target):
    """
    Compute KL divergence loss between prediction and target.
    In this simple implementation, we use MSE as an approximation.
    """
    return torch.mean((pred - target)**2)


def compute_score_loss(pred_score, teacher_score):
    """
    Compute score alignment error using MSE.
    """
    return torch.mean((pred_score - teacher_score)**2)


def get_device():
    """
    Get the appropriate device (GPU or CPU) for training.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
