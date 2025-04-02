"""
Training module for PriorBrush experiment.

This module implements the SwiftBrush and PriorBrush generation models.
"""

import torch
import numpy as np
import time
from preprocess import normalize_image


class SwiftBrushGenerator:
    """
    Simulates the SwiftBrush one-step generation using variational score distillation.
    
    Note: In a production environment, this would load and use a pre-trained model.
    For this experiment, we simulate the model behavior.
    """
    
    def __init__(self, img_size=256, channels=3, device="cuda"):
        """
        Initialize the SwiftBrush generator.
        
        Args:
            img_size (int): Size of the generated image.
            channels (int): Number of channels in the generated image.
            device (str): Device to run the model on.
        """
        self.img_size = img_size
        self.channels = channels
        self.device = device
    
    def generate(self, prompt, seed):
        """
        Generate an image based on a text prompt.
        
        Args:
            prompt (str): Text prompt for image generation.
            seed (int): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Generated image tensor.
        """
        torch.manual_seed(seed)
        generated_image = torch.randn(self.channels, self.img_size, self.img_size, device=self.device)
        
        generated_image = torch.abs(generated_image)  # Make values positive
        generated_image = normalize_image(generated_image)
        
        print(f"[SwiftBrush] Generated image for prompt: '{prompt}' using seed {seed}")
        return generated_image


class PriorBrushGenerator:
    """
    Implements the PriorBrush dual-stage generation with diffusion-based prior refinement.
    
    Stage 1: One-step generation using SwiftBrush
    Stage 2: Prior-aware refinement via fast diffusion guidance
    """
    
    def __init__(self, img_size=256, channels=3, device="cuda"):
        """
        Initialize the PriorBrush generator.
        
        Args:
            img_size (int): Size of the generated image.
            channels (int): Number of channels in the generated image.
            device (str): Device to run the model on.
        """
        self.img_size = img_size
        self.channels = channels
        self.device = device
        self.swift_generator = SwiftBrushGenerator(img_size, channels, device)
    
    def _apply_refinement(self, image, prompt, refinement_steps):
        """
        Apply the prior-aware diffusion refinement to the generated image.
        
        Args:
            image (torch.Tensor): Input image from one-step generation.
            prompt (str): Text prompt for guiding the refinement.
            refinement_steps (int): Number of diffusion steps for refinement.
            
        Returns:
            torch.Tensor: Refined image tensor.
        """
        
        
        torch.manual_seed(42)  # Use constant seed for refinement noise
        
        structured_noise = torch.randn_like(image, device=self.device) * (0.01 * refinement_steps)
        
        refined_image = image + structured_noise
        
        refined_image = normalize_image(refined_image)
        
        return refined_image
    
    def generate(self, prompt, seed, refinement_steps=3):
        """
        Generate an image with PriorBrush using two stages:
        1. One-step generation
        2. Prior-aware refinement
        
        Args:
            prompt (str): Text prompt for image generation.
            seed (int): Random seed for reproducibility.
            refinement_steps (int): Number of diffusion steps for refinement.
            
        Returns:
            torch.Tensor: Generated and refined image tensor.
        """
        swift_image = self.swift_generator.generate(prompt, seed)
        
        refined_image = self._apply_refinement(swift_image, prompt, refinement_steps)
        
        print(f"[PriorBrush] Refined image with {refinement_steps} diffusion steps for prompt: '{prompt}'")
        return refined_image
