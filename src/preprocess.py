"""
Data preprocessing for VG-DD experiments.
This module includes utilities for loading and processing image data.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw

def get_transform(image_size=224):
    """Get standard image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def perturb_image(image):
    """
    Create a perturbed version of an image by adding a black rectangle occlusion.
    
    Args:
        image: PIL Image to perturb
        
    Returns:
        Perturbed PIL Image
    """
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)
    occ_width, occ_height = img_w // 4, img_h // 4
    top_left_x = torch.randint(0, img_w - occ_width, (1,)).item()
    top_left_y = torch.randint(0, img_h - occ_height, (1,)).item()
    draw.rectangle(
        [top_left_x, top_left_y, top_left_x + occ_width, top_left_y + occ_height], 
        fill="black"
    )
    return image

class DummyImageDataset(Dataset):
    """
    Dummy image dataset for testing experiments.
    Generates random images for experiment validation.
    """
    def __init__(self, size=100, image_size=224, transform=None):
        self.size = size
        self.image_size = image_size
        self.transform = transform or get_transform(image_size)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        dummy_array = (torch.rand(self.image_size, self.image_size, 3) * 255).byte().numpy()
        original_img = Image.fromarray(dummy_array, mode="RGB")
        
        perturbed_img = perturb_image(original_img.copy())
        
        original_tensor = self.transform(original_img)
        perturbed_tensor = self.transform(perturbed_img)
        
        return {
            "original": original_tensor,
            "perturbed": perturbed_tensor,
            "idx": idx
        }
