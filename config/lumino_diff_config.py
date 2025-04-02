"""
Configuration for LuminoDiff experiments.
"""

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='LuminoDiff: A Dual-Latent Guided Diffusion Model')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=64, help='Size of generated images')
    
    parser.add_argument('--content_dim', type=int, default=256, help='Content latent dimension')
    parser.add_argument('--brightness_dim', type=int, default=64, help='Brightness latent dimension')
    parser.add_argument('--fusion_dim', type=int, default=128, help='Fusion dimension')
    
    parser.add_argument('--experiment', type=str, default='all', 
                        choices=['ablation', 'brightness', 'fusion', 'all'],
                        help='Which experiment to run')
    
    parser.add_argument('--output_dir', type=str, default='logs', help='Directory to save output logs')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to load/save data')
    
    args = parser.parse_args()
    return args
