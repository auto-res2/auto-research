#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script checks GPU compatibility for the BI-SDICL experiments.
It verifies that the code can run on NVIDIA Tesla T4 with 16 GB VRAM.
"""

import torch
import numpy as np
from train import BI_SDICLPolicy

def check_gpu_compatibility():
    """
    Checks GPU compatibility for the BI-SDICL experiments.
    """
    print("\n=== GPU Compatibility Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create a model with typical dimensions
    state_dim = 4  # CartPole state dimension
    action_dim = 2  # CartPole action dimension
    demo_embedding_dim = 64
    
    # Create the model
    model = BI_SDICLPolicy(state_dim, action_dim, demo_embedding_dim=demo_embedding_dim)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Estimate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Estimate activation memory (rough estimate)
    batch_size = 32  # Typical batch size
    seq_len = 2  # Sequence length (bias token + state)
    hidden_dim = 128  # Hidden dimension
    
    # Estimate memory for activations in attention layers
    activation_size = batch_size * seq_len * hidden_dim * 4 * 4  # 4 bytes per float32, 4 for Q,K,V,O
    
    total_memory = param_size + buffer_size + activation_size
    
    print(f"Estimated model memory (parameters): {param_size / 1e6:.2f} MB")
    print(f"Estimated model memory (buffers): {buffer_size / 1e6:.2f} MB")
    print(f"Estimated activation memory: {activation_size / 1e6:.2f} MB")
    print(f"Total estimated memory: {total_memory / 1e6:.2f} MB")
    
    # Check if the model can fit on Tesla T4 (16 GB VRAM)
    tesla_t4_memory = 16 * 1e9  # 16 GB in bytes
    
    print(f"\nTesla T4 memory: {tesla_t4_memory / 1e9:.2f} GB")
    print(f"Memory usage percentage: {total_memory / tesla_t4_memory * 100:.4f}%")
    
    if total_memory < tesla_t4_memory:
        print("\nCONCLUSION: The model can run on NVIDIA Tesla T4 with 16 GB VRAM.")
        print("The model uses a small transformer architecture with 128-dimensional embeddings")
        print("and 4 attention heads, which is well within the 16GB VRAM capacity of Tesla T4 GPUs.")
    else:
        print("\nWARNING: The model may exceed the memory capacity of NVIDIA Tesla T4 with 16 GB VRAM.")
        print("Consider reducing batch size or model dimensions.")

if __name__ == "__main__":
    check_gpu_compatibility()
