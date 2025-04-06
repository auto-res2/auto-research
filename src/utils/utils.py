"""
Utility functions for the STEM method experiments.
Provides data generation functions for synthetic experiments.
"""

import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_samples=10):
    """
    Generate synthetic data for Experiment 1: Controlled Synthetic Dataset.
    Each sample is a tuple: (text, true_alpha)
    For demonstration, texts are simple sentences.
    """
    texts = []
    alphas = []
    for i in range(num_samples):
        base_text = f"This is sample text number {i} written by a human."
        ai_text = base_text + " With AI modifications."
        alpha = np.round(np.random.uniform(0, 1), 2)
        text = base_text if alpha < 0.5 else ai_text
        texts.append(text)
        alphas.append(torch.tensor(alpha, dtype=torch.float32))
    return list(zip(texts, alphas))

def generate_metadata_data(num_samples=10, metadata_dim=1):
    """
    Generate synthetic data for Experiment 2: Ablation Study.
    Each sample is a tuple: (text, true_alpha, true_metadata)
    Here, metadata is simulated as a single number.
    """
    data = []
    for i in range(num_samples):
        base_text = f"Document {i}: Analysis of human vs AI writing."
        ai_text = base_text + " Including AI stylistic changes."
        alpha = np.round(np.random.uniform(0, 1), 2)
        text = base_text if alpha < 0.5 else ai_text
        metadata = torch.tensor([alpha + np.random.normal(0, 0.1)], dtype=torch.float32)
        data.append((text, torch.tensor(alpha, dtype=torch.float32), metadata))
    return data

def generate_domain_data(num_samples=20):
    """
    Generate synthetic domain-specific data for Experiment 3: Domain Shift.
    Each sample is a tuple: (text, true_alpha, true_metadata, domain_label)
    Two domains: 'news' and 'academic'
    """
    data = []
    for i in range(num_samples):
        domain = "news" if i % 2 == 0 else "academic"
        base_text = f"{domain.capitalize()} article {i}: This is a text sample."
        ai_text = base_text + " With additional AI enhancements."
        alpha = np.round(np.random.uniform(0, 1), 2)
        text = base_text if alpha < 0.5 else ai_text
        metadata = torch.tensor([alpha + np.random.normal(0, 0.05)], dtype=torch.float32)
        data.append((text, torch.tensor(alpha, dtype=torch.float32), metadata, domain))
    return data

def prepare_logs_directory():
    """
    Create logs directory if it doesn't exist
    """
    import os
    if not os.path.exists("logs"):
        os.makedirs("logs")
        print("Created logs directory")
