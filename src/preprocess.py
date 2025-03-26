"""
Preprocessing script for MS-ANO experiments.

This script is responsible for setting up the test prompts and
initializing the CLIP model for evaluation.
"""

import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel

def load_config(config_path):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_clip_model():
    """Initialize and return CLIP processor and model for evaluation."""
    print("Loading CLIP model and processor...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    
    return clip_processor, clip_model

def get_test_prompts(config):
    """Return the test prompts from config."""
    return config.get("prompts", [
        "a cat and a rabbit",
        "a scenic mountain view",
        "a futuristic city",
        "a cozy living room"
    ])

def preprocess(config_path="config/ms_ano_config.json"):
    """Main preprocessing function."""
    print("Starting preprocessing...")
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    config = load_config(config_path)
    
    prompts = get_test_prompts(config)
    print(f"Loaded {len(prompts)} test prompts")
    
    clip_processor, clip_model = setup_clip_model()
    
    return {
        "config": config,
        "prompts": prompts,
        "clip_processor": clip_processor,
        "clip_model": clip_model,
        "n_runs": config.get("n_runs", 3)
    }

if __name__ == "__main__":
    preprocess_output = preprocess()
    print("Preprocessing completed successfully")
    print(f"Config: {preprocess_output['config']}")
    print(f"Prompts: {preprocess_output['prompts']}")
    print(f"N runs: {preprocess_output['n_runs']}")
