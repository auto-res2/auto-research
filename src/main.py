"""
RapidAlign: Main experiment script

This script runs the RapidAlign experiments.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.preprocess import preprocess
from src.evaluate import inference_speed_benchmark, stability_preference_evaluation, zero_shot_switch_experiment
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.rapidalign_config import (
    RANDOM_SEED, 
    INFERENCE_SPEED_CONFIG, 
    STABILITY_CONFIG, 
    BEHAVIOR_SWITCH_CONFIG,
    TEST_CONFIG
)

def test():
    """
    Test each experiment with minimal iterations so that the test finishes immediately.
    """
    print("Running minimal tests for all three experiments...")
    
    inference_speed_benchmark(
        n_trials=TEST_CONFIG['inference_speed']['n_trials'],
        num_steps=TEST_CONFIG['inference_speed']['num_steps']
    )
    
    stability_preference_evaluation(
        num_steps=TEST_CONFIG['stability']['num_steps']
    )
    
    zero_shot_switch_experiment(
        sim_steps=TEST_CONFIG['behavior_switch']['sim_steps'],
        switch_interval=TEST_CONFIG['behavior_switch']['switch_interval']
    )
    
    print("Minimal tests complete.")

def main():
    """
    Main function to run all experiments.
    """
    preprocess()
    
    print("Starting RapidAlign experiments...")
    print("Using PyTorch version:", torch.__version__)
    print("Device available:", "CUDA" if torch.cuda.is_available() else "CPU")
    
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))
        print("CUDA Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    
    test()
    
    
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
