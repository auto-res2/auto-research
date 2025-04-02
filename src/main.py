"""
Main script for running the PriorBrush experiment.

This script orchestrates the complete experiment pipeline from model setup
to running experiments and generating plots.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from train import SwiftBrushGenerator, PriorBrushGenerator
from evaluate import (
    experiment_inference_and_quality,
    experiment_ablation,
    experiment_sensitivity,
    run_quick_test
)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.priorbrush_config import MODEL_CONFIG, EXPERIMENT_CONFIG, OUTPUT_CONFIG


def setup_environment():
    """
    Set up the environment for the experiments.
    
    Returns:
        dict: Environment configuration.
    """
    seed = MODEL_CONFIG["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    logs_dir = Path(OUTPUT_CONFIG["logs_dir"])
    logs_dir.mkdir(exist_ok=True)
    
    return {
        "device": device,
        "seed": seed,
    }


def main():
    """
    Main function to run the PriorBrush experiment.
    """
    print("Starting PriorBrush Experiment")
    print("==============================")
    
    env_config = setup_environment()
    device = env_config["device"]
    
    swift_generator = SwiftBrushGenerator(
        img_size=MODEL_CONFIG["swift_params"]["img_size"],
        channels=MODEL_CONFIG["swift_params"]["channels"],
        device=device
    )
    
    prior_generator = PriorBrushGenerator(
        img_size=MODEL_CONFIG["prior_params"]["img_size"],
        channels=MODEL_CONFIG["prior_params"]["channels"],
        device=device
    )
    
    ablation_path = os.path.join(OUTPUT_CONFIG["figures_dir"], OUTPUT_CONFIG["ablation_plot_name"])
    sensitivity_path = os.path.join(OUTPUT_CONFIG["figures_dir"], OUTPUT_CONFIG["sensitivity_plot_name"])
    
    run_quick_test(swift_generator, prior_generator, EXPERIMENT_CONFIG)
    
    print("\n" + "="*50)
    print("Running Experiment 1: Inference and Quality Comparison")
    experiment_inference_and_quality(
        swift_generator, prior_generator,
        prompt=EXPERIMENT_CONFIG["exp1"]["prompt"],
        seed=env_config["seed"],
        refinement_steps=EXPERIMENT_CONFIG["exp1"]["refinement_steps"],
        num_runs=EXPERIMENT_CONFIG["exp1"]["num_runs"],
        device=device
    )
    
    print("\n" + "="*50)
    print("Running Experiment 2: Ablation Study")
    experiment_ablation(
        swift_generator, prior_generator,
        prompt=EXPERIMENT_CONFIG["exp2"]["prompt"],
        seed=env_config["seed"],
        refinement_steps=EXPERIMENT_CONFIG["exp2"]["refinement_steps"],
        device=device,
        output_path=ablation_path
    )
    
    print("\n" + "="*50)
    print("Running Experiment 3: Sensitivity Analysis")
    experiment_sensitivity(
        swift_generator, prior_generator,
        prompt=EXPERIMENT_CONFIG["exp3"]["prompt"],
        seed=env_config["seed"],
        step_range=EXPERIMENT_CONFIG["exp3"]["step_range"],
        device=device,
        output_path=sensitivity_path
    )
    
    print("\n" + "="*50)
    print("PriorBrush Experiment Completed")
    print("All experiments have been successfully executed.")
    print(f"Figures saved to {OUTPUT_CONFIG['figures_dir']} directory")


if __name__ == "__main__":
    main()
