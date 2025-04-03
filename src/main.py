"""
Main script for running ASMO-Solver experiments.
"""
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.asmo_solver_config import (
    RANDOM_SEED, DEVICE, TEST_MODE,
    EXP1_STEPS, EXP1_TIME_SPAN,
    EXP2_LATENT_DIM, EXP2_TIMESTEPS, EXP2_WINDOW_SIZE, EXP2_MIN_DIM, EXP2_MAX_DIM, EXP2_THRESHOLD,
    EXP3_LATENT_DIM, EXP3_TIME_SPAN, EXP3_TEACHER_STEPS, EXP3_STUDENT_STEPS, EXP3_TRAIN_EPOCHS, EXP3_LEARNING_RATE
)

from src.preprocess import setup_device
from src.train import DiffusionDynamics
from src.evaluate import (
    experiment1, experiment2, experiment3,
    ensure_directory
)

def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def run_experiments():
    """
    Run all experiments for ASMO-Solver.
    """
    print("******** Starting ASMO-Solver Experiments ********")
    
    set_random_seed(RANDOM_SEED)
    device = setup_device(DEVICE)
    
    ensure_directory("logs")
    ensure_directory("models")
    
    print(f"Running in {'TEST' if TEST_MODE else 'FULL'} mode")
    
    x0 = torch.tensor([1.0, 0.0]).to(device)
    dynamics = DiffusionDynamics().to(device)
    errors_base, errors_asmo = experiment1(EXP1_STEPS, EXP1_TIME_SPAN, dynamics, x0)
    
    dims_used, reconstruction_errors = experiment2(
        EXP2_LATENT_DIM, EXP2_TIMESTEPS, EXP2_WINDOW_SIZE,
        EXP2_MIN_DIM, EXP2_MAX_DIM, EXP2_THRESHOLD, RANDOM_SEED
    )
    
    final_error = experiment3(
        EXP3_LATENT_DIM, EXP3_TIME_SPAN, EXP3_TEACHER_STEPS,
        EXP3_STUDENT_STEPS, EXP3_TRAIN_EPOCHS, EXP3_LEARNING_RATE
    )
    
    print("\n******** Experiment Results Summary ********")
    print("Experiment 1: Adaptive Time-Step Adjustment")
    for i, n in enumerate(EXP1_STEPS):
        print(f"  Steps={n}: Base Error={errors_base[i]:.5f}, ASMO Error={errors_asmo[i]:.5f}, "
              f"Improvement={100 * (1 - errors_asmo[i] / errors_base[i]):.2f}%")
    
    print("\nExperiment 2: Dynamic Manifold Construction")
    print(f"  Average dimensions used: {np.mean(dims_used):.2f}")
    print(f"  Mean reconstruction error: {np.mean(reconstruction_errors):.5f}")
    
    print("\nExperiment 3: Teacher-Student Distillation")
    print(f"  Final teacher-student MSE error: {final_error:.5f}")
    
    print("\n******** Experiments Completed Successfully ********")

if __name__ == "__main__":
    run_experiments()
