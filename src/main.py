"""
Main script for running NTEC-G experiments.
Implements three experiments comparing the Base Method with NTEC-G.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_dummy_data, prepare_visualization_data
from src.train import (
    DiffusionGuidanceModel, 
    base_method_guidance, 
    ntec_g_guidance,
    diffusion_step_with_regularization, 
    geometric_regularization
)
from src.evaluate import (
    plot_convergence_curves, 
    plot_samples_grid, 
    visualize_latent_space, 
    generate_samples
)
from config.experiment_config import *

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

def run_experiment1():
    """
    Experiment 1: Speed and Convergence Analysis.
    Compares the Base Method with NTEC-G in terms of convergence speed.
    """
    print("\n==== Experiment 1: Speed and Convergence Analysis ====")
    
    model = DiffusionGuidanceModel()
    
    initial_state = load_dummy_data(BATCH_SIZE, LATENT_DIM)
    
    start = time.time()
    final_state_base, iterations, norm_diffs_base = base_method_guidance(
        initial_state.clone(), 
        model, 
        epsilon=CONVERGENCE_EPSILON, 
        max_iter=MAX_ITERATIONS
    )
    elapsed_base = time.time() - start
    print(f"Base Method finished in {elapsed_base:.4f} seconds after {iterations} iterations.")
    
    start = time.time()
    final_state_ntec, probe_norms = ntec_g_guidance(
        initial_state.clone(), 
        model, 
        probe_steps=PROBE_STEPS, 
        epsilon=CONVERGENCE_EPSILON
    )
    elapsed_ntec = time.time() - start
    print(f"NTEC-G method finished in {elapsed_ntec:.4f} seconds (using {len(probe_norms)} probe iterations).")
    
    plot_convergence_curves(norm_diffs_base, probe_norms)
    
    return norm_diffs_base, probe_norms

def run_experiment2():
    """
    Experiment 2: Sample Fidelity and Quality Evaluation.
    Compares samples generated using both methods.
    """
    print("\n==== Experiment 2: Sample Fidelity and Quality Evaluation ====")
    model = DiffusionGuidanceModel()
    
    print("Generating samples using the Base Method...")
    samples_base = generate_samples(
        lambda x, m: base_method_guidance(x, m)[0],  # Extract only the state
        model, 
        num_samples=NUM_SAMPLES
    )
    
    print("Generating samples using the NTEC-G guidance...")
    samples_ntec = generate_samples(
        lambda x, m: ntec_g_guidance(x, m)[0],  # Extract only the state
        model, 
        num_samples=NUM_SAMPLES
    )
    
    plot_samples_grid(samples_base, title="Samples: Base Method", filename="samples_base.pdf")
    plot_samples_grid(samples_ntec, title="Samples: NTEC-G Method", filename="samples_ntec.pdf")
    
    print("Note: In a real implementation, FID and Inception Score computations would be performed.")
    
    return samples_base, samples_ntec

def run_experiment3():
    """
    Experiment 3: Ablation Study on Geometric Regularization.
    Tests different regularization coefficients.
    """
    print("\n==== Experiment 3: Ablation Study on Geometric Regularization ====")
    model = DiffusionGuidanceModel()
    
    latent_space_collection = {}
    
    for variant, reg_coeff in REG_VARIANTS.items():
        print(f"Running variant '{variant}' with lambda_reg={reg_coeff}")
        latent_collection = []
        
        for i in range(REG_NUM_SAMPLES):
            state = torch.randn(1, LATENT_DIM)
            for t in range(DIFFUSION_STEPS):
                use_reg = (reg_coeff > 0.0)
                state = diffusion_step_with_regularization(
                    state, 
                    model, 
                    lambda_reg=reg_coeff, 
                    use_reg=use_reg
                )
            latent_collection.append(state)
            
            if i < 2:
                print(f"Variant '{variant}', sample {i} completed.")
                
        latent_space_collection[variant] = torch.cat(latent_collection, dim=0)
    
    for variant, latents in latent_space_collection.items():
        plot_filename = f"latent_space_{variant}.pdf"
        visualize_latent_space(
            latents, 
            title=f"Latent Space: {variant}", 
            filename=plot_filename
        )
    
    return latent_space_collection

def run_test():
    """
    Run a quick test of all experiments to verify functionality.
    """
    print("\n********** Running Test **********")
    print("This is a quick test to verify that the code runs correctly.")
    
    global BATCH_SIZE, MAX_ITERATIONS, NUM_SAMPLES, REG_NUM_SAMPLES
    
    BATCH_SIZE = 16
    MAX_ITERATIONS = 10
    NUM_SAMPLES = 4
    REG_NUM_SAMPLES = 5
    
    run_experiment1()
    run_experiment2()
    run_experiment3()
    
    print("\nTest completed successfully.")

if __name__ == "__main__":
    print("NTEC-G: Neural Tangent Extrapolated Characteristic Guidance")
    print("Starting experiments...")
    
    run_test()
    
    print("\n\n********** Running Full Experiments **********")
    
    BATCH_SIZE = 64
    MAX_ITERATIONS = 50
    NUM_SAMPLES = 16
    REG_NUM_SAMPLES = 50
    
    exp1_results = run_experiment1()
    exp2_results = run_experiment2()
    exp3_results = run_experiment3()
    
    print("\nAll experiments completed successfully.")
    print("PDF figures saved in the logs directory.")
    
    sys.exit(0)
