"""
Main script for MS-ANO (Multi-Stage Adaptive Noise Optimization) experiments.

This script implements three experiments:
  1. Baseline Comparison: Semantic Alignment (via CLIP), Image Fidelity (simulated FID)
     and Inference Speed between the base StableDiffusionInitNOPipeline and a new
     MS-ANO-enhanced pipeline.
  2. Ablation Study: Variants of the MS-ANO configuration (full, single-stage, and no prompt
     integration at later stages).
  3. Hyperparameter Sensitivity: Grid search over key hyperparameters to evaluate
     performance metrics.
"""

import os
import time
import argparse
import torch
import numpy as np

from diffusers import StableDiffusionPipeline

from preprocess import preprocess
from train import train_model, MS_ANOPipeline
from evaluate import (
    compute_clip_score, 
    compute_fid, 
    run_baseline_evaluation,
    run_ablation_study,
    run_hyperparameter_study
)

def run_minimal_test(config_path="config/ms_ano_config.json"):
    """
    Run a minimal version of the experiments to validate that the code executes.
    This test uses only one prompt and one run to quickly check functionality.
    """
    print("\nRunning minimal test...")
    
    os.makedirs("logs", exist_ok=True)
    
    preprocess_output = preprocess(config_path)
    config = preprocess_output["config"]
    config["test_mode"] = True  # Ensure test mode is enabled
    config["n_runs"] = 1  # Use only one run
    prompts = ["a cat and a rabbit"]  # Use only one prompt
    clip_processor = preprocess_output["clip_processor"]
    clip_model = preprocess_output["clip_model"]
    
    msano_pipeline = train_model(config, preprocess_output)
    
    print("Generating a test image...")
    output = msano_pipeline(prompts[0], num_inference_steps=10)
    image = output["sample"][0]
    score = compute_clip_score(image, prompts[0], clip_processor, clip_model)
    print(f"CLIP score for test image: {score:.3f}")
    
    print("\nMinimal test finished successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Experiments for MS-ANO method.")
    parser.add_argument("--test", action="store_true", help="Run minimal test and exit.")
    parser.add_argument("--config", type=str, default="config/ms_ano_config.json", 
                       help="Path to configuration file.")
    args = parser.parse_args()
    
    if args.test:
        run_minimal_test(args.config)
        return
    
    print("\n" + "="*80)
    print("Starting MS-ANO Experiments")
    print("="*80)
    
    os.makedirs("logs", exist_ok=True)
    
    print("\nStep 1: Preprocessing...")
    preprocess_output = preprocess(args.config)
    config = preprocess_output["config"]
    prompts = preprocess_output["prompts"]
    n_runs = preprocess_output["n_runs"]
    clip_processor = preprocess_output["clip_processor"]
    clip_model = preprocess_output["clip_model"]
    
    print("\nStep 2: Setting up models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    base_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch_dtype
    )
    base_pipeline.to(device)
    
    msano_pipeline = train_model(config, preprocess_output)
    
    print(f"Running on device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    print("\nStep 3: Running experiments...")
    
    baseline_results = run_baseline_evaluation(
        base_pipeline, msano_pipeline, prompts, n_runs, clip_processor, clip_model
    )
    
    ablation_results = run_ablation_study(
        prompts, n_runs, clip_processor, clip_model
    )
    
    hyperparam_results = run_hyperparameter_study(
        prompts, n_runs, clip_processor, clip_model
    )
    
    print("\n" + "="*80)
    print("All experiments completed successfully")
    print("="*80)
    
    print("\nSummary of Results:")
    print("-" * 40)
    
    print("\n1. Baseline Comparison:")
    for pipeline in baseline_results:
        avg_clip = np.mean([entry["avg_clip_score"] for entry in baseline_results[pipeline]])
        avg_runtime = np.mean([entry["avg_runtime"] for entry in baseline_results[pipeline]])
        print(f"  {pipeline.upper()}: Avg CLIP={avg_clip:.3f}, Avg Runtime={avg_runtime:.3f}s")
    
    print("\n2. Ablation Study:")
    for variant in ablation_results:
        avg_clip = np.mean([entry["avg_clip_score"] for entry in ablation_results[variant]])
        avg_runtime = np.mean([entry["avg_runtime"] for entry in ablation_results[variant]])
        print(f"  {variant}: Avg CLIP={avg_clip:.3f}, Avg Runtime={avg_runtime:.3f}s")
    
    print("\n3. Hyperparameter Sensitivity:")
    for result in hyperparam_results:
        print(f"  Stages={result['stages']}, Threshold={result['clustering_threshold']}, "
              f"Weight={result['attention_weight']}: "
              f"CLIP={result['overall_clip_score']:.3f}, Runtime={result['overall_runtime']:.3f}s")
    
    print("\nExperiment results are saved in the logs directory.")
    return True

if __name__ == "__main__":
    main()
