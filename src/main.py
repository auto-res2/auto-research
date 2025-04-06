"""
Main entry point for the QD²P (Q-Denoising Diffusion Probe) experiment.

This script orchestrates the entire pipeline from data preprocessing to model
training and evaluation. It runs the three experiments described in the paper:
1. Controlled Quality Evaluation on Synthetic Tasks
2. Ablation Study on the Candidate Reweighting Mechanism
3. Efficiency and Distillation: Reducing Inference Overhead
"""

import torch
import logging
import argparse
import sys
import time
from pathlib import Path

from preprocess import set_seed, create_synthetic_data, setup_experiment_directories
from train import SimpleDiffusion, QProbe, train_models, quality_metric
from evaluate import (
    controlled_quality_experiment,
    ablation_candidate_reweighting,
    efficiency_and_distillation,
    test_all_experiments
)

sys.path.append("config")
import qd2p_config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run QD²P experiments")
    parser.add_argument(
        "--test", action="store_true", 
        help="Run quick tests with reduced parameters"
    )
    parser.add_argument(
        "--device", type=str, default=config.DEVICE,
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def print_experiment_header(title):
    """Print a formatted header for an experiment."""
    header_length = len(title) + 10
    print("\n" + "=" * header_length)
    print(f"===  {title}  ===")
    print("=" * header_length + "\n")


def print_system_info():
    """Print information about the system."""
    print("\nSystem Information:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA version: {torch.version.cuda}")
        print(f"- GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"- Device being used: {args.device}")
    print("")


def run_full_experiments(device):
    """Run the full QD²P experiments."""
    logger.info("Starting full QD²P experiments")
    
    data = create_synthetic_data(config.LATENT_DIM, config.BATCH_SIZE, device=device)
    
    print_experiment_header("Model Training")
    models = train_models(config, data, device=device)
    diffusion_model = models["diffusion_model"]
    q_probe = models["q_probe"]
    
    print_experiment_header("Experiment 1: Controlled Quality Evaluation")
    exp1_results = controlled_quality_experiment(
        diffusion_model, q_probe, config, data, device=device
    )
    
    print_experiment_header("Experiment 2: Ablation Study on Candidate Reweighting")
    exp2_results = ablation_candidate_reweighting(
        diffusion_model, q_probe, config, data, device=device
    )
    
    print_experiment_header("Experiment 3: Efficiency and Distillation")
    exp3_results = efficiency_and_distillation(
        diffusion_model, q_probe, config, data, device=device
    )
    
    print_experiment_header("Summary of Results")
    print(f"Experiment 1 - Quality improvement: {(exp1_results['qd2p_scores'][-1] - exp1_results['baseline_scores'][-1]):.4f}")
    print(f"Experiment 2 - Best configuration: {max(exp2_results['results'].items(), key=lambda x: x[1][-1])[0]}")
    print(f"Experiment 3 - Speedup from distillation: {exp3_results['full_qd2p_time'] / exp3_results['distilled_time']:.2f}x")
    print(f"Experiment 3 - Quality difference: {(exp3_results['quality_full'] - exp3_results['quality_distilled']):.4f}")
    
    print("\nGenerated Figures:")
    print(f"- Experiment 1: {exp1_results['figure_path']}")
    print(f"- Experiment 2: {exp2_results['figure_path']}")
    print(f"- Experiment 3 (Latency): {exp3_results['latency_figure_path']}")
    print(f"- Experiment 3 (Quality): {exp3_results['quality_figure_path']}")
    
    return {
        "exp1_results": exp1_results,
        "exp2_results": exp2_results,
        "exp3_results": exp3_results
    }


def run_test_experiments(device):
    """Run quick tests of the QD²P experiments with reduced parameters."""
    logger.info("Starting test QD²P experiments with reduced parameters")
    
    data = create_synthetic_data(config.TEST_LATENT_DIM, config.BATCH_SIZE, device=device)
    
    diffusion_model = SimpleDiffusion(config.TEST_LATENT_DIM).to(device)
    q_probe = QProbe(config.TEST_LATENT_DIM).to(device)
    
    print_experiment_header("Quick Tests of All Experiments")
    test_time = test_all_experiments(diffusion_model, q_probe, config, data, device=device)
    
    print(f"\nAll tests completed in {test_time:.4f} seconds")
    print("\nTest run successful. For full experiments, run without the --test flag.")


if __name__ == "__main__":
    args = parse_arguments()
    
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    setup_experiment_directories()
    
    print_system_info()
    
    start_time = time.time()
    
    if args.test:
        run_test_experiments(args.device)
    else:
        run_full_experiments(args.device)
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\nExperiment completed successfully!")
