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
    
    print("\n" + "=" * 50)
    print("QD²P: Q-Denoising Diffusion Probe Experiment")
    print("=" * 50)
    print(f"\nExperiment Configuration:")
    print(f"- Latent Dimension: {config.LATENT_DIM}")
    print(f"- Batch Size: {config.BATCH_SIZE}")
    print(f"- Diffusion Steps: {config.DIFFUSION_STEPS}")
    print(f"- Candidate Count: {config.CANDIDATE_COUNT}")
    print(f"- Random Seed: {config.RANDOM_SEED}")
    print(f"- Device: {device}")
    print("\nStarting data generation...")
    
    data = create_synthetic_data(config.LATENT_DIM, config.BATCH_SIZE, device=device)
    print(f"Synthetic data created with shape: {data['latent'].shape}")
    
    print_experiment_header("Model Training")
    print("Training SimpleDiffusion model and QProbe...")
    print(f"- Optimizer: Adam with learning rate 0.01")
    print(f"- Loss function: MSE between Q-values and quality metrics")
    print(f"- Training for {config.DIFFUSION_STEPS} diffusion steps")
    
    models = train_models(config, data, device=device)
    diffusion_model = models["diffusion_model"]
    q_probe = models["q_probe"]
    print(f"Training completed in {models['train_time']:.2f} seconds")
    
    print_experiment_header("Experiment 1: Controlled Quality Evaluation")
    print("Running controlled quality evaluation on synthetic tasks...")
    print(f"- Comparing baseline diffusion vs. QD²P with {config.CANDIDATE_COUNT} candidates")
    print(f"- Measuring quality improvement over {config.DIFFUSION_STEPS} denoising steps")
    print(f"- Using cosine similarity as quality metric")
    exp1_results = controlled_quality_experiment(
        diffusion_model, q_probe, config, data, device=device
    )
    print("\nExperiment 1 Results:")
    print(f"- Initial baseline quality: {exp1_results['baseline_scores'][0]:.4f}")
    print(f"- Final baseline quality: {exp1_results['baseline_scores'][-1]:.4f}")
    print(f"- Initial QD²P quality: {exp1_results['qd2p_scores'][0]:.4f}")
    print(f"- Final QD²P quality: {exp1_results['qd2p_scores'][-1]:.4f}")
    print(f"- Quality improvement: {(exp1_results['qd2p_scores'][-1] - exp1_results['baseline_scores'][-1]):.4f}")
    print(f"- Figure saved to: {exp1_results['figure_path']}")
    
    print_experiment_header("Experiment 2: Ablation Study on Candidate Reweighting")
    print("Running ablation study on candidate reweighting mechanism...")
    print(f"- Testing {len(config.CANDIDATE_COUNTS)} different candidate counts: {config.CANDIDATE_COUNTS}")
    print(f"- Testing {len(config.TEMPERATURES)} different temperature values: {config.TEMPERATURES}")
    print(f"- Evaluating {len(config.CANDIDATE_COUNTS) * len(config.TEMPERATURES)} total configurations")
    exp2_results = ablation_candidate_reweighting(
        diffusion_model, q_probe, config, data, device=device
    )
    print("\nExperiment 2 Results:")
    best_config = max(exp2_results['results'].items(), key=lambda x: x[1][-1])[0]
    best_score = max(exp2_results['results'].items(), key=lambda x: x[1][-1])[1][-1]
    print(f"- Best configuration: {best_config} with final quality score: {best_score:.4f}")
    for config_name, scores in exp2_results['results'].items():
        print(f"- {config_name}: initial={scores[0]:.4f}, final={scores[-1]:.4f}")
    print(f"- Figure saved to: {exp2_results['figure_path']}")
    
    print_experiment_header("Experiment 3: Efficiency and Distillation")
    print("Running efficiency and distillation experiment...")
    print(f"- Comparing full QD²P vs. distilled version")
    print(f"- Measuring inference time and quality metrics")
    print(f"- Using lightweight adapter for distillation")
    exp3_results = efficiency_and_distillation(
        diffusion_model, q_probe, config, data, device=device
    )
    print("\nExperiment 3 Results:")
    print(f"- Full QD²P Inference Time: {exp3_results['full_qd2p_time']:.4f} seconds")
    print(f"- Distilled Inference Time: {exp3_results['distilled_time']:.4f} seconds")
    print(f"- Speedup from distillation: {exp3_results['full_qd2p_time'] / exp3_results['distilled_time']:.2f}x")
    print(f"- Full QD²P Quality: {exp3_results['quality_full']:.4f}")
    print(f"- Distilled Quality: {exp3_results['quality_distilled']:.4f}")
    print(f"- Quality difference: {(exp3_results['quality_full'] - exp3_results['quality_distilled']):.4f}")
    print(f"- Latency figure saved to: {exp3_results['latency_figure_path']}")
    print(f"- Quality figure saved to: {exp3_results['quality_figure_path']}")
    
    print_experiment_header("Summary of Results")
    print("QD²P Experiment Summary:")
    print("\nExperiment 1 - Controlled Quality Evaluation:")
    print(f"- Quality improvement: {(exp1_results['qd2p_scores'][-1] - exp1_results['baseline_scores'][-1]):.4f}")
    print(f"- Baseline final quality: {exp1_results['baseline_scores'][-1]:.4f}")
    print(f"- QD²P final quality: {exp1_results['qd2p_scores'][-1]:.4f}")
    
    print("\nExperiment 2 - Ablation Study:")
    best_config = max(exp2_results['results'].items(), key=lambda x: x[1][-1])[0]
    best_score = max(exp2_results['results'].items(), key=lambda x: x[1][-1])[1][-1]
    print(f"- Best configuration: {best_config}")
    print(f"- Best configuration score: {best_score:.4f}")
    print(f"- Total configurations tested: {len(exp2_results['results'])}")
    
    print("\nExperiment 3 - Efficiency and Distillation:")
    print(f"- Speedup from distillation: {exp3_results['full_qd2p_time'] / exp3_results['distilled_time']:.2f}x")
    print(f"- Quality retention: {(exp3_results['quality_distilled'] / exp3_results['quality_full'] * 100):.1f}%")
    
    print("\nGenerated Figures:")
    print(f"- Experiment 1 (Quality Evaluation): {exp1_results['figure_path']}")
    print(f"- Experiment 2 (Ablation Study): {exp2_results['figure_path']}")
    print(f"- Experiment 3 (Latency Comparison): {exp3_results['latency_figure_path']}")
    print(f"- Experiment 3 (Quality Comparison): {exp3_results['quality_figure_path']}")
    
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
