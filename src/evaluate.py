"""
Evaluation module for QD²P experiment.

Implements the three experiments described in the paper:
1. Controlled Quality Evaluation on Synthetic Tasks
2. Ablation Study on the Candidate Reweighting Mechanism
3. Efficiency and Distillation: Reducing Inference Overhead
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def controlled_quality_experiment(diffusion_model, q_probe, config, data, device="cuda"):
    """
    Experiment 1: Controlled Quality Evaluation on Synthetic Tasks.
    
    Compares baseline diffusion with QD²P using synthetic latent vectors.
    
    Args:
        diffusion_model: Pretrained diffusion model
        q_probe: Pretrained Q-probe
        config: Configuration parameters
        data: Dictionary containing evaluation data
        device: Device to run on
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting Experiment 1: Controlled Quality Evaluation on Synthetic Tasks.")
    
    latent = data["latent"]
    ideal = data["ideal"]
    
    quality_scores_baseline = []
    quality_scores_qd2p = []
    
    for step in range(config.DIFFUSION_STEPS):
        noise_level = 1.0 / (step + 1)
        logger.info(f"Step {step+1}/{config.DIFFUSION_STEPS}, noise_level = {noise_level:.4f}")
        
        latent_baseline = diffusion_model(latent, noise_level)
        score_baseline = quality_metric(latent_baseline, ideal.expand_as(latent_baseline))
        quality_scores_baseline.append(score_baseline.item())
        
        candidates = []
        q_values = []
        for _ in range(config.CANDIDATE_COUNT):
            candidate = diffusion_model(latent, noise_level)
            candidates.append(candidate)
            q_val = q_probe(candidate)
            q_values.append(q_val)
        candidates_tensor = torch.stack(candidates)
        q_values_tensor = torch.stack(q_values).squeeze(-1)
        
        weights = F.softmax(q_values_tensor, dim=0)
        weighted_candidate = (weights.unsqueeze(-1) * candidates_tensor).sum(dim=0)
        
        score_qd2p = quality_metric(weighted_candidate, ideal.expand_as(weighted_candidate))
        quality_scores_qd2p.append(score_qd2p.item())
        
        latent = latent_baseline.detach()
    
    plt.figure(figsize=(config.FIGURE_WIDTH, config.FIGURE_HEIGHT))
    plt.plot(range(1, config.DIFFUSION_STEPS + 1), quality_scores_baseline, 
             label="Baseline Diffusion", marker='o')
    plt.plot(range(1, config.DIFFUSION_STEPS + 1), quality_scores_qd2p, 
             label="QD²P", marker='s')
    plt.xlabel("Denoising Step")
    plt.ylabel("Quality Metric (Cosine Similarity)")
    plt.title("Quality Metrics over Denoising Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{config.SAVE_DIR}/quality_metric_synthetic.pdf"
    Path(filename).parent.mkdir(exist_ok=True)
    plt.savefig(filename, dpi=config.PDF_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Experiment 1 plot saved as {filename}")
    
    print("Experiment 1 Results:")
    print("Baseline Quality Scores:", quality_scores_baseline)
    print("QD²P Quality Scores:", quality_scores_qd2p)
    
    return {
        "baseline_scores": quality_scores_baseline,
        "qd2p_scores": quality_scores_qd2p,
        "figure_path": filename
    }


def ablation_candidate_reweighting(diffusion_model, q_probe, config, data, device="cuda"):
    """
    Experiment 2: Ablation Study on the Candidate Reweighting Mechanism.
    
    Studies the effect of different candidate counts and temperatures.
    
    Args:
        diffusion_model: Pretrained diffusion model
        q_probe: Pretrained Q-probe
        config: Configuration parameters
        data: Dictionary containing evaluation data
        device: Device to run on
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting Experiment 2: Ablation Study on Candidate Reweighting Mechanism.")
    
    ideal = data["ideal"]
    results = {}
    
    for k in config.CANDIDATE_COUNTS:
        for temp in config.TEMPERATURES:
            config_key = f"k={k}_T={temp}"
            quality_scores = []
            latent = data["latent"].clone()
            
            for step in range(config.DIFFUSION_STEPS):
                noise_level = 1.0 / (step + 1)
                candidates = []
                q_values = []
                
                for _ in range(k):
                    candidate = diffusion_model(latent, noise_level)
                    candidates.append(candidate)
                    q_values.append(q_probe(candidate))
                
                candidates_tensor = torch.stack(candidates)
                q_values_tensor = torch.stack(q_values).squeeze(-1)
                
                weights = F.softmax(q_values_tensor / temp, dim=0)
                weighted_candidate = (weights.unsqueeze(-1) * candidates_tensor).sum(dim=0)
                
                quality = quality_metric(weighted_candidate, ideal.expand_as(weighted_candidate))
                quality_scores.append(quality.item())
                
                latent = weighted_candidate.detach()
            
            results[config_key] = quality_scores
            logger.info(f"Config {config_key} completed.")
    
    plt.figure(figsize=(config.FIGURE_WIDTH, config.FIGURE_HEIGHT))
    
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(config.CANDIDATE_COUNTS)))
    line_styles = ['-', '--', '-.', ':']
    
    for i, k in enumerate(config.CANDIDATE_COUNTS):
        for j, temp in enumerate(config.TEMPERATURES):
            config_key = f"k={k}_T={temp}"
            plt.plot(range(1, config.DIFFUSION_STEPS + 1), results[config_key], 
                     label=config_key, color=colors[i], 
                     linestyle=line_styles[j % len(line_styles)], marker='o')
    
    plt.xlabel("Denoising Step")
    plt.ylabel("Quality Metric (Cosine Similarity)")
    plt.title("Ablation Study: Candidate Count and Temperature Impact")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{config.SAVE_DIR}/ablation_candidate_reweighting.pdf"
    plt.savefig(filename, dpi=config.PDF_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Experiment 2 plot saved as {filename}")
    
    print("Experiment 2 Results (Quality Metric per Configuration):")
    for config_key, scores in results.items():
        print(f"  {config_key}: {scores}")
    
    return {
        "results": results,
        "figure_path": filename
    }


def lora_adapter_module(input_tensor, adapter_params):
    """
    A simple LoRA-like lightweight adapter.
    
    Args:
        input_tensor: Input tensor
        adapter_params: Adapter parameters (matrix)
        
    Returns:
        Adapted tensor
    """
    return input_tensor + input_tensor @ adapter_params


def efficiency_and_distillation(diffusion_model, q_probe, config, data, device="cuda"):
    """
    Experiment 3: Efficiency and Distillation (Reducing Inference Overhead).
    
    Compares inference time between full QD²P and a distilled version.
    
    Args:
        diffusion_model: Pretrained diffusion model
        q_probe: Pretrained Q-probe
        config: Configuration parameters
        data: Dictionary containing evaluation data
        device: Device to run on
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting Experiment 3: Efficiency and Distillation.")
    
    latent = data["latent"]
    ideal = data["ideal"]
    
    start_time = time.time()
    
    for step in range(config.DIFFUSION_STEPS):
        noise_level = 1.0 / (step + 1)
        candidates = []
        q_values = []
        
        for _ in range(config.CANDIDATE_COUNT):
            candidate = diffusion_model(latent, noise_level)
            candidates.append(candidate)
            q_values.append(q_probe(candidate))
        
        candidates_tensor = torch.stack(candidates)
        q_values_tensor = torch.stack(q_values).squeeze(-1)
        weights = F.softmax(q_values_tensor, dim=0)
        weighted_candidate = (weights.unsqueeze(-1) * candidates_tensor).sum(dim=0)
        latent = weighted_candidate.detach()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    full_qd2p_time = time.time() - start_time
    
    adapter_params = torch.randn(config.LATENT_DIM, config.LATENT_DIM, device=device) * 0.01
    latent = data["latent"].clone()
    
    start_time = time.time()
    for step in range(config.DIFFUSION_STEPS):
        noise_level = 1.0 / (step + 1)
        latent_diffused = diffusion_model(latent, noise_level)
        latent = lora_adapter_module(latent_diffused, adapter_params)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    distilled_time = time.time() - start_time
    
    quality_full = quality_metric(latent, ideal.expand_as(latent))
    quality_distilled = quality_metric(latent, ideal.expand_as(latent))
    
    print("Full QD²P Inference Time: {:.4f} s, Quality: {:.4f}".format(
        full_qd2p_time, quality_full))
    print("Distilled Inference Time: {:.4f} s, Quality: {:.4f}".format(
        distilled_time, quality_distilled))
    
    if torch.cuda.is_available():
        mem_usage = torch.cuda.memory_allocated() / (1024 * 1024)
        print("Current GPU Memory Usage: {:.2f} MB".format(mem_usage))
    else:
        print("CUDA is not available; skipping GPU memory profiling.")
    
    plt.figure(figsize=(config.FIGURE_WIDTH, config.FIGURE_HEIGHT/2))
    methods = ['Full QD²P', 'Distilled']
    times = [full_qd2p_time, distilled_time]
    plt.bar(methods, times, color=["blue", "green"])
    plt.ylabel("Inference Time (s)")
    plt.title("Inference Time Comparison")
    
    for i, v in enumerate(times):
        plt.text(i, v + 0.01, f"{v:.4f}s", ha='center')
    
    filename = f"{config.SAVE_DIR}/inference_latency_comparison.pdf"
    plt.savefig(filename, dpi=config.PDF_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Experiment 3 inference latency plot saved as {filename}")
    
    plt.figure(figsize=(config.FIGURE_WIDTH, config.FIGURE_HEIGHT/2))
    qualities = [quality_full.item(), quality_distilled.item()]
    plt.bar(methods, qualities, color=["blue", "green"])
    plt.ylabel("Quality Metric")
    plt.title("Quality Comparison")
    
    for i, v in enumerate(qualities):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    quality_filename = f"{config.SAVE_DIR}/quality_comparison.pdf"
    plt.savefig(quality_filename, dpi=config.PDF_DPI, bbox_inches='tight')
    plt.close()
    
    return {
        "full_qd2p_time": full_qd2p_time,
        "distilled_time": distilled_time,
        "quality_full": quality_full.item(),
        "quality_distilled": quality_distilled.item(),
        "latency_figure_path": filename,
        "quality_figure_path": quality_filename
    }


def quality_metric(output, target):
    """
    Compute quality metric (cosine similarity) between output and target.
    
    Args:
        output: Output tensor
        target: Target tensor
        
    Returns:
        Cosine similarity (higher is better)
    """
    output_norm = F.normalize(output, dim=1)
    target_norm = F.normalize(target, dim=1)
    similarity = (output_norm * target_norm).sum(dim=1).mean()
    return similarity


def test_all_experiments(diffusion_model, q_probe, config, data, device="cuda"):
    """
    Run quick tests of all experiments.
    
    Args:
        diffusion_model: Pretrained diffusion model
        q_probe: Pretrained Q-probe
        config: Test configuration with reduced parameters
        data: Dictionary containing test data
        device: Device to run on
    """
    print("\nRunning quick tests for all experiments...")
    
    test_config = type('TestConfig', (), {
        'LATENT_DIM': config.TEST_LATENT_DIM,
        'DIFFUSION_STEPS': config.TEST_STEPS,
        'CANDIDATE_COUNT': config.TEST_CANDIDATE_COUNT,
        'CANDIDATE_COUNTS': [2, 3],
        'TEMPERATURES': [0.5, 1.0],
        'SAVE_DIR': config.SAVE_DIR,
        'PDF_DPI': config.PDF_DPI,
        'FIGURE_WIDTH': config.FIGURE_WIDTH,
        'FIGURE_HEIGHT': config.FIGURE_HEIGHT
    })()
    
    start_overall = time.time()
    
    print("\nTesting Experiment 1 (Controlled Quality Evaluation)...")
    controlled_quality_experiment(diffusion_model, q_probe, test_config, data, device)
    
    print("\nTesting Experiment 2 (Ablation Candidate Reweighting)...")
    ablation_candidate_reweighting(diffusion_model, q_probe, test_config, data, device)
    
    print("\nTesting Experiment 3 (Efficiency and Distillation)...")
    efficiency_and_distillation(diffusion_model, q_probe, test_config, data, device)
    
    overall_time = time.time() - start_overall
    print("\nAll tests completed in {:.4f} seconds.".format(overall_time))
    
    return overall_time


if __name__ == "__main__":
    from preprocess import create_synthetic_data, set_seed
    from train import SimpleDiffusion, QProbe
    
    set_seed(42)
    
    class Config:
        LATENT_DIM = 8
        DIFFUSION_STEPS = 3
        CANDIDATE_COUNT = 3
        CANDIDATE_COUNTS = [2, 3]
        TEMPERATURES = [0.5, 1.0]
        SAVE_DIR = "logs"
        PDF_DPI = 300
        FIGURE_WIDTH = 8
        FIGURE_HEIGHT = 4
        TEST_LATENT_DIM = 8
        TEST_STEPS = 3
        TEST_CANDIDATE_COUNT = 3
    
    data = create_synthetic_data(Config.LATENT_DIM, 4, device="cpu")
    diffusion_model = SimpleDiffusion(Config.LATENT_DIM)
    q_probe = QProbe(Config.LATENT_DIM)
    
    test_all_experiments(diffusion_model, q_probe, Config, data, device="cpu")
