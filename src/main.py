"""
SpectralGraph-ST: Spectral Graph–Enhanced Scene Transformer

This script implements three experiments:
  Experiment 1: Robustness to Noisy Object Detections.
  Experiment 2: Computational Efficiency Analysis.
  Experiment 3: Module Ablation Study.

The implementation uses dummy scene graph models (EGTR and SpectralGraph-ST)
and synthetic data. All plots are saved as high-quality PDF files.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage
import sys
import random

from preprocess import load_dataset, add_noise_to_bboxes, add_noise_to_confidences
from train import EGTR, SpectralGraphST
from evaluate import evaluate_model, profile_pipeline, save_plot

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.experiment_config import EXPERIMENT_CONFIG, MODEL_CONFIGS, NOISE_LEVELS
    config = EXPERIMENT_CONFIG
    print(f"Loaded configuration from config/experiment_config.py")
except ImportError:
    print("Configuration file not found, using defaults.")
    config = {
        "random_seed": 42,
        "num_images": 20,
        "test_mode": False
    }
    MODEL_CONFIGS = {
        "egtr": {},
        "spectral_graph_st": {
            "use_spectral_filter": True,
            "use_stochastic_sampling": True
        }
    }
    NOISE_LEVELS = [(0.0, 0.0), (5.0, 0.1), (10.0, 0.2), (15.0, 0.3)]

random_seed = config.get("random_seed", 42)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("config", exist_ok=True)


def experiment_noise_robustness():
    """
    Experiment to test model robustness to noisy object detections.
    Compares EGTR and SpectralGraph-ST performance under different noise levels.
    """
    print("\n=== Experiment 1: Robustness to Noisy Object Detections ===")
    
    num_images = 5 if config.get("test_mode", False) else config.get("num_images", 20)
    dataset = load_dataset("VisualGenome", num_images=num_images)
    
    model_egtr = EGTR()
    model_st = SpectralGraphST()
    
    performance_egtr = []
    performance_st = []
    
    for bbox_noise, flip_prob in NOISE_LEVELS:
        print(f"Testing noise level: bbox_std = {bbox_noise}, flip_prob = {flip_prob}")
        metrics_list_egtr = []
        metrics_list_st = []
        
        for image, gt in dataset:
            detections = image["detections"]
            
            noisy_bboxes = add_noise_to_bboxes(detections['bboxes'], noise_std=bbox_noise)
            noisy_confidences = add_noise_to_confidences(detections['confidences'], flip_prob=flip_prob)
            
            noisy_detections = {
                "bboxes": noisy_bboxes,
                "confidences": noisy_confidences,
                "labels": detections['labels']
            }
            
            scene_graph_egtr = model_egtr(image, noisy_detections)
            scene_graph_st = model_st(image, noisy_detections)
            
            metric_egtr = evaluate_model(scene_graph_egtr, gt)
            metric_st = evaluate_model(scene_graph_st, gt)
            metrics_list_egtr.append(metric_egtr)
            metrics_list_st.append(metric_st)
        
        avg_egtr = np.mean(np.array(metrics_list_egtr)[:, 0])
        avg_st = np.mean(np.array(metrics_list_st)[:, 0])
        performance_egtr.append(avg_egtr)
        performance_st.append(avg_st)
        print(f"  EGTR avg recall: {avg_egtr:.3f}, SpectralGraph-ST avg recall: {avg_st:.3f}")
    
    noise_labels = [f"{n[0]}std_{n[1]}flip" for n in NOISE_LEVELS]
    plt.figure(figsize=(6, 4))
    plt.plot(noise_labels, performance_egtr, label="EGTR", marker="o")
    plt.plot(noise_labels, performance_st, label="SpectralGraph-ST", marker="o")
    plt.xlabel("Noise Level (bbox noise std & flip probability)")
    plt.ylabel("Average Recall@K")
    plt.legend()
    plt.title("Performance vs Noise Level")
    
    save_plot(plt, "accuracy_noise_robustness_pair1", "Performance vs Noise Level")


def deterministic_candidate_edge_search(model, image, detections):
    """Simulate full exhaustive search for candidate relation edges."""
    return model.full_relation_search(image, detections)

def stochastic_spectral_sampling(model, image, detections):
    """Use the stochastic spectral sampling search."""
    return model.spectral_relation_search(image, detections)

def experiment_efficiency():
    """
    Experiment to compare computational efficiency of deterministic search vs stochastic sampling.
    Measures execution time and memory usage.
    """
    print("\n=== Experiment 2: Computational Efficiency Analysis ===")
    
    image = torch.randn(3, 224, 224)
    num_objects = 50 if not config.get("test_mode", False) else 20
    synthetic_detections = {
        "bboxes": np.random.rand(num_objects, 4) * 224,
        "confidences": np.random.rand(num_objects),
        "labels": np.random.randint(0, 20, size=(num_objects,))
    }
    
    model_st = SpectralGraphST()
    
    dt_time, dt_memory, dt_result = profile_pipeline(
        deterministic_candidate_edge_search, model_st, image, synthetic_detections
    )
    
    st_time, st_memory, st_result = profile_pipeline(
        stochastic_spectral_sampling, model_st, image, synthetic_detections
    )
    
    print(f"Deterministic Search - Time: {dt_time:.4f}s, Memory: {dt_memory:.2f} MB")
    print(f"Stochastic Sampling - Time: {st_time:.4f}s, Memory: {st_memory:.2f} MB")
    
    variants = ["Deterministic", "Stochastic"]
    times = [dt_time, st_time]
    memories = [dt_memory, st_memory]
    
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(variants, times, color=["red", "green"])
    plt.ylabel("Execution Time (s)")
    plt.title("Runtime Comparison")
    
    plt.subplot(1, 2, 2)
    plt.bar(variants, memories, color=["red", "green"])
    plt.ylabel("Peak Memory (MB)")
    plt.title("Memory Usage Comparison")
    
    save_plot(plt, "inference_latency_efficiency_pair1", "Efficiency Analysis")


def experiment_module_ablation():
    """
    Experiment to study the impact of different SpectralGraph-ST modules.
    Compares full model against variants with components disabled.
    """
    print("\n=== Experiment 3: Module Ablation Study ===")
    
    num_images = 5 if config.get("test_mode", False) else config.get("num_images", 20)
    dataset = load_dataset("VisualGenome", num_images=num_images)
    
    config_full = MODEL_CONFIGS.get("spectral_graph_st", {
        "use_spectral_filter": True, 
        "use_stochastic_sampling": True
    })
    
    config_no_spectral = MODEL_CONFIGS.get("spectral_graph_st_no_filter", {
        "use_spectral_filter": False, 
        "use_stochastic_sampling": True
    })
    
    config_deterministic = MODEL_CONFIGS.get("spectral_graph_st_no_sampling", {
        "use_spectral_filter": True, 
        "use_stochastic_sampling": False
    })
    
    model_full = SpectralGraphST(config=config_full)
    model_no_spectral = SpectralGraphST(config=config_no_spectral)
    model_deterministic = SpectralGraphST(config=config_deterministic)
    
    results_full = []
    results_no_spectral = []
    results_deterministic = []
    
    for image, gt in dataset:
        detections = image["detections"]
        
        output_full = model_full(image, detections)
        output_no_spec = model_no_spectral(image, detections)
        output_det = model_deterministic(image, detections)
        
        results_full.append(evaluate_model(output_full, gt))
        results_no_spectral.append(evaluate_model(output_no_spec, gt))
        results_deterministic.append(evaluate_model(output_det, gt))
    
    avg_full = np.mean(results_full, axis=0)
    avg_no_spectral = np.mean(results_no_spectral, axis=0)
    avg_deterministic = np.mean(results_deterministic, axis=0)
    
    print("Average Metrics (Recall@K, mAP, Relation F1):")
    print(f"  Full Model:         {np.round(avg_full, 3)}")
    print(f"  No Spectral Filter: {np.round(avg_no_spectral, 3)}")
    print(f"  Deterministic Samp: {np.round(avg_deterministic, 3)}")
    
    variants = ["Full", "No Spectral Filter", "Deterministic Sampling"]
    metrics_names = ["Recall@K", "mAP", "Relation F1"]
    
    x = np.arange(len(variants))
    width = 0.25
    
    plt.figure(figsize=(8, 5))
    for i in range(3):
        vals = [avg_full[i], avg_no_spectral[i], avg_deterministic[i]]
        plt.bar(x + i*width, vals, width, label=metrics_names[i])
    
    plt.xticks(x + width, variants)
    plt.xlabel("Model Variant")
    plt.ylabel("Performance Metric")
    plt.title("Module Ablation Study Results")
    plt.legend()
    
    save_plot(plt, "module_ablation_results_pair1", "Module Ablation Study")


def main():
    """
    Main function to run all experiments.
    """
    print("=" * 80)
    print("SpectralGraph-ST (Spectral Graph–Enhanced Scene Transformer) Experiments")
    print("=" * 80)
    
    test_mode = config.get("test_mode", False)
    if test_mode:
        print("\nRunning in TEST MODE with reduced dataset size and iterations.\n")
    
    try:
        experiment_noise_robustness()
        experiment_efficiency()
        experiment_module_ablation()
        
        print("\n" + "=" * 80)
        print("All experiments completed successfully!")
        print("Results and plots saved to logs/ directory.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during experiment execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
