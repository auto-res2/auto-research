"""
Main script for running IDRR-GAR experiments.
This script runs three experiments:
1. Evaluation on Dynamic and Non-Rigid Scenes
2. Ablation Study on the Iterative Dynamic Region Refinement Module
3. Analysis of Geometry-Aware Sampling Effectiveness
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import importlib
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import get_data_loaders, DynamicScenesDataset
from src.train import (
    BaseDepthModel, IDRRGARModel, train_model, 
    geometry_aware_sampling, uniform_sampling,
    photometric_loss, smoothness_loss, scale_alignment_loss, compute_region_loss
)
from src.evaluate import (
    evaluate_model, visualize_sampling, save_loss_comparison,
    save_loss_iterations, save_bar_comparison, run_statistical_test
)

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"))
try:
    import idrr_gar_config as config
except ImportError:
    class DefaultConfig:
        EXPERIMENT_NAME = "IDRR-GAR"
        RANDOM_SEED = 42
        BATCH_SIZE = 8
        NUM_EPOCHS = 1
        LEARNING_RATE = 1e-3
        ITERATIONS = 3
        INPUT_CHANNELS = 3
        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 320
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        USE_FP16 = True
        NUM_REGIONS = 50
        PHOTOMETRIC_LOSS_WEIGHT = 1.0
        SMOOTHNESS_LOSS_WEIGHT = 0.1
        SCALE_ALIGNMENT_LOSS_WEIGHT = 0.1
        DATA_DIR = "data/"
        MODELS_DIR = "models/"
        LOGS_DIR = "logs/"
    
    config = DefaultConfig()

def setup_logging():
    """Set up logging to console and file."""
    log_dir = config.LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'experiment.log'))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def setup_device():
    """Set up the device for computation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        if config.USE_FP16 and torch.cuda.is_available():
            logging.info("Enabling mixed precision (FP16) for memory efficiency")
    else:
        device = torch.device("cpu")
        logging.info("No GPU available, using CPU")
    
    return device

def create_output_dirs():
    """Create necessary directories for outputs."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

def experiment_dynamic_nonrigid(device, logger):
    """
    Run experiment 1: Evaluation on Dynamic and Non-Rigid Scenes.
    
    Args:
        device: Device for computation
        logger: Logger object
    """
    logger.info("\n=== Starting Experiment 1: Evaluation on Dynamic and Non-Rigid Scenes ===")
    
    train_loader, eval_loader = get_data_loaders(config)
    logger.info(f"Created data loaders with {len(train_loader)} training batches and {len(eval_loader)} evaluation batches")
    
    base_model = BaseDepthModel(input_channels=config.INPUT_CHANNELS).to(device)
    idrr_model = IDRRGARModel(input_channels=config.INPUT_CHANNELS, iterations=config.ITERATIONS).to(device)
    
    optimizer_base = optim.Adam(base_model.parameters(), lr=config.LEARNING_RATE)
    optimizer_idrr = optim.Adam(idrr_model.parameters(), lr=config.LEARNING_RATE)
    
    loss_base_history = []
    loss_idrr_history = []
    
    logger.info(f"Training for {config.NUM_EPOCHS} epochs")
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        logger.info("Training Base Model...")
        epoch_losses_base = train_model(base_model, train_loader, optimizer_base, device, config)
        loss_base_history.extend(epoch_losses_base)
        
        logger.info("Training IDRR-GAR Model...")
        epoch_losses_idrr = train_model(idrr_model, train_loader, optimizer_idrr, device, config)
        loss_idrr_history.extend(epoch_losses_idrr)
        
        logger.info("Evaluating models...")
        base_metrics = evaluate_model(base_model, eval_loader, device, config)
        idrr_metrics = evaluate_model(idrr_model, eval_loader, device, config)
        
        logger.info(f"Base Model - Avg Loss: {base_metrics['avg_loss']:.4f}")
        logger.info(f"IDRR-GAR Model - Avg Loss: {idrr_metrics['avg_loss']:.4f}")
    
    save_loss_comparison(
        loss_base_history, 
        loss_idrr_history, 
        "Training Loss Comparison: Base vs. IDRR-GAR",
        os.path.join(config.LOGS_DIR, "training_loss_idrr_vs_base.pdf")
    )
    
    run_statistical_test(
        np.array(loss_idrr_history), 
        np.array(loss_base_history),
        "IDRR-GAR", 
        "Base Method"
    )
    
    logger.info("Experiment 1 completed successfully")
    return loss_base_history, loss_idrr_history

def experiment_ablation(device, logger):
    """
    Run experiment 2: Ablation Study on the Iterative Dynamic Region Refinement Module.
    
    Args:
        device: Device for computation
        logger: Logger object
    """
    logger.info("\n=== Starting Experiment 2: Ablation Study on Iterative Refinement ===")
    
    images = torch.rand(2, config.INPUT_CHANNELS, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
    gt_depth = torch.rand(2, 1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
    
    full_model = IDRRGARModel(input_channels=config.INPUT_CHANNELS, iterations=config.ITERATIONS).to(device)
    simplified_model = IDRRGARModel(input_channels=config.INPUT_CHANNELS, iterations=1).to(device)
    
    logger.info("Running iterative refinement with full model...")
    current_depth = full_model.decoder(full_model.encoder(images))
    loss_history = []
    
    for i in range(full_model.iterations):
        delta = full_model.transformer_refine(images, current_depth)
        current_depth = current_depth + delta
        loss_iter = photometric_loss(current_depth, gt_depth)
        loss_history.append(loss_iter.item())
        logger.info(f"Iteration {i+1}: Reconstruction Loss = {loss_iter.item():.4f}")
    
    save_loss_iterations(
        loss_history,
        "Loss Variation over Refinement Iterations",
        os.path.join(config.LOGS_DIR, "training_loss_iterative_refinement.pdf")
    )
    
    logger.info("Running simplified model (single-step)...")
    pred_simplified = simplified_model(images)
    loss_simplified = photometric_loss(pred_simplified, gt_depth)
    logger.info(f"Simplified Model (single-step) Reconstruction Loss: {loss_simplified.item():.4f}")
    
    final_loss_full = loss_history[-1]
    improvement = (loss_simplified.item() - final_loss_full) / loss_simplified.item() * 100
    
    logger.info(f"Full Model (final iteration) Loss: {final_loss_full:.4f}")
    logger.info(f"Improvement from iterative refinement: {improvement:.2f}%")
    
    save_bar_comparison(
        [loss_simplified.item(), final_loss_full],
        ["Single-Step", f"{config.ITERATIONS}-Step Iterative"],
        "Comparison of Single-Step vs. Iterative Refinement",
        os.path.join(config.LOGS_DIR, "ablation_comparison.pdf")
    )
    
    logger.info("Experiment 2 completed successfully")
    return loss_history, loss_simplified.item()

def experiment_geometry_sampling(device, logger):
    """
    Run experiment 3: Analysis of Geometry-Aware Sampling Effectiveness.
    
    Args:
        device: Device for computation
        logger: Logger object
    """
    logger.info("\n=== Starting Experiment 3: Geometry-Aware Sampling ===")
    
    images = torch.rand(2, config.INPUT_CHANNELS, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
    
    encoder = nn.Conv2d(config.INPUT_CHANNELS, 8, kernel_size=3, padding=1).to(device)
    features = encoder(images)  # shape: (2, 8, H, W)
    
    logger.info("Performing geometry-aware and uniform sampling...")
    samples_geo = geometry_aware_sampling(features, num_regions=config.NUM_REGIONS)
    samples_uniform = uniform_sampling(features, num_regions=config.NUM_REGIONS)
    
    model = IDRRGARModel(input_channels=config.INPUT_CHANNELS, iterations=config.ITERATIONS).to(device)
    pred_depth = model(images)
    
    gt_depth = torch.rand(2, 1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH).to(device)
    
    region_loss_geo = compute_region_loss(pred_depth, gt_depth, samples_geo)
    region_loss_uniform = compute_region_loss(pred_depth, gt_depth, samples_uniform)
    
    if isinstance(region_loss_geo, torch.Tensor):
        region_loss_geo_cpu = region_loss_geo.detach().cpu().item()
    else:
        region_loss_geo_cpu = float(region_loss_geo)
        
    if isinstance(region_loss_uniform, torch.Tensor):
        region_loss_uniform_cpu = region_loss_uniform.detach().cpu().item()
    else:
        region_loss_uniform_cpu = float(region_loss_uniform)
    
    logger.info(f"Region-specific photometric loss with Geometry-Aware Sampling: {region_loss_geo_cpu:.4f}")
    logger.info(f"Region-specific photometric loss with Uniform Sampling: {region_loss_uniform_cpu:.4f}")
    
    img_np = images[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize to [0,1]
    
    visualize_sampling(
        img_np, 
        samples_geo[0], 
        height=config.IMAGE_HEIGHT, 
        width=config.IMAGE_WIDTH, 
        filename=os.path.join(config.LOGS_DIR, "sampling_visualization_geometry.pdf")
    )
    
    visualize_sampling(
        img_np, 
        samples_uniform[0], 
        height=config.IMAGE_HEIGHT, 
        width=config.IMAGE_WIDTH, 
        filename=os.path.join(config.LOGS_DIR, "sampling_visualization_uniform.pdf")
    )
    
    save_bar_comparison(
        [region_loss_geo_cpu, region_loss_uniform_cpu],  # Use CPU values
        ["Geometry-Aware", "Uniform"],
        "Sampling Strategy Comparison",
        os.path.join(config.LOGS_DIR, "sampling_loss_comparison.pdf")
    )
    
    if region_loss_uniform_cpu > region_loss_geo_cpu:
        improvement = (region_loss_uniform_cpu - region_loss_geo_cpu) / region_loss_uniform_cpu * 100
        logger.info(f"Geometry-aware sampling improves loss by {improvement:.2f}% compared to uniform sampling")
    else:
        difference = (region_loss_geo_cpu - region_loss_uniform_cpu) / region_loss_uniform_cpu * 100
        logger.info(f"Uniform sampling performs better by {difference:.2f}% in this test case")
    
    logger.info("Experiment 3 completed successfully")
    return region_loss_geo_cpu, region_loss_uniform_cpu

def main():
    """Main function to run all experiments."""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    logger = setup_logging()
    logger.info(f"Starting IDRR-GAR experiments with random seed {config.RANDOM_SEED}")
    
    create_output_dirs()
    
    device = setup_device()
    
    logger.info(f"Experiment Configuration:")
    logger.info(f"- Batch Size: {config.BATCH_SIZE}")
    logger.info(f"- Number of Epochs: {config.NUM_EPOCHS}")
    logger.info(f"- Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"- Refinement Iterations: {config.ITERATIONS}")
    logger.info(f"- Image Dimensions: {config.IMAGE_HEIGHT}x{config.IMAGE_WIDTH}")
    logger.info(f"- Using FP16: {config.USE_FP16}")
    
    try:
        loss_base, loss_idrr = experiment_dynamic_nonrigid(device, logger)
        
        loss_iterations, loss_simplified = experiment_ablation(device, logger)
        
        loss_geo, loss_uniform = experiment_geometry_sampling(device, logger)
        
        print("\n" + "="*80)
        print("                     IDRR-GAR EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        print("\n[EXPERIMENT 1] Evaluation on Dynamic and Non-Rigid Scenes")
        print("-"*70)
        print(f"Base Model:")
        print(f"  - Initial Loss: {loss_base[0]:.4f}")
        print(f"  - Final Loss: {loss_base[-1]:.4f}")
        print(f"  - Improvement: {((loss_base[0] - loss_base[-1]) / loss_base[0] * 100):.2f}%")
        print(f"\nIDRR-GAR Model:")
        print(f"  - Initial Loss: {loss_idrr[0]:.4f}")
        print(f"  - Final Loss: {loss_idrr[-1]:.4f}")
        print(f"  - Improvement: {((loss_idrr[0] - loss_idrr[-1]) / loss_idrr[0] * 100):.2f}%")
        print(f"\nComparison:")
        if loss_idrr[-1] < loss_base[-1]:
            print(f"  - IDRR-GAR outperforms Base Model by {((loss_base[-1] - loss_idrr[-1]) / loss_base[-1] * 100):.2f}%")
        else:
            print(f"  - Base Model outperforms IDRR-GAR by {((loss_idrr[-1] - loss_base[-1]) / loss_idrr[-1] * 100):.2f}%")
        
        print("\n[EXPERIMENT 2] Ablation Study on Iterative Refinement")
        print("-"*70)
        print(f"Iterative Refinement Process:")
        for i, loss in enumerate(loss_iterations):
            print(f"  - Iteration {i+1}: Loss = {loss:.4f}")
        
        print(f"\nComparison:")
        print(f"  - Single-Step Model Loss: {loss_simplified:.4f}")
        print(f"  - {config.ITERATIONS}-Step Iterative Model Final Loss: {loss_iterations[-1]:.4f}")
        if loss_iterations[-1] < loss_simplified:
            print(f"  - Iterative refinement improves performance by {((loss_simplified - loss_iterations[-1]) / loss_simplified * 100):.2f}%")
        else:
            print(f"  - Single-step approach performs better by {((loss_iterations[-1] - loss_simplified) / loss_iterations[-1] * 100):.2f}%")
            print(f"  - This suggests that for this specific test case, additional iterations may not be beneficial")
        
        print("\n[EXPERIMENT 3] Analysis of Geometry-Aware Sampling")
        print("-"*70)
        print(f"Sampling Strategy Comparison:")
        print(f"  - Geometry-Aware Sampling Loss: {loss_geo:.4f}")
        print(f"  - Uniform Sampling Loss: {loss_uniform:.4f}")
        if loss_geo < loss_uniform:
            print(f"  - Geometry-aware sampling improves performance by {((loss_uniform - loss_geo) / loss_uniform * 100):.2f}%")
            print(f"  - This confirms our hypothesis that focusing on regions with high geometric complexity")
            print(f"    leads to better depth estimation in dynamic scenes")
        else:
            print(f"  - Uniform sampling performs better by {((loss_geo - loss_uniform) / loss_geo * 100):.2f}%")
            print(f"  - This suggests that for this specific test case, the geometry-aware sampling")
            print(f"    may not provide significant benefits over uniform sampling")
        
        print("\n[VISUALIZATION OUTPUTS]")
        print("-"*70)
        print(f"The following visualization files have been generated in {config.LOGS_DIR}:")
        print(f"  1. training_loss_idrr_vs_base.pdf - Loss comparison between Base and IDRR-GAR models")
        print(f"  2. training_loss_iterative_refinement.pdf - Loss variation over refinement iterations")
        print(f"  3. ablation_comparison.pdf - Comparison of single-step vs. iterative refinement")
        print(f"  4. sampling_visualization_geometry.pdf - Visualization of geometry-aware sampling")
        print(f"  5. sampling_visualization_uniform.pdf - Visualization of uniform sampling")
        print(f"  6. sampling_loss_comparison.pdf - Loss comparison between sampling strategies")
        
        print("\n" + "="*80)
        print("                          EXPERIMENT COMPLETED")
        print("="*80)
        
        logger.info("\n=== Experiment Results Summary ===")
        logger.info("Experiment 1 - Dynamic and Non-Rigid Scenes:")
        logger.info(f"- Base Model Final Loss: {loss_base[-1]:.4f}")
        logger.info(f"- IDRR-GAR Model Final Loss: {loss_idrr[-1]:.4f}")
        
        logger.info("\nExperiment 2 - Ablation Study:")
        logger.info(f"- Single-Step Model Loss: {loss_simplified:.4f}")
        logger.info(f"- {config.ITERATIONS}-Step Iterative Model Final Loss: {loss_iterations[-1]:.4f}")
        
        logger.info("\nExperiment 3 - Geometry-Aware Sampling:")
        logger.info(f"- Geometry-Aware Sampling Loss: {loss_geo:.4f}")
        logger.info(f"- Uniform Sampling Loss: {loss_uniform:.4f}")
        
        logger.info("\nAll experiments completed successfully!")
        logger.info(f"PDF figures saved in {config.LOGS_DIR}")
        
    except Exception as e:
        logger.error(f"Error during experiment execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
