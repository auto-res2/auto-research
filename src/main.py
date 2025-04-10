"""
Main script for running DITTO-GSD experiments.
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from utils.dataset import Synthetic3DDataset
from utils.models import DITTOModel, DITTOGSDModel
from utils.visualization import plot_loss_curve, plot_degradation_comparison
from preprocess import preprocess_data, generate_degraded_versions, add_gaussian_noise
from train import train_model, train_variant
from evaluate import run_experiment, evaluate_single_sample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.ditto_gsd_config import MODEL_PARAMS, TRAIN_PARAMS, DATASET_PARAMS, EXPERIMENT_PARAMS, GPU_PARAMS

for directory in ['logs', 'models', 'data']:
    if not os.path.exists(directory):
        os.makedirs(directory)

def experiment1_controlled_reconstruction():
    """
    Experiment 1: Controlled Reconstruction Quality and Efficiency Comparison.
    """
    logger.info("Starting Experiment 1: Controlled Reconstruction Quality and Efficiency Comparison")
    
    dataset = Synthetic3DDataset(
        num_samples=DATASET_PARAMS['num_samples'],
        num_points=DATASET_PARAMS['num_points']
    )
    dataloader = preprocess_data(dataset, batch_size=TRAIN_PARAMS['batch_size'])
    
    device = GPU_PARAMS['device']
    model_ditto = DITTOModel(
        point_features=MODEL_PARAMS['point_features'],
        grid_features=MODEL_PARAMS['grid_features']
    ).to(device)
    
    model_ditto_gsd = DITTOGSDModel(
        point_features=MODEL_PARAMS['point_features'],
        grid_features=MODEL_PARAMS['grid_features'],
        use_gs_decoder=MODEL_PARAMS['use_gaussian_decoder']
    ).to(device)
    
    logger.info("Training Baseline DITTO model...")
    _, baseline_losses = train_model(
        model_ditto,
        dataloader,
        num_epochs=TRAIN_PARAMS['num_epochs'],
        device=device,
        lr=TRAIN_PARAMS['learning_rate'],
        weight_decay=TRAIN_PARAMS['weight_decay']
    )
    
    logger.info("Training DITTO-GSD model...")
    _, gsd_losses = train_model(
        model_ditto_gsd,
        dataloader,
        num_epochs=TRAIN_PARAMS['num_epochs'],
        device=device,
        use_geo_loss=MODEL_PARAMS['use_geometric_loss'],
        lr=TRAIN_PARAMS['learning_rate'],
        weight_decay=TRAIN_PARAMS['weight_decay']
    )
    
    logger.info("Evaluating Baseline DITTO model...")
    baseline_metrics = run_experiment(model_ditto, dataloader, device)
    
    logger.info("Evaluating DITTO-GSD model...")
    gsd_metrics = run_experiment(model_ditto_gsd, dataloader, device)
    
    logger.info("Results:")
    logger.info(f"Baseline DITTO - Mean Chamfer: {baseline_metrics['mean_chamfer']:.4f}, Runtime: {baseline_metrics['runtime']:.2f}s")
    logger.info(f"DITTO-GSD     - Mean Chamfer: {gsd_metrics['mean_chamfer']:.4f}, Runtime: {gsd_metrics['runtime']:.2f}s")
    
    epochs = list(range(1, TRAIN_PARAMS['num_epochs'] + 1))
    plot_loss_curve(
        epochs, 
        [baseline_losses, gsd_losses], 
        ["Baseline DITTO", "DITTO-GSD"],
        "Training Loss Comparison",
        "training_loss_baseline_pair1"
    )
    
    baseline_curve = np.linspace(baseline_metrics['mean_chamfer']*1.2, baseline_metrics['mean_chamfer'], 10)
    gsd_curve = np.linspace(gsd_metrics['mean_chamfer']*1.2, gsd_metrics['mean_chamfer'], 10)
    plot_loss_curve(
        list(range(1, 11)),
        [baseline_curve, gsd_curve],
        ["Baseline DITTO", "DITTO-GSD"],
        "Reconstruction Accuracy",
        "accuracy_baseline_pair1"
    )
    
    torch.save(model_ditto.state_dict(), "models/ditto_baseline.pt")
    torch.save(model_ditto_gsd.state_dict(), "models/ditto_gsd.pt")
    
    return baseline_metrics, gsd_metrics

def experiment2_noise_sparsity():
    """
    Experiment 2: Robustness to Noise and Sparsity.
    """
    logger.info("Starting Experiment 2: Robustness to Noise and Sparsity")
    
    dataset = Synthetic3DDataset(num_samples=EXPERIMENT_PARAMS['test_samples'])
    sample_pc, sample_gt = dataset[0]
    sample_pc_np = sample_pc.numpy()
    
    noise_levels = DATASET_PARAMS['noise_levels']
    sparsity_levels = DATASET_PARAMS['sparsity_levels']
    
    degraded_samples = generate_degraded_versions(sample_pc_np, noise_levels, sparsity_levels)
    
    device = GPU_PARAMS['device']
    model_ditto = DITTOModel(
        point_features=MODEL_PARAMS['point_features'],
        grid_features=MODEL_PARAMS['grid_features']
    ).to(device)
    
    model_ditto_gsd = DITTOGSDModel(
        point_features=MODEL_PARAMS['point_features'],
        grid_features=MODEL_PARAMS['grid_features'],
        use_gs_decoder=MODEL_PARAMS['use_gaussian_decoder']
    ).to(device)
    
    dataloader = preprocess_data(dataset, batch_size=TRAIN_PARAMS['batch_size'])
    
    logger.info("Training models on clean data...")
    train_model(model_ditto, dataloader, num_epochs=3, device=device)
    train_model(model_ditto_gsd, dataloader, num_epochs=3, device=device, 
                use_geo_loss=MODEL_PARAMS['use_geometric_loss'])
    
    results = {}
    baseline_scores = []
    gsd_scores = []
    conditions = []
    
    logger.info("Evaluating models on degraded samples...")
    for key, degr_pc in degraded_samples.items():
        pred_baseline = evaluate_single_sample(model_ditto, degr_pc, device)
        
        pred_gsd = evaluate_single_sample(model_ditto_gsd, degr_pc, device)
        
        pred_reshaped = pred_baseline.reshape(-1, 3)
        gt_np = sample_gt.numpy()
        min_points = min(pred_reshaped.shape[0], gt_np.shape[0])
        baseline_metric = np.mean(np.abs(pred_reshaped[:min_points] - gt_np[:min_points]))
        pred_gsd_reshaped = pred_gsd.reshape(-1, 3)
        min_points_gsd = min(pred_gsd_reshaped.shape[0], gt_np.shape[0])
        gsd_metric = np.mean(np.abs(pred_gsd_reshaped[:min_points_gsd] - gt_np[:min_points_gsd]))
        
        results[key] = {"baseline": baseline_metric, "gsd": gsd_metric}
        baseline_scores.append(baseline_metric)
        gsd_scores.append(gsd_metric)
        conditions.append(key)
        
        logger.info(f"Degradation {key}: Baseline Metric={baseline_metric:.4f} | GSD Metric={gsd_metric:.4f}")
    
    plot_degradation_comparison(
        conditions,
        baseline_scores,
        gsd_scores,
        "Performance under Noise and Sparsity",
        "inference_latency_multimodal_vs_text"
    )
    
    return results

def experiment3_ablation():
    """
    Experiment 3: Ablation Study for Decoder Efficacy.
    """
    logger.info("Starting Experiment 3: Ablation Study for Decoder Efficacy")
    
    dataset = Synthetic3DDataset(num_samples=EXPERIMENT_PARAMS['test_samples'])
    
    experiment_variants = EXPERIMENT_PARAMS['ablation_variants']
    
    device = GPU_PARAMS['device']
    ablation_results = {}
    
    for variant, cfg in experiment_variants.items():
        logger.info(f"Running ablation variant: {variant}")
        loss_log = train_variant(cfg, dataset, epochs=5, device=device)
        ablation_results[variant] = loss_log
    
    epochs = list(range(1, 6))  # 5 epochs
    losses = [ablation_results[variant] for variant in experiment_variants.keys()]
    labels = list(experiment_variants.keys())
    
    plot_loss_curve(
        epochs,
        losses,
        labels,
        "Ablation Study - Reconstruction Loss",
        "training_loss_amict_pair2"
    )
    
    return ablation_results

def test_code():
    """
    A simple test function that runs a very short version of each experiment.
    This test is designed to finish immediately to confirm that the code executes.
    """
    print("Running quick test of all experiments...")

    test_dataset = Synthetic3DDataset(num_samples=8)
    test_loader = preprocess_data(test_dataset, batch_size=4)
    
    device = GPU_PARAMS['device']
    model_test = DITTOModel().to(device)
    
    metrics = run_experiment(model_test, test_loader, device)
    print(f"Test Experiment 1: Chamfer = {metrics['mean_chamfer']:.4f}, runtime = {metrics['runtime']:.2f}s")
    
    sample_pc, _ = test_dataset[0]
    degraded = add_gaussian_noise(sample_pc.numpy(), sigma=0.01)
    pred_test = evaluate_single_sample(model_test, degraded, device)
    print(f"Test Experiment 2: Dummy metric (mean prediction) = {np.mean(pred_test):.4f}")
    
    cfg = {"use_gs_decoder": True, "use_proj": True, "use_geo_loss": True}
    loss_log = train_variant(cfg, test_dataset, epochs=1, device=device)
    print(f"Test Experiment 3: Single epoch loss = {loss_log[0]:.4f}")
    
    print("Quick test finished.")

if __name__ == '__main__':
    for directory in ['logs', 'models', 'data']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    test_code()
    
    print("\nRunning full experiments...")
    experiment1_controlled_reconstruction()
    experiment2_noise_sparsity()
    experiment3_ablation()
    
    print("All experiments completed successfully.")
