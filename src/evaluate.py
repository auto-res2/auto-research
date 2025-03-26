"""Evaluation code for RG-MDS experiments."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from src.utils.metrics import compute_mIoU, compute_boundary_f1
from src.utils.segmentation import base_segmentation, rg_mds_segmentation, limited_prompt_segmentation
from src.utils.reference import retrieve_reference

def experiment1(dataset, feature_extractor, transform, device, max_samples=20, save_plot=True, output_dir='./logs'):
    """
    Experiment 1: Comparative Evaluation on Standard Benchmarks.
    
    Args:
        dataset: Dataset for evaluation
        feature_extractor: Model for feature extraction
        transform: Transforms to apply
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
        save_plot: Whether to save plots
        output_dir: Directory to save outputs
        
    Returns:
        dict: Results of the experiment
    """
    print("\nExperiment 1: Comparative Evaluation on Standard Benchmarks")
    
    from src.train import create_reference_gallery
    reference_gallery = create_reference_gallery(dataset, feature_extractor, transform, device)
    
    results = {'base_mIoU': [], 'rg_mds_mIoU': [], 'base_boundary': [], 'rg_mds_boundary': []}
    
    print(f"Evaluating {min(max_samples, len(dataset))} samples...")
    
    for idx in tqdm(range(min(max_samples, len(dataset))), desc="Experiment 1"):
        img_tensor, gt_tensor = dataset[idx]
        from torchvision.transforms.functional import to_pil_image
        img = to_pil_image(img_tensor)
        gt_mask = gt_tensor.squeeze().numpy()
        
        reference = retrieve_reference(img, reference_gallery, feature_extractor, transform)
        
        mask_base = base_segmentation(img)
        mask_rgmds = rg_mds_segmentation(img, reference, weighting_mode='adaptive')
        
        results['base_mIoU'].append(compute_mIoU(mask_base, gt_mask))
        results['rg_mds_mIoU'].append(compute_mIoU(mask_rgmds, gt_mask))
        results['base_boundary'].append(compute_boundary_f1(mask_base, gt_mask))
        results['rg_mds_boundary'].append(compute_boundary_f1(mask_rgmds, gt_mask))
    
    avg_base_mIoU = np.mean(results['base_mIoU'])
    avg_rgmds_mIoU = np.mean(results['rg_mds_mIoU'])
    avg_base_boundary = np.mean(results['base_boundary'])
    avg_rgmds_boundary = np.mean(results['rg_mds_boundary'])
    
    print("Results:")
    print(f"Base Method mIoU: {avg_base_mIoU:.3f}")
    print(f"RG-MDS mIoU: {avg_rgmds_mIoU:.3f}")
    print(f"Base Method Boundary F1: {avg_base_boundary:.3f}")
    print(f"RG-MDS Boundary F1: {avg_rgmds_boundary:.3f}")
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "experiment1_results.pdf")
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        methods = ['Base', 'RG-MDS']
        miou_scores = [avg_base_mIoU, avg_rgmds_mIoU]
        boundary_scores = [avg_base_boundary, avg_rgmds_boundary]
        
        ax[0].bar(methods, miou_scores, color=['skyblue', 'salmon'])
        ax[0].set_title('Mean IoU Comparison')
        ax[0].set_ylabel('mIoU')
        
        ax[1].bar(methods, boundary_scores, color=['skyblue', 'salmon'])
        ax[1].set_title('Boundary F1 Comparison')
        ax[1].set_ylabel('Boundary F1 Score')
        
        plt.tight_layout()
        plt.savefig(plot_path, format="pdf")
        print(f"Experiment 1 plots saved as '{plot_path}'.")
        plt.close()
        
    return {
        'base_mIoU': avg_base_mIoU,
        'rg_mds_mIoU': avg_rgmds_mIoU,
        'base_boundary': avg_base_boundary,
        'rg_mds_boundary': avg_rgmds_boundary
    }

def experiment2(dataset, feature_extractor, transform, device, max_samples=20, save_plot=True, output_dir='./logs'):
    """
    Experiment 2: Adaptive Weighting Ablation Study.
    
    Args:
        dataset: Dataset for evaluation
        feature_extractor: Model for feature extraction
        transform: Transforms to apply
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
        save_plot: Whether to save plots
        output_dir: Directory to save outputs
        
    Returns:
        dict: Results of the experiment
    """
    print("\nExperiment 2: Adaptive Weighting Ablation Study")
    
    from src.train import create_reference_gallery
    reference_gallery = create_reference_gallery(dataset, feature_extractor, transform, device)
    
    results_ablation = {'adaptive_mIoU': [], 'fixed_mIoU': []}
    
    print(f"Ablation evaluation on {min(max_samples, len(dataset))} samples...")
    for idx in tqdm(range(min(max_samples, len(dataset))), desc="Experiment 2"):
        img_tensor, gt_tensor = dataset[idx]
        from torchvision.transforms.functional import to_pil_image
        img = to_pil_image(img_tensor)
        gt_mask = gt_tensor.squeeze().numpy()
        
        reference = retrieve_reference(img, reference_gallery, feature_extractor, transform)
        
        mask_adaptive = rg_mds_segmentation(img, reference, weighting_mode='adaptive')
        mask_fixed = rg_mds_segmentation(img, reference, weighting_mode='fixed')
        
        results_ablation['adaptive_mIoU'].append(compute_mIoU(mask_adaptive, gt_mask))
        results_ablation['fixed_mIoU'].append(compute_mIoU(mask_fixed, gt_mask))
    
    avg_adaptive_mIoU = np.mean(results_ablation['adaptive_mIoU'])
    avg_fixed_mIoU = np.mean(results_ablation['fixed_mIoU'])
    
    print("Adaptive Weighting mIoU: {:.3f}".format(avg_adaptive_mIoU))
    print("Fixed Weighting mIoU: {:.3f}".format(avg_fixed_mIoU))
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "experiment2_ablation.pdf")
        
        fig, ax = plt.subplots(figsize=(5,4))
        methods = ['Adaptive', 'Fixed']
        miou_scores = [avg_adaptive_mIoU, avg_fixed_mIoU]
        ax.bar(methods, miou_scores, color=['seagreen', 'orchid'])
        ax.set_title('Adaptive vs Fixed Weighting (mIoU)')
        ax.set_ylabel('mIoU')
        plt.tight_layout()
        plt.savefig(plot_path, format="pdf")
        print(f"Experiment 2 plot saved as '{plot_path}'.")
        plt.close()
        
    return {
        'adaptive_mIoU': avg_adaptive_mIoU,
        'fixed_mIoU': avg_fixed_mIoU
    }

def experiment3(dataset, feature_extractor, transform, device, max_samples=5, save_plot=True, output_dir='./logs'):
    """
    Experiment 3: Robustness to Limited Token Expressiveness.
    
    Args:
        dataset: Dataset for evaluation
        feature_extractor: Model for feature extraction
        transform: Transforms to apply
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
        save_plot: Whether to save plots
        output_dir: Directory to save outputs
        
    Returns:
        dict: Results of the experiment
    """
    print("\nExperiment 3: Robustness to Limited Token Expressiveness")
    
    from src.train import create_reference_gallery
    reference_gallery = create_reference_gallery(dataset, feature_extractor, transform, device)
    
    miou_base_list = []
    miou_rgmds_list = []
    
    for idx in range(min(max_samples, len(dataset))):
        img_tensor, gt_tensor = dataset[idx]
        from torchvision.transforms.functional import to_pil_image
        img = to_pil_image(img_tensor)
        gt_mask = gt_tensor.squeeze().numpy()
        
        noisy_mask_base = limited_prompt_segmentation(img, prompt="incomplete description")
        reference = retrieve_reference(img, reference_gallery, feature_extractor, transform)
        mask_rgmds = rg_mds_segmentation(img, reference, weighting_mode='adaptive')
        
        miou_base = compute_mIoU(noisy_mask_base, gt_mask)
        miou_rgmds = compute_mIoU(mask_rgmds, gt_mask)
        miou_base_list.append(miou_base)
        miou_rgmds_list.append(miou_rgmds)
        
        print(f"Image {idx}: Base (noisy) mIoU = {miou_base:.3f}, RG-MDS mIoU = {miou_rgmds:.3f}")
        
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"experiment3_sample_{idx}.pdf")
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].set_title("Ground Truth")
            axs[0].imshow(gt_mask, cmap='gray')
            axs[0].axis('off')
            axs[1].set_title("Base (Limited Token)")
            axs[1].imshow(noisy_mask_base, cmap='gray')
            axs[1].axis('off')
            axs[2].set_title("RG-MDS")
            axs[2].imshow(mask_rgmds, cmap='gray')
            axs[2].axis('off')
            plt.suptitle(f"Sample {idx} Segmentation")
            plt.tight_layout()
            plt.savefig(plot_path, format="pdf")
            print(f"Saved qualitative plot as '{plot_path}'.")
            plt.close()
    
    avg_miou_base = np.mean(miou_base_list)
    avg_miou_rgmds = np.mean(miou_rgmds_list)
    print(f"\nAverage mIoU for Base (Limited Token): {avg_miou_base:.3f}")
    print(f"Average mIoU for RG-MDS: {avg_miou_rgmds:.3f}")
    
    if save_plot:
        plot_path = os.path.join(output_dir, "experiment3_aggregated.pdf")
        
        fig, ax = plt.subplots(figsize=(5,4))
        methods = ['Base (Noisy)', 'RG-MDS']
        avg_scores = [avg_miou_base, avg_miou_rgmds]
        ax.bar(methods, avg_scores, color=['lightcoral', 'mediumseagreen'])
        ax.set_title('Aggregated mIoU Comparison')
        ax.set_ylabel('mIoU')
        plt.tight_layout()
        plt.savefig(plot_path, format="pdf")
        print(f"Aggregated results plot saved as '{plot_path}'.")
        plt.close()
        
    return {
        'base_mIoU': avg_miou_base,
        'rg_mds_mIoU': avg_miou_rgmds
    }

def run_tests(dataset, feature_extractor, transform, device, output_dir='./logs'):
    """
    Run quick tests for all experiments to verify that code works.
    
    Args:
        dataset: Dataset for testing
        feature_extractor: Model for feature extraction
        transform: Transforms to apply
        device: Device to run on
        output_dir: Directory to save outputs
        
    Returns:
        bool: True if all tests pass
    """
    print("\nRunning quick tests for all experiments...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        experiment1(dataset, feature_extractor, transform, device, max_samples=3, save_plot=False, output_dir=output_dir)
        experiment2(dataset, feature_extractor, transform, device, max_samples=3, save_plot=False, output_dir=output_dir)
        experiment3(dataset, feature_extractor, transform, device, max_samples=2, save_plot=False, output_dir=output_dir)
        print("All tests passed successfully!")
        return True
    except Exception as e:
        print(f"A test failed: {e}")
        return False
