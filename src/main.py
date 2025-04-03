"""Main script for running GraphDiffLayout experiments."""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import torch

from src.preprocess import generate_random_layout, create_test_layouts, generate_ground_truth
from src.train import dummy_generate_image_noise_collage, dummy_generate_image_graph_diff_layout
from src.evaluate import (
    evaluate_layout_text_alignment, evaluate_small_object_fidelity, 
    evaluate_scalability, plot_small_object_ssim, compute_region_ssim
)
from src.utils.visualization import visualize_comparison
from config.experiment_config import EXPERIMENT_PARAMS

def setup_environment():
    """Set up the environment for experiments."""
    os.makedirs('logs', exist_ok=True)
    
    random_seed = EXPERIMENT_PARAMS.get('random_seed', 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    
    print("===== Environment Information =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    print("==================================\n")

def experiment1():
    """
    Experiment 1: Layout and Text Alignment Comparison.
    Compare the layout and text alignment capabilities of NoiseCollage and GraphDiffLayout.
    """
    print("=== Experiment 1: Layout and Text Alignment Comparison ===")
    
    layouts = create_test_layouts()
    
    img_nc_dict = {}
    img_gdl_dict = {}
    
    for name, layout in layouts.items():
        img_nc = dummy_generate_image_noise_collage(layout)
        img_gdl = dummy_generate_image_graph_diff_layout(layout)
        
        img_nc_dict[name] = img_nc
        img_gdl_dict[name] = img_gdl
        
        visualize_comparison(
            img_nc, img_gdl,
            f"NoiseCollage - {name}", f"GraphDiffLayout - {name}",
            f"Layout and Text Alignment Comparison ({name})",
            f"layout_text_alignment_{name}"
        )
    
    all_iou_results = evaluate_layout_text_alignment(layouts, img_nc_dict, img_gdl_dict)
    
    return all_iou_results

def experiment2():
    """
    Experiment 2: Fidelity for Small and Hard-to-Generate Objects.
    Compare the fidelity of small and hard-to-generate objects between NoiseCollage and GraphDiffLayout.
    """
    print("\n=== Experiment 2: Fidelity for Small and Hard-to-Generate Objects ===")
    
    layout = [
        {'bbox': [30, 30, 20, 20], 'label': 'small1'},
        {'bbox': [80, 40, 20, 20], 'label': 'small2'},
        {'bbox': [130, 50, 20, 20], 'label': 'small3'}
    ]
    
    ground_truth_img = generate_ground_truth(layout)
    generated_img_nc = dummy_generate_image_noise_collage(layout)
    generated_img_gdl = dummy_generate_image_graph_diff_layout(layout)
    
    ssim_nc, ssim_gdl = evaluate_small_object_fidelity(
        layout, ground_truth_img, generated_img_nc, generated_img_gdl
    )
    
    plot_small_object_ssim(layout, 
                          [compute_region_ssim(generated_img_nc, ground_truth_img, obj['bbox']) for obj in layout],
                          [compute_region_ssim(generated_img_gdl, ground_truth_img, obj['bbox']) for obj in layout])
    
    return ssim_nc, ssim_gdl

def experiment3():
    """
    Experiment 3: Scalability and Runtime Efficiency.
    Compare the scalability and runtime efficiency of NoiseCollage and GraphDiffLayout.
    """
    print("\n=== Experiment 3: Scalability and Runtime Efficiency ===")
    
    object_counts = EXPERIMENT_PARAMS.get('object_counts', [5, 10, 20, 50])
    
    counts, runtime_nc, runtime_gdl = evaluate_scalability(
        lambda n: generate_random_layout(n), object_counts
    )
    
    return counts, runtime_nc, runtime_gdl

def run_test():
    """
    Test function that runs a quick version of all experiments.
    """
    print("===== Running Quick Test of Experiments =====")
    
    iou_results = experiment1()
    
    ssim_nc_avg, ssim_gdl_avg = experiment2()
    
    counts, runtimes_nc, runtimes_gdl = experiment3()
    
    print("\n===== Summary of Experiments =====")
    print("Experiment 1: IoU across different layouts:")
    for item in iou_results:
        layout_name, iou_nc, iou_gdl = item
        print(f"  {layout_name}: NoiseCollage IoU = {iou_nc:.3f}, GraphDiffLayout IoU = {iou_gdl:.3f}")
    
    print(f"Experiment 2: Average SSIM: NoiseCollage = {ssim_nc_avg:.3f}, GraphDiffLayout = {ssim_gdl_avg:.3f}")
    
    print("Experiment 3: Inference times (s):")
    for idx, num in enumerate(counts):
        print(f"  {num} objects: NoiseCollage = {runtimes_nc[idx]:.4f}, GraphDiffLayout = {runtimes_gdl[idx]:.4f}")
    
    print("===== Experiments Completed =====")

if __name__ == '__main__':
    setup_environment()
    
    run_test()
