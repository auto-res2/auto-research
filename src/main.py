"""Main script for running GraphDiffLayout experiments."""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import torch

from preprocess import generate_random_layout, create_test_layouts, generate_ground_truth
from train import dummy_generate_image_noise_collage, dummy_generate_image_graph_diff_layout
from evaluate import (
    evaluate_layout_text_alignment, evaluate_small_object_fidelity, 
    evaluate_scalability, plot_small_object_ssim, compute_region_ssim
)
from utils.visualization import visualize_comparison
import sys
sys.path.append('..')
from config.experiment_config import EXPERIMENT_PARAMS

def setup_environment():
    """Set up the environment for experiments."""
    os.makedirs('logs', exist_ok=True)
    
    print("\n" + "="*50)
    print("GRAPHDIFFLAYOUT EXPERIMENT SETUP".center(50))
    print("="*50)
    
    print("\n📋 EXPERIMENT CONFIGURATION:")
    for key, value in EXPERIMENT_PARAMS.items():
        print(f"  • {key}: {value}")
    
    random_seed = EXPERIMENT_PARAMS.get('random_seed', 42)
    print(f"\n🔄 Setting random seed to: {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    
    print("\n💻 ENVIRONMENT INFORMATION:")
    print(f"  • PyTorch version: {torch.__version__}")
    print(f"  • CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  • CUDA device count: {torch.cuda.device_count()}")
        print(f"  • CUDA device capability: {torch.cuda.get_device_capability(0)}")
    
    print(f"  • Working directory: {os.getcwd()}")
    print(f"  • Python version: {sys.version.split()[0]}")
    
    print("\n" + "="*50 + "\n")

def experiment1():
    """
    Experiment 1: Layout and Text Alignment Comparison.
    Compare the layout and text alignment capabilities of NoiseCollage and GraphDiffLayout.
    """
    print("\n" + "="*50)
    print("EXPERIMENT 1: LAYOUT AND TEXT ALIGNMENT".center(50))
    print("="*50)
    
    print("\n🔍 Creating test layouts...")
    layouts = create_test_layouts()
    print(f"  • Created {len(layouts)} test layouts: {', '.join(layouts.keys())}")
    
    print("\n🖼️ Generating images for each layout...")
    img_nc_dict = {}
    img_gdl_dict = {}
    
    for name, layout in layouts.items():
        print(f"\n  • Processing layout: '{name}' with {len(layout)} objects")
        print(f"    - Objects: {', '.join([obj['label'] for obj in layout])}")
        
        print(f"    - Generating NoiseCollage image...")
        start_time = time.time()
        img_nc = dummy_generate_image_noise_collage(layout)
        nc_time = time.time() - start_time
        print(f"      ✓ Generated in {nc_time:.4f} seconds")
        
        print(f"    - Generating GraphDiffLayout image...")
        start_time = time.time()
        img_gdl = dummy_generate_image_graph_diff_layout(layout)
        gdl_time = time.time() - start_time
        print(f"      ✓ Generated in {gdl_time:.4f} seconds")
        
        img_nc_dict[name] = img_nc
        img_gdl_dict[name] = img_gdl
        
        print(f"    - Saving comparison visualization...")
        filepath = visualize_comparison(
            img_nc, img_gdl,
            f"NoiseCollage - {name}", f"GraphDiffLayout - {name}",
            f"Layout and Text Alignment Comparison ({name})",
            f"layout_text_alignment_{name}"
        )
        print(f"      ✓ Saved to: {filepath}")
    
    print("\n📊 Evaluating layout and text alignment...")
    all_iou_results = evaluate_layout_text_alignment(layouts, img_nc_dict, img_gdl_dict)
    
    print("\n📈 EXPERIMENT 1 RESULTS SUMMARY:")
    for layout_name, iou_nc, iou_gdl in all_iou_results:
        improvement = ((iou_gdl - iou_nc) / iou_nc) * 100
        print(f"  • {layout_name} layout:")
        print(f"    - NoiseCollage IoU: {iou_nc:.4f}")
        print(f"    - GraphDiffLayout IoU: {iou_gdl:.4f}")
        print(f"    - Improvement: {improvement:.2f}%")
    
    print("\n" + "-"*50)
    return all_iou_results

def experiment2():
    """
    Experiment 2: Fidelity for Small and Hard-to-Generate Objects.
    Compare the fidelity of small and hard-to-generate objects between NoiseCollage and GraphDiffLayout.
    """
    print("\n" + "="*50)
    print("EXPERIMENT 2: SMALL OBJECT FIDELITY".center(50))
    print("="*50)
    
    print("\n🔍 Creating small object layout...")
    layout = [
        {'bbox': [30, 30, 20, 20], 'label': 'small1'},
        {'bbox': [80, 40, 20, 20], 'label': 'small2'},
        {'bbox': [130, 50, 20, 20], 'label': 'small3'}
    ]
    print(f"  • Created layout with {len(layout)} small objects")
    for i, obj in enumerate(layout):
        print(f"    - Object {i+1}: {obj['label']} at position ({obj['bbox'][0]}, {obj['bbox'][1]}) with size {obj['bbox'][2]}x{obj['bbox'][3]}")
    
    print("\n🎯 Generating ground truth image...")
    start_time = time.time()
    ground_truth_img = generate_ground_truth(layout)
    print(f"  • Ground truth generated in {time.time() - start_time:.4f} seconds")
    
    print("\n🖼️ Generating model outputs...")
    print("  • Generating NoiseCollage image...")
    start_time = time.time()
    generated_img_nc = dummy_generate_image_noise_collage(layout)
    nc_time = time.time() - start_time
    print(f"    ✓ Generated in {nc_time:.4f} seconds")
    
    print("  • Generating GraphDiffLayout image...")
    start_time = time.time()
    generated_img_gdl = dummy_generate_image_graph_diff_layout(layout)
    gdl_time = time.time() - start_time
    print(f"    ✓ Generated in {gdl_time:.4f} seconds")
    
    print("\n📊 Evaluating small object fidelity...")
    ssim_nc, ssim_gdl = evaluate_small_object_fidelity(
        layout, ground_truth_img, generated_img_nc, generated_img_gdl
    )
    
    print("\n📈 Computing SSIM scores for individual objects...")
    ssim_scores_nc = [compute_region_ssim(generated_img_nc, ground_truth_img, obj['bbox']) for obj in layout]
    ssim_scores_gdl = [compute_region_ssim(generated_img_gdl, ground_truth_img, obj['bbox']) for obj in layout]
    
    for i, (obj, nc_score, gdl_score) in enumerate(zip(layout, ssim_scores_nc, ssim_scores_gdl)):
        improvement = ((gdl_score - nc_score) / nc_score) * 100 if nc_score > 0 else float('inf')
        print(f"  • Object {obj['label']}:")
        print(f"    - NoiseCollage SSIM: {nc_score:.4f}")
        print(f"    - GraphDiffLayout SSIM: {gdl_score:.4f}")
        print(f"    - Improvement: {improvement:.2f}%")
    
    print("\n📊 Plotting SSIM comparison...")
    filepath = plot_small_object_ssim(layout, ssim_scores_nc, ssim_scores_gdl)
    print(f"  • Plot saved to: {filepath}")
    
    print("\n📈 EXPERIMENT 2 RESULTS SUMMARY:")
    improvement = ((ssim_gdl - ssim_nc) / ssim_nc) * 100 if ssim_nc > 0 else float('inf')
    print(f"  • Average SSIM for NoiseCollage: {ssim_nc:.4f}")
    print(f"  • Average SSIM for GraphDiffLayout: {ssim_gdl:.4f}")
    print(f"  • Overall improvement: {improvement:.2f}%")
    
    print("\n" + "-"*50)
    return ssim_nc, ssim_gdl

def experiment3():
    """
    Experiment 3: Scalability and Runtime Efficiency.
    Compare the scalability and runtime efficiency of NoiseCollage and GraphDiffLayout.
    """
    print("\n" + "="*50)
    print("EXPERIMENT 3: SCALABILITY AND RUNTIME".center(50))
    print("="*50)
    
    print("\n🔢 Setting up scalability test...")
    object_counts = EXPERIMENT_PARAMS.get('object_counts', [5, 10, 20, 50])
    print(f"  • Testing with object counts: {object_counts}")
    
    print("\n⏱️ Measuring inference times...")
    from evaluate import evaluate_scalability
    
    start_time = time.time()
    counts, runtime_nc, runtime_gdl = evaluate_scalability(
        lambda n: generate_random_layout(n), object_counts
    )
    total_time = time.time() - start_time
    print(f"  • Completed all measurements in {total_time:.2f} seconds")
    
    print("\n📈 EXPERIMENT 3 RESULTS SUMMARY:")
    print("  • Runtime comparison (seconds):")
    print("    ---------------------------------------------")
    print("    | Objects | NoiseCollage | GraphDiffLayout |")
    print("    ---------------------------------------------")
    for i, count in enumerate(counts):
        nc_time = runtime_nc[i]
        gdl_time = runtime_gdl[i]
        speedup = (nc_time / gdl_time) if gdl_time > 0 else float('inf')
        print(f"    |   {count:3d}   |    {nc_time:.4f}    |     {gdl_time:.4f}      |")
    print("    ---------------------------------------------")
    
    print("\n  • Performance analysis:")
    for i, count in enumerate(counts):
        nc_time = runtime_nc[i]
        gdl_time = runtime_gdl[i]
        if gdl_time < nc_time:
            speedup = (nc_time / gdl_time) - 1
            print(f"    - For {count} objects: GraphDiffLayout is {speedup:.2f}x faster")
        else:
            slowdown = (gdl_time / nc_time) - 1
            print(f"    - For {count} objects: GraphDiffLayout is {slowdown:.2f}x slower")
    
    print("\n" + "-"*50)
    return counts, runtime_nc, runtime_gdl

def run_test():
    """
    Test function that runs a quick version of all experiments.
    """
    print("\n" + "="*70)
    print("GRAPHDIFFLAYOUT RESEARCH EXPERIMENTS".center(70))
    print("="*70)
    
    print("\n📋 RUNNING ALL EXPERIMENTS:")
    print("  This test will run three experiments to evaluate GraphDiffLayout against NoiseCollage:")
    print("  1. Layout and Text Alignment Comparison")
    print("  2. Fidelity for Small and Hard-to-Generate Objects")
    print("  3. Scalability and Runtime Efficiency")
    
    print("\n" + "="*70)
    print("STARTING EXPERIMENT SUITE".center(70))
    print("="*70)
    
    start_time = time.time()
    
    print("\n🧪 Running Experiment 1: Layout and Text Alignment...")
    exp1_start = time.time()
    iou_results = experiment1()
    exp1_time = time.time() - exp1_start
    print(f"  ✓ Completed in {exp1_time:.2f} seconds")
    
    print("\n🧪 Running Experiment 2: Small Object Fidelity...")
    exp2_start = time.time()
    ssim_nc_avg, ssim_gdl_avg = experiment2()
    exp2_time = time.time() - exp2_start
    print(f"  ✓ Completed in {exp2_time:.2f} seconds")
    
    print("\n🧪 Running Experiment 3: Scalability and Runtime...")
    exp3_start = time.time()
    counts, runtimes_nc, runtimes_gdl = experiment3()
    exp3_time = time.time() - exp3_start
    print(f"  ✓ Completed in {exp3_time:.2f} seconds")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY".center(70))
    print("="*70)
    
    print("\n📊 EXPERIMENT 1: Layout and Text Alignment")
    print("  • Results by layout type:")
    for item in iou_results:
        layout_name, iou_nc, iou_gdl = item
        improvement = ((iou_gdl - iou_nc) / iou_nc) * 100 if iou_nc > 0 else float('inf')
        print(f"    - {layout_name}: NoiseCollage IoU = {iou_nc:.4f}, GraphDiffLayout IoU = {iou_gdl:.4f} ({improvement:.2f}% improvement)")
    
    print("\n📊 EXPERIMENT 2: Small Object Fidelity")
    improvement = ((ssim_gdl_avg - ssim_nc_avg) / ssim_nc_avg) * 100 if ssim_nc_avg > 0 else float('inf')
    print(f"  • Average SSIM: NoiseCollage = {ssim_nc_avg:.4f}, GraphDiffLayout = {ssim_gdl_avg:.4f}")
    print(f"  • Overall improvement: {improvement:.2f}%")
    
    print("\n📊 EXPERIMENT 3: Scalability and Runtime")
    print("  • Runtime comparison by object count:")
    for idx, num in enumerate(counts):
        nc_time = runtimes_nc[idx]
        gdl_time = runtimes_gdl[idx]
        if gdl_time < nc_time:
            speedup = (nc_time / gdl_time) - 1
            print(f"    - {num} objects: NoiseCollage = {nc_time:.4f}s, GraphDiffLayout = {gdl_time:.4f}s ({speedup:.2f}x faster)")
        else:
            slowdown = (gdl_time / nc_time) - 1
            print(f"    - {num} objects: NoiseCollage = {nc_time:.4f}s, GraphDiffLayout = {gdl_time:.4f}s ({slowdown:.2f}x slower)")
    
    print("\n🔬 RESEARCH CONCLUSIONS:")
    
    avg_iou_nc = np.mean([r[1] for r in iou_results])
    avg_iou_gdl = np.mean([r[2] for r in iou_results])
    iou_improvement = ((avg_iou_gdl - avg_iou_nc) / avg_iou_nc) * 100 if avg_iou_nc > 0 else float('inf')
    
    avg_runtime_nc = np.mean(runtimes_nc)
    avg_runtime_gdl = np.mean(runtimes_gdl)
    runtime_ratio = avg_runtime_gdl / avg_runtime_nc
    
    print(f"  • Layout Alignment: GraphDiffLayout shows {iou_improvement:.2f}% better IoU scores on average")
    print(f"  • Small Object Fidelity: GraphDiffLayout shows {improvement:.2f}% better SSIM scores")
    print(f"  • Runtime Efficiency: GraphDiffLayout is {runtime_ratio:.2f}x the runtime of NoiseCollage")
    
    print(f"\n⏱️ Total experiment runtime: {total_time:.2f} seconds")
    print(f"  • Experiment 1: {exp1_time:.2f}s ({exp1_time/total_time*100:.1f}%)")
    print(f"  • Experiment 2: {exp2_time:.2f}s ({exp2_time/total_time*100:.1f}%)")
    print(f"  • Experiment 3: {exp3_time:.2f}s ({exp3_time/total_time*100:.1f}%)")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY".center(70))
    print("="*70)

if __name__ == '__main__':
    setup_environment()
    
    run_test()
