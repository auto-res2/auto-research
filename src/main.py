"""
Implementation of experiments to evaluate the Spatially-Constrained Normal Diffusion (SCND) method.

Experiments include:
  1. Ablation study on the Masked-Attention Guidance Module.
  2. Evaluation of Progressive Spatial Refinement.
  3. Analysis of Integrated Score Distillation Sampling (SDS) with Spatial Mask Alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import time

from preprocess import get_dummy_data
from train import DiffusionModel, diffusion_process, sds_loss_fn
from evaluate import compute_metrics, save_comparison_plot, save_loss_curve

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.scnd_config import *

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("paper", exist_ok=True)

def experiment1_ablation_study():
    """
    Experiment 1: Ablation Study on the Masked-Attention Guidance Module.
    Compares models with and without the Masked-Attention module.
    """
    print("Experiment 1: Ablation Study on the Masked-Attention Guidance Module")
    print("----------------------------------------------------------------------")
    
    model_full = DiffusionModel(use_masked_attn=True)
    model_baseline = DiffusionModel(use_masked_attn=False)
    model_full.eval()
    model_baseline.eval()
    
    outputs_full = []
    outputs_baseline = []
    metrics_list = []
    
    for i in range(NUM_SAMPLES):
        image, semantic_mask = get_dummy_data(batch_size=1, height=IMAGE_SIZE, width=IMAGE_SIZE)
        with torch.no_grad():
            out_full = model_full(image, semantic_mask)
            out_base = model_baseline(image)
        ssim_full, psnr_full = compute_metrics(out_full, image)
        ssim_base, psnr_base = compute_metrics(out_base, image)
        metrics_list.append((ssim_full, psnr_full, ssim_base, psnr_base))
        outputs_full.append(out_full)
        outputs_baseline.append(out_base)
        print(f"Sample {i+1}:")
        print(f"  Full SCND:   SSIM = {ssim_full:.4f}  PSNR = {psnr_full:.2f}")
        print(f"  Baseline:    SSIM = {ssim_base:.4f}  PSNR = {psnr_base:.2f}")
    
    save_comparison_plot(
        [outputs_full[-1], outputs_baseline[-1]], 
        ["Full SCND Output", "Baseline Output"],
        "paper/experiment1_outputs.pdf"
    )
    
    print("Experiment 1 completed. Output visualization saved as paper/experiment1_outputs.pdf\n")

def experiment2_progressive_refinement():
    """
    Experiment 2: Evaluation of Progressive Spatial Refinement.
    Analyzes the diffusion process with progressive refinement.
    """
    print("Experiment 2: Evaluation of Progressive Spatial Refinement")
    print("--------------------------------------------------------")
    
    model_full = DiffusionModel(use_masked_attn=True)
    model_full.eval()
    
    x_init = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    _, semantic_mask = get_dummy_data(batch_size=1, height=IMAGE_SIZE, width=IMAGE_SIZE)
    
    print(f"Running diffusion process for {NUM_DIFFUSION_STEPS} steps...")
    intermediates, losses = diffusion_process(
        model_full, x_init, semantic_mask, num_steps=NUM_DIFFUSION_STEPS
    )
    
    save_loss_curve(
        losses,
        "Diffusion Step",
        "Loss",
        "Spatial Consistency Loss over Diffusion Steps",
        "paper/experiment2_loss_curve.pdf"
    )
    
    save_comparison_plot(
        [intermediates[2], intermediates[-1]], 
        [f"Early Stage Output (Step 3)", f"Late Stage Output (Step {NUM_DIFFUSION_STEPS})"],
        "paper/experiment2_stage_comparison.pdf",
        figsize=(12,6)
    )
    
    print("Experiment 2 completed.")
    print("Loss evolution curve saved as paper/experiment2_loss_curve.pdf")
    print("Stage comparison saved as paper/experiment2_stage_comparison.pdf\n")

def experiment3_sds_integration():
    """
    Experiment 3: Analysis of Integrated Score Distillation Sampling (SDS) with Spatial Mask Alignment.
    Simulates training with the SDS loss function including spatial alignment.
    """
    print("Experiment 3: Analysis of Integrated SDS with Spatial Mask Alignment")
    print("----------------------------------------------------------------")
    
    model_full = DiffusionModel(use_masked_attn=True)
    optimizer = torch.optim.Adam(model_full.parameters(), lr=LEARNING_RATE)
    
    image, semantic_mask = get_dummy_data(batch_size=1, height=IMAGE_SIZE, width=IMAGE_SIZE)
    
    loss_values = []
    print(f"Running SDS training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        
        generated_output = model_full(image, semantic_mask)
        
        attention_map = torch.sigmoid(torch.randn_like(semantic_mask))
        
        loss = sds_loss_fn(generated_output, image, attention_map, semantic_mask, alpha=SDS_ALPHA)
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_values.append(loss_val)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Total Loss: {loss_val:.4f}")
    
    save_loss_curve(
        loss_values,
        "Epoch",
        "Total Loss",
        "SDS Loss Convergence",
        "paper/experiment3_sds_loss.pdf"
    )
    
    torch.save(model_full.state_dict(), "models/scnd_model.pt")
    
    print("Experiment 3 completed. SDS training loss plot saved as paper/experiment3_sds_loss.pdf")
    print("Model saved as models/scnd_model.pt\n")

def test():
    """
    Run minimal versions of each experiment to verify that the code runs correctly.
    """
    print("\nStarting minimal test run of all experiments...\n")
    
    test_height = TEST_IMAGE_SIZE
    test_width = TEST_IMAGE_SIZE
    
    print("Running Experiment 1 (Ablation Study, single sample)...")
    model_full = DiffusionModel(use_masked_attn=True)
    model_baseline = DiffusionModel(use_masked_attn=False)
    model_full.eval()
    model_baseline.eval()
    image, semantic_mask = get_dummy_data(batch_size=1, height=test_height, width=test_width)
    with torch.no_grad():
        out_full = model_full(image, semantic_mask)
        out_baseline = model_baseline(image)
    ssim_full, psnr_full = compute_metrics(out_full, image)
    ssim_base, psnr_base = compute_metrics(out_baseline, image)
    print(f"Test Sample Metrics:")
    print(f"  Full SCND:   SSIM = {ssim_full:.4f}  PSNR = {psnr_full:.2f}")
    print(f"  Baseline:    SSIM = {ssim_base:.4f}  PSNR = {psnr_base:.2f}\n")
    
    print("Running Experiment 2 (Progressive Refinement, few steps)...")
    x_init = torch.randn(1, 3, test_height, test_width)
    _, semantic_mask_small = get_dummy_data(batch_size=1, height=test_height, width=test_width)
    intermediates, losses = diffusion_process(model_full, x_init, semantic_mask_small, num_steps=TEST_STEPS)
    print("Loss values at each step (minitest):", losses, "\n")
    
    print("Running Experiment 3 (SDS Integration, few epochs)...")
    optimizer = torch.optim.Adam(model_full.parameters(), lr=LEARNING_RATE)
    image_small, semantic_mask_small = get_dummy_data(batch_size=1, height=test_height, width=test_width)
    for epoch in range(TEST_EPOCHS):
        optimizer.zero_grad()
        generated_output = model_full(image_small, semantic_mask_small)
        attention_map = torch.sigmoid(torch.randn_like(semantic_mask_small))
        loss = sds_loss_fn(generated_output, image_small, attention_map, semantic_mask_small, alpha=SDS_ALPHA)
        loss.backward()
        optimizer.step()
        print(f"Test Epoch {epoch+1}/{TEST_EPOCHS}, Loss: {loss.item():.4f}")
    
    print("\nMinimal test run completed successfully.")

if __name__ == "__main__":
    start_time = time.time()
    print("="*80)
    print("Starting SCND Experiment Suite")
    print("="*80)
    
    test()
    
    experiment1_ablation_study()
    experiment2_progressive_refinement()
    experiment3_sds_integration()
    
    elapsed_time = time.time() - start_time
    print(f"All experiments completed in {elapsed_time:.2f} seconds")
    print("="*80)
