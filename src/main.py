"""
Main script for the Progressive Brightness Distillation Diffusion experiment.
"""

import os
import random
import numpy as np
import torch

from config.pbd_diffusion_config import (
    SEED,
    DEVICE,
    OUTPUT_DIR,
    LOGS_DIR,
    MODELS_DIR
)
from src.preprocess import prepare_data, set_seed
from src.train import train_teacher_student_model
from src.evaluate import evaluate_dual_stage_correction, perform_ablation_study

def experiment1_dual_stage_evaluation(train_loader, test_loader, train_gt_loader, test_gt_loader):
    """Run Experiment 1: Dual-Stage Brightness Correction Evaluation."""
    print("\n=== Experiment 1: Dual-Stage Brightness Correction Evaluation ===")
    
    results = evaluate_dual_stage_correction(train_loader, train_gt_loader)
    
    print("Experiment 1 completed.\n")
    return results

def experiment2_teacher_student_distillation(train_loader, test_loader):
    """Run Experiment 2: Teacher-Student Progressive Distillation Study."""
    print("\n=== Experiment 2: Teacher-Student Progressive Distillation Study ===")
    
    teacher_network, student_network, loss_history = train_teacher_student_model(train_loader, test_loader)
    
    print("Experiment 2 completed.\n")
    return teacher_network, student_network, loss_history

def experiment3_ablation_study(train_loader, test_loader, train_gt_loader, test_gt_loader):
    """Run Experiment 3: Ablation Study on Progressive Refinement Components."""
    print("\n=== Experiment 3: Ablation Study on Progressive Refinement Components ===")
    
    results = perform_ablation_study(train_loader, train_gt_loader)
    
    print("Experiment 3 completed.\n")
    return results

def run_experiments():
    """Run all experiments."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    set_seed(SEED)
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print("\nPreparing data...")
    train_loader, test_loader, train_gt_loader, test_gt_loader = prepare_data()
    print("Data preparation completed.")
    
    experiment1_dual_stage_evaluation(train_loader, test_loader, train_gt_loader, test_gt_loader)
    experiment2_teacher_student_distillation(train_loader, test_loader)
    experiment3_ablation_study(train_loader, test_loader, train_gt_loader, test_gt_loader)
    
    print("\nAll experiments completed successfully.")

def test_code():
    """
    Run a quick test of the experiments.
    This function runs with minimal computation to verify that the code works.
    """
    print("Running quick test of experiments...\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    set_seed(SEED)
    
    train_loader, test_loader, train_gt_loader, test_gt_loader = prepare_data()
    
    evaluate_dual_stage_correction(train_loader, train_gt_loader)
    
    train_teacher_student_model(train_loader, test_loader, num_epochs=1)
    
    perform_ablation_study(train_loader, train_gt_loader)
    
    print("Quick test completed. All experiments ran without error.")

if __name__ == '__main__':
    test_code()
    
    # run_experiments()
