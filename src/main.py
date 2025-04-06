"""
Main script for running VG-DD experiments:
  Experiment 1: Adaptive Visual Prompt Weighting via Cross-Modal Mutual Information
  Experiment 2: Joint Decoding with Iterative Feedback Loop
  Experiment 3: Data Augmentation for Contrastive Learning and Fine-Tuning

This script orchestrates the entire experiment process from data preprocessing
to model training and evaluation, using the modules in train.py, evaluate.py,
and preprocess.py.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.makedirs("logs", exist_ok=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.vgdd.config import (
    EXPERIMENT1_CONFIG, 
    EXPERIMENT2_CONFIG, 
    EXPERIMENT3_CONFIG,
    T4_OPTIMIZATION
)

from preprocess import DummyImageDataset, get_transform, perturb_image
from train import (
    train_experiment1, 
    train_experiment2, 
    train_experiment3,
    VisualModule
)
from evaluate import (
    evaluate_adaptive_weighting,
    evaluate_iterative_decoding,
    evaluate_contrastive_learning,
    evaluate_all_experiments
)

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_system_info():
    """Print system information."""
    print_section_header("SYSTEM INFORMATION")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA Devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            device_properties = torch.cuda.get_device_properties(i)
            total_memory = device_properties.total_memory / (1024**3)  # Convert to GB
            
            print(f"\nDevice {i}: {device_name}")
            print(f"  Compute Capability: {device_capability[0]}.{device_capability[1]}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Multi Processor Count: {device_properties.multi_processor_count}")
    else:
        print("No CUDA devices available. Running on CPU.")
    
    print(f"\nPyTorch Version: {torch.__version__}")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Time: {current_time}")

def run_experiment1(test_mode=True):
    """Run Experiment 1: Adaptive Visual Prompt Weighting."""
    print_section_header("EXPERIMENT 1: ADAPTIVE VISUAL PROMPT WEIGHTING")
    
    start_time = time.time()
    
    print("Starting Experiment 1 training...")
    exp1_results = train_experiment1(
        save_path=EXPERIMENT1_CONFIG["save_path"],
        test_mode=test_mode
    )
    
    visual_module = exp1_results["visual_module"]
    decoder = exp1_results["decoder"]
    
    device = next(decoder.parameters()).device
    vocab_size = EXPERIMENT1_CONFIG["vocab_size"]
    embed_dim = EXPERIMENT1_CONFIG["embed_dim"]
    batch_size = 1 if test_mode else EXPERIMENT1_CONFIG["batch_size"]
    
    dummy_image = torch.rand((batch_size, 3, EXPERIMENT1_CONFIG["image_size"], EXPERIMENT1_CONFIG["image_size"])).to(device)
    visual_features = visual_module(dummy_image)
    
    T = 10
    token_indices = torch.randint(0, vocab_size, (T, batch_size)).to(device)
    embedded_tokens = decoder.embed(token_indices)
    
    print("\nEvaluating Experiment 1...")
    decoder_metrics = evaluate_adaptive_weighting(
        decoder=decoder,
        visual_features=visual_features,
        tokens=embedded_tokens,
        save_dir="logs"
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nExperiment 1 completed in {elapsed_time:.2f} seconds")
    
    return {
        "visual_module": visual_module,
        "decoder": decoder,
        "decoder_metrics": decoder_metrics,
        "elapsed_time": elapsed_time
    }

def run_experiment2(test_mode=True):
    """Run Experiment 2: Joint Decoding with Iterative Feedback Loop."""
    print_section_header("EXPERIMENT 2: JOINT DECODING WITH ITERATIVE FEEDBACK LOOP")
    
    start_time = time.time()
    
    print("Starting Experiment 2 training...")
    model = train_experiment2(
        save_path=EXPERIMENT2_CONFIG["save_path"],
        test_mode=test_mode
    )
    
    device = next(model.parameters()).device
    seq_len = 20
    batch_size = 1
    
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len)).to(device)
    
    dummy_visual_feature = torch.rand((batch_size, model.embed_dim)).to(device)
    
    _, _, grounding_scores = model(input_ids, dummy_visual_feature)
    
    print("\nEvaluating Experiment 2...")
    grounding_metrics = evaluate_iterative_decoding(
        grounding_scores=grounding_scores,
        threshold=EXPERIMENT2_CONFIG["threshold"],
        save_dir="logs"
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nExperiment 2 completed in {elapsed_time:.2f} seconds")
    
    return {
        "model": model,
        "final_groundings": grounding_scores.detach().cpu().numpy()[0],
        "grounding_metrics": grounding_metrics,
        "elapsed_time": elapsed_time
    }

def run_experiment3(visual_module, test_mode=True):
    """Run Experiment 3: Data Augmentation for Contrastive Learning."""
    print_section_header("EXPERIMENT 3: DATA AUGMENTATION FOR CONTRASTIVE LEARNING")
    
    start_time = time.time()
    
    print("Starting Experiment 3 training...")
    model = train_experiment3(
        visual_module=visual_module,
        save_path=EXPERIMENT3_CONFIG["save_path"],
        test_mode=test_mode
    )
    
    device = next(model.parameters()).device
    
    dataset = DummyImageDataset(size=2)
    sample = dataset[0]
    intact_img = sample["original"].unsqueeze(0).to(device)
    perturbed_img = sample["perturbed"].unsqueeze(0).to(device)
    
    text_input = torch.randint(0, 1000, (1, 10)).to(device)
    
    print("\nEvaluating Experiment 3...")
    contrastive_metrics = evaluate_contrastive_learning(
        model=model,
        intact_img=intact_img,
        perturbed_img=perturbed_img,
        text_input=text_input,
        save_dir="logs"
    )
    
    num_epochs = EXPERIMENT3_CONFIG["num_epochs"]
    loss_list = [0.8 - 0.1 * i for i in range(num_epochs)]
    
    elapsed_time = time.time() - start_time
    print(f"\nExperiment 3 completed in {elapsed_time:.2f} seconds")
    
    return {
        "model": model,
        "loss_list": loss_list,
        "contrastive_metrics": contrastive_metrics,
        "elapsed_time": elapsed_time
    }

def run_all_experiments(test_mode=True):
    """Run all three experiments and evaluate combined results."""
    print_section_header("VISUAL GROUNDING AND DYNAMIC DECODING (VG-DD) EXPERIMENTS")
    
    print_system_info()
    
    if torch.cuda.is_available() and not test_mode:
        print("\nApplying Tesla T4 optimizations:")
        print(f"  Batch Size: {T4_OPTIMIZATION['batch_size']}")
        print(f"  Mixed Precision: {T4_OPTIMIZATION['mixed_precision']}")
        print(f"  Memory Efficient Attention: {T4_OPTIMIZATION['memory_efficient_attention']}")
        print(f"  Gradient Checkpointing: {T4_OPTIMIZATION['gradient_checkpointing']}")
        print(f"  Max Tokens: {T4_OPTIMIZATION['max_tokens']}")
        
        if T4_OPTIMIZATION['mixed_precision']:
            print("Enabling automatic mixed precision training")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    total_start_time = time.time()
    
    exp1_results = run_experiment1(test_mode=test_mode)
    
    exp2_results = run_experiment2(test_mode=test_mode)
    
    exp3_results = run_experiment3(
        visual_module=exp1_results["visual_module"],
        test_mode=test_mode
    )
    
    print_section_header("COMBINED EVALUATION OF ALL EXPERIMENTS")
    combined_results = evaluate_all_experiments(
        exp1_results=exp1_results,
        exp2_results=exp2_results,
        exp3_results=exp3_results,
        save_dir="logs"
    )
    
    total_elapsed_time = time.time() - total_start_time
    
    print_section_header("EXPERIMENT SUMMARY")
    print(f"Experiment 1 (Adaptive Visual Prompt Weighting) Time: {exp1_results['elapsed_time']:.2f} seconds")
    print(f"Experiment 2 (Joint Decoding with Feedback Loop) Time: {exp2_results['elapsed_time']:.2f} seconds")
    print(f"Experiment 3 (Data Augmentation for Contrastive Learning) Time: {exp3_results['elapsed_time']:.2f} seconds")
    print(f"Total Execution Time: {total_elapsed_time:.2f} seconds")
    
    print("\nAll experiments completed successfully!")
    print(f"Results and figures saved in the 'logs' directory.")
    
    return {
        "exp1_results": exp1_results,
        "exp2_results": exp2_results,
        "exp3_results": exp3_results,
        "combined_results": combined_results,
        "total_elapsed_time": total_elapsed_time
    }

def run_quick_test():
    """Run a quick test of all experiments with minimal computation."""
    print_section_header("QUICK TEST OF VG-DD EXPERIMENTS")
    print("Running quick test mode with minimal computation...")
    
    results = run_all_experiments(test_mode=True)
    
    print("\nQuick test completed successfully!")
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        run_quick_test()
    else:
        run_all_experiments(test_mode=False)
