#!/usr/bin/env python
"""
Evaluation module for Tweedie-Guided Global Consistent Video Editing (TG-GCVE).

This module implements:
1. Model evaluation metrics (CLIP score, LPIPS, temporal consistency)
2. Visualization of editing results
3. Comparison of different model variants
4. Evaluation of adaptive fusion and iterative refinement
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
import lpips
from torchvision import transforms
import cv2

# Import from other modules
from train import TGGCVEModule, visualize_global_context, compute_clip_score
from preprocess import preprocess_for_training, add_noise

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# Evaluation Metrics
# --------------------------

def compute_lpips_score(output_frames, target_frames, lpips_model=None):
    """
    Compute LPIPS perceptual similarity score between output and target frames.
    
    Args:
        output_frames (torch.Tensor): Output frames from the model
        target_frames (torch.Tensor): Target frames (ground truth)
        lpips_model: LPIPS model (if None, will be initialized)
        
    Returns:
        float: Average LPIPS score
    """
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Ensure inputs are in the correct format
    if output_frames.dim() == 5:  # [B, T, C, H, W]
        batch_size, seq_len = output_frames.shape[:2]
        output_frames = output_frames.view(-1, *output_frames.shape[2:])
        target_frames = target_frames.view(-1, *target_frames.shape[2:])
    
    # Compute LPIPS
    with torch.no_grad():
        lpips_scores = lpips_model(output_frames, target_frames)
        avg_lpips = lpips_scores.mean().item()
    
    return avg_lpips

def compute_temporal_consistency(frames):
    """
    Compute temporal consistency score for a sequence of frames.
    Lower score indicates better temporal consistency.
    
    Args:
        frames (torch.Tensor): Sequence of frames [B, T, C, H, W]
        
    Returns:
        float: Temporal consistency score
    """
    batch_size, seq_len, channels, height, width = frames.shape
    
    # Compute frame differences
    frame_diffs = []
    for i in range(1, seq_len):
        prev_frame = frames[:, i-1]
        curr_frame = frames[:, i]
        diff = F.mse_loss(prev_frame, curr_frame, reduction='none').mean(dim=[1, 2, 3])
        frame_diffs.append(diff)
    
    # Stack frame differences
    frame_diffs = torch.stack(frame_diffs, dim=1)  # [B, T-1]
    
    # Compute variance of frame differences (lower is better)
    consistency_score = frame_diffs.var(dim=1).mean().item()
    
    return consistency_score

def compute_warping_error(frames, flow_model=None):
    """
    Compute warping error between consecutive frames using optical flow.
    Lower error indicates better temporal consistency.
    
    Args:
        frames (torch.Tensor): Sequence of frames [B, T, C, H, W]
        flow_model: Optical flow model (if None, will use OpenCV)
        
    Returns:
        float: Average warping error
    """
    batch_size, seq_len, channels, height, width = frames.shape
    
    # Convert to numpy for OpenCV processing
    frames_np = frames.detach().cpu().numpy()
    
    total_error = 0.0
    count = 0
    
    for b in range(batch_size):
        for t in range(1, seq_len):
            # Get consecutive frames
            prev_frame = (frames_np[b, t-1] * 255).transpose(1, 2, 0).astype(np.uint8)
            curr_frame = (frames_np[b, t] * 255).transpose(1, 2, 0).astype(np.uint8)
            
            # Convert to grayscale for optical flow
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Warp previous frame to current frame
            h, w = prev_gray.shape
            flow_map = np.column_stack((
                np.tile(np.arange(w), h),
                np.repeat(np.arange(h), w)
            )).reshape(h, w, 2).astype(np.float32)
            flow_map += flow
            
            # Ensure flow_map is within bounds
            flow_map[:, :, 0] = np.clip(flow_map[:, :, 0], 0, w - 1)
            flow_map[:, :, 1] = np.clip(flow_map[:, :, 1], 0, h - 1)
            
            # Warp previous frame
            warped_frame = np.zeros_like(prev_frame)
            for c in range(channels):
                warped_frame[:, :, c] = cv2.remap(
                    prev_frame[:, :, c], flow_map[:, :, 0], flow_map[:, :, 1],
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Compute warping error
            error = np.mean((warped_frame.astype(np.float32) - curr_frame.astype(np.float32)) ** 2)
            total_error += error
            count += 1
    
    # Compute average warping error
    avg_error = total_error / count if count > 0 else 0.0
    
    return avg_error

# --------------------------
# Evaluation Functions
# --------------------------

def evaluate_model(model, dataloader, clip_model=None, lpips_model=None, target_text="edited video"):
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader for evaluation data
        clip_model: CLIP model (if None, will be initialized)
        lpips_model: LPIPS model (if None, will be initialized)
        target_text (str): Target text description for CLIP score
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Initialize models if not provided
    if clip_model is None:
        clip_model, _ = clip.load("ViT-B/32", device=device)
    
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Initialize metrics
    total_clip_score = 0.0
    total_lpips_score = 0.0
    total_temporal_consistency = 0.0
    total_warping_error = 0.0
    
    # Evaluate on dataset
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Get frames from batch
            frames = batch['frames'].to(device)
            batch_size, seq_len, channels, height, width = frames.shape
            
            # Reshape to [B*T, C, H, W] for processing individual frames
            flat_frames = frames.view(-1, channels, height, width)
            
            # Add noise to frames
            noise_level = torch.tensor(0.1).to(device)
            noisy_frames = add_noise(flat_frames, noise_level)
            
            # Forward pass
            outputs, _, _, _, _ = model(noisy_frames, noise_level)
            
            # Reshape outputs back to [B, T, C, H, W]
            outputs = outputs.view(batch_size, seq_len, channels, height, width)
            
            # Compute CLIP score for first frame of each sequence
            for b in range(batch_size):
                clip_score = compute_clip_score(
                    outputs[b, 0].unsqueeze(0),
                    text_description=target_text,
                    clip_model=clip_model
                )
                total_clip_score += clip_score
            
            # Compute LPIPS score
            lpips_score = compute_lpips_score(
                outputs.view(-1, channels, height, width),
                frames.view(-1, channels, height, width),
                lpips_model=lpips_model
            )
            total_lpips_score += lpips_score * batch_size
            
            # Compute temporal consistency
            temporal_consistency = compute_temporal_consistency(outputs)
            total_temporal_consistency += temporal_consistency * batch_size
            
            # Compute warping error
            warping_error = compute_warping_error(outputs)
            total_warping_error += warping_error * batch_size
            
            # Only evaluate on a few batches for efficiency
            if batch_idx >= 5:
                break
    
    # Compute average metrics
    num_samples = (batch_idx + 1) * batch_size
    avg_clip_score = total_clip_score / num_samples
    avg_lpips_score = total_lpips_score / num_samples
    avg_temporal_consistency = total_temporal_consistency / num_samples
    avg_warping_error = total_warping_error / num_samples
    
    # Print evaluation results
    print(f"\nEvaluation Results:")
    print(f"CLIP Score: {avg_clip_score:.4f}")
    print(f"LPIPS Score: {avg_lpips_score:.4f}")
    print(f"Temporal Consistency: {avg_temporal_consistency:.4f}")
    print(f"Warping Error: {avg_warping_error:.4f}")
    
    return {
        'clip_score': avg_clip_score,
        'lpips_score': avg_lpips_score,
        'temporal_consistency': avg_temporal_consistency,
        'warping_error': avg_warping_error
    }

def evaluate_variants(variants, data_dir, target_text="edited video"):
    """
    Evaluate multiple model variants and compare their performance.
    
    Args:
        variants (dict): Dictionary of model variants
        data_dir (str): Directory containing evaluation data
        target_text (str): Target text description for CLIP score
        
    Returns:
        dict: Evaluation metrics for each variant
    """
    print(f"\nEvaluating model variants on data from {data_dir}")
    
    # Preprocess data
    eval_dataloader = preprocess_for_training(
        data_dir=data_dir,
        batch_size=2,
        frame_size=(256, 256),
        sequence_length=16,

    )
    
    # Initialize models
    clip_model, _ = clip.load("ViT-B/32", device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Evaluate each variant
    results = {}
    for name, config in variants.items():
        print(f"\n---- Evaluating variant: {name} ----")
        
        # Create model
        model = TGGCVEModule(
            in_channels=3,
            hidden_dim=64,
            use_refined_stage=config["use_refined_stage"],
            use_consistency_loss=config["use_consistency_loss"],
            use_global_context=config["use_global_context"],
            iterative_refinement=config.get("iterative_refinement", False),
            num_iterations=config.get("num_iterations", 3)
        ).to(device)
        
        # Load model weights if available
        model_path = os.path.join("models", f"{name}.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"No pretrained weights found at {model_path}, using random initialization")
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            dataloader=eval_dataloader,
            clip_model=clip_model,
            lpips_model=lpips_model,
            target_text=target_text
        )
        
        results[name] = metrics
    
    # Compare variants
    print("\nVariant Comparison:")
    for metric in ['clip_score', 'lpips_score', 'temporal_consistency', 'warping_error']:
        print(f"\n{metric.upper()}:")
        for name, metrics in results.items():
            print(f"{name}: {metrics[metric]:.4f}")
    
    return results

def visualize_editing_results(model, input_frames, target_text="edited video"):
    """
    Visualize the results of video editing using the model.
    
    Args:
        model (nn.Module): Trained model
        input_frames (torch.Tensor): Input frames [B, T, C, H, W]
        target_text (str): Target text description
        
    Returns:
        None (displays visualization)
    """
    model.eval()
    
    batch_size, seq_len, channels, height, width = input_frames.shape
    
    # Reshape to [B*T, C, H, W] for processing individual frames
    flat_frames = input_frames.view(-1, channels, height, width)
    
    # Add noise to frames
    noise_level = torch.tensor(0.1).to(device)
    noisy_frames = add_noise(flat_frames, noise_level)
    
    # Forward pass
    with torch.no_grad():
        outputs, out_first, out_refined, attn_map, intermediates = model(
            noisy_frames, 
            noise_level=noise_level,
            global_editing_intensity=1.0
        )
    
    # Reshape outputs back to [B, T, C, H, W]
    outputs = outputs.view(batch_size, seq_len, channels, height, width)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot input and output frames
    for t in range(min(seq_len, 5)):
        # Input frame
        plt.subplot(2, 5, t + 1)
        input_np = input_frames[0, t].permute(1, 2, 0).cpu().numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)
        plt.imshow(input_np)
        plt.title(f"Input Frame {t+1}")
        plt.axis("off")
        
        # Output frame
        plt.subplot(2, 5, t + 6)
        output_np = outputs[0, t].permute(1, 2, 0).cpu().numpy()
        output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min() + 1e-8)
        plt.imshow(output_np)
        plt.title(f"Output Frame {t+1}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Compute and display metrics
    clip_model, _ = clip.load("ViT-B/32", device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # CLIP score
    clip_score = compute_clip_score(
        outputs[0, 0].unsqueeze(0),
        text_description=target_text,
        clip_model=clip_model
    )
    
    # LPIPS score
    lpips_score = compute_lpips_score(
        outputs.view(-1, channels, height, width),
        input_frames.view(-1, channels, height, width),
        lpips_model=lpips_model
    )
    
    # Temporal consistency
    temporal_consistency = compute_temporal_consistency(outputs)
    
    # Warping error
    warping_error = compute_warping_error(outputs)
    
    print(f"\nEditing Metrics:")
    print(f"CLIP Score (semantic fidelity): {clip_score:.4f}")
    print(f"LPIPS Score (perceptual similarity): {lpips_score:.4f}")
    print(f"Temporal Consistency: {temporal_consistency:.4f}")
    print(f"Warping Error: {warping_error:.4f}")
    
    # Visualize attention maps if available
    if attn_map is not None:
        plt.figure(figsize=(15, 5))
        for t in range(min(seq_len, 5)):
            plt.subplot(1, 5, t + 1)
            attn_np = attn_map[t].cpu().numpy()
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
            plt.imshow(attn_np, cmap="viridis")
            plt.title(f"Attention Map {t+1}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    # Visualize iterative refinement if available
    if intermediates is not None:
        num_iterations = len(intermediates)
        plt.figure(figsize=(15, 5))
        for i in range(min(num_iterations, 5)):
            plt.subplot(1, 5, i + 1)
            iter_np = intermediates[i][0].permute(1, 2, 0).cpu().numpy()
            iter_np = (iter_np - iter_np.min()) / (iter_np.max() - iter_np.min() + 1e-8)
            plt.imshow(iter_np)
            plt.title(f"Iteration {i+1}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

def evaluate_adaptive_fusion(data_dir, target_text="edited video"):
    """
    Evaluate the effect of adaptive fusion and iterative refinement.
    
    Args:
        data_dir (str): Directory containing evaluation data
        target_text (str): Target text description for CLIP score
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating adaptive fusion and iterative refinement on data from {data_dir}")
    
    # Preprocess data
    eval_dataloader = preprocess_for_training(
        data_dir=data_dir,
        batch_size=2,
        frame_size=(256, 256),
        sequence_length=16,

    )
    
    # Define variants
    variants = {
        "baseline": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": False,
            "iterative_refinement": False
        },
        "with_fusion": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True,
            "iterative_refinement": False
        },
        "with_refinement": {
            "use_refined_stage": True, 
            "use_consistency_loss": True, 
            "use_global_context": True,
            "iterative_refinement": True,
            "num_iterations": 3
        }
    }
    
    # Evaluate variants
    results = evaluate_variants(variants, data_dir, target_text)
    
    # Analyze results
    print("\nAdaptive Fusion and Iterative Refinement Analysis:")
    
    # Compare baseline vs. with_fusion
    fusion_improvement = {}
    for metric in ['clip_score', 'lpips_score', 'temporal_consistency', 'warping_error']:
        baseline_value = results['baseline'][metric]
        fusion_value = results['with_fusion'][metric]
        
        if metric in ['clip_score']:
            # Higher is better
            improvement = ((fusion_value - baseline_value) / baseline_value) * 100
        else:
            # Lower is better
            improvement = ((baseline_value - fusion_value) / baseline_value) * 100
        
        fusion_improvement[metric] = improvement
    
    # Compare with_fusion vs. with_refinement
    refinement_improvement = {}
    for metric in ['clip_score', 'lpips_score', 'temporal_consistency', 'warping_error']:
        fusion_value = results['with_fusion'][metric]
        refinement_value = results['with_refinement'][metric]
        
        if metric in ['clip_score']:
            # Higher is better
            improvement = ((refinement_value - fusion_value) / fusion_value) * 100
        else:
            # Lower is better
            improvement = ((fusion_value - refinement_value) / fusion_value) * 100
        
        refinement_improvement[metric] = improvement
    
    # Print improvement analysis
    print("\nAdaptive Fusion Improvement over Baseline:")
    for metric, improvement in fusion_improvement.items():
        print(f"{metric}: {improvement:.2f}%")
    
    print("\nIterative Refinement Improvement over Fusion:")
    for metric, improvement in refinement_improvement.items():
        print(f"{metric}: {improvement:.2f}%")
    
    return {
        'results': results,
        'fusion_improvement': fusion_improvement,
        'refinement_improvement': refinement_improvement
    }

# --------------------------
# Main Function
# --------------------------

if __name__ == "__main__":
    # Example usage
    data_dir = "data/sample_videos"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory: {data_dir}")
        print("Please add video files to this directory for evaluation")
    
    # If data directory exists and contains video files, evaluate the model
    elif any(f.endswith(('.mp4', '.avi', '.mov')) for f in os.listdir(data_dir)):
        # Define model variants
        variants = {
            "full_TG_GCVE": {
                "use_refined_stage": True, 
                "use_consistency_loss": True, 
                "use_global_context": True
            },
            "no_consistency": {
                "use_refined_stage": True, 
                "use_consistency_loss": False, 
                "use_global_context": True
            },
            "with_refinement": {
                "use_refined_stage": True, 
                "use_consistency_loss": True, 
                "use_global_context": True,
                "iterative_refinement": True,
                "num_iterations": 3
            }
        }
        
        # Evaluate variants
        results = evaluate_variants(variants, data_dir)
        
        # Evaluate adaptive fusion and iterative refinement
        fusion_results = evaluate_adaptive_fusion(data_dir)
        
        print("\nEvaluation completed successfully.")
    else:
        print(f"No video files found in {data_dir}")
        print("Please add video files (mp4, avi, mov) to this directory for evaluation")
