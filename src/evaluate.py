"""
Evaluation script for the PTDA model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.ptda_model import PTDAModel, AblatedPTDAModel, BaselineModel
from src.utils.data import load_video_frames, create_dummy_video, preprocess_frame_for_model
from src.utils.metrics import compute_ssim, compute_psnr, compute_temporal_consistency, compute_temporal_consistency_long
from src.utils.visualization import save_comparison_plot, save_error_map, save_metrics_plot
from config.ptda.config import MODEL_CONFIG, EXPERIMENT_CONFIG, PATHS


def load_model(model_path, model_type='ptda', device='cuda'):
    """
    Load a model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model to load ('ptda', 'ablated', or 'baseline')
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if model_type == 'ptda':
        model = PTDAModel(
            include_latent=MODEL_CONFIG['include_latent'],
            latent_dim=MODEL_CONFIG['latent_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        )
    elif model_type == 'ablated':
        model = AblatedPTDAModel(
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        )
    elif model_type == 'baseline':
        model = BaselineModel(
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def run_inference(model, frames, device='cuda'):
    """
    Run inference on a list of frames using the given model.
    
    Args:
        model: Model to use for inference
        frames: List of frames (numpy arrays)
        device: Device to run inference on
        
    Returns:
        List of generated frames (numpy arrays)
    """
    model.eval()
    generated_frames = []
    
    with torch.no_grad():
        for frame in tqdm(frames, desc="Running inference"):
            tensor_frame = preprocess_frame_for_model(frame, device=device)
            
            output = model(tensor_frame)
            if isinstance(output, tuple):
                output = output[0]  # For models that output (reconstruction, latent)
            
            output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            generated_frames.append(output_np)
    
    return generated_frames


def generate_long_video(model, initial_frame, num_frames=10, device='cuda'):
    """
    Generate a long video sequence starting from the given initial frame.
    
    Args:
        model: Model to use for generation
        initial_frame: Initial frame (numpy array)
        num_frames: Number of frames to generate
        device: Device to run generation on
        
    Returns:
        List of generated frames (numpy arrays)
    """
    model.eval()
    generated_frames = [initial_frame]
    current_frame = initial_frame
    
    with torch.no_grad():
        for i in tqdm(range(num_frames - 1), desc="Generating video"):
            tensor_frame = preprocess_frame_for_model(current_frame, device=device)
            
            output = model(tensor_frame)
            if isinstance(output, tuple):
                output = output[0]  # For models that output (reconstruction, latent)
            
            next_frame = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            generated_frames.append(next_frame)
            current_frame = next_frame
    
    return generated_frames


def evaluate_experiment_1(ptda_model, baseline_model, device='cuda'):
    """
    Evaluate Experiment 1: Dynamic Background Synthesis and Consistency Testing.
    
    Args:
        ptda_model: PTDA model
        baseline_model: Baseline model
        device: Device to run evaluation on
    """
    print("Evaluating Experiment 1: Dynamic Background Synthesis and Consistency Testing...")
    
    results_dir = os.path.join(PATHS['results_dir'], 'experiment_1')
    figures_dir = os.path.join(PATHS['figures_dir'], 'experiment_1')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    video_path = os.path.join(PATHS['data_dir'], 'dummy_video_0.mp4')
    if not os.path.exists(video_path):
        create_dummy_video(video_path, num_frames=EXPERIMENT_CONFIG['experiment_1']['num_frames'])
    
    frames = load_video_frames(video_path, num_frames=EXPERIMENT_CONFIG['experiment_1']['num_frames'])
    print(f"Loaded {len(frames)} frames from video.")
    
    ptda_generated = run_inference(ptda_model, frames, device=device)
    baseline_generated = run_inference(baseline_model, frames, device=device)
    
    ptda_ssim = compute_ssim(frames, ptda_generated)
    ptda_psnr = compute_psnr(frames, ptda_generated)
    baseline_ssim = compute_ssim(frames, baseline_generated)
    baseline_psnr = compute_psnr(frames, baseline_generated)
    
    print(f"PTDA SSIM: {ptda_ssim:.4f}   PSNR: {ptda_psnr:.2f}")
    print(f"Baseline SSIM: {baseline_ssim:.4f}   PSNR: {baseline_psnr:.2f}")
    
    ptda_temporal_consistency = compute_temporal_consistency(ptda_generated)
    baseline_temporal_consistency = compute_temporal_consistency(baseline_generated)
    
    print(f"PTDA Temporal Consistency Score: {ptda_temporal_consistency:.4f}")
    print(f"Baseline Temporal Consistency Score: {baseline_temporal_consistency:.4f}")
    
    metrics = {
        'ptda_ssim': ptda_ssim,
        'ptda_psnr': ptda_psnr,
        'ptda_temporal_consistency': ptda_temporal_consistency,
        'baseline_ssim': baseline_ssim,
        'baseline_psnr': baseline_psnr,
        'baseline_temporal_consistency': baseline_temporal_consistency,
    }
    
    metrics_dict = {
        'PTDA SSIM': [ptda_ssim],
        'Baseline SSIM': [baseline_ssim],
        'PTDA PSNR': [ptda_psnr / 50],  # Scale down for visualization
        'Baseline PSNR': [baseline_psnr / 50],  # Scale down for visualization
        'PTDA TC': [ptda_temporal_consistency],
        'Baseline TC': [baseline_temporal_consistency],
    }
    save_metrics_plot(metrics_dict, "Experiment 1 Metrics", os.path.join(figures_dir, "metrics.pdf"))
    
    save_comparison_plot(
        frames[:5],
        ptda_generated[:5],
        "Original vs. PTDA Generated Frames",
        os.path.join(figures_dir, "ptda_comparison.pdf")
    )
    save_comparison_plot(
        frames[:5],
        baseline_generated[:5],
        "Original vs. Baseline Generated Frames",
        os.path.join(figures_dir, "baseline_comparison.pdf")
    )
    
    if len(ptda_generated) >= 4:
        save_error_map(
            ptda_generated[2],
            ptda_generated[3],
            "Temporal Error Map (PTDA) between Frame 3 and 4",
            os.path.join(figures_dir, "ptda_error_map.pdf")
        )
    
    print("Experiment 1 evaluation completed.\n")
    return metrics


def evaluate_experiment_2(ptda_model, ablated_model, device='cuda'):
    """
    Evaluate Experiment 2: Latent Variable Integration Ablation Study.
    
    Args:
        ptda_model: PTDA model
        ablated_model: Ablated PTDA model
        device: Device to run evaluation on
    """
    print("Evaluating Experiment 2: Latent Variable Integration Ablation Study...")
    
    results_dir = os.path.join(PATHS['results_dir'], 'experiment_2')
    figures_dir = os.path.join(PATHS['figures_dir'], 'experiment_2')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    from src.utils.data import VideoFrameDataset, get_dataloader
    dataset = VideoFrameDataset(
        num_samples=EXPERIMENT_CONFIG['experiment_2']['num_samples'],
        num_frames=EXPERIMENT_CONFIG['experiment_2']['num_frames'],
        height=64,
        width=64
    )
    dataloader = get_dataloader(dataset, batch_size=2, shuffle=True)
    
    batch = next(iter(dataloader)).to(device)
    
    batch_size = batch.size(0)
    first_frames = batch[:, 0].view(batch_size, 3, 64, 64)  # Explicitly reshape to ensure correct dimensions
    
    ptda_model.eval()
    with torch.no_grad():
        if hasattr(ptda_model, 'include_latent') and ptda_model.include_latent:
            ptda_output, ptda_latent = ptda_model(first_frames)
        else:
            ptda_output = ptda_model(first_frames)
            ptda_latent = None
    
    ablated_model.eval()
    with torch.no_grad():
        ablated_output = ablated_model(first_frames)
    
    ptda_output_np = (ptda_output.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    ablated_output_np = (ablated_output.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    input_np = (batch[:, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    
    from src.utils.metrics import compute_ssim, compute_psnr
    ptda_ssim = compute_ssim(input_np, ptda_output_np)
    ptda_psnr = compute_psnr(input_np, ptda_output_np)
    ablated_ssim = compute_ssim(input_np, ablated_output_np)
    ablated_psnr = compute_psnr(input_np, ablated_output_np)
    
    print(f"PTDA SSIM: {ptda_ssim:.4f}   PSNR: {ptda_psnr:.2f}")
    print(f"Ablated SSIM: {ablated_ssim:.4f}   PSNR: {ablated_psnr:.2f}")
    
    metrics = {
        'ptda_ssim': ptda_ssim,
        'ptda_psnr': ptda_psnr,
        'ablated_ssim': ablated_ssim,
        'ablated_psnr': ablated_psnr,
    }
    
    metrics_dict = {
        'PTDA SSIM': [ptda_ssim],
        'Ablated SSIM': [ablated_ssim],
        'PTDA PSNR': [ptda_psnr / 50],  # Scale down for visualization
        'Ablated PSNR': [ablated_psnr / 50],  # Scale down for visualization
    }
    save_metrics_plot(metrics_dict, "Experiment 2 Metrics", os.path.join(figures_dir, "metrics.pdf"))
    
    if ptda_latent is not None:
        from src.utils.visualization import visualize_latent_space
        visualize_latent_space(
            [ptda_latent],
            "PTDA Latent Space",
            os.path.join(figures_dir, "ptda_latent_space.pdf")
        )
    
    print("Experiment 2 evaluation completed.\n")
    return metrics


def evaluate_experiment_3(ptda_model, baseline_model, device='cuda'):
    """
    Evaluate Experiment 3: Long-Range Temporal Coherence Evaluation.
    
    Args:
        ptda_model: PTDA model
        baseline_model: Baseline model
        device: Device to run evaluation on
    """
    print("Evaluating Experiment 3: Long-Range Temporal Coherence Evaluation...")
    
    results_dir = os.path.join(PATHS['results_dir'], 'experiment_3')
    figures_dir = os.path.join(PATHS['figures_dir'], 'experiment_3')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    initial_frame_path = os.path.join(PATHS['data_dir'], 'initial_frame.jpg')
    if not os.path.exists(initial_frame_path):
        initial_frame = np.uint8(np.random.rand(256, 256, 3) * 255)
        os.makedirs(os.path.dirname(initial_frame_path), exist_ok=True)
        plt.imsave(initial_frame_path, initial_frame)
    else:
        initial_frame = plt.imread(initial_frame_path)
        if initial_frame.dtype == np.float32:
            initial_frame = (initial_frame * 255).astype(np.uint8)
    
    num_frames = EXPERIMENT_CONFIG['experiment_3']['num_frames']
    window_size = EXPERIMENT_CONFIG['experiment_3']['window_size']
    
    ptda_video = generate_long_video(ptda_model, initial_frame, num_frames=num_frames, device=device)
    baseline_video = generate_long_video(baseline_model, initial_frame, num_frames=num_frames, device=device)
    
    ptda_consistency = compute_temporal_consistency_long(ptda_video, window_size=window_size)
    baseline_consistency = compute_temporal_consistency_long(baseline_video, window_size=window_size)
    
    print(f"Long-range Temporal Consistency (PTDA): {ptda_consistency:.4f}")
    print(f"Long-range Temporal Consistency (Baseline): {baseline_consistency:.4f}")
    
    metrics = {
        'ptda_long_range_consistency': ptda_consistency,
        'baseline_long_range_consistency': baseline_consistency,
    }
    
    metrics_dict = {
        'PTDA': [ptda_consistency],
        'Baseline': [baseline_consistency],
    }
    save_metrics_plot(metrics_dict, "Long-range Temporal Consistency", os.path.join(figures_dir, "metrics.pdf"))
    
    save_comparison_plot(
        [initial_frame] + ptda_video[1:5],
        ptda_video[:5],
        "PTDA Generated Sequence",
        os.path.join(figures_dir, "ptda_sequence.pdf")
    )
    save_comparison_plot(
        [initial_frame] + baseline_video[1:5],
        baseline_video[:5],
        "Baseline Generated Sequence",
        os.path.join(figures_dir, "baseline_sequence.pdf")
    )
    
    sample_frame_ptda = ptda_video[4].copy()
    plt.figure(figsize=(10, 8))
    plt.imshow(sample_frame_ptda)
    plt.title("Sample Frame from PTDA Generated Sequence (Frame 4)")
    plt.savefig(os.path.join(figures_dir, "ptda_sample_frame.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Experiment 3 evaluation completed.\n")
    return metrics


def main():
    """
    Main function for model evaluation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    os.makedirs(PATHS['figures_dir'], exist_ok=True)
    
    ptda_model = PTDAModel(
        include_latent=MODEL_CONFIG['include_latent'],
        latent_dim=MODEL_CONFIG['latent_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    ablated_model = AblatedPTDAModel(
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    baseline_model = BaselineModel(
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    exp1_metrics = evaluate_experiment_1(ptda_model, baseline_model, device=device)
    exp2_metrics = evaluate_experiment_2(ptda_model, ablated_model, device=device)
    exp3_metrics = evaluate_experiment_3(ptda_model, baseline_model, device=device)
    
    print("\nEvaluation Summary:")
    print("Experiment 1 - Dynamic Background Synthesis:")
    print(f"  PTDA SSIM: {exp1_metrics['ptda_ssim']:.4f}, PSNR: {exp1_metrics['ptda_psnr']:.2f}")
    print(f"  Baseline SSIM: {exp1_metrics['baseline_ssim']:.4f}, PSNR: {exp1_metrics['baseline_psnr']:.2f}")
    
    print("\nExperiment 2 - Latent Variable Integration Ablation Study:")
    print(f"  PTDA SSIM: {exp2_metrics['ptda_ssim']:.4f}, PSNR: {exp2_metrics['ptda_psnr']:.2f}")
    print(f"  Ablated SSIM: {exp2_metrics['ablated_ssim']:.4f}, PSNR: {exp2_metrics['ablated_psnr']:.2f}")
    
    print("\nExperiment 3 - Long-Range Temporal Coherence:")
    print(f"  PTDA Consistency: {exp3_metrics['ptda_long_range_consistency']:.4f}")
    print(f"  Baseline Consistency: {exp3_metrics['baseline_long_range_consistency']:.4f}")


if __name__ == "__main__":
    main()
