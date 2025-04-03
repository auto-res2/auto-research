"""
Evaluation module for SPCDD MRI Super-Resolution.

This module handles the evaluation of trained models and visualization of results.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.preprocess import AnatomyExtractor
from src.train import (
    DiffusionModel, 
    DiffusionModelWithIntensity, 
    TeacherModel, 
    StudentModel
)
from src.utils.metrics import (
    calculate_psnr, 
    calculate_ssim, 
    plot_comparison, 
    plot_histograms, 
    plot_training_curve
)
from src.utils.visualization import (
    tensor_to_numpy, 
    plot_image_grid, 
    plot_bar_comparison
)

def evaluate_ablation_study(config, models, extractors, val_loader, save_dir="results/ablation"):
    """
    Evaluate models from the ablation study and generate comparison plots.
    
    Args:
        config: Configuration dictionary
        models: Dictionary of trained models with and without anatomical prior
        extractors: Dictionary of trained extractors (or None)
        val_loader: DataLoader for validation data
        save_dir: Directory to save results
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        model.eval()
    for name, extractor in extractors.items():
        if extractor is not None:
            extractor.eval()
    
    metrics = {
        'with_prior': {'psnr': [], 'ssim': []},
        'without_prior': {'psnr': [], 'ssim': []}
    }
    
    sample_images = {
        'input': None,
        'target': None,
        'with_prior': None,
        'without_prior': None
    }
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Evaluating ablation models")):
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            outputs = {}
            
            if extractors['with_prior'] is not None:
                prior = extractors['with_prior'](img_15T)
                outputs['with_prior'] = models['with_prior'](img_15T, anatomy_prior=prior)
            else:
                outputs['with_prior'] = models['with_prior'](img_15T)
            
            outputs['without_prior'] = models['without_prior'](img_15T)
            
            for name, output in outputs.items():
                output_np = output.cpu().numpy()
                target_np = target_7T.cpu().numpy()
                
                for j in range(output_np.shape[0]):
                    psnr = calculate_psnr(output_np[j, 0], target_np[j, 0])
                    ssim = calculate_ssim(output_np[j, 0], target_np[j, 0], data_range=1.0)
                    metrics[name]['psnr'].append(psnr)
                    metrics[name]['ssim'].append(ssim)
            
            if i == 0:
                sample_images['input'] = img_15T.cpu().numpy()
                sample_images['target'] = target_7T.cpu().numpy()
                sample_images['with_prior'] = outputs['with_prior'].cpu().numpy()
                sample_images['without_prior'] = outputs['without_prior'].cpu().numpy()
    
    avg_metrics = {}
    for name in metrics:
        avg_metrics[name] = {
            'psnr': np.mean(metrics[name]['psnr']),
            'ssim': np.mean(metrics[name]['ssim'])
        }
        print(f"Model {name}: PSNR = {avg_metrics[name]['psnr']:.2f}, SSIM = {avg_metrics[name]['ssim']:.4f}")
    
    plot_bar_comparison(
        [avg_metrics['with_prior']['psnr'], avg_metrics['without_prior']['psnr']],
        ['With Anatomical Prior', 'Without Anatomical Prior'],
        title="PSNR Comparison - Ablation Study",
        ylabel="PSNR (dB)",
        save_path=os.path.join(save_dir, "psnr_comparison_ablation.pdf")
    )
    
    plot_bar_comparison(
        [avg_metrics['with_prior']['ssim'], avg_metrics['without_prior']['ssim']],
        ['With Anatomical Prior', 'Without Anatomical Prior'],
        title="SSIM Comparison - Ablation Study",
        ylabel="SSIM",
        save_path=os.path.join(save_dir, "ssim_comparison_ablation.pdf")
    )
    
    if (sample_images['input'] is not None and 
        sample_images['with_prior'] is not None and
        sample_images['without_prior'] is not None and
        sample_images['target'] is not None):
        
        for i in range(min(3, sample_images['input'].shape[0])):
            plot_comparison(
                sample_images['with_prior'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"comparison_with_prior_{i}.pdf"),
                title="With Anatomical Prior"
            )
            
            plot_comparison(
                sample_images['without_prior'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"comparison_without_prior_{i}.pdf"),
                title="Without Anatomical Prior"
            )
            
            plot_image_grid(
                [
                    sample_images['input'][i, 0],
                    sample_images['with_prior'][i, 0],
                    sample_images['without_prior'][i, 0],
                    sample_images['target'][i, 0]
                ],
                titles=[
                    "Input (1.5T)",
                    "With Prior",
                    "Without Prior",
                    "Target (7T)"
                ],
                rows=1, cols=4,
                save_path=os.path.join(save_dir, f"grid_comparison_{i}.pdf")
            )
    
    return avg_metrics

def evaluate_intensity_modulation(config, models, val_loader, save_dir="results/intensity"):
    """
    Evaluate models with and without intensity modulation and generate comparison plots.
    
    Args:
        config: Configuration dictionary
        models: Dictionary of trained models with and without intensity modulation
        val_loader: DataLoader for validation data
        save_dir: Directory to save results
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        model.eval()
    
    metrics = {
        'with_intensity': {'psnr': [], 'ssim': []},
        'without_intensity': {'psnr': [], 'ssim': []}
    }
    
    sample_images = {
        'input': None,
        'target': None,
        'with_intensity': None,
        'without_intensity': None
    }
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Evaluating intensity modulation models")):
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            outputs = {
                'with_intensity': models['with_intensity'](img_15T),
                'without_intensity': models['without_intensity'](img_15T)
            }
            
            for name, output in outputs.items():
                output_np = output.cpu().numpy()
                target_np = target_7T.cpu().numpy()
                
                for j in range(output_np.shape[0]):
                    psnr = calculate_psnr(output_np[j, 0], target_np[j, 0])
                    ssim = calculate_ssim(output_np[j, 0], target_np[j, 0], data_range=1.0)
                    metrics[name]['psnr'].append(psnr)
                    metrics[name]['ssim'].append(ssim)
            
            if i == 0:
                sample_images['input'] = img_15T.cpu().numpy()
                sample_images['target'] = target_7T.cpu().numpy()
                sample_images['with_intensity'] = outputs['with_intensity'].cpu().numpy()
                sample_images['without_intensity'] = outputs['without_intensity'].cpu().numpy()
    
    avg_metrics = {}
    for name in metrics:
        avg_metrics[name] = {
            'psnr': np.mean(metrics[name]['psnr']),
            'ssim': np.mean(metrics[name]['ssim'])
        }
        print(f"Model {name}: PSNR = {avg_metrics[name]['psnr']:.2f}, SSIM = {avg_metrics[name]['ssim']:.4f}")
    
    plot_bar_comparison(
        [avg_metrics['with_intensity']['psnr'], avg_metrics['without_intensity']['psnr']],
        ['With Intensity Modulation', 'Without Intensity Modulation'],
        title="PSNR Comparison - Intensity Modulation",
        ylabel="PSNR (dB)",
        save_path=os.path.join(save_dir, "psnr_comparison_intensity.pdf")
    )
    
    plot_bar_comparison(
        [avg_metrics['with_intensity']['ssim'], avg_metrics['without_intensity']['ssim']],
        ['With Intensity Modulation', 'Without Intensity Modulation'],
        title="SSIM Comparison - Intensity Modulation",
        ylabel="SSIM",
        save_path=os.path.join(save_dir, "ssim_comparison_intensity.pdf")
    )
    
    if (sample_images['input'] is not None and 
        sample_images['with_intensity'] is not None and
        sample_images['without_intensity'] is not None and
        sample_images['target'] is not None):
        
        for i in range(min(3, sample_images['input'].shape[0])):
            plot_comparison(
                sample_images['with_intensity'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"comparison_with_intensity_{i}.pdf"),
                title="With Intensity Modulation"
            )
            
            plot_comparison(
                sample_images['without_intensity'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"comparison_without_intensity_{i}.pdf"),
                title="Without Intensity Modulation"
            )
            
            plot_image_grid(
                [
                    sample_images['input'][i, 0],
                    sample_images['with_intensity'][i, 0],
                    sample_images['without_intensity'][i, 0],
                    sample_images['target'][i, 0]
                ],
                titles=[
                    "Input (1.5T)",
                    "With Intensity Mod",
                    "Without Intensity Mod",
                    "Target (7T)"
                ],
                rows=1, cols=4,
                save_path=os.path.join(save_dir, f"grid_comparison_intensity_{i}.pdf")
            )
            
            plot_histograms(
                sample_images['with_intensity'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"histogram_with_intensity_{i}.pdf"),
                title="Intensity Distribution - With Modulation"
            )
            
            plot_histograms(
                sample_images['without_intensity'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"histogram_without_intensity_{i}.pdf"),
                title="Intensity Distribution - Without Modulation"
            )
    
    return avg_metrics

def evaluate_distillation(config, teacher, student, val_loader, save_dir="results/distillation"):
    """
    Evaluate teacher and student models and generate comparison plots.
    
    Args:
        config: Configuration dictionary
        teacher: Trained teacher model
        student: Trained student model
        val_loader: DataLoader for validation data
        save_dir: Directory to save results
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    
    teacher.eval()
    student.eval()
    
    metrics = {
        'teacher': {'psnr': [], 'ssim': []},
        'student': {'psnr': [], 'ssim': []}
    }
    
    sample_images = {
        'input': None,
        'target': None,
        'teacher': None,
        'student': None
    }
    
    inference_times = {
        'teacher': [],
        'student': []
    }
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Evaluating distillation models")):
            img_15T, target_7T, _ = data
            img_15T = img_15T.to(device)
            target_7T = target_7T.to(device)
            
            teacher_output, _ = teacher(img_15T)
            student_output, _ = student(img_15T)
            
            outputs = {
                'teacher': teacher_output,
                'student': student_output
            }
            
            for name, output in outputs.items():
                output_np = output.cpu().numpy()
                target_np = target_7T.cpu().numpy()
                
                for j in range(output_np.shape[0]):
                    psnr = calculate_psnr(output_np[j, 0], target_np[j, 0])
                    ssim = calculate_ssim(output_np[j, 0], target_np[j, 0], data_range=1.0)
                    metrics[name]['psnr'].append(psnr)
                    metrics[name]['ssim'].append(ssim)
            
            if i == 0:
                sample_images['input'] = img_15T.cpu().numpy()
                sample_images['target'] = target_7T.cpu().numpy()
                sample_images['teacher'] = teacher_output.cpu().numpy()
                sample_images['student'] = student_output.cpu().numpy()
    
    avg_metrics = {}
    for name in metrics:
        avg_metrics[name] = {
            'psnr': np.mean(metrics[name]['psnr']),
            'ssim': np.mean(metrics[name]['ssim'])
        }
        print(f"Model {name}: PSNR = {avg_metrics[name]['psnr']:.2f}, SSIM = {avg_metrics[name]['ssim']:.4f}")
    
    sample_input = torch.rand((1, 1, config.image_size, config.image_size)).to(device)
    
    with torch.no_grad():
        _ = teacher(sample_input)
        _ = student(sample_input)
    
    import time
    iterations = 10
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = teacher(sample_input)
    teacher_time = (time.time() - start_time) / iterations
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = student(sample_input)
    student_time = (time.time() - start_time) / iterations
    
    print(f"Inference time: Teacher = {teacher_time:.4f}s, Student = {student_time:.4f}s")
    print(f"Speedup: {teacher_time / student_time:.2f}x")
    
    plot_bar_comparison(
        [avg_metrics['teacher']['psnr'], avg_metrics['student']['psnr']],
        ['Teacher Model', 'Student Model'],
        title="PSNR Comparison - Distillation",
        ylabel="PSNR (dB)",
        save_path=os.path.join(save_dir, "psnr_comparison_distillation.pdf")
    )
    
    plot_bar_comparison(
        [avg_metrics['teacher']['ssim'], avg_metrics['student']['ssim']],
        ['Teacher Model', 'Student Model'],
        title="SSIM Comparison - Distillation",
        ylabel="SSIM",
        save_path=os.path.join(save_dir, "ssim_comparison_distillation.pdf")
    )
    
    plot_bar_comparison(
        [teacher_time, student_time],
        ['Teacher Model', 'Student Model'],
        title="Inference Time Comparison",
        ylabel="Time (seconds)",
        save_path=os.path.join(save_dir, "inference_time_comparison.pdf")
    )
    
    if (sample_images['input'] is not None and 
        sample_images['teacher'] is not None and
        sample_images['student'] is not None and
        sample_images['target'] is not None):
        
        for i in range(min(3, sample_images['input'].shape[0])):
            plot_comparison(
                sample_images['teacher'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"comparison_teacher_{i}.pdf"),
                title="Teacher Model Output"
            )
            
            plot_comparison(
                sample_images['student'][i, 0],
                sample_images['target'][i, 0],
                save_path=os.path.join(save_dir, f"comparison_student_{i}.pdf"),
                title="Student Model Output"
            )
            
            plot_image_grid(
                [
                    sample_images['input'][i, 0],
                    sample_images['teacher'][i, 0],
                    sample_images['student'][i, 0],
                    sample_images['target'][i, 0]
                ],
                titles=[
                    "Input (1.5T)",
                    "Teacher Output",
                    "Student Output",
                    "Target (7T)"
                ],
                rows=1, cols=4,
                save_path=os.path.join(save_dir, f"grid_comparison_distillation_{i}.pdf")
            )
    
    return avg_metrics, {'teacher': teacher_time, 'student': student_time}

def evaluate_model_memory_usage(config, models, device):
    """
    Evaluate memory usage of different models.
    
    Args:
        config: Configuration dictionary
        models: Dictionary of models to evaluate
        device: Device to run evaluation on
        
    Returns:
        memory_usage: Dictionary of memory usage for each model
    """
    memory_usage = {}
    
    sample_input = torch.rand((1, 1, config.image_size, config.image_size)).to(device)
    
    for name, model in models.items():
        model.to(device)
        model.eval()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(sample_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        else:
            memory_used = 0
            
        memory_usage[name] = memory_used
        print(f"Model {name}: Memory usage = {memory_used:.2f} MB")
    
    if device.type == 'cuda':
        plot_bar_comparison(
            list(memory_usage.values()),
            list(memory_usage.keys()),
            title="Memory Usage Comparison",
            ylabel="Memory (MB)",
            save_path="results/memory_usage_comparison.pdf"
        )
    
    return memory_usage

def save_model(model, path):
    """
    Save a trained model to disk.
    
    Args:
        model: The model to save
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """
    Load a trained model from disk.
    
    Args:
        model: The model to load weights into
        path: Path to the saved model
        device: Device to load the model onto
        
    Returns:
        model: The loaded model
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model
