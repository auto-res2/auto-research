"""
Model evaluation module for A2Diff experiments.
"""
import time
import torch
import numpy as np
from tqdm import tqdm

def generate_samples(model, dataloader, adaptive=False, severity_estimator=None, config=None):
    """
    Generate samples using either fixed or adaptive schedule.
    
    Args:
        model: Diffusion model
        dataloader: DataLoader for the dataset
        adaptive: Whether to use adaptive schedule
        severity_estimator: Severity estimator network
        config: Experiment configuration
        
    Returns:
        all_samples: Generated samples
        all_num_steps: Number of steps taken
        all_inference_times: Inference times
    """
    all_samples = []
    all_num_steps = []
    all_inference_times = []
    
    # Default base schedule if config is None
    default_base_schedule = [10, 10, 3, 2, 2]
    default_severity_threshold = 0.5
    
    for batch in tqdm(dataloader, desc="Experiment 1 Batch"):
        if len(batch) == 2:
            images, _ = batch
            images = images.to(model.device)
        else:
            images, _, _ = batch
            images = images.to(model.device)
            
        tic = time.time()
        
        # Use default schedule if config is None
        base_schedule = default_base_schedule
        if config is not None:
            base_schedule = config['model']['base_schedule']
        
        if adaptive and severity_estimator is not None:
            with torch.no_grad():
                # Use the batch mean severity; could be done per image in full-scale experiments
                severity_score = severity_estimator(images).mean().item()
            
            # Adapted schedule based on severity
            threshold = default_severity_threshold
            if config is not None:
                threshold = config['model']['severity_threshold']
                
            if severity_score > threshold:
                schedule = [t+1 for t in base_schedule]
                print(f"Adaptive schedule chosen (severity={severity_score:.3f}): {schedule}")
            else:
                schedule = base_schedule
                print(f"Adaptive schedule chosen (severity={severity_score:.3f}): {schedule}")
        else:
            schedule = base_schedule
        
        gen, steps_taken = model.sample(schedule=schedule, batch_size=images.shape[0])
        toc = time.time()
        
        all_samples.append(gen)
        all_num_steps.append(steps_taken)
        all_inference_times.append(toc-tic)
        
        print(f"Batch processed. Steps: {steps_taken} Inference time: {toc-tic:.4f} sec")
        
    return torch.cat(all_samples), all_num_steps, all_inference_times

def generate_samples_variant(model, dataloader, variant='full', severity_estimator=None, config=None):
    """
    Generate samples using different variants of the schedule.
    
    Args:
        model: Diffusion model
        dataloader: DataLoader for the dataset
        variant: Variant of the schedule to use ('full', 'fixed_extra', 'random')
        severity_estimator: Severity estimator network
        config: Experiment configuration
        
    Returns:
        all_samples: Generated samples
        all_num_steps: Number of steps taken
        all_inference_times: Inference times
    """
    all_samples = []
    all_num_steps = []
    all_inference_times = []
    
    # Default base schedule if config is None
    default_base_schedule = [10, 10, 3, 2, 2]
    default_severity_threshold = 0.5
    
    for batch in tqdm(dataloader, desc=f"Experiment 2 ({variant}) Batch"):
        if len(batch) == 2:
            images, _ = batch
            images = images.to(model.device)
        else:
            images, _, _ = batch
            images = images.to(model.device)
            
        tic = time.time()
        
        # Use default schedule if config is None
        base_schedule = default_base_schedule
        if config is not None:
            base_schedule = config['model']['base_schedule']
        
        if variant == 'full' and severity_estimator is not None:
            with torch.no_grad():
                severity_score = severity_estimator(images).mean().item()
            
            # Adapted schedule based on severity
            threshold = default_severity_threshold
            if config is not None:
                threshold = config['model']['severity_threshold']
                
            if severity_score > threshold:
                schedule = [t+1 for t in base_schedule]
                print(f"Adaptive schedule chosen (severity={severity_score:.3f}): {schedule}")
            else:
                schedule = base_schedule
                print(f"Adaptive schedule chosen (severity={severity_score:.3f}): {schedule}")
        elif variant == 'fixed_extra':
            schedule = [t+1 for t in base_schedule]
            print(f"Fixed extra schedule: {schedule}")
        elif variant == 'random':
            rand_score = np.random.uniform(0, 1)
            if rand_score > 0.5:
                schedule = [t+1 for t in base_schedule]
                print(f"Random adaptation (score={rand_score:.3f}): {schedule}")
            else:
                schedule = base_schedule
                print(f"Random adaptation (score={rand_score:.3f}): {base_schedule}")
        else:
            schedule = base_schedule
            print(f"Using default schedule: {base_schedule}")
            
        gen, steps_taken = model.sample(schedule=schedule, batch_size=images.shape[0])
        toc = time.time()
        all_samples.append(gen)
        all_num_steps.append(steps_taken)
        all_inference_times.append(toc-tic)
        
        print(f"Variant {variant}: Batch processed. Steps: {steps_taken} Inference time: {toc-tic:.4f} sec")
        
    return torch.cat(all_samples), all_num_steps, all_inference_times

def generate_samples_with_degradation(model, dataloader, adaptive=False, severity_estimator=None, config=None):
    """
    Generate samples with degradation for the robustness experiment.
    
    Args:
        model: Diffusion model
        dataloader: DataLoader for the dataset
        adaptive: Whether to use adaptive schedule
        severity_estimator: Severity estimator network
        config: Experiment configuration
        
    Returns:
        all_samples: Generated samples
        all_num_steps: Number of steps taken
        all_inference_times: Inference times
        all_severity_scores: Severity scores
        degradation_flags_all: Degradation flags
    """
    all_samples = []
    all_num_steps = []
    all_inference_times = []
    all_severity_scores = []  # list of severity (per batch average)
    degradation_flags_all = []
    
    # Default base schedule if config is None
    default_base_schedule = [10, 10, 3, 2, 2]
    default_severity_threshold = 0.5
    
    for batch in tqdm(dataloader, desc="Experiment 3 Batch"):
        images, _, flags = batch
        images = images.to(model.device)
        tic = time.time()
        
        # Use default schedule if config is None
        base_schedule = default_base_schedule
        if config is not None:
            base_schedule = config['model']['base_schedule']
        
        if adaptive and severity_estimator is not None:
            with torch.no_grad():
                scores = severity_estimator(images)
                avg_score = scores.mean().item()
            
            # Adapted schedule based on severity
            threshold = default_severity_threshold
            if config is not None:
                threshold = config['model']['severity_threshold']
                
            if avg_score > threshold:
                schedule = [t+1 for t in base_schedule]
                print(f"Adaptive schedule chosen (severity={avg_score:.3f}): {schedule}")
            else:
                schedule = base_schedule
                print(f"Adaptive schedule chosen (severity={avg_score:.3f}): {schedule}")
            
            all_severity_scores.append(avg_score)
        else:
            schedule = base_schedule
        
        gen, steps_taken = model.sample(schedule=schedule, batch_size=images.shape[0])
        toc = time.time()
        
        all_samples.append(gen)
        all_num_steps.append(steps_taken)
        all_inference_times.append(toc-tic)
        degradation_flags_all.extend(flags)
        
        print(f"Degradation mode: Batch processed. Steps: {steps_taken} Inference time: {toc-tic:.4f} sec")
        
    # For simplicity, we return avg severity score per batch (if adaptive)
    return torch.cat(all_samples), all_num_steps, all_inference_times, all_severity_scores, degradation_flags_all
