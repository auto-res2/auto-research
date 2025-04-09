import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to evaluate on
        
    Returns:
        float: Average loss
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            if hasattr(model, 'forward') and 'steps' in model.forward.__code__.co_varnames:
                output = model(images, steps=10)
            else:
                output = model(images)
                
            loss = ((output - images) ** 2).mean().item()
            losses.append(loss)
            
    avg_loss = sum(losses) / len(losses)
    print(f"Evaluation - Avg Loss: {avg_loss:.4f}")
    return avg_loss

def profile_inference(model, images, device, name="model", steps=None):
    """Profile inference time and memory usage.
    
    Args:
        model: Model to profile
        images: Input images
        device: Device to profile on
        name: Model name for logging
        steps: Number of fixed point steps (if applicable)
        
    Returns:
        Tuple: Output, elapsed time, memory usage
    """
    model.eval()
    images = images.to(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(device)
        
    with torch.no_grad():
        if steps is not None and hasattr(model, 'forward') and 'steps' in model.forward.__code__.co_varnames:
            _ = model(images, steps=steps)
        else:
            _ = model(images)
    
    start_time = time.time()
    with torch.no_grad():
        if steps is not None and hasattr(model, 'forward') and 'steps' in model.forward.__code__.co_varnames:
            output = model(images, steps=steps)
        else:
            output = model(images)
    elapsed = time.time() - start_time
    
    max_memory = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    
    print(f"[{name}] Inference time: {elapsed:.4f}s, Max GPU Memory: {max_memory / 1024 / 1024:.2f} MB")
    return output, elapsed, max_memory

def plot_training_loss(losses, title, filename, ylabel="Loss"):
    """Plot and save training loss curve.
    
    Args:
        losses: List of loss values
        title: Plot title
        filename: Filename to save plot to
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

def plot_comparison_barplot(data, labels, title, filename, ylabel):
    """Create and save bar plot comparison.
    
    Args:
        data: Data values
        labels: Bar labels
        title: Plot title
        filename: Filename to save plot to
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x=labels, y=data)
    
    for i, v in enumerate(data):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

def visualize_outputs(original, teacher_output, student_output, filename):
    """Visualize and compare model outputs.
    
    Args:
        original: Original input image
        teacher_output: Teacher model output
        student_output: Student model output
        filename: Filename to save visualization to
    """
    def tensor_to_image(tensor):
        if tensor.dim() == 4:  # batch dimension
            tensor = tensor[0]  # take first image in batch
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        if img.min() < 0:
            img = (img + 1) / 2
        return np.clip(img, 0, 1)
    
    orig_img = tensor_to_image(original)
    teacher_img = tensor_to_image(teacher_output)
    student_img = tensor_to_image(student_output)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig_img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(teacher_img)
    axes[1].set_title("Teacher Output")
    axes[1].axis('off')
    
    axes[2].imshow(student_img)
    axes[2].set_title("Student Output")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
    plt.close()
