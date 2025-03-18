import torch
import matplotlib.pyplot as plt
import os

def evaluate_cifar10_model(model, dataloader, criterion, device='cuda'):
    """
    Evaluate a model on CIFAR10 dataset
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        dataloader: Data loader for evaluation
        criterion: Loss function
        device (str): Device to use for evaluation
        
    Returns:
        tuple: (loss, accuracy) on the dataset
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total_examples += inputs.size(0)
    
    loss = running_loss / total_examples
    accuracy = correct.double() / total_examples
    
    return loss, accuracy.item()

def plot_quadratic_convergence(results, save_path=None):
    """
    Plot convergence curves for quadratic optimization
    
    Args:
        results (dict): Dictionary containing loss histories for different optimizers
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 4))
    for name, res in results.items():
        plt.plot(res['losses'], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Convergence on the Quadratic Objective")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved quadratic convergence plot as '{save_path}'.")
    else:
        plt.show()

def plot_momentum_norm(momentum_norms, name='ACM', save_path=None):
    """
    Plot momentum norm evolution
    
    Args:
        momentum_norms (list): List of momentum norm values
        name (str): Name of the optimizer
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 4))
    plt.plot(momentum_norms, label=f'{name} Momentum Norm')
    plt.xlabel("Iteration")
    plt.ylabel("Momentum Norm")
    plt.title(f"Momentum Evolution in {name}")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved momentum norm plot as '{save_path}'.")
    else:
        plt.show()

def calculate_perplexity(loss):
    """
    Calculate perplexity from loss for language modeling
    
    Args:
        loss (float): Cross-entropy loss
        
    Returns:
        float: Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()
