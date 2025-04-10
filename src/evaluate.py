"""
Evaluation module for DITTO-GSD experiments.
"""
import time
import numpy as np
import torch

try:
    from pytorch3d.loss import chamfer_distance
except ImportError:
    print("pytorch3d not available, using custom chamfer distance implementation")
    
    def chamfer_distance(x, y):
        """
        Custom implementation of chamfer distance without pytorch3d.
        
        Args:
            x: First point cloud (B, N, 3)
            y: Second point cloud (B, N, 3)
            
        Returns:
            Tuple of (loss, _)
        """
        x_expanded = x.unsqueeze(2)  # (B, N, 1, 3)
        y_expanded = y.unsqueeze(1)  # (B, 1, N, 3)
        
        dist = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)  # (B, N, N)
        
        x_to_y = torch.min(dist, dim=2)[0]  # (B, N)
        y_to_x = torch.min(dist, dim=1)[0]  # (B, N)
        
        loss = torch.mean(x_to_y) + torch.mean(y_to_x)
        
        return loss, None

def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: Dataloader for evaluation data.
        device: Device to use for evaluation.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    model = model.to(device)
    model.eval()
    
    losses = []
    chamfer_scores = []
    
    start_time = time.time()
    with torch.no_grad():
        for pc, gt in dataloader:
            pc = pc.to(device)
            gt = gt.to(device)
            
            pred = model(pc)
            
            chamfer_loss, _ = chamfer_distance(pred.unsqueeze(0), gt.unsqueeze(0))
            
            losses.append(chamfer_loss.item())
            chamfer_scores.append(chamfer_loss.item())
    
    elapsed_time = time.time() - start_time
    
    metrics = {
        'mean_loss': np.mean(losses),
        'mean_chamfer': np.mean(chamfer_scores),
        'runtime': elapsed_time
    }
    
    return metrics

def evaluate_single_sample(model, pc_np, device='cuda'):
    """
    Evaluate a single point cloud sample.
    
    Args:
        model: Model to evaluate.
        pc_np: Input point cloud (numpy array).
        device: Device to use for evaluation.
        
    Returns:
        Predicted point cloud as numpy array.
    """
    model = model.to(device)
    model.eval()
    
    pc_tensor = torch.tensor(pc_np, dtype=torch.float32).to(device).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(pc_tensor)
    
    return pred.cpu().numpy()

def run_experiment(model, dataloader, device='cuda'):
    """
    Run reconstruction experiment and compute metrics.
    
    Args:
        model: Model to evaluate.
        dataloader: Dataloader for evaluation.
        device: Device to use for evaluation.
        
    Returns:
        Metrics dictionary.
    """
    return evaluate_model(model, dataloader, device)
