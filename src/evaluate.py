import torch
import numpy as np
from typing import Dict

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    mse = torch.nn.functional.mse_loss(pred * mask, target * mask).item()
    mae = torch.nn.functional.l1_loss(pred * mask, target * mask).item()
    
    relative_error = torch.norm(pred * mask - target * mask) / torch.norm(target * mask)
    relative_error = relative_error.item()
    
    return {
        "MSE": mse,
        "MAE": mae,
        "Relative_Error": relative_error
    }

def evaluate_model(model: torch.nn.Module, test_data: torch.Tensor, 
                  test_mask: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(test_data, test_mask)
        metrics = compute_metrics(pred, test_data, test_mask)
    return metrics
