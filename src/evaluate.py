import torch
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import LearnableGatedPooling

def evaluate_model(
    model: LearnableGatedPooling,
    test_data: Tuple[torch.Tensor, torch.Tensor]
) -> Dict[str, float]:
    model.eval()
    X_test, y_test = test_data
    
    with torch.no_grad():
        predictions, gate_values = model(X_test)
        pred_labels = torch.argmax(predictions, dim=1)
    
    accuracy = accuracy_score(y_test.numpy(), pred_labels.numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test.numpy(),
        pred_labels.numpy(),
        average='weighted'
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'avg_gate_value': float(gate_values.mean().item())
    }
