import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import numpy as np

def evaluate_cifar10(model: nn.Module,
                    test_loader: DataLoader,
                    device: torch.device) -> Dict[str, float]:
    """Evaluate model on CIFAR-10 test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        'test_loss': test_loss / len(test_loader),
        'test_accuracy': 100. * correct / total
    }

def evaluate_ptb(model: nn.Module,
                val_data: torch.Tensor,
                device: torch.device,
                batch_size: int = 20,
                bptt: int = 35) -> Dict[str, float]:
    """Evaluate model on PTB validation set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.
    hidden = model.init_hidden(batch_size)
    
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target
    
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, bptt):
            data, targets = get_batch(val_data, i)
            data, targets = data.to(device), targets.to(device)
            
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, output.size(-1)), targets)
            total_loss += loss.item()
    
    val_loss = total_loss / (len(val_data) // bptt)
    return {
        'val_loss': val_loss,
        'val_perplexity': np.exp(val_loss)
    }
