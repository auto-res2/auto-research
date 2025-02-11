import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any
import yaml

from utils.optimizers import HybridOptimizer
from utils.models import SimpleCNN, SimpleRNN

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    experiment_name: str
) -> Dict[str, float]:
    """Train model with specified optimizer and configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer
    optimizer_config = config['optimizer']
    if optimizer_config['name'] == 'hybrid':
        optimizer = HybridOptimizer(
            model.parameters(),
            lr=optimizer_config['lr'],
            betas=optimizer_config['betas'],
            momentum=optimizer_config['momentum'],
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config['momentum'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['name'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(f'logs/{experiment_name}')
    
    best_val_loss = float('inf')
    metrics = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                writer.add_scalar('train/batch_loss', loss.item(), 
                                epoch * len(train_loader) + batch_idx)
        
        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_acc'].append(val_accuracy)
        
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/accuracy', val_accuracy, epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 
                      Path('models') / f'{experiment_name}_best.pt')
        
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]} - '
              f'Train Loss: {avg_train_loss:.4f} - '
              f'Val Loss: {avg_val_loss:.4f} - '
              f'Val Acc: {val_accuracy:.2f}%')
    
    writer.close()
    
    # Return final metrics values
    return {
        'final_train_loss': metrics['train_loss'][-1],
        'final_val_loss': metrics['val_loss'][-1],
        'final_val_acc': metrics['val_acc'][-1],
        'best_val_loss': best_val_loss
    }
