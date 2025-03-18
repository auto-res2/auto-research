import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy
import matplotlib.pyplot as plt
from src.utils.optimizer import ACMOptimizer

def train_cifar10_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=2, device='cuda'):
    """
    Train a model on CIFAR10 dataset
    
    Args:
        model (nn.Module): PyTorch model to train
        dataloaders (dict): Dictionary containing train and validation data loaders
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        num_epochs (int): Number of epochs to train
        device (str): Device to use for training ('cuda' or 'cpu')
        
    Returns:
        tuple: (trained_model, history) containing the trained model and training history
    """
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total_examples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total_examples += inputs.size(0)

            epoch_loss = running_loss / total_examples
            epoch_acc = correct.double() / total_examples
            print(f"{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_acc'].append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_model_wts)
    return model, history

def optimize_quadratic(x, A, b, optimizer, num_iters=200):
    """
    Optimize a quadratic function f(x) = 0.5 x^T A x + b^T x
    
    Args:
        x (Tensor): Parameter vector to optimize
        A (Tensor): Quadratic term matrix
        b (Tensor): Linear term vector
        optimizer: Optimizer to use
        num_iters (int): Number of iterations
        
    Returns:
        tuple: (losses, momentum_norms) containing loss history and momentum norm history
    """
    def quadratic_loss(x, A, b):
        return 0.5 * torch.matmul(x.unsqueeze(0), torch.matmul(A, x.unsqueeze(1))).squeeze() + torch.dot(b, x)
    
    losses = []
    momentum_norms = []
    
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = quadratic_loss(x, A, b)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # Record momentum norm if available
        state = optimizer.state[x]
        if 'momentum_buffer' in state:
            momentum_norms.append(state['momentum_buffer'].norm().item())
        else:
            momentum_norms.append(0)
            
    return losses, momentum_norms

def train_transformer_epoch(model, optimizer, scheduler, data_source, criterion, vocab_size, device='cuda', bptt=35, use_quick_test=False):
    """
    Train a transformer model for one epoch on language modeling task
    
    Args:
        model (nn.Module): Transformer model
        optimizer: Optimizer to use
        scheduler: Learning rate scheduler
        data_source (Tensor): Training data
        criterion: Loss function
        vocab_size (int): Size of vocabulary
        device (str): Device to use for training
        bptt (int): Sequence length for language modeling
        use_quick_test (bool): If True, limit the number of iterations
        
    Returns:
        float: Total loss for the epoch
    """
    model.train()
    total_loss = 0.0
    ntokens = vocab_size
    iter_steps = 0
    
    # Get batch function for language modeling
    def get_batch(source, i, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target
    
    # For quick test, limit iterations; otherwise full epoch
    max_iter = 5 if use_quick_test else (data_source.size(0) - 1) // bptt
    
    for i in range(0, data_source.size(0) - 1, bptt):
        if iter_steps >= max_iter:
            break
            
        data, targets = get_batch(data_source, i, bptt)
        data = data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        iter_steps += 1
        print(f"Iteration {iter_steps}: Loss {loss.item():.3f}")
        
    scheduler.step()
    return total_loss

def evaluate_transformer(model, data_source, criterion, vocab_size, device='cuda', bptt=35):
    """
    Evaluate a transformer model on language modeling task
    
    Args:
        model (nn.Module): Transformer model
        data_source (Tensor): Validation or test data
        criterion: Loss function
        vocab_size (int): Size of vocabulary
        device (str): Device to use for evaluation
        bptt (int): Sequence length for language modeling
        
    Returns:
        float: Total loss on the dataset
    """
    model.eval()
    total_loss = 0.0
    ntokens = vocab_size
    
    # Get batch function for language modeling
    def get_batch(source, i, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            total_loss += loss.item()
            
    return total_loss
