import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
from tqdm import tqdm
from torch.optim import SGD, Adam

# Import our custom optimizer
from utils.optimizers import HybridOptimizer

# Import MADGRAD and AggMo optimizers
try:
    from madgrad import MADGRAD
except ImportError:
    # Implement MADGRAD if not available
    class MADGRAD(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0, eps=1e-6):
            defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
            super(MADGRAD, self).__init__(params, defaults)
            
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
                
            for group in self.param_groups:
                lr, momentum, weight_decay, eps = group['lr'], group['momentum'], group['weight_decay'], group['eps']
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad.data
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state['sum'] = torch.zeros_like(p.data)
                        state['grad_sum_sq'] = torch.zeros_like(p.data)
                        state['s'] = torch.zeros_like(p.data)
                        state['x0'] = torch.clone(p.data)
                        state['step'] = 0
                    
                    state['step'] += 1
                    step = state['step']
                    
                    if weight_decay != 0:
                        grad.add_(p.data, alpha=weight_decay)
                    
                    # Update gradient sum
                    state['sum'].add_(grad)
                    
                    # Update gradient sum squared
                    state['grad_sum_sq'].addcmul_(grad, grad, value=1.0)
                    
                    # Compute update
                    sqrt_norm = state['grad_sum_sq'].sqrt().add_(eps)
                    update = state['sum'] / sqrt_norm
                    
                    # Update parameters
                    p.data.copy_(state['x0'] - lr * update)
                    
            return loss

try:
    from aggmo import AggMo
except ImportError:
    # Implement AggMo if not available
    class AggMo(torch.optim.Optimizer):
        def __init__(self, params, lr=0.1, betas=[0.0, 0.9, 0.99], weight_decay=0):
            defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
            super(AggMo, self).__init__(params, defaults)
            
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
                
            for group in self.param_groups:
                lr, betas, weight_decay = group['lr'], group['betas'], group['weight_decay']
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad.data
                    state = self.state[p]
                    
                    # Initialize state
                    if len(state) == 0:
                        state['momentum_buffers'] = [torch.zeros_like(p.data) for _ in range(len(betas))]
                    
                    momentum_buffers = state['momentum_buffers']
                    
                    if weight_decay != 0:
                        grad.add_(p.data, alpha=weight_decay)
                    
                    # Update momentum buffers
                    for i, beta in enumerate(betas):
                        momentum_buffers[i].mul_(beta).add_(grad)
                    
                    # Compute update
                    update = torch.zeros_like(p.data)
                    for buf in momentum_buffers:
                        update.add_(buf)
                    update.div_(len(betas))
                    
                    # Update parameters
                    p.data.add_(update, alpha=-lr)
                    
            return loss

# CNN model for CIFAR-10
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# LSTM model for PTB
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.init_weights()
        
    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        
    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.fc(output.view(-1, self.hidden_dim))
        return decoded, hidden
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

def train_cifar10(model, train_loader, test_loader, optimizer_name, lr=0.01, epochs=10, device='cuda'):
    """
    Train a model on CIFAR-10 dataset.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer_name: Name of the optimizer to use
        lr: Learning rate
        epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        dict: Training history with metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer based on name
    if optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'madgrad':
        optimizer = MADGRAD(model.parameters(), lr=lr)
    elif optimizer_name == 'aggmo':
        optimizer = AggMo(model.parameters(), lr=lr, betas=[0.0, 0.9, 0.99])
    elif optimizer_name == 'hybrid':
        optimizer = HybridOptimizer(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss/(batch_idx+1), 
                'acc': 100.*correct/total
            })
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate test metrics
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {train_loss:.4f} - train_acc: {train_acc:.2f}% - test_loss: {test_loss:.4f} - test_acc: {test_acc:.2f}%")
    
    return history

def train_ptb(model, train_loader, valid_loader, optimizer_name, lr=20, epochs=10, clip=0.25, device='cuda'):
    """
    Train a language model on PTB dataset.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        optimizer_name: Name of the optimizer to use
        lr: Learning rate
        epochs: Number of training epochs
        clip: Gradient clipping value
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        dict: Training history with metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer based on name
    if optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'madgrad':
        optimizer = MADGRAD(model.parameters(), lr=lr)
    elif optimizer_name == 'aggmo':
        optimizer = AggMo(model.parameters(), lr=lr, betas=[0.0, 0.9, 0.99])
    elif optimizer_name == 'hybrid':
        optimizer = HybridOptimizer(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_ppl': [],
        'valid_loss': [],
        'valid_ppl': [],
        'epoch_times': []
    }
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        hidden = model.init_hidden(train_loader.batch_size, device)
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            # Detach hidden states
            hidden = tuple([h.detach() for h in hidden])
            
            model.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            ppl = math.exp(total_loss / (batch_idx + 1))
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'ppl': ppl
            })
        
        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(train_loss)
        
        # Evaluate on validation set
        valid_loss = evaluate_ptb(model, valid_loader, criterion, device)
        valid_ppl = math.exp(valid_loss)
        epoch_time = time.time() - start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['valid_loss'].append(valid_loss)
        history['valid_ppl'].append(valid_ppl)
        history['epoch_times'].append(epoch_time)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {train_loss:.4f} - train_ppl: {train_ppl:.2f} - valid_loss: {valid_loss:.4f} - valid_ppl: {valid_ppl:.2f}")
    
    return history

def evaluate_ptb(model, data_loader, criterion, device):
    """Evaluate the model on the given data loader"""
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(data_loader.batch_size, device)
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            hidden = tuple([h.detach() for h in hidden])
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)
