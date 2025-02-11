import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import time
from utils.optimizers import HybridOptimizer
from madgrad import MADGRAD
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RNNLM(nn.Module):
    def __init__(self, ntoken: int, ninp: int, nhid: int, nlayers: int):
        super(RNNLM, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=0.5)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

def get_optimizer(name: str, model_params, **kwargs) -> torch.optim.Optimizer:
    """Get optimizer by name."""
    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'madgrad': MADGRAD,
        'hybrid': HybridOptimizer
    }
    return optimizers[name.lower()](model_params, **kwargs)

def train_cifar10(model: nn.Module,
                  train_loader: DataLoader,
                  optimizer_name: str,
                  device: torch.device,
                  epochs: int = 10,
                  **optim_kwargs) -> Dict[str, Any]:
    """Train model on CIFAR-10."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), **optim_kwargs)
    
    metrics = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)
        
    return metrics

def train_ptb(model: nn.Module,
              train_data: torch.Tensor,
              optimizer_name: str,
              device: torch.device,
              batch_size: int = 20,
              bptt: int = 35,
              epochs: int = 10,
              **optim_kwargs) -> Dict[str, Any]:
    """Train model on PTB."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), **optim_kwargs)
    
    metrics = {'train_loss': [], 'train_ppl': []}
    
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        hidden = model.init_hidden(batch_size)
        
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            data, targets = data.to(device), targets.to(device)
            
            hidden = tuple(h.detach() for h in hidden)
            model.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, output.size(-1)), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_loss = total_loss / (len(train_data) // bptt)
        epoch_ppl = np.exp(epoch_loss)
        metrics['train_loss'].append(epoch_loss)
        metrics['train_ppl'].append(epoch_ppl)
        
    return metrics
