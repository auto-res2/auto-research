import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import List, Dict
import math

class HybridOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.0, 0.9, 0.99], momentum=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super(HybridOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, betas, momentum, eps, weight_decay = (
                group['lr'], group['betas'], group['momentum'], 
                group['eps'], group['weight_decay']
            )
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                    
                param_state = self.state[p]
                
                if len(param_state) == 0:
                    param_state['momentum_buffers'] = {
                        beta: torch.zeros_like(p.data) 
                        for beta in betas
                    }
                    param_state['dual_avg_buffer'] = torch.zeros_like(p.data)
                
                avg_updates = torch.zeros_like(p.data)
                
                for beta in betas:
                    buf = param_state['momentum_buffers'][beta]
                    buf.mul_(beta).add_(d_p)
                    avg_updates.add_(buf)
                
                dual_avg_buffer = param_state['dual_avg_buffer']
                dual_avg_buffer.add_(d_p)
                
                p.data.sub_(
                    lr * avg_updates / len(betas) + 
                    dual_avg_buffer * eps
                )
                
        return loss

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, trainloader, optimizer, criterion, device, epochs=10):
    """Train the model and return training history."""
    model.train()
    history = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        history.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_acc})
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%')
    
    return history
