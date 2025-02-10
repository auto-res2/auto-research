import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AggMoOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.0, 0.9, 0.99], weight_decay=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(AggMoOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffers'] = [torch.zeros_like(p.data) for _ in group['betas']]

                for i, beta in enumerate(group['betas']):
                    buf = state['momentum_buffers'][i]
                    buf.mul_(beta).add_(grad)
                    p.data.add_(buf, alpha=-group['lr']/len(group['betas']))
        return loss

class MADGRADOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super(MADGRADOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_sum_sq'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                state['grad_sum_sq'].addcmul_(grad, grad, value=1)
                denom = state['grad_sum_sq'].sqrt().add_(group['eps'])
                
                s = state['s']
                s.mul_(group['momentum']).addcdiv_(grad, denom)
                p.data.add_(s, alpha=-group['lr'])
        return loss

class HybridOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.0, 0.9, 0.99], momentum=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

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

def train_model(model, trainloader, testloader, optimizer, criterion, epochs=5):
    """Train the model and track metrics."""
    model.train()
    metrics = {
        'train_losses': [],
        'test_losses': [],
        'test_accuracies': [],
        'convergence_rate': []
    }
    
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        # Calculate metrics
        avg_train_loss = epoch_loss / num_batches
        metrics['train_losses'].append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_test_loss = test_loss / len(testloader)
        test_accuracy = 100 * correct / total
        metrics['test_losses'].append(avg_test_loss)
        metrics['test_accuracies'].append(test_accuracy)
        
        # Calculate convergence rate (change in loss)
        if epoch > 0:
            conv_rate = abs(metrics['train_losses'][-1] - metrics['train_losses'][-2])
            metrics['convergence_rate'].append(conv_rate)
        
        model.train()
    
    return metrics
