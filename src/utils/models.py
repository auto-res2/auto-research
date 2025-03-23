"""Model definitions for the ACM optimizer experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNCIFAR10(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, config):
        super(SimpleCNNCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, config['conv1_channels'], 5, 1)
        self.bn1 = nn.BatchNorm2d(config['conv1_channels'])
        self.conv2 = nn.Conv2d(config['conv1_channels'], config['conv2_channels'], 5, 1)
        self.bn2 = nn.BatchNorm2d(config['conv2_channels'])
        self.fc1 = nn.Linear(config['conv2_channels'] * 5 * 5, config['fc1_size'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.fc2 = nn.Linear(config['fc1_size'], 10)
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn2(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleCNNMNIST(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self, config):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, config['conv1_channels'], 3, 1)
        self.conv2 = nn.Conv2d(config['conv1_channels'], config['conv2_channels'], 3, 1)
        self.dropout1 = nn.Dropout2d(config['dropout_rate'])
        self.dropout2 = nn.Dropout2d(config['dropout_rate'])
        self.fc1 = nn.Linear(config['conv2_channels'] * 12 * 12, config['fc1_size'])
        self.fc2 = nn.Linear(config['fc1_size'], 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
