"""
Neural network models for the NSRPP method.
"""

import torch
import torch.nn as nn
import torch.utils.data as data

class SurrogateNet(nn.Module):
    """
    Two-layer surrogate network with dropout for the NSRPP method.
    """
    def __init__(self, input_dim=1, hidden_dim=32, dropout_p=0.1):
        super(SurrogateNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class OneLayerSurrogate(nn.Module):
    """
    One-layer surrogate network for ablation study.
    """
    def __init__(self, input_dim=1):
        super(OneLayerSurrogate, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

class MultiLayerSurrogate(nn.Module):
    """
    Multi-layer surrogate network for ablation study.
    """
    def __init__(self, input_dim=1, hidden_dim=32):
        super(MultiLayerSurrogate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class PretrainedSurrogate(nn.Module):
    """
    Surrogate network with frozen pretrained layers for ablation study.
    """
    def __init__(self, input_dim=1, hidden_dim=32):
        super(PretrainedSurrogate, self).__init__()
        self.pretrained = nn.Linear(input_dim, hidden_dim)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        x = self.pretrained(x)
        return self.net(x)

class ThetaDataset(data.Dataset):
    """
    Dataset class for surrogate model training.
    """
    def __init__(self, thetas, risks):
        self.thetas = torch.tensor(thetas, dtype=torch.float32)
        self.risks = torch.tensor(risks, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.thetas)
    def __getitem__(self, idx):
        return self.thetas[idx], self.risks[idx]
