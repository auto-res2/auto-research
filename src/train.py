import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
from model import LearnableGatedPooling
from preprocess import load_and_preprocess_data

def train_model(config: Dict[str, Any]) -> LearnableGatedPooling:
    model = LearnableGatedPooling(
        input_dim=config['model']['input_dim'],
        seq_len=config['model']['seq_len']
    )
    
    X, y = load_and_preprocess_data(Path('data/train.txt'))
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(
        dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(config['model']['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    return model
