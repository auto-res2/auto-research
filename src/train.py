import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm
from models.hybrid_optimizer import HybridOptimizer

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                config: Dict) -> None:
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
