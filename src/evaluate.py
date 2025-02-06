import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm

def evaluate_model(model: nn.Module,
                  test_loader: torch.utils.data.DataLoader) -> Dict:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            output = model(batch)
            loss = criterion(output, batch)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return {'test_loss': avg_loss}
