import torch
import torch.nn as nn
from torch.optim import Adam
from models.learnable_gated_pooling import LearnableGatedPooling

def train_model(model, train_loader, val_loader, num_epochs=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.float().to(device)
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)
