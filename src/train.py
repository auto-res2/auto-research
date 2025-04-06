"""
Training module for the STEM method.
Defines model architectures and training functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.preprocess import extract_embeddings

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class FeatureExtractor(nn.Module):
    """Simple fully connected layer with ReLU for feature projection."""
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.fc(x))

class STEMModel(nn.Module):
    """
    STEM Model: Uses subtractive mixture mechanism.
    
    The model represents document density as the square of a (potentially subtractive) mixture:
    f(x) = [ (1 – α) gₕ(x) – α gₐᵢ(x) ]²
    """
    def __init__(self, hidden_dim):
        super(STEMModel, self).__init__()
        self.g_h = FeatureExtractor(768, hidden_dim)  # 768 from roberta-base embeddings
        self.g_ai = FeatureExtractor(768, hidden_dim)
    
    def forward(self, embeddings, alpha):
        human_features = self.g_h(embeddings)
        ai_features = self.g_ai(embeddings)
        mixed = ((1 - alpha) * human_features - alpha * ai_features) ** 2
        out = mixed.mean(dim=1)
        return out

class AdditiveModel(nn.Module):
    """
    Base Additive Model: Uses additive mechanism for comparison.
    """
    def __init__(self, hidden_dim):
        super(AdditiveModel, self).__init__()
        self.g_h = FeatureExtractor(768, hidden_dim)
        self.g_ai = FeatureExtractor(768, hidden_dim)

    def forward(self, embeddings, alpha):
        human_features = self.g_h(embeddings)
        ai_features = self.g_ai(embeddings)
        mixed = (1 - alpha) * human_features + alpha * ai_features
        out = mixed.mean(dim=1)
        return out

class STEMWithMetadataModel(STEMModel):
    """
    STEM model with metadata auxiliary learning.
    """
    def __init__(self, hidden_dim, metadata_dim):
        super(STEMWithMetadataModel, self).__init__(hidden_dim)
        self.metadata_predictor = nn.Linear(hidden_dim, metadata_dim)
    
    def forward(self, embeddings, alpha):
        human_features = self.g_h(embeddings)
        ai_features = self.g_ai(embeddings)
        mixed = ((1 - alpha) * human_features - alpha * ai_features) ** 2
        feature_repr = human_features  # dummy representation
        output = mixed.mean(dim=1)
        metadata_pred = self.metadata_predictor(feature_repr)
        return output, metadata_pred

class AdditiveWithMetadataModel(AdditiveModel):
    """
    Additive Model with Auxiliary Metadata Loss.
    """
    def __init__(self, hidden_dim, metadata_dim):
        super(AdditiveWithMetadataModel, self).__init__(hidden_dim)
        self.metadata_predictor = nn.Linear(hidden_dim, metadata_dim)
        
    def forward(self, embeddings, alpha):
        mixed_feature = super().forward(embeddings, alpha)
        metadata_pred = self.metadata_predictor(mixed_feature.unsqueeze(1))
        return mixed_feature, metadata_pred

def train_model(model, data, num_epochs=5, lr=1e-3, use_metadata=False):
    """
    Training loop for a given model over synthetic data.
    data: list of tuples.
      - For standard experiment: (text, true_alpha)
      - For metadata experiment: (text, true_alpha, true_metadata)
    Returns loss_history (list of loss values per epoch).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    lambda_aux = 0.5  # weight for metadata auxiliary loss if used
    loss_history = []

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_losses = []
        for sample in data:
            optimizer.zero_grad()
            if use_metadata:
                text, true_alpha, true_metadata = sample
            else:
                text, true_alpha = sample

            embeddings = extract_embeddings([text])  # batch size 1
            alpha_tensor = true_alpha.view(-1, 1)
            if use_metadata and hasattr(model, 'metadata_predictor'):
                output, metadata_pred = model(embeddings, alpha_tensor)
                primary_loss = criterion(output, torch.zeros_like(output))
                aux_loss = criterion(metadata_pred, true_metadata)
                loss = primary_loss + lambda_aux * aux_loss
            else:
                output = model(embeddings, alpha_tensor)
                loss = criterion(output, torch.zeros_like(output))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    return loss_history
