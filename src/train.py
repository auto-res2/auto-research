"""
Model implementations and training for ProtoSurvPath.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.protosurvpath_config import *
from src.utils.model_components import GeneExpressionEncoder, VisionEncoder, CrossAttentionFusion, SimpleFusion
from src.utils.visualization import plot_training_loss, plot_metric

class ProtoSurvPath(nn.Module):
    """
    ProtoSurvPath: Prototype-based multimodal framework for survival prediction.
    Uses dual prototyping with cross-attention fusion.
    """
    
    def __init__(self, gene_input_dim=GENE_INPUT_DIM, image_channels=IMAGE_CHANNELS, hidden_dim=HIDDEN_DIM):
        """
        Initialize the ProtoSurvPath model.
        
        Args:
            gene_input_dim: Dimension of gene expression data
            image_channels: Number of channels in histology images
            hidden_dim: Hidden dimension for encoders
        """
        super().__init__()
        self.gene_encoder = GeneExpressionEncoder(gene_input_dim, hidden_dim)
        self.vision_encoder = VisionEncoder(image_channels, hidden_dim)
        self.fusion = CrossAttentionFusion(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)  # risk score for survival analysis
    
    def forward(self, gene, image):
        """
        Forward pass through the model.
        
        Args:
            gene: Gene expression data
            image: Histology image data
        
        Returns:
            risk: Predicted risk score
        """
        gene_feat = self.gene_encoder(gene)
        image_feat = self.vision_encoder(image)
        fused_feat = self.fusion(gene_feat, image_feat)
        risk = self.output(fused_feat)
        return risk

class BaseMethod(nn.Module):
    """
    Baseline method with simple concatenation of gene and image features.
    """
    
    def __init__(self, gene_input_dim=GENE_INPUT_DIM, image_channels=IMAGE_CHANNELS, hidden_dim=HIDDEN_DIM):
        """
        Initialize the BaseMethod model.
        
        Args:
            gene_input_dim: Dimension of gene expression data
            image_channels: Number of channels in histology images
            hidden_dim: Hidden dimension for encoders
        """
        super().__init__()
        self.gene_encoder = GeneExpressionEncoder(gene_input_dim, hidden_dim)
        self.vision_encoder = VisionEncoder(image_channels, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, gene, image):
        """
        Forward pass through the model.
        
        Args:
            gene: Gene expression data
            image: Histology image data
        
        Returns:
            risk: Predicted risk score
        """
        gene_feat = self.gene_encoder(gene)
        image_feat = self.vision_encoder(image)
        cat_feat = torch.cat([gene_feat, image_feat], dim=-1)
        feat = torch.relu(self.fc(cat_feat))
        risk = self.output(feat)
        return risk

class PANTHER(nn.Module):
    """
    PANTHER model: Alternative fusion method with predefined priors.
    """
    
    def __init__(self, gene_input_dim=GENE_INPUT_DIM, image_channels=IMAGE_CHANNELS, hidden_dim=HIDDEN_DIM):
        """
        Initialize the PANTHER model.
        
        Args:
            gene_input_dim: Dimension of gene expression data
            image_channels: Number of channels in histology images
            hidden_dim: Hidden dimension for encoders
        """
        super().__init__()
        self.gene_encoder = GeneExpressionEncoder(gene_input_dim, hidden_dim)
        self.vision_encoder = VisionEncoder(image_channels, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, gene, image):
        """
        Forward pass through the model.
        
        Args:
            gene: Gene expression data
            image: Histology image data
        
        Returns:
            risk: Predicted risk score
        """
        gene_feat = self.gene_encoder(gene)
        image_feat = self.vision_encoder(image)
        cat_feat = torch.cat([gene_feat, image_feat], dim=-1)
        feat = torch.relu(self.fc1(cat_feat))
        feat = torch.relu(self.fc2(feat))
        risk = self.output(feat)
        return risk

class ProtoSurvPathSimpleFusion(nn.Module):
    """
    ProtoSurvPath variant with simple fusion (for ablation studies).
    """
    
    def __init__(self, gene_input_dim=GENE_INPUT_DIM, image_channels=IMAGE_CHANNELS, hidden_dim=HIDDEN_DIM):
        """
        Initialize the ProtoSurvPathSimpleFusion model.
        
        Args:
            gene_input_dim: Dimension of gene expression data
            image_channels: Number of channels in histology images
            hidden_dim: Hidden dimension for encoders
        """
        super().__init__()
        self.gene_encoder = GeneExpressionEncoder(gene_input_dim, hidden_dim)
        self.vision_encoder = VisionEncoder(image_channels, hidden_dim)
        self.fusion = SimpleFusion(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, gene, image):
        """
        Forward pass through the model.
        
        Args:
            gene: Gene expression data
            image: Histology image data
        
        Returns:
            risk: Predicted risk score
        """
        gene_feat = self.gene_encoder(gene)
        image_feat = self.vision_encoder(image)
        fused_feat = self.fusion(gene_feat, image_feat)
        risk = self.output(fused_feat)
        return risk

def dummy_survival_loss(risk, time=None, event=None):
    """
    Dummy survival loss function.
    In a real implementation, this would be replaced with a proper
    survival loss like negative Cox partial log-likelihood.
    
    Args:
        risk: Predicted risk scores
        time: Survival times (unused in dummy loss)
        event: Event indicators (unused in dummy loss)
    
    Returns:
        loss: Mean risk score as dummy loss
    """
    return risk.mean()

def train_one_epoch(model, dataloader, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for parameter updates
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        risk = model(batch['gene'], batch['image'])
        loss = dummy_survival_loss(risk, batch['time'], batch['event'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_model(model, dataloader):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
    
    Returns:
        c_index: Concordance index (dummy implementation)
    """
    model.eval()
    risks = []
    events = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            risk = model(batch['gene'], batch['image'])
            risks.extend(risk.squeeze().tolist())
            events.extend(batch['event'])
    
    risks = np.array(risks)
    events = np.array(events)
    
    c_index = np.mean(risks > np.median(risks))
    return c_index

def train_model(model, train_loader, test_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, model_name="ProtoSurvPath"):
    """
    Train and evaluate a model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epochs: Number of training epochs
        lr: Learning rate for optimizer
        model_name: Name of the model for saving and logging
    
    Returns:
        model: Trained model
        loss_history: List of training losses per epoch
        metric_history: List of evaluation metrics per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    metric_history = []
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print(f"Training {model_name} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer)
        
        metric = evaluate_model(model, test_loader)
        
        loss_history.append(loss)
        metric_history.append(metric)
        
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}: Loss = {loss:.4f}, c-index = {metric:.4f}")
        
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f"models/{model_name}_epoch_{epoch+1}.pt")
    
    plot_training_loss(loss_history, title=f"{model_name} Training Loss", condition=model_name.lower())
    plot_metric(metric_history, title=f"{model_name} c-index", condition=model_name.lower())
    
    return model, loss_history, metric_history

def cross_validation_experiment(ModelClass, model_name="ProtoSurvPath", epochs=NUM_EPOCHS, hidden_dim=HIDDEN_DIM, quick_test=QUICK_TEST):
    """
    Run a cross-validation experiment for a model.
    
    Args:
        ModelClass: Model class to instantiate
        model_name: Name of the model for saving and logging
        epochs: Number of training epochs
        hidden_dim: Hidden dimension for encoders
        quick_test: Flag for quick test with reduced dataset
    
    Returns:
        loss_history: List of training losses per epoch
        metric_history: List of evaluation metrics per epoch
        model: Trained model
        test_loader: DataLoader for test data
    """
    from src.preprocess import load_or_generate_data
    
    global QUICK_TEST
    QUICK_TEST = quick_test
    
    train_loader, test_loader = load_or_generate_data()
    
    model = ModelClass(gene_input_dim=GENE_INPUT_DIM, image_channels=IMAGE_CHANNELS, hidden_dim=hidden_dim)
    
    _, loss_history, metric_history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=epochs,
        lr=LEARNING_RATE,
        model_name=model_name
    )
    
    return loss_history, metric_history, model, test_loader

if __name__ == "__main__":
    from src.preprocess import load_or_generate_data
    
    QUICK_TEST = True
    
    train_loader, test_loader = load_or_generate_data()
    
    model = ProtoSurvPath()
    
    train_model(model, train_loader, test_loader, num_epochs=2)
