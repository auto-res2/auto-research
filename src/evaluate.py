import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloader, criterion=None):
    """Evaluate a model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    results = {
        'accuracy': accuracy * 100,  # Convert to percentage
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    if criterion:
        results['loss'] = running_loss / len(dataloader)
    
    return results

def plot_confusion_matrix(labels, predictions, num_classes, class_names=None, filename="confusion_matrix.pdf"):
    """Plot and save a confusion matrix."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig(os.path.join("logs", filename), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved as '{os.path.join('logs', filename)}'")

def plot_feature_discrepancy(strengths, discrepancies, filename="feature_discrepancy.pdf"):
    """Plot and save feature discrepancy vs. diffusion strength."""
    plt.figure(figsize=(10, 6))
    plt.plot(strengths, discrepancies, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Diffusion Strength')
    plt.ylabel('Average Feature Discrepancy')
    plt.title('Effect of Diffusion Strength on Feature Discrepancy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(strengths)
    plt.tight_layout()
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig(os.path.join("logs", filename), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature discrepancy plot saved as '{os.path.join('logs', filename)}'")

def get_feature_extractor():
    """Get a pretrained ResNet18 model for feature extraction."""
    feature_extractor = models.resnet18(pretrained=True).eval()
    feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
    feature_extractor = feature_extractor.to(DEVICE)
    return feature_extractor

def extract_features(image, feature_extractor, preprocess=None):
    """Extract features from an image using a feature extractor."""
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    return features.squeeze().cpu().numpy()

def print_classification_report(labels, predictions, class_names=None):
    """Print a classification report."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(max(max(labels), max(predictions)) + 1)]
    
    report = classification_report(labels, predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)
