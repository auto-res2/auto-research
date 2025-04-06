import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def check_gpu_memory():
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: Allocated: {gpu_mem_alloc:.2f} GB, Reserved: {gpu_mem_reserved:.2f} GB")
        if gpu_mem_reserved > 14.0:  # Keep below 16GB for Tesla T4
            print("WARNING: GPU memory usage is high, consider reducing batch size or model size.")
    else:
        print("GPU not available, using CPU.")

def get_classifier_model(num_classes=4):
    """Create a ResNet18 model for classification."""
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    return model.to(DEVICE)

def train_model(model, loader, num_epochs=3, output_dir="logs"):
    """Train a model on the provided data loader."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    loss_history = []
    accuracy_history = []
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(loader)
        accuracy = 100 * correct / total
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")
        
        if epoch % 2 == 0:
            check_gpu_memory()
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "classifier_model.pth"))
    
    return model, loss_history, accuracy_history

def plot_training_curves(loss_history, accuracy_history=None, title="Training Curves", filename="training_curves.pdf"):
    """Plot and save training loss and accuracy curves."""
    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, 'b-', marker='o', label='Training Loss')
    
    if accuracy_history:
        plt.plot(epochs, accuracy_history, 'r-', marker='s', label='Training Accuracy')
        plt.ylabel('Loss / Accuracy (%)')
    else:
        plt.ylabel('Loss')
        
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig(os.path.join("logs", filename), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved as '{os.path.join('logs', filename)}'")
