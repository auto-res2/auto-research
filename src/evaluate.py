import torch
import torch.nn.functional as F

def evaluate_model(model, testloader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    
    return {
        'test_loss': avg_loss,
        'test_accuracy': accuracy
    }
