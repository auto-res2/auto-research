import torch

def evaluate_model(model, testloader, criterion):
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(testloader)
    
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Average test loss: {avg_loss:.3f}')
    
    return accuracy, avg_loss
