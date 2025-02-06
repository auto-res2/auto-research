import torch
from preprocess import get_dataloaders
from models.learnable_gated_pooling import LearnableGatedPooling
from train import train_model
from evaluate import evaluate_metrics

def main():
    # Set parameters
    input_dim = 768
    seq_len = 10
    batch_size = 32
    num_epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)
    
    print("Initializing model...")
    model = LearnableGatedPooling(input_dim, seq_len)
    
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs, device)
    
    print("\nEvaluating model...")
    metrics = evaluate_metrics(trained_model, test_loader, device)
    
    print("\nFinal Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
