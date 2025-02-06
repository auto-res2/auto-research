import torch
from preprocess import generate_synthetic_data
from train import IntegratedMemoryLRMC, train_model
from evaluate import evaluate_model
import json
import os

def main():
    # Experiment parameters
    d1, d2 = 100, 100
    rank = 10
    max_iter = 50
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Generating synthetic data...")
    train_data, train_mask = generate_synthetic_data(d1, d2, rank)
    test_data, test_mask = generate_synthetic_data(d1, d2, rank)
    
    print("Initializing model...")
    model = IntegratedMemoryLRMC(d1, d2, rank, max_iter).to(device)
    
    print("Training model...")
    model = train_model(model, train_data, train_mask, num_epochs)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, test_data, test_mask)
    
    print("\nFinal Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/integrated_memory_lrmc.pt")

if __name__ == "__main__":
    main()
