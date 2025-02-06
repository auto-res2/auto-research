import yaml
import torch
from preprocess import create_dataloaders
from train import train_model
from evaluate import evaluate_model
from models.hybrid_optimizer import HybridOptimizer

def main():
    # Load configuration
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config)
    
    # Initialize model
    model = HybridOptimizer(
        d1=config['model']['d1'],
        d2=config['model']['d2'],
        r=config['model']['r'],
        alpha=config['model']['alpha'],
        p=config['model']['p'],
        max_iter=config['model']['max_iter'],
        zeta0=config['model']['zeta0'],
        eta0=config['model']['eta0'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train model
    train_model(model, train_loader, config)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader)
    print(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main()
