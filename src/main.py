import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import time
import copy
import os
import sys

from src.utils.optimizer import ACMOptimizer
from src.preprocess import load_cifar10, prepare_synthetic_quadratic, load_wikitext2
from src.train import (
    train_cifar10_model, optimize_quadratic, 
    train_transformer_epoch, evaluate_transformer
)
from src.evaluate import (
    evaluate_cifar10_model, plot_quadratic_convergence, 
    plot_momentum_norm, calculate_perplexity
)

# Import configurations
from config.cifar10_config import *
from config.quadratic_config import *
from config.transformer_config import *

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

def experiment_cifar10(use_quick_test=False):
    """
    Experiment 1: CIFAR10 Classification using ResNet18
    
    Args:
        use_quick_test (bool): If True, use a small subset of data and fewer epochs
        
    Returns:
        dict: Training history
    """
    print("\n=== Experiment 1: CIFAR10 Classification using ResNet18 ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    dataloaders = load_cifar10(batch_size=BATCH_SIZE, use_quick_test=use_quick_test)
    
    # Initialize ResNet18 model
    model = models.resnet18(num_classes=NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate ACM optimizer
    optimizer = ACMOptimizer(
        model.parameters(), 
        lr=LEARNING_RATE, 
        beta=MOMENTUM_BETA, 
        curvature_influence=CURVATURE_INFLUENCE, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    # Set number of epochs based on quick test flag
    num_epochs = QUICK_TEST_EPOCHS if use_quick_test else NUM_EPOCHS
    
    # Train model
    start_time = time.time()
    model_trained, history = train_cifar10_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        num_epochs=num_epochs, device=device
    )
    elapsed_time = time.time() - start_time
    
    print(f"Training finished. Total time (seconds): {elapsed_time:.2f}")
    
    # Evaluate model
    val_loss, val_acc = evaluate_cifar10_model(model_trained, dataloaders['val'], criterion, device)
    print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Save model
    torch.save(model_trained.state_dict(), 'models/cifar10_resnet18.pth')
    print("Model saved to 'models/cifar10_resnet18.pth'")
    
    return history

def experiment_synthetic_quadratic(use_quick_test=False):
    """
    Experiment 2: Synthetic Quadratic Minimization
    
    Args:
        use_quick_test (bool): If True, use fewer iterations
        
    Returns:
        dict: Results for different optimizers
    """
    print("\n=== Experiment 2: Synthetic Quadratic Minimization ===")
    
    # Prepare quadratic problem
    A, b, x_init = prepare_synthetic_quadratic()
    
    # Set number of iterations based on quick test flag
    num_iters = QUICK_TEST_ITERATIONS if use_quick_test else NUM_ITERATIONS
    
    # Initialize optimizers
    optimizers = {
        'ACM': ACMOptimizer([x_init], lr=ACM_LEARNING_RATE, beta=ACM_BETA, curvature_influence=ACM_CURVATURE_INFLUENCE),
        'SGD': torch.optim.SGD([x_init], lr=SGD_LEARNING_RATE),
        'Adam': torch.optim.Adam([x_init], lr=ADAM_LEARNING_RATE),
    }
    
    results = {}
    for name, opt in optimizers.items():
        # Reset initial value of x for each optimizer run
        x = x_init.clone().detach().requires_grad_(True)
        print(f"Running optimizer: {name}")
        losses, momentum_norms = optimize_quadratic(x, A, b, opt, num_iters=num_iters)
        results[name] = {'losses': losses, 'momentum_norms': momentum_norms}
        print(f"Final loss for {name}: {losses[-1]:.4f}")
    
    # Plot results
    plot_quadratic_convergence(results, save_path='logs/quadratic_loss_convergence.png')
    
    # Plot momentum norms for ACM
    if 'ACM' in results:
        plot_momentum_norm(results['ACM']['momentum_norms'], name='ACM', save_path='logs/acm_momentum_norm.png')
    
    return results

def experiment_transformer_wikitext2(use_quick_test=False):
    """
    Experiment 3: Transformer Language Modeling on WikiText-2
    
    Args:
        use_quick_test (bool): If True, use a small subset of data and fewer epochs
        
    Returns:
        model: Trained transformer model
    """
    print("\n=== Experiment 3: Transformer Language Modeling on WikiText-2 ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    train_data, valid_data, test_data, vocab = load_wikitext2(
        batch_size=BATCH_SIZE, 
        bptt=BPTT, 
        use_quick_test=use_quick_test
    )
    
    # Define a simple Transformer-based language model
    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, embed_size=EMBED_SIZE, nhead=NHEAD, nhid=NHID, nlayers=NLAYERS, dropout=DROPOUT):
            super(TransformerModel, self).__init__()
            self.model_type = 'Transformer'
            self.encoder = nn.Embedding(vocab_size, embed_size)
            encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
            self.decoder = nn.Linear(embed_size, vocab_size)
            self.embed_size = embed_size

        def forward(self, src):
            # src shape: [seq_len, batch_size]
            src = self.encoder(src) * (self.embed_size ** 0.5)
            output = self.transformer_encoder(src)
            output = self.decoder(output)
            return output
    
    vocab_size = len(vocab)
    model = TransformerModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate ACM optimizer
    optimizer = ACMOptimizer(
        model.parameters(), 
        lr=LEARNING_RATE, 
        beta=BETA, 
        curvature_influence=CURVATURE_INFLUENCE
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    # Set number of epochs based on quick test flag
    num_epochs = QUICK_TEST_EPOCHS if use_quick_test else NUM_EPOCHS
    
    # Training loop
    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_transformer_epoch(
            model, optimizer, scheduler, train_data, criterion, 
            vocab_size, device=device, bptt=BPTT, use_quick_test=use_quick_test
        )
        
        val_loss = evaluate_transformer(model, valid_data, criterion, vocab_size, device=device, bptt=BPTT)
        perplexity = calculate_perplexity(val_loss)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:.3f}, Perplexity: {perplexity:.2f}")
    
    # Evaluate on test data
    test_loss = evaluate_transformer(model, test_data, criterion, vocab_size, device=device, bptt=BPTT)
    test_perplexity = calculate_perplexity(test_loss)
    print(f"\nTest Loss: {test_loss:.3f}, Test Perplexity: {test_perplexity:.2f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/transformer_wikitext2.pth')
    print("Model saved to 'models/transformer_wikitext2.pth'")
    
    return model

def quick_test():
    """
    Run a quick test of all experiments to ensure the code works
    """
    print("\n********** Quick Test Start **********")
    print("Running quick tests for all experiments...")
    
    try:
        _ = experiment_cifar10(use_quick_test=True)
        print("CIFAR10 experiment quick test completed successfully.")
    except Exception as e:
        print(f"CIFAR10 experiment quick test failed: {str(e)}")
    
    try:
        _ = experiment_synthetic_quadratic(use_quick_test=True)
        print("Synthetic quadratic experiment quick test completed successfully.")
    except Exception as e:
        print(f"Synthetic quadratic experiment quick test failed: {str(e)}")
    
    try:
        _ = experiment_transformer_wikitext2(use_quick_test=True)
        print("Transformer language modeling experiment quick test completed successfully.")
    except Exception as e:
        print(f"Transformer language modeling experiment quick test failed: {str(e)}")
    
    print("Quick test completed.")
    print("********** Quick Test End **********\n")

if __name__ == '__main__':
    # Create a flag for quick testing
    run_quick_test = False
    
    if run_quick_test:
        quick_test()
    else:
        # Run the full experiments
        print("Running full experiments...")
        
        # Experiment 1: CIFAR10 Classification
        _ = experiment_cifar10(use_quick_test=False)
        
        # Experiment 2: Synthetic Quadratic Optimization
        _ = experiment_synthetic_quadratic(use_quick_test=False)
        
        # Experiment 3: Transformer Language Modeling
        _ = experiment_transformer_wikitext2(use_quick_test=False)
        
        print("\nAll experiments completed successfully.")
    
    # Print list of required libraries for clarity
    print("\nRequired Python libraries for this experiment:")
    print("torch, torchvision, torchtext, matplotlib, time, copy")
    
    print("\nExperiment completed.")
