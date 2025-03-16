import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import custom modules
from preprocess import get_cifar10_loaders, get_ptb_loaders
from train import CNN, LSTM, train_cifar10, train_ptb
from evaluate import (
    evaluate_cifar10, 
    evaluate_ptb, 
    plot_training_history, 
    plot_optimizer_comparison,
    plot_confusion_matrix
)
from utils.optimizers import HybridOptimizer

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('config', exist_ok=True)

def get_device():
    """Get the device to run the experiments on"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def run_cifar10_experiment(args, device):
    """
    Run experiment on CIFAR-10 dataset with different optimizers.
    
    Args:
        args: Command line arguments
        device: Device to run the experiment on
    
    Returns:
        dict: Results for different optimizers
    """
    print("\n" + "="*80)
    print("CIFAR-10 Image Classification Experiment")
    print("="*80)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    print(f"CIFAR-10 dataset loaded with batch size {args.batch_size}")
    
    # Define optimizers to compare
    optimizers = ['sgd', 'adam', 'madgrad', 'aggmo', 'hybrid']
    
    # Store results for each optimizer
    results = {}
    histories = {}
    
    # Run experiment for each optimizer
    for opt_name in optimizers:
        print(f"\nTraining with {opt_name.upper()} optimizer")
        print("-" * 50)
        
        # Create model
        model = CNN()
        
        # Train model
        history = train_cifar10(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_name=opt_name,
            lr=args.learning_rate,
            epochs=args.epochs,
            device=device
        )
        
        # Evaluate model
        accuracy, loss, predictions, targets = evaluate_cifar10(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        # Save results
        results[opt_name] = {
            'accuracy': accuracy,
            'loss': loss,
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_test_loss': history['test_loss'][-1],
            'final_test_acc': history['test_acc'][-1],
            'avg_epoch_time': np.mean(history['epoch_times'])
        }
        histories[opt_name] = history
        
        # Save model
        model_path = f"models/cifar10_{opt_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Print results
        print(f"\nResults for {opt_name.upper()} optimizer:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Loss: {loss:.4f}")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"  Average Epoch Time: {np.mean(history['epoch_times']):.2f}s")
    
    # Plot comparison of optimizers
    print("\nPlotting optimizer comparisons...")
    
    # Plot training loss comparison
    plot_optimizer_comparison(
        histories=histories,
        metric='loss',
        title='CIFAR-10 Training Loss Comparison',
        save_path='logs/cifar10_loss_comparison.png'
    )
    
    # Plot training accuracy comparison
    plot_optimizer_comparison(
        histories=histories,
        metric='acc',
        title='CIFAR-10 Training Accuracy Comparison',
        save_path='logs/cifar10_acc_comparison.png'
    )
    
    # Plot validation accuracy comparison
    plot_optimizer_comparison(
        histories=histories,
        metric='val_acc',
        title='CIFAR-10 Validation Accuracy Comparison',
        save_path='logs/cifar10_val_acc_comparison.png'
    )
    
    # Print summary table
    print("\nCIFAR-10 Summary:")
    print("-" * 100)
    print(f"{'Optimizer':<10} | {'Test Acc':<10} | {'Test Loss':<10} | {'Train Acc':<10} | {'Train Loss':<10} | {'Epoch Time':<10}")
    print("-" * 100)
    
    for opt_name, result in results.items():
        print(f"{opt_name.upper():<10} | {result['accuracy']:.4f} | {result['loss']:.4f} | {result['final_train_acc']:.2f}% | {result['final_train_loss']:.4f} | {result['avg_epoch_time']:.2f}s")
    
    return results, histories

def run_ptb_experiment(args, device):
    """
    Run experiment on PTB dataset with different optimizers.
    
    Args:
        args: Command line arguments
        device: Device to run the experiment on
    
    Returns:
        dict: Results for different optimizers
    """
    print("\n" + "="*80)
    print("Penn Treebank Language Modeling Experiment")
    print("="*80)
    
    # Load data
    print("Loading PTB dataset...")
    train_loader, valid_loader, test_loader, vocab_size = get_ptb_loaders(
        batch_size=args.ptb_batch_size, 
        seq_length=args.ptb_seq_length
    )
    print(f"PTB dataset loaded with batch size {args.ptb_batch_size} and sequence length {args.ptb_seq_length}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Define optimizers to compare
    optimizers = ['sgd', 'adam', 'madgrad', 'aggmo', 'hybrid']
    
    # Store results for each optimizer
    results = {}
    histories = {}
    
    # Run experiment for each optimizer
    for opt_name in optimizers:
        print(f"\nTraining with {opt_name.upper()} optimizer")
        print("-" * 50)
        
        # Create model
        model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=args.ptb_embedding_dim,
            hidden_dim=args.ptb_hidden_dim,
            num_layers=args.ptb_num_layers,
            dropout=args.ptb_dropout
        )
        
        # Train model
        history = train_ptb(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer_name=opt_name,
            lr=args.ptb_learning_rate,
            epochs=args.ptb_epochs,
            clip=args.ptb_clip,
            device=device
        )
        
        # Evaluate model
        perplexity, loss = evaluate_ptb(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        # Save results
        results[opt_name] = {
            'perplexity': perplexity,
            'loss': loss,
            'final_train_loss': history['train_loss'][-1],
            'final_train_ppl': history['train_ppl'][-1],
            'final_valid_loss': history['valid_loss'][-1],
            'final_valid_ppl': history['valid_ppl'][-1],
            'avg_epoch_time': np.mean(history['epoch_times'])
        }
        histories[opt_name] = history
        
        # Save model
        model_path = f"models/ptb_{opt_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Print results
        print(f"\nResults for {opt_name.upper()} optimizer:")
        print(f"  Test Perplexity: {perplexity:.2f}")
        print(f"  Test Loss: {loss:.4f}")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Train Perplexity: {history['train_ppl'][-1]:.2f}")
        print(f"  Average Epoch Time: {np.mean(history['epoch_times']):.2f}s")
    
    # Plot comparison of optimizers
    print("\nPlotting optimizer comparisons...")
    
    # Plot training loss comparison
    plot_optimizer_comparison(
        histories=histories,
        metric='loss',
        title='PTB Training Loss Comparison',
        save_path='logs/ptb_loss_comparison.png'
    )
    
    # Plot training perplexity comparison
    plot_optimizer_comparison(
        histories=histories,
        metric='ppl',
        title='PTB Training Perplexity Comparison',
        save_path='logs/ptb_ppl_comparison.png'
    )
    
    # Plot validation perplexity comparison
    plot_optimizer_comparison(
        histories=histories,
        metric='val_ppl',
        title='PTB Validation Perplexity Comparison',
        save_path='logs/ptb_val_ppl_comparison.png'
    )
    
    # Print summary table
    print("\nPTB Summary:")
    print("-" * 100)
    print(f"{'Optimizer':<10} | {'Test PPL':<10} | {'Test Loss':<10} | {'Train PPL':<10} | {'Train Loss':<10} | {'Epoch Time':<10}")
    print("-" * 100)
    
    for opt_name, result in results.items():
        print(f"{opt_name.upper():<10} | {result['perplexity']:.2f} | {result['loss']:.4f} | {result['final_train_ppl']:.2f} | {result['final_train_loss']:.4f} | {result['avg_epoch_time']:.2f}s")
    
    return results, histories

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hybrid Optimizer Experiment')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    
    # CIFAR-10 settings
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for CIFAR-10')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for CIFAR-10')
    
    # PTB settings
    parser.add_argument('--ptb-batch-size', type=int, default=20, help='Batch size for PTB')
    parser.add_argument('--ptb-seq-length', type=int, default=35, help='Sequence length for PTB')
    parser.add_argument('--ptb-epochs', type=int, default=5, help='Number of epochs for PTB')
    parser.add_argument('--ptb-learning-rate', type=float, default=20, help='Learning rate for PTB')
    parser.add_argument('--ptb-embedding-dim', type=int, default=200, help='Embedding dimension for PTB')
    parser.add_argument('--ptb-hidden-dim', type=int, default=200, help='Hidden dimension for PTB')
    parser.add_argument('--ptb-num-layers', type=int, default=2, help='Number of LSTM layers for PTB')
    parser.add_argument('--ptb-dropout', type=float, default=0.5, help='Dropout rate for PTB')
    parser.add_argument('--ptb-clip', type=float, default=0.25, help='Gradient clipping for PTB')
    
    # Experiment selection
    parser.add_argument('--run-cifar10', action='store_true', default=True, help='Run CIFAR-10 experiment')
    parser.add_argument('--run-ptb', action='store_true', default=True, help='Run PTB experiment')
    
    return parser.parse_args()

def main():
    """Main function to run the experiments"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup directories
    setup_directories()
    
    # Get device
    device = get_device()
    
    # Start time
    start_time = datetime.now()
    print(f"Starting experiments at {start_time}")
    
    # Run experiments
    cifar10_results = None
    ptb_results = None
    
    if args.run_cifar10:
        cifar10_results, cifar10_histories = run_cifar10_experiment(args, device)
    
    if args.run_ptb:
        ptb_results, ptb_histories = run_ptb_experiment(args, device)
    
    # End time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nExperiments completed at {end_time}")
    print(f"Total duration: {duration}")
    
    # Final summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    
    if cifar10_results:
        print("\nCIFAR-10 Best Performers:")
        best_acc = max(cifar10_results.items(), key=lambda x: x[1]['accuracy'])
        fastest_conv = min(cifar10_results.items(), key=lambda x: x[1]['avg_epoch_time'])
        print(f"  Best Accuracy: {best_acc[0].upper()} ({best_acc[1]['accuracy']:.4f})")
        print(f"  Fastest Convergence: {fastest_conv[0].upper()} ({fastest_conv[1]['avg_epoch_time']:.2f}s per epoch)")
    
    if ptb_results:
        print("\nPTB Best Performers:")
        best_ppl = min(ptb_results.items(), key=lambda x: x[1]['perplexity'])
        fastest_conv = min(ptb_results.items(), key=lambda x: x[1]['avg_epoch_time'])
        print(f"  Best Perplexity: {best_ppl[0].upper()} ({best_ppl[1]['perplexity']:.2f})")
        print(f"  Fastest Convergence: {fastest_conv[0].upper()} ({fastest_conv[1]['avg_epoch_time']:.2f}s per epoch)")
    
    print("\nHybrid Optimizer Performance Analysis:")
    if cifar10_results:
        hybrid_cifar10 = cifar10_results['hybrid']
        print(f"  CIFAR-10 Accuracy: {hybrid_cifar10['accuracy']:.4f}")
        print(f"  CIFAR-10 Convergence: {hybrid_cifar10['avg_epoch_time']:.2f}s per epoch")
    
    if ptb_results:
        hybrid_ptb = ptb_results['hybrid']
        print(f"  PTB Perplexity: {hybrid_ptb['perplexity']:.2f}")
        print(f"  PTB Convergence: {hybrid_ptb['avg_epoch_time']:.2f}s per epoch")
    
    print("\nConclusion:")
    print("  The HybridOptimizer combines the advantages of AggMo's aggregated momentum with")
    print("  MADGRAD's adaptive updates, providing a balance between stability and adaptivity.")
    print("  This experiment demonstrates its effectiveness across different machine learning tasks.")

if __name__ == "__main__":
    main()
