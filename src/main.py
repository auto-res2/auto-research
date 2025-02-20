import torch
import json
from pathlib import Path
from models import SimpleCNN, PTBLanguageModel
from train import get_cifar10_loaders
from evaluate import get_ptb_data, evaluate_optimizers
from preprocess import get_cifar10_transforms, get_ptb_processor

def run_experiment(config_path=None):
    """
    Run the complete experiment comparing different optimizers on CIFAR-10 and PTB.
    
    Args:
        config_path: Optional path to a JSON config file with experiment parameters
    """
    # Default configuration
    config = {
        'epochs': 10,
        'batch_size': 128,
        'learning_rate': 0.01,
        'save_dir': 'results',
        'tasks': ['cifar10', 'ptb']
    }
    
    # Load custom config if provided
    if config_path:
        with open(config_path) as f:
            config.update(json.load(f))
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Save experiment configuration
    with open(save_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    results = {}
    
    # Run experiments for each task
    for task in config['tasks']:
        print(f"\n{'='*80}\nRunning experiments on {task.upper()}\n{'='*80}")
        
        task_results = evaluate_optimizers(
            task=task,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate'],
            save_dir=str(save_dir / task)
        )
        
        results[task] = task_results
    
    # Save combined results
    with open(save_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final summary
    print("\nExperiment Summary:")
    print("=" * 80)
    
    for task in config['tasks']:
        print(f"\n{task.upper()} Results:")
        print("-" * 40)
        
        if task == 'cifar10':
            print(f"{'Optimizer':<10} {'Best Test Acc':<15} {'Convergence Epoch'}")
            print("-" * 40)
            for opt_name, opt_results in results[task].items():
                print(f"{opt_name:<10} {opt_results['best_test_acc']:<15.2f} "
                      f"{opt_results['convergence_epoch']}")
        else:  # ptb
            print(f"{'Optimizer':<10} {'Best Val PPL':<15} {'Convergence Epoch'}")
            print("-" * 40)
            for opt_name, opt_results in results[task].items():
                print(f"{opt_name:<10} {opt_results['best_val_ppl']:<15.2f} "
                      f"{opt_results['convergence_epoch']}")
    
    print("\nExperiment completed successfully!")
    print(f"Results saved in: {save_dir}")

if __name__ == '__main__':
    # Run with default configuration for a quick test
    run_experiment()
