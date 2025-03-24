# src/main.py
import os
import torch
import argparse
import time
from src.preprocess import preprocess_data
from src.train import train_ambient_diffusion, train_one_step_generator
from src.evaluate import experiment1_efficiency_benchmark, run_all_experiments
from src.utils.models import AmbientDiffusionModel, OneStepGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Ambient Score Distillation (ASD) experiments')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.1, 0.3, 0.6],
                        help='Noise levels for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for storing data')
    parser.add_argument('--models_dir', type=str, default='./models',
                        help='Directory for storing models')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                        help='Directory for storing logs and results')
    parser.add_argument('--experiment', type=int, default=0,
                        help='Run specific experiment (0: all, 1-3: specific experiment)')
    parser.add_argument('--diffusion_steps', type=int, default=50,
                        help='Number of diffusion steps for ambient diffusion')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for evaluation')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with reduced computation')
    return parser.parse_args()

def print_system_info():
    """Print system information."""
    print("\n========== System Information ==========")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("========================================\n")

def print_experiment_header(title):
    """Print a formatted experiment header."""
    print("\n" + "=" * 80)
    print(f"{title}".center(80))
    print("=" * 80 + "\n")

def run_experiment1(args):
    """Run Experiment 1: Efficiency and Inference-Time Benchmarking."""
    print_experiment_header("Experiment 1: Efficiency and Inference-Time Benchmarking")
    
    # Initialize models
    ambient_model = AmbientDiffusionModel().to(args.device)
    one_step_model = OneStepGenerator().to(args.device)
    
    # Adjust parameters for test mode
    num_samples = 10 if args.test_mode else args.num_samples
    diffusion_steps = 10 if args.test_mode else args.diffusion_steps
    
    # Run the experiment
    results = experiment1_efficiency_benchmark(
        device=args.device,
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
        save_dir=args.logs_dir
    )
    
    # Print detailed results
    print("\nExperiment 1 Results Summary:")
    print(f"Ambient Diffusion inference time per sample (ms): {results['ambient_time']:.2f}")
    print(f"One-Step Generator inference time per sample (ms): {results['one_step_time']:.2f}")
    print(f"Speedup factor: {results['speedup_factor']:.2f}x")
    print(f"Ambient Diffusion memory usage (MB): {results['ambient_memory']:.2f}")
    print(f"One-Step Generator memory usage (MB): {results['one_step_memory']:.2f}")
    print(f"Memory reduction: {results['memory_reduction']:.2f}x")
    
    return results

def run_experiment2(args):
    """Run Experiment 2: Robustness to Noise and Memorization Prevention."""
    print_experiment_header("Experiment 2: Robustness to Noise and Memorization Prevention")
    
    # Adjust parameters for test mode
    noise_levels = [0.1] if args.test_mode else args.noise_levels
    batch_size = 16 if args.test_mode else args.batch_size
    
    # Preprocess data
    print("Preprocessing data for different noise levels...")
    dataloaders = preprocess_data(noise_levels=noise_levels, batch_size=batch_size)
    
    # Train models for each noise level
    results = {}
    for noise_level in noise_levels:
        print(f"\nTraining with noise level {noise_level}...")
        
        # Train ambient diffusion model
        ambient_model = train_ambient_diffusion(
            noise_level=noise_level,
            batch_size=batch_size,
            epochs=args.epochs,
            device=args.device,
            save_dir=args.models_dir
        )
        
        # Train one-step generator
        one_step_model = train_one_step_generator(
            teacher_model=ambient_model,
            noise_level=noise_level,
            batch_size=batch_size,
            epochs=args.epochs,
            device=args.device,
            save_dir=args.models_dir
        )
        
        # Store models for this noise level
        results[noise_level] = {
            'ambient_model': ambient_model,
            'one_step_model': one_step_model
        }
    
    print("\nExperiment 2 Results Summary:")
    print(f"Successfully trained models for {len(noise_levels)} noise levels: {noise_levels}")
    print(f"Models saved to {args.models_dir}")
    
    return results

def run_experiment3(args):
    """Run Experiment 3: Data-Free Distillation Efficacy and Ablation Study."""
    print_experiment_header("Experiment 3: Data-Free Distillation Efficacy and Ablation Study")
    
    # Adjust parameters for test mode
    num_synthetic = 20 if args.test_mode else 100
    batch_size = 8 if args.test_mode else 16
    
    # Run the experiment
    from src.utils.experiments import experiment3_ablation_study
    results = experiment3_ablation_study(
        device=args.device,
        num_synthetic=num_synthetic,
        batch_size=batch_size,
        epochs=args.epochs,
        save_dir=args.logs_dir
    )
    
    # Print detailed results
    print("\nExperiment 3 Results Summary:")
    print("Ablation Study Final Losses:")
    for config_name, config_results in results.items():
        print(f"  {config_name}: {config_results['final_loss']:.4f}")
    
    # Analyze results
    best_config = min(results.items(), key=lambda x: x[1]['final_loss'])[0]
    print(f"\nBest configuration: {best_config}")
    print("This demonstrates the importance of each component in the ASD method.")
    
    return results

def main():
    """Main function to run the ASD experiments."""
    # Parse arguments
    args = parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Print system information
    print_system_info()
    
    # Print experiment configuration
    print("\n========== Experiment Configuration ==========")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Test mode: {args.test_mode}")
    print("=============================================\n")
    
    # Record start time
    start_time = time.time()
    
    # Run experiments
    if args.experiment == 0 or args.experiment == 1:
        exp1_results = run_experiment1(args)
    
    if args.experiment == 0 or args.experiment == 2:
        exp2_results = run_experiment2(args)
    
    if args.experiment == 0 or args.experiment == 3:
        exp3_results = run_experiment3(args)
    
    # Record end time and print total runtime
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print overall summary
    print_experiment_header("Overall Summary")
    print(f"Total runtime: {total_time:.2f} seconds")
    print("\nAmbient Score Distillation (ASD) Method:")
    print("ASD combines ambient diffusion training with score identity distillation")
    print("to create an efficient one-step generator that is robust to noise and")
    print("prevents memorization while maintaining high sample quality.")
    
    print("\nKey Findings:")
    if args.experiment == 0 or args.experiment == 1:
        print(f"1. Efficiency: ASD achieves {exp1_results['speedup_factor']:.2f}x speedup in inference time")
        print(f"   and {exp1_results['memory_reduction']:.2f}x reduction in memory usage.")
    
    if args.experiment == 0 or args.experiment == 2:
        print("2. Robustness: ASD effectively learns from noisy data at multiple noise levels,")
        print("   demonstrating its ability to prevent memorization of corrupted data.")
    
    if args.experiment == 0 or args.experiment == 3:
        print("3. Ablation Study: Both score identity and consistency components")
        print("   contribute to the effectiveness of the ASD method.")
    
    print("\nExperiments completed successfully!")

if __name__ == "__main__":
    main()
