#!/usr/bin/env python3
"""
Implementation of experiments comparing HACP vs. Base Method
using a Minigrid environment (via gymnasium). The experiments include:
- Experiment 1: End-to-End Performance in Reinforcement Learning 
- Experiment 2: Visualization of Adaptive Abstract Representations
- Experiment 3: Ablation Study (removing the planning module)

Each experiment prints/logs intermediate results and plots figures.
A minimal test function is provided for a quick run.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import time
from datetime import datetime

# Import from other scripts
from preprocess import create_environment, get_environment_info, prepare_data_dir
from train import BaseMethod, HACP, HACP_Ablated, train_model
from evaluate import collect_representations, plot_tsne, evaluate_model

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from config.experiment_config import *
except ImportError:
    # Default configuration if import fails
    ENV_NAME = "MiniGrid-Empty-8x8-v0"
    FALLBACK_ENV = "CartPole-v1"
    HIDDEN_SIZE = 128
    PLANNING_TEMPERATURE = 0.8
    NUM_EPISODES = 100
    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    DEVICE = "cuda"
    EVAL_EPISODES = 10
    NUM_STEPS_REPRESENTATION = 200
    TEST_HIDDEN_SIZE = 64
    TEST_NUM_EPISODES = 5

def setup():
    """Set up the experiment environment."""
    # Create directory structure if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    prepare_data_dir()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check for CUDA availability
    global DEVICE
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead.")
        DEVICE = "cpu"
    
    # Print system information
    print("\n----- System Information -----")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return DEVICE

def experiment_end_to_end(env, input_shape, output_size, hidden_size, num_episodes):
    """
    Experiment 1: Train both Base Method and HACP on the environment and compare rewards.
    
    Args:
        env: Gymnasium environment
        input_shape: Shape of environment observations
        output_size: Number of possible actions
        hidden_size: Size of hidden layers
        num_episodes: Number of episodes to train for
        
    Returns:
        tuple: (base_model, hacp_model, base_rewards, hacp_rewards)
    """
    print("\n----- Experiment 1: End-to-End Performance -----")
    print("Training Base Method...")
    base_model = BaseMethod(input_shape, hidden_size, output_size)
    base_rewards = train_model(base_model, env, num_episodes=num_episodes, device=DEVICE)
    
    print("\nTraining HACP...")
    hacp_model = HACP(input_shape, hidden_size, output_size, planning_temperature=PLANNING_TEMPERATURE)
    hacp_rewards = train_model(hacp_model, env, num_episodes=num_episodes, device=DEVICE)
    
    # Print summary statistics
    print("\nTraining complete. Summary of rewards:")
    print(f"Base Method:  Mean Reward = {np.mean(base_rewards):.2f}, Std Dev = {np.std(base_rewards):.2f}")
    print(f"HACP:         Mean Reward = {np.mean(hacp_rewards):.2f}, Std Dev = {np.std(hacp_rewards):.2f}")
    
    # Plot reward curves for visual comparison
    plt.figure(figsize=(8, 6))
    plt.plot(base_rewards, label="Base Method")
    plt.plot(hacp_rewards, label="HACP", linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title("Experiment 1: Reward vs. Episode")
    plt.legend()
    
    # Save plot to logs directory
    plt.savefig("logs/reward_comparison.png")
    
    # Evaluate models
    print("\nEvaluating Base Method...")
    base_mean, base_std = evaluate_model(base_model, env, num_episodes=EVAL_EPISODES, device=DEVICE)
    print(f"Base Method Evaluation: Mean Reward = {base_mean:.2f}, Std Dev = {base_std:.2f}")
    
    print("\nEvaluating HACP...")
    hacp_mean, hacp_std = evaluate_model(hacp_model, env, num_episodes=EVAL_EPISODES, device=DEVICE)
    print(f"HACP Evaluation: Mean Reward = {hacp_mean:.2f}, Std Dev = {hacp_std:.2f}")
    
    return base_model, hacp_model, base_rewards, hacp_rewards

def experiment_visualization(env, base_model, hacp_model):
    """
    Experiment 2: Visualize and compare the learned abstract representations.
    
    Args:
        env: Gymnasium environment
        base_model: Trained Base Method model
        hacp_model: Trained HACP model
    """
    print("\n----- Experiment 2: Representation Visualization -----")
    print("Collecting representations from Base Method...")
    base_repr, base_labels = collect_representations(base_model, env, num_steps=NUM_STEPS_REPRESENTATION, device=DEVICE)
    print("Collecting representations from HACP...")
    hacp_repr, hacp_labels = collect_representations(hacp_model, env, num_steps=NUM_STEPS_REPRESENTATION, device=DEVICE)
    
    print("Performing t-SNE dimensionality reduction and plotting...")
    plot_tsne(base_repr, base_labels, hacp_repr, hacp_labels)
    print("Visualization saved to logs/tsne_visualization.png")

def experiment_ablation(env, input_shape, output_size, hidden_size, num_episodes):
    """
    Experiment 3: Ablation study comparing HACP with and without its planning module.
    
    Args:
        env: Gymnasium environment
        input_shape: Shape of environment observations
        output_size: Number of possible actions
        hidden_size: Size of hidden layers
        num_episodes: Number of episodes to train for
    """
    print("\n----- Experiment 3: Ablation Study -----")
    print("Training full HACP...")
    hacp_model = HACP(input_shape, hidden_size, output_size, planning_temperature=PLANNING_TEMPERATURE)
    full_rewards = train_model(hacp_model, env, num_episodes=num_episodes, device=DEVICE)
    
    print("\nTraining ablated HACP (planning module removed)...")
    hacp_ablated_model = HACP_Ablated(input_shape, hidden_size, output_size)
    ablated_rewards = train_model(hacp_ablated_model, env, num_episodes=num_episodes, device=DEVICE)
    
    print("\nAblation Study complete. Summary of rewards:")
    print(f"Full HACP:    Mean Reward = {np.mean(full_rewards):.2f}, Std Dev = {np.std(full_rewards):.2f}")
    print(f"Ablated HACP: Mean Reward = {np.mean(ablated_rewards):.2f}, Std Dev = {np.std(ablated_rewards):.2f}")
    
    # Plotting the comparison
    plt.figure(figsize=(8, 6))
    plt.plot(full_rewards, label="Full HACP")
    plt.plot(ablated_rewards, label="HACP Ablated", linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title("Experiment 3: Ablation Study - Effect of Planning Module")
    plt.legend()
    
    # Save plot to logs directory
    plt.savefig("logs/ablation_study.png")

def test_code():
    """
    Run a minimal test to verify that the code executes correctly.
    This runs a very short training loop for each experiment.
    """
    print("\n=== Running Quick Test ===")
    # Create environment
    try:
        env = create_environment(ENV_NAME)
    except Exception as e:
        print(f"Error creating environment: {e}")
        env = create_environment(FALLBACK_ENV)
    
    # Get observation shape and action space sizes
    input_shape, output_size = get_environment_info(env)
    
    # Run a short end-to-end training experiment
    base_model, hacp_model, _, _ = experiment_end_to_end(
        env, input_shape, output_size, TEST_HIDDEN_SIZE, TEST_NUM_EPISODES
    )
    
    # Run a quick visualization experiment
    experiment_visualization(env, base_model, hacp_model)
    
    # Run ablation study with a short training loop
    experiment_ablation(env, input_shape, output_size, TEST_HIDDEN_SIZE, TEST_NUM_EPISODES)
    
    print("\n=== Test Complete ===")

def main():
    """Main function to run the full experiment suite."""
    start_time = time.time()
    print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup the environment
    setup()
    
    # Create environment
    try:
        env = create_environment(ENV_NAME)
    except Exception as e:
        print(f"Error creating environment: {e}")
        env = create_environment(FALLBACK_ENV)
    
    # Get observation shape and action space sizes
    input_shape, output_size = get_environment_info(env)
    print(f"Environment: observation shape = {input_shape}, action space size = {output_size}")
    
    # Run full experiments or test code based on command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_code()
    else:
        # Run the full experiment suite
        base_model, hacp_model, _, _ = experiment_end_to_end(
            env, input_shape, output_size, HIDDEN_SIZE, NUM_EPISODES
        )
        experiment_visualization(env, base_model, hacp_model)
        experiment_ablation(env, input_shape, output_size, HIDDEN_SIZE, NUM_EPISODES)
    
    elapsed_time = time.time() - start_time
    print(f"\nExperiment completed in {elapsed_time:.2f} seconds.")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
