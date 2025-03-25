#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements the evaluation components for the BI-SDICL (Bias-Integrated Sequential 
Decision In-Context Learner) experiments, including evaluation routines for the three experiments:
1. Robustness under Environmental Stochasticity
2. Ablation Study of the Bias Conversion Module
3. Interpretability and Diagnostic Visualization of Bias Integration
"""

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocess import NoisyEnvWrapper, generate_demonstration_tokens, prepare_environment
from train import BI_SDICLPolicy, BaseMethodPolicy, create_policy

# ================================
# Evaluation Routines
# ================================
def evaluate_agent(env, policy, demo_tokens=None, num_episodes=20):
    """
    Evaluates a trained policy on the given environment.
    
    Args:
        env: Gym environment
        policy: Policy network to evaluate
        demo_tokens: Demonstration tokens for BI-SDICL
        num_episodes: Number of episodes to evaluate for
        
    Returns:
        List of cumulative episode rewards
    """
    policy.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            done = False
            ep_reward = 0.0
            while not done:
                logits = policy(state, demo_tokens)
                action = logits.argmax(dim=-1).item()
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = torch.FloatTensor(next_state).unsqueeze(0)
            rewards.append(ep_reward)
    return rewards

def evaluate_with_bias_perturbation(env, policy, demo_tokens, perturb=False):
    """
    Evaluate one episode with the option to perturb (e.g., zero out) the bias vector.
    
    Args:
        env: Gym environment
        policy: BI-SDICL policy
        demo_tokens: Demonstration tokens
        perturb: Whether to perturb the bias vector
        
    Returns:
        Total reward for the episode
    """
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    done = False
    total_reward = 0.0
    while not done:
        # Get bias vector from demonstration tokens.
        bias = policy.get_bias_vector(state, demo_tokens)
        if perturb:
            # Option: zero out the bias vector.
            bias = torch.zeros_like(bias)
        # Forward pass: combine bias and state embedding.
        state_emb = policy.state_embedding(state).unsqueeze(1)
        x = torch.cat([bias, state_emb], dim=1)
        attended, _ = policy.attention(x, x, x)
        out = attended[:, -1, :]
        logits = policy.decision_head(out)
        action = logits.argmax(dim=-1).item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = torch.FloatTensor(next_state).unsqueeze(0)
    return total_reward

# ================================
# Experiment 1: Robustness under Environmental Stochasticity
# ================================
def experiment_robustness(noise_levels=None, num_episodes=20, seed=42):
    """
    Evaluates the robustness of BI-SDICL under different levels of environmental stochasticity.
    
    Args:
        noise_levels: List of noise levels to evaluate
        num_episodes: Number of episodes to evaluate for each noise level
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("\n=== Experiment 1: Robustness under Environmental Stochasticity ===")
    avg_rewards = []
    
    # Create environment to get state and action dimensions
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Dummy demonstration tokens: one batch of 10 demos with embedding dimension 64.
    demo_tokens = generate_demonstration_tokens(batch_size=1, num_demos=10, demo_embedding_dim=64, seed=seed)
    
    # Use BI‑SDICL with bias conversion (full version).
    results = {}
    for noise in noise_levels:
        print(f"\n-- Testing noise level: {noise} --")
        env = NoisyEnvWrapper(gym.make("CartPole-v1"), noise_level=noise)
        policy = create_policy(state_dim, action_dim, policy_type="bi_sdicl", use_bias_conversion=True)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        
        # Train the policy
        from train import train_agent
        train_rewards = train_agent(env, policy, optimizer, num_episodes=num_episodes, 
                                   demo_tokens=demo_tokens, print_interval=10)
        
        # Evaluate the policy
        eval_rewards = evaluate_agent(env, policy, demo_tokens=demo_tokens, num_episodes=10)
        avg_reward = np.mean(eval_rewards)
        print(f"Average reward under noise level {noise}: {avg_reward:.2f}")
        avg_rewards.append(avg_reward)
        
        results[noise] = {
            'train_rewards': train_rewards,
            'eval_rewards': eval_rewards,
            'avg_eval_reward': avg_reward
        }
        
        env.close()
    
    # Plot the average rewards vs noise level.
    plt.figure()
    plt.plot(noise_levels, avg_rewards, marker='o')
    plt.title("Robustness: Avg. Cumulative Reward vs Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Avg. Cumulative Reward")
    plt.grid(True)
    plt.savefig("logs/experiment1_robustness.png")
    plt.close()
    
    return results

# ================================
# Experiment 2: Ablation Study of the Bias Conversion Module
# ================================
def experiment_ablation(noise_level=0.2, num_episodes=20, seed=42):
    """
    Performs an ablation study of the Bias Conversion Module.
    
    Args:
        noise_level: Noise level for the environment
        num_episodes: Number of episodes to train for
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results
    """
    print("\n=== Experiment 2: Ablation Study of the Bias Conversion Module ===")
    env = NoisyEnvWrapper(gym.make("CartPole-v1"), noise_level=noise_level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    demo_tokens = generate_demonstration_tokens(batch_size=1, num_demos=10, demo_embedding_dim=64, seed=seed)

    # Variant A: Full BI‑SDICL (with bias conversion)
    policy_full = create_policy(state_dim, action_dim, policy_type="bi_sdicl", use_bias_conversion=True)
    # Variant B: BI‑SDICL without bias conversion
    policy_no_conversion = create_policy(state_dim, action_dim, policy_type="bi_sdicl", use_bias_conversion=False)
    # Variant C: Base method (ignore demo tokens completely)
    policy_base = create_policy(state_dim, action_dim, policy_type="base")

    # Use the same training settings for all three variants
    optimizer_full = torch.optim.Adam(policy_full.parameters(), lr=1e-3)
    optimizer_no_conversion = torch.optim.Adam(policy_no_conversion.parameters(), lr=1e-3)
    optimizer_base = torch.optim.Adam(policy_base.parameters(), lr=1e-3)

    # Train all three variants
    from train import train_agent
    print("\n[Training Variant A - Full BI‑SDICL]")
    rewards_full = train_agent(env, policy_full, optimizer_full, num_episodes=num_episodes, 
                              demo_tokens=demo_tokens, print_interval=10)
    
    print("\n[Training Variant B - Without Bias Conversion]")
    rewards_no_conv = train_agent(env, policy_no_conversion, optimizer_no_conversion, 
                                 num_episodes=num_episodes, demo_tokens=demo_tokens, print_interval=10)
    
    print("\n[Training Variant C - Base Method]")
    rewards_base = train_agent(env, policy_base, optimizer_base, num_episodes=num_episodes, 
                              demo_tokens=None, print_interval=10)
    
    # Evaluate all three variants
    eval_rewards_full = evaluate_agent(env, policy_full, demo_tokens=demo_tokens)
    eval_rewards_no_conv = evaluate_agent(env, policy_no_conversion, demo_tokens=demo_tokens)
    eval_rewards_base = evaluate_agent(env, policy_base, demo_tokens=None)
    
    avg_full = np.mean(eval_rewards_full)
    avg_no_conv = np.mean(eval_rewards_no_conv)
    avg_base = np.mean(eval_rewards_base)
    
    print("\nAblation Study Results:")
    print(f"Variant A (Full BI‑SDICL) average reward: {avg_full:.2f}")
    print(f"Variant B (No Conversion) average reward: {avg_no_conv:.2f}")
    print(f"Variant C (Base Method) average reward: {avg_base:.2f}")

    # Visualize as a bar plot.
    plt.figure()
    labels = ["Full BI‑SDICL", "No Conversion", "Base Method"]
    avg_rewards = [avg_full, avg_no_conv, avg_base]
    plt.bar(labels, avg_rewards, color=["green", "orange", "red"])
    plt.title("Ablation Study: Average Reward Comparison")
    plt.ylabel("Avg. Cumulative Reward")
    plt.savefig("logs/experiment2_ablation.png")
    plt.close()
    
    results = {
        'full_bi_sdicl': {
            'train_rewards': rewards_full,
            'eval_rewards': eval_rewards_full,
            'avg_eval_reward': avg_full
        },
        'no_conversion': {
            'train_rewards': rewards_no_conv,
            'eval_rewards': eval_rewards_no_conv,
            'avg_eval_reward': avg_no_conv
        },
        'base_method': {
            'train_rewards': rewards_base,
            'eval_rewards': eval_rewards_base,
            'avg_eval_reward': avg_base
        }
    }
    
    env.close()
    return results

# ================================
# Experiment 3: Interpretability and Bias Diagnostic Visualization
# ================================
def experiment_interpretability(noise_level=0.2, num_episodes=20, seed=42):
    """
    Performs interpretability and diagnostic visualization of bias integration.
    
    Args:
        noise_level: Noise level for the environment
        num_episodes: Number of episodes to train for
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results
    """
    print("\n=== Experiment 3: Interpretability and Diagnostic Visualization ===")
    env = NoisyEnvWrapper(gym.make("CartPole-v1"), noise_level=noise_level)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    demo_tokens = generate_demonstration_tokens(batch_size=1, num_demos=10, demo_embedding_dim=64, seed=seed)

    # Create and train a BI‑SDICL policy.
    policy = create_policy(state_dim, action_dim, policy_type="bi_sdicl", use_bias_conversion=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    print("\n[Training BI‑SDICL policy for interpretability diagnostics]")
    from train import train_agent
    train_rewards = train_agent(env, policy, optimizer, num_episodes=num_episodes, 
                               demo_tokens=demo_tokens, print_interval=10)

    # Evaluate under two conditions: original bias vs. perturbed bias.
    baseline_rewards = []
    perturbed_rewards = []
    num_test_episodes = 20
    for _ in range(num_test_episodes):
        r_base = evaluate_with_bias_perturbation(env, policy, demo_tokens, perturb=False)
        r_pert = evaluate_with_bias_perturbation(env, policy, demo_tokens, perturb=True)
        baseline_rewards.append(r_base)
        perturbed_rewards.append(r_pert)

    print(f"\nAverage reward with original bias: {np.mean(baseline_rewards):.2f}")
    print(f"Average reward with perturbed bias: {np.mean(perturbed_rewards):.2f}")

    # Box plot to visualize the difference.
    plt.figure()
    plt.boxplot([baseline_rewards, perturbed_rewards])
    plt.xticks([1, 2], ["Original Bias", "Perturbed Bias"])
    plt.title("Effect of Bias Perturbation on Episode Reward")
    plt.ylabel("Cumulative Episode Reward")
    plt.savefig("logs/experiment3_interpretability.png")
    plt.close()
    
    results = {
        'train_rewards': train_rewards,
        'baseline_rewards': baseline_rewards,
        'perturbed_rewards': perturbed_rewards,
        'avg_baseline_reward': np.mean(baseline_rewards),
        'avg_perturbed_reward': np.mean(perturbed_rewards)
    }
    
    env.close()
    return results

def run_all_experiments(quick_test=False):
    """
    Runs all three experiments.
    
    Args:
        quick_test: Whether to run a quick test with fewer episodes
        
    Returns:
        Dictionary with results from all experiments
    """
    num_episodes = 5 if quick_test else 20
    seed = 42
    
    # Run Experiment 1: Robustness
    if quick_test:
        noise_levels = [0.0, 0.2]  # Fewer noise levels for quick test
    else:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results_exp1 = experiment_robustness(noise_levels=noise_levels, num_episodes=num_episodes, seed=seed)
    
    # Run Experiment 2: Ablation Study
    results_exp2 = experiment_ablation(noise_level=0.2, num_episodes=num_episodes, seed=seed)
    
    # Run Experiment 3: Interpretability
    results_exp3 = experiment_interpretability(noise_level=0.2, num_episodes=num_episodes, seed=seed)
    
    return {
        'experiment1_robustness': results_exp1,
        'experiment2_ablation': results_exp2,
        'experiment3_interpretability': results_exp3
    }

if __name__ == "__main__":
    # Test the evaluation components
    import torch.optim as optim
    
    # Create environment and policy
    env = prepare_environment(noise_level=0.1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create BI-SDICL policy
    policy = create_policy(state_dim, action_dim, policy_type="bi_sdicl")
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Generate demonstration tokens
    demo_tokens = generate_demonstration_tokens(seed=42)
    
    # Train for a few episodes
    from train import train_agent
    print("Training BI-SDICL policy...")
    train_agent(env, policy, optimizer, num_episodes=5, demo_tokens=demo_tokens)
    
    # Evaluate the policy
    print("Evaluating BI-SDICL policy...")
    rewards = evaluate_agent(env, policy, demo_tokens=demo_tokens, num_episodes=5)
    print(f"Evaluation rewards: {rewards}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    
    # Run a quick test of all experiments
    print("\nRunning quick test of all experiments...")
    run_all_experiments(quick_test=True)
    
    env.close()
