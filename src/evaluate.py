"""
RapidAlign: Evaluation module

This module implements the evaluation experiments for RapidAlign.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import gym
import os
from typing import List, Tuple, Dict, Any

os.makedirs("logs", exist_ok=True)

class SimpleGridEnv:
    """
    A simple 2D GridWorld environment simulation using gym's style interface.
    """
    def __init__(self, grid_size=10):
        """
        Initialize the environment.
        
        Args:
            grid_size (int): Size of the grid
        """
        self.size = grid_size
        self.reset()
        
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            np.ndarray: Initial agent position
        """
        self.agent_pos = np.array([self.size // 2, self.size // 2], dtype=float)
        return self.agent_pos.copy()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Action to take
            
        Returns:
            Tuple: Next state, reward, done flag, info dictionary
        """
        self.agent_pos = np.clip(self.agent_pos + action, 0, self.size - 1)
        reward = 0  
        done = False
        return self.agent_pos.copy(), reward, done, {}

def evaluate_trajectory(ground_truth, generated):
    """
    Compute L2 norm differences and cosine similarity between trajectory steps.
    
    Args:
        ground_truth (List[torch.Tensor]): Ground truth trajectory
        generated (List[torch.Tensor]): Generated trajectory
        
    Returns:
        Tuple[List[float], List[float]]: L2 errors and cosine similarities
    """
    l2_errors = []
    cosine_sims = []
    for gt, gen in zip(ground_truth, generated):
        l2_errors.append(torch.norm(gt - gen).item())
        sim = cosine_similarity(gt.reshape(1, -1).numpy(), gen.reshape(1, -1).numpy())[0][0]
        cosine_sims.append(sim)
    return l2_errors, cosine_sims

def behavior_switch_controller(t, switch_interval=5):
    """
    Behavior switch controller based on elapsed time.
    
    Args:
        t (float): Current time
        switch_interval (int): Time interval between behavior switches
        
    Returns:
        np.ndarray: Target action
    """
    if int(t / switch_interval) % 2 == 0:
        return np.array([-1, 0])
    else:
        return np.array([1, 0])

def rapidalign_grid_planner(current_pos, target_action, num_steps=10, noise_scale=0.05):
    """
    RapidAlign-based planner adapted for grid movement in the 2D environment.
    
    Args:
        current_pos (np.ndarray): Current position
        target_action (np.ndarray): Target action
        num_steps (int): Number of planning steps
        noise_scale (float): Scale of noise to add
        
    Returns:
        List[np.ndarray]: Planned trajectory
    """
    pos = torch.tensor(current_pos, dtype=torch.float32)
    trajectory = [pos.clone().numpy()]
    dt = 0.1
    def system(x):
        return torch.tensor(target_action, dtype=torch.float32) - 0.05 * x
    for step in range(num_steps):
        pred = pos + dt * system(pos) + noise_scale * torch.randn_like(pos)
        pos = pos + (dt/2) * (system(pos) + system(pred))
        trajectory.append(pos.clone().numpy())
    return trajectory

def inference_speed_benchmark(n_trials=100, num_steps=50):
    """
    Inference speed benchmark experiment.
    
    Args:
        n_trials (int): Number of trials
        num_steps (int): Number of steps per trial
    """
    from src.train import generate_latent_vector, ddim_sampler, rapidalign_sampler
    
    ddim_times = []
    rapidalign_times = []
    
    for trial in range(n_trials):
        latent_init = generate_latent_vector()
        
        start_time = time.perf_counter()
        _ = ddim_sampler(latent_init, num_steps=num_steps)
        ddim_times.append(time.perf_counter() - start_time)
        
        start_time = time.perf_counter()
        _ = rapidalign_sampler(latent_init, num_steps=num_steps)
        rapidalign_times.append(time.perf_counter() - start_time)
    
    avg_ddim = np.mean(ddim_times)
    avg_rapidalign = np.mean(rapidalign_times)
    
    print("=== Experiment 1: Inference Speed Benchmarking ===")
    print(f"Average DDIM inference time over {n_trials} trials: {avg_ddim:.6f} sec")
    print(f"Average RapidAlign inference time over {n_trials} trials: {avg_rapidalign:.6f} sec")
    
    plt.figure(figsize=(8, 5))
    plt.hist(ddim_times, bins=20, alpha=0.6, label="DDIM")
    plt.hist(rapidalign_times, bins=20, alpha=0.6, label="RapidAlign")
    plt.title("Inference Speed Distribution")
    plt.xlabel("Inference time (sec)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/inference_speed_benchmark.pdf")
    plt.close()

def stability_preference_evaluation(num_steps=50):
    """
    Stability and preference alignment evaluation experiment.
    
    Args:
        num_steps (int): Number of steps
    """
    from src.train import generate_latent_vector, ground_truth_trajectory, ddim_sampler, rapidalign_sampler
    
    latent_init = torch.randn(16)
    gt_traj = ground_truth_trajectory(latent_init, num_steps)
    
    noise_levels = [0.05, 0.1, 0.2]
    results = {}
    
    print("\n=== Experiment 2: Stability and Preference Alignment Quality ===")
    for noise in noise_levels:
        ddim_traj = ddim_sampler(latent_init, num_steps=num_steps, noise_scale=noise)
        rapid_traj = rapidalign_sampler(latent_init, num_steps=num_steps, noise_scale=noise)
        
        ddim_l2, ddim_cos = evaluate_trajectory(gt_traj, ddim_traj)
        rapid_l2, rapid_cos = evaluate_trajectory(gt_traj, rapid_traj)
        
        results[noise] = {
            'ddim_l2': ddim_l2,
            'ddim_cos': ddim_cos,
            'rapid_l2': rapid_l2,
            'rapid_cos': rapid_cos,
        }
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(ddim_l2, label='DDIM L2 Error')
        plt.plot(rapid_l2, label='RapidAlign L2 Error')
        plt.title(f"L2 Error (Noise level: {noise})")
        plt.xlabel("Step")
        plt.ylabel("L2 norm error")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(ddim_cos, label='DDIM Cosine Similarity')
        plt.plot(rapid_cos, label='RapidAlign Cosine Similarity')
        plt.title(f"Cosine Similarity (Noise level: {noise})")
        plt.xlabel("Step")
        plt.ylabel("Cosine similarity")
        plt.legend()
        plt.tight_layout()
        pdf_filename = f"logs/alignment_quality_noise{str(noise).replace('.','p')}.pdf"
        plt.savefig(pdf_filename)
        plt.close()
        
        print(f"Noise level {noise}: Finished plotting results to {pdf_filename}")

def zero_shot_switch_experiment(sim_steps=50, switch_interval=5):
    """
    Zero-shot behavior switching experiment in a gridworld environment.
    
    Args:
        sim_steps (int): Number of simulation steps
        switch_interval (int): Time interval between behavior switches
    """
    env = SimpleGridEnv(grid_size=10)
    pos = env.reset()
    all_positions = [pos.copy()]
    switch_times = []
    
    print("\n=== Experiment 3: Zero-Shot Behavior Switching ===")
    start_time = time.perf_counter()
    
    for t in range(sim_steps):
        current_time = time.perf_counter() - start_time
        target = behavior_switch_controller(current_time, switch_interval=switch_interval)
        switch_times.append(current_time)
        
        traj = rapidalign_grid_planner(pos, target, num_steps=3, noise_scale=0.05)
        pos = traj[1]
        all_positions.append(pos)
        
        env.agent_pos = pos.copy()
    
    all_positions = np.array(all_positions)
    plt.figure(figsize=(6,6))
    plt.plot(all_positions[:,0], all_positions[:,1], marker='o', linestyle='-')
    plt.title("Agent Trajectory with Zero-Shot Behavior Switching")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.tight_layout()
    pdf_filename = "logs/trajectory_zero_shot_switch.pdf"
    plt.savefig(pdf_filename)
    plt.close()
    
    print(f"Zero-shot behavior switching experiment complete. Trajectory saved as {pdf_filename}")
