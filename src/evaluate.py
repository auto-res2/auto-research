"""
Evaluation script for HBFG-SE3 experiments.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

from utils.se3_utils import compute_energy, compute_rmsd, generate_conformations, tensor_rmsd, create_protein_batch
from utils.diffusion import ForcePredictor, BootstrappedForceNetwork, HBFGSE3Diffuser

def run_base_method(input_data, num_iterations=100, device='cuda'):
    """Simulate the Base Method with a coarse and a fine stage."""
    coarse_time = 0.0
    fine_time = 0.0

    # Coarse stage
    t0 = time.time()
    for i in range(num_iterations):
        # Simulated computation: diffusion iteration.
        dummy = torch.sin(input_data + i)
    coarse_time = time.time() - t0

    # Fine stage (simulated MD/refinement stage)
    t0 = time.time()
    for i in range(num_iterations // 2):
        dummy = torch.cos(input_data + i)
    fine_time = time.time() - t0

    return coarse_time, fine_time

def run_hbfg_se3_method(input_data, diffuser, num_iterations=100, device='cuda'):
    """Run the HBFG-SE3 Method with bootstrapped guidance."""
    coarse_time = 0.0
    fine_time = 0.0

    # Coarse stage (fewer iterations due to faster convergence)
    t0 = time.time()
    coarse_result = diffuser.run_coarse_diffusion(
        input_data, 
        num_steps=num_iterations // 2, 
        guidance_scale=1.0
    )
    coarse_time = time.time() - t0

    # Fine stage with bootstrapped guidance
    t0 = time.time()
    fine_result = diffuser.run_fine_diffusion(
        coarse_result, 
        num_steps=num_iterations // 4, 
        guidance_scale=2.0, 
        bootstrap=True
    )
    fine_time = time.time() - t0

    return coarse_time, fine_time, fine_result

def benchmark_methods(trained_model, experiment_config, device):
    """Benchmark the Base and the HBFG-SE3 methods."""
    print("\n--- Experiment 1: Efficiency and Runtime Benchmarking ---")
    
    # Parameters
    num_iterations = experiment_config['experiment']['efficiency']['num_iterations']
    
    # Create a fixed dummy tensor simulating a protein input
    batch_size = 2
    num_atoms = 10
    input_data = create_protein_batch(batch_size, num_atoms, device)
    
    # Warm-up GPU (if using CUDA)
    if torch.cuda.is_available():
        for _ in range(5):
            _ = input_data * 2
    
    # Benchmark the Base Method
    start_time = time.time()
    base_coarse, base_fine = run_base_method(input_data, num_iterations=num_iterations, device=device)
    base_total = time.time() - start_time
    base_gpu_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    
    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark the HBFG-SE3 Method
    start_time = time.time()
    hbfg_coarse, hbfg_fine, _ = run_hbfg_se3_method(
        input_data, 
        trained_model['diffuser'], 
        num_iterations=num_iterations,
        device=device
    )
    hbfg_total = time.time() - start_time
    hbfg_gpu_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    
    # Log the results
    print("Base Method:  Coarse Time = {:.4f}s, Fine Time = {:.4f}s, Total Time = {:.4f}s, GPU Memory = {} bytes".format(
          base_coarse, base_fine, base_total, base_gpu_mem))
    print("HBFG-SE3 Method:  Coarse Time = {:.4f}s, Fine Time = {:.4f}s, Total Time = {:.4f}s, GPU Memory = {} bytes".format(
          hbfg_coarse, hbfg_fine, hbfg_total, hbfg_gpu_mem))
    
    # Save results to file
    os.makedirs('logs', exist_ok=True)
    with open('logs/benchmark_results.txt', 'w') as f:
        f.write("Base Method:  Coarse Time = {:.4f}s, Fine Time = {:.4f}s, Total Time = {:.4f}s, GPU Memory = {} bytes\n".format(
                base_coarse, base_fine, base_total, base_gpu_mem))
        f.write("HBFG-SE3 Method:  Coarse Time = {:.4f}s, Fine Time = {:.4f}s, Total Time = {:.4f}s, GPU Memory = {} bytes\n".format(
                hbfg_coarse, hbfg_fine, hbfg_total, hbfg_gpu_mem))
    
    # Create and save figure
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Base Method', 'HBFG-SE3']
    coarse_times = [base_coarse, hbfg_coarse]
    fine_times = [base_fine, hbfg_fine]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, coarse_times, width, label='Coarse Stage')
    ax.bar(x + width/2, fine_times, width, label='Fine Stage')
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('logs/runtime_comparison.png')
    
    return {
        'base': {'coarse': base_coarse, 'fine': base_fine, 'total': base_total, 'memory': base_gpu_mem},
        'hbfg': {'coarse': hbfg_coarse, 'fine': hbfg_fine, 'total': hbfg_total, 'memory': hbfg_gpu_mem}
    }

def evaluate_quality_and_diversity(experiment_config):
    """Generate ensembles, evaluate energies, compute pairwise RMSD, and perform clustering."""
    print("\n--- Experiment 2: Quality and Diversity of Protein Conformations ---")
    
    # Parameters
    num_samples = experiment_config['experiment']['quality_diversity']['num_samples']
    num_atoms = experiment_config['experiment']['quality_diversity']['num_atoms']
    
    # Generate ensembles for each method
    confs_hbfg = generate_conformations(method='HBFG-SE3', num_samples=num_samples, num_atoms=num_atoms)
    confs_base = generate_conformations(method='Base', num_samples=num_samples, num_atoms=num_atoms)
    
    # Evaluate energies
    energies_hbfg = [compute_energy(conf) for conf in confs_hbfg]
    energies_base = [compute_energy(conf) for conf in confs_base]
    
    print("Average Energy - HBFG-SE3: {:.4f}, Base: {:.4f}".format(
        np.mean(energies_hbfg), np.mean(energies_base)))
    
    # Compute RMSD matrix for HBFG-SE3 ensemble
    num_confs = len(confs_hbfg)
    rmsd_matrix = np.zeros((num_confs, num_confs))
    for i in range(num_confs):
        for j in range(i, num_confs):
            rmsd_val = compute_rmsd(confs_hbfg[i], confs_hbfg[j])
            rmsd_matrix[i, j] = rmsd_val
            rmsd_matrix[j, i] = rmsd_val
    
    # Perform clustering on the RMSD matrix (diversity evaluation)
    clustering = AgglomerativeClustering(n_clusters=5, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(rmsd_matrix)
    print("Cluster distribution (HBFG-SE3):", np.bincount(labels))
    
    # Extract RMSD values for histogram
    rmsd_values = []
    for i in range(num_confs):
        for j in range(i + 1, num_confs):
            rmsd_values.append(rmsd_matrix[i, j])
    
    # Create and save histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(rmsd_values, bins=20)
    plt.title('RMSD Distribution for HBFG-SE3 Ensemble')
    plt.xlabel('RMSD')
    plt.ylabel('Frequency')
    plt.savefig('logs/rmsd_distribution.png')
    
    # Save results to file
    with open('logs/quality_diversity_results.txt', 'w') as f:
        f.write("Average Energy - HBFG-SE3: {:.4f}, Base: {:.4f}\n".format(
            np.mean(energies_hbfg), np.mean(energies_base)))
        f.write("Cluster distribution (HBFG-SE3): {}\n".format(np.bincount(labels)))
    
    return {
        'energies': {'hbfg': np.mean(energies_hbfg), 'base': np.mean(energies_base)},
        'clustering': np.bincount(labels).tolist(),
        'rmsd_values': rmsd_values
    }

def run_ablation_study(trained_model, experiment_config, device):
    """Run an ablation study on the bootstrapped guidance."""
    print("\n--- Experiment 3: Ablation Study on Bootstrapped Guidance Update ---")
    
    # Parameters
    num_steps = experiment_config['experiment']['ablation']['num_steps']
    
    # Create initial conformation
    num_atoms = 10
    batch_size = 1
    initial_conf = create_protein_batch(batch_size, num_atoms, device)
    
    # Create two guidance networks with the same initialization
    force_net_full = BootstrappedForceNetwork().to(device)
    force_net_fixed = BootstrappedForceNetwork().to(device)
    force_net_fixed.load_state_dict(force_net_full.state_dict())
    
    # Create diffusers
    diffuser_full = HBFGSE3Diffuser(force_net_full, device=device)
    diffuser_fixed = HBFGSE3Diffuser(force_net_fixed, device=device)
    
    # Run diffusion with bootstrapped guidance
    print("Running diffusion with bootstrapped guidance...")
    start_time = time.time()
    result_full = diffuser_full.run_fine_diffusion(
        initial_conf, 
        num_steps=num_steps, 
        guidance_scale=2.0, 
        bootstrap=True
    )
    time_full = time.time() - start_time
    
    # Run diffusion with fixed guidance
    print("Running diffusion with fixed guidance...")
    start_time = time.time()
    result_fixed = diffuser_fixed.run_fine_diffusion(
        initial_conf, 
        num_steps=num_steps, 
        guidance_scale=2.0, 
        bootstrap=False
    )
    time_fixed = time.time() - start_time
    
    # Compute energy for both results
    with torch.no_grad():
        energy_full = torch.mean(torch.sum(force_net_full(result_full)**2, dim=-1)).item()
        energy_fixed = torch.mean(torch.sum(force_net_fixed(result_fixed)**2, dim=-1)).item()
    
    # Compute RMSD against a dummy ground truth structure (here zero tensor)
    ground_truth = torch.zeros_like(initial_conf)
    rmsd_full = tensor_rmsd(result_full, ground_truth)
    rmsd_fixed = tensor_rmsd(result_fixed, ground_truth)
    
    # Log results
    print("RMSD compared to ground truth (simulated):")
    print("  Bootstrapped Guidance: {:.4f}, Fixed Guidance: {:.4f}".format(rmsd_full, rmsd_fixed))
    print("Energy values:")
    print("  Bootstrapped Guidance: {:.4f}, Fixed Guidance: {:.4f}".format(energy_full, energy_fixed))
    print("Runtime:")
    print("  Bootstrapped Guidance: {:.4f}s, Fixed Guidance: {:.4f}s".format(time_full, time_fixed))
    
    # Save results to file
    with open('logs/ablation_results.txt', 'w') as f:
        f.write("RMSD compared to ground truth (simulated):\n")
        f.write("  Bootstrapped Guidance: {:.4f}, Fixed Guidance: {:.4f}\n".format(rmsd_full, rmsd_fixed))
        f.write("Energy values:\n")
        f.write("  Bootstrapped Guidance: {:.4f}, Fixed Guidance: {:.4f}\n".format(energy_full, energy_fixed))
        f.write("Runtime:\n")
        f.write("  Bootstrapped Guidance: {:.4f}s, Fixed Guidance: {:.4f}s\n".format(time_full, time_fixed))
    
    # Create and save figure for energy comparison
    plt.figure(figsize=(10, 6))
    methods = ['Bootstrapped Guidance', 'Fixed Guidance']
    energies = [energy_full, energy_fixed]
    rmsds = [rmsd_full, rmsd_fixed]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy plot
    ax1.bar(methods, energies)
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Comparison')
    
    # RMSD plot
    ax2.bar(methods, rmsds)
    ax2.set_ylabel('RMSD to Ground Truth')
    ax2.set_title('RMSD Comparison')
    
    fig.tight_layout()
    plt.savefig('logs/ablation_comparison.png')
    
    return {
        'bootstrapped': {'energy': energy_full, 'rmsd': rmsd_full, 'time': time_full},
        'fixed': {'energy': energy_fixed, 'rmsd': rmsd_fixed, 'time': time_fixed}
    }

def evaluate_model(trained_model, experiment_config, device):
    """Run all evaluation experiments."""
    results = {}
    
    # Run efficiency benchmark
    if experiment_config['experiment']['run_efficiency_benchmark']:
        results['benchmark'] = benchmark_methods(trained_model, experiment_config, device)
    
    # Run quality and diversity evaluation
    if experiment_config['experiment']['run_quality_diversity']:
        results['quality_diversity'] = evaluate_quality_and_diversity(experiment_config)
    
    # Run ablation study
    if experiment_config['experiment']['run_ablation_study']:
        results['ablation'] = run_ablation_study(trained_model, experiment_config, device)
    
    return results
