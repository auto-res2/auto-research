"""
Script to compare NSRPP and Nelder-Mead optimization methods.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.train import run_nsrpp, run_nelder_mead
from src.evaluate import evaluate_nelder_mead_comparison
from src.utils.plotting import save_figure

np.random.seed(42)
torch.manual_seed(42)

os.makedirs("logs/figures", exist_ok=True)

print("Running NSRPP method...")
nsrpp_risk_history = run_nsrpp(num_iters=10, delta=0.1, learning_rate=0.1, init_theta=0.0)

print("\nRunning Nelder-Mead optimization...")
nm_risk_history = run_nelder_mead(num_iters=10, init_theta=0.0)

metrics = evaluate_nelder_mead_comparison(nsrpp_risk_history, nm_risk_history, 'logs/figures')

print('\n=== Nelder-Mead vs NSRPP Comparison Results ===')
print(f'NSRPP Final Risk: {metrics["nsrpp_final_risk"]:.4f}')
print(f'Nelder-Mead Final Risk: {metrics["nm_final_risk"]:.4f}')
print(f'Risk Reduction (NSRPP vs NM): {metrics["risk_reduction_percent"]:.2f}%')
print(f'NSRPP Convergence Iteration: {metrics["nsrpp_convergence_iter"]}')
print(f'Nelder-Mead Convergence Iteration: {metrics["nm_convergence_iter"]}')

plt.figure(figsize=(10, 6))
iterations = range(len(nsrpp_risk_history))
plt.plot(iterations, nsrpp_risk_history, label="NSRPP", linewidth=2)
plt.plot(iterations, nm_risk_history, label="Nelder-Mead", linewidth=2, linestyle="--")
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("True Risk", fontsize=12)
plt.title("Risk Convergence: NSRPP vs. Nelder-Mead", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
save_figure("detailed_nsrpp_vs_nelder_mead_comparison", 'logs/figures')

print("\nComparison complete. Results and figures saved in logs/figures directory.")
