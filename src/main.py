"""
Main script for APESGP experiments.

This script orchestrates the execution of all APESGP experiments, from data preprocessing to model training and evaluation.
"""

import time
import warnings
import os

from train import run_experiment_1
from evaluate import run_experiment_2, run_experiment_3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.apesgp_config import EXP1_ITERATIONS, EXP2_ITERATIONS, EXP3_ITERATIONS, EXP3_SEED_RUNS
from config.apesgp_config import TEST_ITERATIONS, TEST_SEED_RUNS

warnings.filterwarnings("ignore")

def ensure_directory(directory):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)

def run_all_experiments(test_mode=False):
    """
    Run all three experiments with configurable parameters.
    If test_mode is True, use reduced iterations for a quick test.
    """
    print("\n========================= Running All APESGP Experiments =========================")
    start_time = time.time()
    
    ensure_directory("logs")
    
    if test_mode:
        num_iterations = TEST_ITERATIONS
        num_seed_runs = TEST_SEED_RUNS
    else:
        num_iterations = EXP1_ITERATIONS
        num_seed_runs = EXP3_SEED_RUNS
    
    print("\nRunning Experiment 1: APESGP vs Baseline")
    run_experiment_1(num_iterations=num_iterations, use_baseline=False)
    run_experiment_1(num_iterations=num_iterations, use_baseline=True)
    
    print("\nRunning Experiment 2: Cost-Aware Acquisition Analysis")
    run_experiment_2(num_iterations=num_iterations)
    
    print("\nRunning Experiment 3: Sensitivity to GP Hyperparameters")
    run_experiment_3(num_iterations=num_iterations, num_seed_runs=num_seed_runs)
    
    end_time = time.time()
    print("\nAll experiments executed in {:.2f} seconds.".format(end_time - start_time))
    print("Experiment results and figures saved in the logs directory.")

if __name__ == '__main__':
    run_all_experiments(test_mode=True)
    
    print("\n========================= Completed Test Run =========================")
    print("For a full run, update the test_mode parameter to False in main.py or update config settings.")
