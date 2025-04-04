"""
Preprocessing module for APESGP experiments.

This module defines the synthetic functions for the experiments.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.apesgp_config import D, NUM_SUBFUNCTIONS, SUB_DIM, COST_FULL_DEFAULT, COST_PARTIAL_DEFAULT

def synthetic_subfunction(x):
    """
    A simple quadratic function in the subspace.
    """
    return np.sum((x - 0.5)**2)

def evaluate_full(x, cost_full=COST_FULL_DEFAULT):
    """
    Full evaluation: compute the sum over all decomposed subfunctions.
    """
    total = 0.0
    for i in range(NUM_SUBFUNCTIONS):
        xi = x[i*SUB_DIM:(i+1)*SUB_DIM]
        total += synthetic_subfunction(xi)
    return total

def evaluate_partial(x, idx, cost_partial=COST_PARTIAL_DEFAULT):
    """
    Partial evaluation: evaluate only one subfunction, adding noise to simulate a lower-fidelity evaluation.
    """
    xi = x[idx*SUB_DIM:(idx+1)*SUB_DIM]
    value = synthetic_subfunction(xi)
    noise = np.random.normal(0, 0.1)  # add noise
    return value + noise

def generate_candidate(dimension=D):
    """
    Generate a random candidate solution for optimization.
    """
    return np.random.rand(dimension)
