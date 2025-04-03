"""
Configuration for MML-BO experiments.
"""

EXP1_CONFIG = {
    'iters': 100,
    'init_val': [0.0],
    'levels': 3,
    'step_size': 0.1,
    'noise_std_quad': 0.1,  # Noise level for quadratic function
    'noise_std_sin': 1.0,   # Noise level for sinusoidal function
}

EXP2_CONFIG = {
    'iters': 50,
    'levels': 3,
    'fixed_tau': 0.1,
}

EXP3_CONFIG = {
    'epochs': 100,
    'lr': 0.01,
    'batch_size': 64,
    'embedding_dim': 1,
    'n_samples': 100,
    'feature_dim': 10,
}

DEVICE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_limit': 16,  # in GB, for Tesla T4
}
