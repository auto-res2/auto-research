"""
Configuration for CEDP experiments
"""

EXPERIMENT_CONFIG = {
    'batch_size': 32,
    'seed': 42,
    'max_samples': 128,  # Limit number of samples for quicker testing
}

CEDP_CONFIG = {
    'noise_level': 0.3,
    'iterations': 5,
    'consistency_threshold': 0.01,
    'adaptive_decay_factor': 0.9,
}

GPU_CONFIG = {
    'precision': 'float32',  # Use 'float16' for faster execution if needed
}
