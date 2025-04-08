
RANDOM_SEED = 42  # For reproducibility

USE_GPU = True  # Set to False to use CPU
DEVICE = 'cuda' if USE_GPU else 'cpu'

INFERENCE_SPEED_CONFIG = {
    'n_trials': 100,  # Number of trials for benchmarking
    'num_steps': 50,  # Number of steps for sampling
    'latent_dim': 16,  # Dimension of latent vectors
}

STABILITY_CONFIG = {
    'num_steps': 50,  # Number of steps for sampling
    'latent_dim': 16,  # Dimension of latent vectors
    'noise_levels': [0.05, 0.1, 0.2],  # Different noise levels to test
}

BEHAVIOR_SWITCH_CONFIG = {
    'sim_steps': 50,  # Number of simulation steps
    'switch_interval': 5,  # Time interval between behavior switches
    'grid_size': 10,  # Size of the grid environment
    'planning_steps': 3,  # Number of steps for planning
    'noise_scale': 0.05,  # Scale of noise for planning
}

TEST_CONFIG = {
    'inference_speed': {
        'n_trials': 10, 
        'num_steps': 10
    },
    'stability': {
        'num_steps': 10
    },
    'behavior_switch': {
        'sim_steps': 10, 
        'switch_interval': 2
    }
}
