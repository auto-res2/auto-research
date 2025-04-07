"""
Configuration for PurifyCov++ experiments.
"""

MODEL_CONFIG = {
    'num_classes': 10,           # Number of classes in CIFAR-10
    'diffusion_steps': 20,       # Default diffusion steps for purification
    'batch_size': 32,            # Batch size for training and evaluation
    'epsilon': 0.03,             # Epsilon for adversarial example generation
    'quick_test': False          # Whether to run a quick test with a small subset of data
}

EXPERIMENT_CONFIG = {
    'exp1_timesteps': 20,                 # Timesteps for Experiment 1
    'exp2_record_timepoints': [0, 5, 10, 15, 20],  # Timepoints to record PSNR in Experiment 2
    'exp3_diffusion_steps_list': [10, 20, 30]      # List of diffusion steps for Experiment 3
}
