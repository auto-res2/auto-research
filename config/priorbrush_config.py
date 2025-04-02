"""
Configuration for PriorBrush experiment.
"""

MODEL_CONFIG = {
    "seed": 42,
    "device": "cuda",  # Use CUDA for GPU acceleration
    
    "swift_params": {
        "img_size": 256,  # Image size for generation
        "channels": 3,    # RGB channels
    },
    
    "prior_params": {
        "img_size": 256,  # Image size for generation
        "channels": 3,    # RGB channels
        "refinement_steps": 3,  # Default number of refinement steps
    },
}

EXPERIMENT_CONFIG = {
    "exp1": {
        "prompt": "A futuristic cityscape at dusk with neon lights.",
        "num_runs": 5,  # Number of runs for averaging metrics
        "refinement_steps": 3,  # Refinement steps for PriorBrush
    },
    
    "exp2": {
        "prompt": "A surreal landscape with floating islands and waterfalls.",
        "refinement_steps": 3,  # Refinement steps for PriorBrush with refinement
    },
    
    "exp3": {
        "prompt": "An abstract painting with vibrant colors and dynamic brushstrokes.",
        "step_range": [2, 3, 5],  # Different refinement steps to test
    },
    
    "quick_test": {
        "prompt": "A test scene of a minimalist landscape.",
        "refinement_steps": 2,
        "step_range": [2, 3],
    },
}

OUTPUT_CONFIG = {
    "logs_dir": "logs",
    "figures_dir": "logs",  # Save figures in logs directory for GitHub Actions to find
    "ablation_plot_name": "ablation_study_small.pdf",
    "sensitivity_plot_name": "sensitivity_analysis_small.pdf",
}
