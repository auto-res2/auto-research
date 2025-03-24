"""
Configuration file for DPC-3D experiment
"""

# General experiment configuration
EXPERIMENT_CONFIG = {
    "seed": 42,
    "device": "cuda",
    "log_interval": 10,
    "num_workers": 4,
    "checkpoint_interval": 100,
}

# Dataset configuration
DATA_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "batch_size": 32,
    "max_atoms": 50,
    "max_conformers": 10,
    "smiles_path": "data/molecules.csv",
    "processed_data_dir": "data/processed/",
}

# Model configuration
MODEL_CONFIG = {
    # 1D Language Model (LM) configuration
    "lm": {
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "vocab_size": 128,  # SELFIES vocabulary size
        "max_seq_len": 200,
    },
    
    # Diffusion model configuration
    "diffusion": {
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "beta_schedule": "cosine",
        "num_diffusion_steps": 1000,
        "sampling_steps": 100,  # Fewer steps for faster sampling
    },
    
    # Dynamic prompt tuning configuration
    "prompt_tuning": {
        "prompt_dim": 256,
        "update_interval": 10,  # Update prompt every N diffusion steps
        "prompt_lr": 1e-4,
    },
    
    # Bayesian adaptation module
    "bayesian": {
        "use_bayesian": True,
        "n_samples": 10,
        "uncertainty_threshold": 0.5,
    },
}

# Training configuration
TRAIN_CONFIG = {
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "epochs": 50,
    "warmup_steps": 1000,
    "gradient_clip": 1.0,
    "early_stopping_patience": 10,
}

# Evaluation configuration
EVAL_CONFIG = {
    "metrics": ["rmsd", "validity", "energy", "diversity"],
    "num_samples": 100,
    "save_conformers": True,
}

# Experiment variants for ablation study
VARIANTS = {
    "full": {"prompt_tuning": True, "bayesian": True},
    "static_prompt": {"prompt_tuning": False, "bayesian": True},
    "no_bayesian": {"prompt_tuning": True, "bayesian": False},
}
