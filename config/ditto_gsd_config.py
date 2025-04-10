"""
Configuration parameters for the DITTO-GSD experiments.
"""
import torch

MODEL_PARAMS = {
    "point_features": 128,
    "grid_features": 64,
    "use_gaussian_decoder": True,
    "use_geometric_loss": True
}

TRAIN_PARAMS = {
    "batch_size": 4,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "weight_decay": 1e-5
}

DATASET_PARAMS = {
    "num_samples": 100,
    "num_points": 1024,
    "noise_levels": [0.01, 0.02, 0.05],
    "sparsity_levels": [1.0, 0.75, 0.5, 0.25]
}

EXPERIMENT_PARAMS = {
    "test_samples": 10,
    "ablation_variants": {
        "baseline_full": {"use_gs_decoder": False, "use_proj": True, "use_geo_loss": True},
        "gs_decoder_only": {"use_gs_decoder": True, "use_proj": True, "use_geo_loss": False},
        "gs_decoder_geo": {"use_gs_decoder": True, "use_proj": True, "use_geo_loss": True},
        "no_proj": {"use_gs_decoder": True, "use_proj": False, "use_geo_loss": True}
    }
}

GPU_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "precision": "float32"  # Use float32 for Tesla T4 compatibility
}
