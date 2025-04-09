"""
Configuration for SpectralGraph-ST experiments.
"""

EXPERIMENT_CONFIG = {
    "random_seed": 42,
    "num_images": 20,
    "test_mode": False,  # Set to True for a quick test run
}

MODEL_CONFIGS = {
    "egtr": {},
    "spectral_graph_st": {
        "use_spectral_filter": True,
        "use_stochastic_sampling": True
    },
    "spectral_graph_st_no_filter": {
        "use_spectral_filter": False,
        "use_stochastic_sampling": True
    },
    "spectral_graph_st_no_sampling": {
        "use_spectral_filter": True,
        "use_stochastic_sampling": False
    }
}

NOISE_LEVELS = [(0.0, 0.0), (5.0, 0.1), (10.0, 0.2), (15.0, 0.3)]
