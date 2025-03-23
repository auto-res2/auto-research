"""Configuration parameters for the ACM optimizer experiment."""

# Optimizer configurations
OPTIMIZER_CONFIG = {
    'acm': {
        'lr': 0.01,
        'beta': 0.9,
        'curvature_influence': 0.1,
    },
    'adam': {
        'lr': 0.001,
        'betas': (0.9, 0.999),
    },
    'sgd_mom': {
        'lr': 0.01,
        'momentum': 0.9,
    }
}

# Experiment configurations
EXPERIMENT_CONFIG = {
    'synthetic': {
        'num_iters': 100,
        'log_interval': 20,
    },
    'cifar10': {
        'batch_size': 128,
        'num_epochs': 10,
        'log_interval': 100,
    },
    'mnist': {
        'batch_size': 128,
        'num_epochs': 5,
        'log_interval': 100,
    }
}

# Model configurations
MODEL_CONFIG = {
    'cnn_cifar10': {
        'conv1_channels': 32,
        'conv2_channels': 64,
        'fc1_size': 512,
        'dropout_rate': 0.5,
    },
    'cnn_mnist': {
        'conv1_channels': 32,
        'conv2_channels': 64,
        'fc1_size': 128,
        'dropout_rate': 0.25,
    }
}
