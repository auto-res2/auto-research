"""Configuration for ACM optimizer experiments."""

# General configuration
RANDOM_SEED = 42
DEVICE = "cuda"  # or "cpu" if GPU is not available
QUICK_TEST = True  # Set to True for quick testing with minimal iterations

# Synthetic experiment configuration
SYNTHETIC_ITERS = 1000
SYNTHETIC_QUICK_ITERS = 20

# CIFAR-10 experiment configuration
CIFAR_BATCH_SIZE = 128
CIFAR_EPOCHS = 20
CIFAR_QUICK_EPOCHS = 1
CIFAR_LEARNING_RATE = 0.001
CIFAR_WEIGHT_DECAY = 5e-4

# MNIST experiment configuration
MNIST_BATCH_SIZE = 256
MNIST_EPOCHS = 10
MNIST_QUICK_EPOCHS = 1
MNIST_LEARNING_RATE = 0.001

# Optimizer configurations
OPTIMIZERS = {
    "ACM": {
        "lr": 0.001,
        "beta": 0.9,
        "curvature_influence": 0.1,
    },
    "Adam": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
    },
    "SGD_momentum": {
        "lr": 0.001,
        "momentum": 0.9,
    }
}
