# Configuration for ACM Optimizer Experiments

This directory contains configuration files for running the Adaptive Curvature Momentum (ACM) optimizer experiments.

## Files

- `experiment_config.py`: Main configuration file with parameters for all experiments

## Parameters

### Synthetic Function Benchmarking

- `iterations`: Maximum number of iterations for optimization
- `loss_threshold`: Early stopping threshold
- `lr`: Learning rate
- `beta1`: Momentum decay parameter
- `beta2`: Squared gradient decay parameter
- `curvature_coeff`: Coefficient for curvature estimation

### CIFAR-10 Classification

- `epochs`: Number of training epochs
- `batch_size`: Batch size for training (optimized for Tesla T4 16GB VRAM)
- `num_workers`: Number of data loader workers
- `lr`: Learning rate
- `beta1`: Momentum decay parameter
- `beta2`: Squared gradient decay parameter
- `curvature_coeff`: Coefficient for curvature estimation
- `weight_decay`: Weight decay (L2 regularization)
- `use_subset`: Whether to use a subset of data for quick testing
- `subset_batches`: Number of batches to use in subset mode

### Ablation Studies

- `iterations`: Maximum number of iterations for optimization
- `loss_threshold`: Early stopping threshold

## Usage

The configuration is automatically loaded by `src/main.py` when running experiments.

To run a quick test with reduced iterations/epochs:

```bash
python src/main.py --quick
```

To run a specific experiment:

```bash
python src/main.py --exp 1  # Run synthetic benchmarking
python src/main.py --exp 2  # Run CIFAR-10 classification
python src/main.py --exp 3  # Run ablation studies
```
