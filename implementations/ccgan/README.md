# Context-Conditional GAN (CCGAN)

Implementation of Context-Conditional Generative Adversarial Networks for semi-supervised learning with in-painting.

## Overview

This implementation is based on the approach described in "Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks" by Emily Denton, Sam Gross, and Rob Fergus. The model performs image in-painting by:

1. Removing random patches from input images
2. Training a generator to fill in the missing regions based on surrounding context
3. Using a discriminator to judge whether in-painted images are real or fake

## Architecture

- **Generator**: Encoder-decoder architecture that takes masked images and generates content for missing regions
- **Discriminator**: Convolutional network that classifies images as real or in-painted
- **Training**: Adversarial loss combined with reconstruction loss for the masked regions

## Configuration

- **Learning Rate**: 0.0001
- **Training Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 128x128
- **Dataset**: CIFAR-10 (can be easily replaced with other datasets)

## Features

- Random mask generation with variable sizes (32x32 to 64x64 patches)
- Weighted reconstruction loss to ensure quality in-painting
- Sample image generation during training
- Model checkpointing
- GPU support with automatic fallback to CPU

## Usage

```bash
cd implementations/ccgan/
python3 ccgan.py
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Output

- **Checkpoints**: Saved to `checkpoints/` directory
- **Sample Images**: Saved to `samples/` directory every 10 epochs
- **Training Progress**: Printed to console with loss values

## Customization

You can easily modify:
- Dataset by changing the `torchvision.datasets` call
- Mask sizes by adjusting `mask_size_range` parameter
- Network architectures in the `Generator` and `Discriminator` classes
- Loss weights and training hyperparameters
