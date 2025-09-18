# Context-Conditional GAN (CCGAN) Implementation

This implementation is based on the paper "Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks" by Emily Denton, Sam Gross, and Rob Fergus.

## Overview

The Context-Conditional GAN performs semi-supervised learning through in-painting. The model:

1. Takes images with random patches removed
2. Uses a generator network to fill in the holes based on surrounding pixels
3. Uses a discriminator network to judge if images are real (unaltered) or fake (in-painted)
4. Acts as a regularizer for standard supervised training

## Architecture

### Generator
- Encoder-decoder architecture with skip connections
- Downsamples input to capture context
- Upsamples to reconstruct missing regions
- Uses L1 loss for pixel-level reconstruction and adversarial loss

### Discriminator
- Convolutional neural network
- Distinguishes between real and in-painted images
- Provides adversarial signal to the generator

## Training Parameters

- Learning rate: 0.0001
- Training epochs: 100
- Batch size: 16
- Image size: 128x128
- Optimizer: Adam with betas=(0.5, 0.999)

## Features

- Random mask generation for in-painting
- L1 reconstruction loss for better pixel-level accuracy
- Sample image generation during training
- Model checkpointing
- Support for CIFAR-10 dataset (with fallback to custom datasets)

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python ccgan.py
```

## Output

- Sample images saved in `samples/` directory every 10 epochs
- Model checkpoints saved in `checkpoints/` directory every 25 epochs
- Final trained model saved as `ccgan_final.pth`

## File Structure

```
ccgan_implementation/
├── ccgan.py              # Main implementation
├── requirements.txt      # Dependencies
├── README.md            # This file
├── samples/             # Generated sample images (created during training)
├── checkpoints/         # Model checkpoints (created during training)
└── data/               # Dataset directory (CIFAR-10 will be downloaded here)
```

## Implementation Details

The implementation follows the paper's methodology:

1. **In-painting Task**: Random square patches are removed from images
2. **Generator Training**: Learns to fill holes using surrounding context
3. **Discriminator Training**: Learns to distinguish real from in-painted images
4. **Semi-supervised Learning**: The in-painting task acts as a regularizer

The model uses a combination of adversarial loss and L1 reconstruction loss to ensure both realistic and accurate in-painting results.
