#!/usr/bin/env python3
"""
Configuration for HACP experiments.
"""

# Environment configuration
ENV_NAME = "MiniGrid-Empty-8x8-v0"
FALLBACK_ENV = "CartPole-v1"

# Model configuration
HIDDEN_SIZE = 128
PLANNING_TEMPERATURE = 0.8

# Training configuration
NUM_EPISODES = 100
LEARNING_RATE = 1e-3
GAMMA = 0.99
DEVICE = "cuda"  # Use "cpu" if no GPU is available

# Evaluation configuration
EVAL_EPISODES = 10
NUM_STEPS_REPRESENTATION = 200

# Test configuration (for quick tests)
TEST_HIDDEN_SIZE = 64
TEST_NUM_EPISODES = 5
