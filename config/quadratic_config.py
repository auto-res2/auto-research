"""
Configuration for synthetic quadratic optimization experiment
"""

# Problem parameters
DIMENSION = 10
EIGENVALUE_MIN = 1
EIGENVALUE_MAX = 10

# Optimization parameters
NUM_ITERATIONS = 200
ACM_LEARNING_RATE = 0.1
ACM_BETA = 0.9
ACM_CURVATURE_INFLUENCE = 0.2
SGD_LEARNING_RATE = 0.001
ADAM_LEARNING_RATE = 0.01

# Quick test parameters
QUICK_TEST_ITERATIONS = 20
