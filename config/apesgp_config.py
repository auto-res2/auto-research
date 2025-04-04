"""
Configuration file for APESGP experiments.
"""

D = 50  # overall dimension (can be set to any value in [50,100] for a full experiment)
NUM_SUBFUNCTIONS = 5
SUB_DIM = D // NUM_SUBFUNCTIONS  # each subfunction gets an equal partition

COST_FULL_DEFAULT = 1.0
COST_PARTIAL_DEFAULT = 0.2

EXP1_ITERATIONS = 50
EXP2_ITERATIONS = 50
EXP3_ITERATIONS = 50
EXP3_SEED_RUNS = 3

TEST_ITERATIONS = 20
TEST_SEED_RUNS = 2
