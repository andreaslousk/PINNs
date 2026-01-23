"""Configuration for Black-Scholes European Call Option"""
import torch

# ============================================================================
# Black-Scholes Parameters
# ============================================================================

# Option parameters
K = 100.0          # Strike price
T = 1.0            # Time to expiration (years)
r = 0.05           # Risk-free rate
sigma = 0.2        # Volatility
q = 0.02           # Continuous dividend yield

# Grid parameters
M = 1000           # Number of steps in spot price
N = 1000           # Number of steps in time

# ============================================================================
# Training Data Parameters
# ============================================================================

# Data filtering (by delta)
LOWER_DELTA_BOUND = 0.35
UPPER_DELTA_BOUND = 0.65

# Number of training points
NUM_COLLOCATION_POINTS = 15000   # Interior PDE points
NUM_EXPIRATION_POINTS = 300      # Points at expiration (t=T)
NUM_BOUNDARY_POINTS = 300        # Points at boundary (S=0)

# ============================================================================
# Training Parameters
# ============================================================================

NUM_EPOCHS = 531
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# Loss weights
LOSS_WEIGHT_F = 100        # Physics (PDE) loss weight
LOSS_WEIGHT_B = 1          # Boundary condition loss weight
LOSS_WEIGHT_EXP = 1        # Expiration condition loss weight

# Gradient clipping
GRAD_CLIP_MAX_NORM = 1.0

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 100
SCHEDULER_GAMMA = 0.95

# ============================================================================
# Model Architecture
# ============================================================================

INPUT_DIM = 2              # (S, t) - spot price and time
OUTPUT_DIM = 1             # Option price
HIDDEN_DIM = 50            # Number of neurons per layer
NUM_LAYERS = 10            # Number of hidden layers
ACTIVATION = 'relu'        # Activation function

# ============================================================================
# Output Settings
# ============================================================================

OUTPUT_DIR = 'outputs/black_scholes'
MODEL_SAVE_PATH = 'outputs/black_scholes/checkpoints/best_model.pth'
FINAL_MODEL_PATH = 'outputs/black_scholes/checkpoints/trained_model.pth'

SAVE_BEST_MODEL = True
SAVE_FREQ = 10             # Print progress every N epochs

# ============================================================================
# Reproducibility
# ============================================================================

SEED = 42

# ============================================================================
# Device
# ============================================================================

# Device selection (automatic)
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
