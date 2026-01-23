"""Configuration for Taylor-Green vortex problem"""
import numpy as np

# ============================================================================
# Problem Parameters
# ============================================================================

# Domain (Taylor-Green uses [0, 2π] × [0, 2π])
X_MIN = 0.0
X_MAX = 2 * np.pi
Y_MIN = 0.0
Y_MAX = 2 * np.pi
T_MAX = 3.0  # Simulate up to t=3.0

# Physical parameters
RHO = 1.0      # Density
NU = 0.01      # Kinematic viscosity

# ============================================================================
# Training Parameters
# ============================================================================

# Number of training points
N_COLLOCATION = 10000  # Interior points for physics loss
N_BOUNDARY = 1000      # Points per boundary for BC loss
N_INITIAL = 1000       # Points at t=0 for initial condition

# Training settings
NUM_EPOCHS = 10000
LEARNING_RATE = 1e-3
BATCH_SIZE = None      # None = full batch

# Loss weights
LAMBDA_PHYSICS = 1.0
LAMBDA_BOUNDARY = 100.0
LAMBDA_IC = 100.0

# ============================================================================
# Model Architecture
# ============================================================================

INPUT_DIM = 3          # (x, y, t)
OUTPUT_DIM = 3         # (u, v, p)
HIDDEN_DIM = 128
NUM_LAYERS = 4
ACTIVATION = 'relu'    # 'tanh', 'relu', 'gelu', 'sin'

# ============================================================================
# Output Settings
# ============================================================================

OUTPUT_DIR = 'outputs/taylor_green'
SAVE_FREQ = 1000       # Save checkpoint every N epochs
EVAL_FREQ = 100        # Evaluate every N epochs

# Visualization
GRID_SIZE = 64         # Grid resolution for visualization
NUM_FRAMES = 100       # Number of frames for animation

# ============================================================================
# Device
# ============================================================================

DEVICE = 'mps'         # 'cpu', 'cuda', or 'mps' (for Apple Silicon)