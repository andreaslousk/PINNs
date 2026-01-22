# Physics-Informed Neural Networks (PINNs) Repository

A modular repository for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs).

## Overview

This repository provides a flexible framework for training PINNs on various PDE problems. It includes:

- **Reusable core components**: Model architectures, training loops, data generation
- **Multiple PDE problems**: Navier-Stokes (Taylor-Green vortex, lid-driven cavity), Black-Scholes
- **Exact solutions**: Analytical benchmarks for validation
- **Visualization tools**: Generate images and videos of solutions

## Repository Structure

```
pinns-repository/
├── core/                      # Shared PINN infrastructure
│   ├── models/               # Neural network architectures
│   ├── training/             # Training loops and utilities
│   ├── data/                 # Point generation
│   └── evaluation/           # Error metrics and visualization
│
├── problems/                  # Problem-specific implementations
│   ├── navier_stokes/        # Fluid dynamics problems
│   │   ├── taylor_green/     # Taylor-Green vortex
│   │   └── lid_driven/       # Lid-driven cavity
│   └── black_scholes/        # Financial PDEs
│
├── scripts/                   # Command-line scripts
│   ├── train.py              # Universal training script
│   └── evaluate.py           # Evaluation script
│
└── outputs/                   # Training outputs
    └── [problem_name]/
        ├── checkpoints/      # Saved models
        ├── logs/             # Training logs
        └── figures/          # Visualizations
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pinns-repository.git
cd pinns-repository

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- (Optional) CUDA for GPU acceleration

## Quick Start

### Training a Model

Train the Taylor-Green vortex problem:

```bash
python scripts/train.py --problem taylor_green --epochs 10000
```

Train the lid-driven cavity problem:

```bash
python scripts/train.py --problem lid_driven --epochs 10000
```

### Command-Line Options

```bash
python scripts/train.py --help

Options:
  --problem        Problem to solve (taylor_green, lid_driven, etc.)
  --epochs         Number of training epochs (default: 10000)
  --config         Path to config file (optional)
  --device         Device to use (cpu, cuda, mps)
  --output-dir     Output directory for results
  --save-freq      Save checkpoint every N epochs (default: 1000)
  --eval-freq      Evaluate every N epochs (default: 100)
```

## Problems

### 1. Taylor-Green Vortex

**Description**: 2D decaying vortex with exact analytical solution. Perfect for validating PINN implementation.

**Equations**: Incompressible Navier-Stokes
- Domain: [0, 2π] × [0, 2π]
- Boundary conditions: Periodic
- Exact solution available

**Features**:
- Quantitative error metrics (L2 error)
- Beautiful vortex visualization
- Good starting problem

**Run**:
```bash
python scripts/train.py --problem taylor_green --epochs 10000
```

### 2. Lid-Driven Cavity

**Description**: Classic CFD benchmark. Fluid in a square cavity with moving top wall.

**Equations**: Incompressible Navier-Stokes
- Domain: [0, 1] × [0, 1]
- Boundary conditions: No-slip walls, moving top lid
- Benchmark: Ghia et al. (1982) reference data

**Features**:
- Realistic flow problem
- Comparison with numerical benchmark
- More challenging than Taylor-Green

**Run**:
```bash
python scripts/train.py --problem lid_driven --epochs 10000
```

### 3. Black-Scholes (Coming Soon)

European and American option pricing.

## Project Workflow

### 1. Problem Setup

Each problem defines:
- Physics loss (PDE residuals)
- Boundary conditions
- Initial conditions
- Exact solution (if available)

### 2. Training

The universal training script:
1. Loads problem configuration
2. Generates training points
3. Initializes PINN model
4. Trains using physics-informed losses
5. Saves checkpoints and logs

### 3. Evaluation

After training:
- Compare with exact solution or benchmarks
- Generate visualizations
- Compute error metrics

## Adding a New Problem

To add a new PDE problem:

1. **Create problem directory**:
```bash
mkdir -p problems/your_problem/case_name
```

2. **Create required files**:
```python
# problems/your_problem/__init__.py
# problems/your_problem/losses.py        # Physics loss
# problems/your_problem/case_name/setup.py
# problems/your_problem/case_name/train.py
```

3. **Add to training script**:
Update `scripts/train.py` problem map.

4. **Run**:
```bash
python scripts/train.py --problem case_name
```

## Configuration

### Problem-Specific Config

Each problem has its own configuration file with:
- Domain parameters
- Physical parameters (viscosity, density, etc.)
- Training hyperparameters
- Loss weights

### Global Config

Base configuration in `configs/base_config.yaml`:
- Model architecture
- Optimizer settings
- Training parameters

## Model Architecture

### Base PINN

Fully-connected feedforward network:
- Input: (x, y, t)
- Hidden layers: 4 layers × 128 neurons
- Activation: Tanh
- Output: (u, v, p) or problem-specific

### Advanced Models

- **AdaptivePINN**: Learnable activation frequencies
- **ResidualPINN**: Deeper networks with skip connections

## Training Details

### Loss Function

Total loss = λ₁ · Physics Loss + λ₂ · Boundary Loss + λ₃ · Initial Loss

Where:
- **Physics Loss**: PDE residuals at collocation points
- **Boundary Loss**: Boundary condition violations
- **Initial Loss**: Initial condition errors

### Training Points

- **Collocation points**: Random interior points (10,000)
- **Boundary points**: Points on domain boundaries (1,000)
- **Initial points**: Points at t=0 (1,000)

### Automatic Differentiation

PINNs use automatic differentiation to compute:
- First derivatives: ∂u/∂x, ∂u/∂y, ∂u/∂t
- Second derivatives: ∂²u/∂x², ∂²u/∂y²

No finite differences needed!

## Visualization

Generate animations of solutions:
```python
from core.evaluation.visualization import create_animation

create_animation(
    model=model,
    problem='taylor_green',
    output_path='outputs/animation.mp4'
)
```

## Results

After training, find outputs in:
```
outputs/[problem_name]/
├── checkpoints/
│   ├── checkpoint_epoch_1000.pt
│   ├── checkpoint_epoch_2000.pt
│   └── final_model.pt
├── logs/
│   └── training_history.json
└── figures/
    ├── velocity_field.png
    ├── error_plot.png
    └── animation.mp4
```

## Tips for Success

1. **Start simple**: Begin with Taylor-Green vortex (has exact solution)
2. **Monitor losses**: Watch all three loss components during training
3. **Tune weights**: Adjust λ values if one loss dominates
4. **Increase resolution gradually**: Start with fewer points, increase later
5. **Use GPU**: Training is much faster on GPU (CUDA or MPS)

## Common Issues

### High Physics Loss
- Increase collocation points
- Reduce learning rate
- Increase λ_physics weight

### Boundary Conditions Not Satisfied
- Increase λ_boundary weight
- Add more boundary points
- Check boundary loss implementation

### Training Instability
- Reduce learning rate
- Use learning rate scheduler
- Check for NaN gradients

## References

### Papers
- Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems"
- Ghia et al. (1982): "High-Re solutions for incompressible flow using the Navier-Stokes equations"

### Resources
- [PINN Tutorial](https://github.com/maziarraissi/PINNs)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new problems
4. Submit a pull request

## License

MIT License - see LICENSE file

## Citation

If you use this code in your research, please cite:

```bibtex
@software{PINNs,
  title = {Physics-Informed Neural Networks Repository},
  author = {Andreas Louskos},
  year = {2026},
  url = {https://github.com/andreaslousk/PINNs}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact andreaslousk@gmail.com.