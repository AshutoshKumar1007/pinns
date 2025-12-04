# Physics-Informed Neural Networks (PINNs) for Biharmonic PDEs

Implementation of Physics-Informed Neural Networks (PINNs) and Deep Ritz Method (DRM) for solving fourth-order biharmonic partial differential equations.

## Table of Contents
- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Phase I: PINNs Implementation](#phase-i-pinns-implementation)
- [Phase II: Deep Ritz Method (DRM)](#phase-ii-deep-ritz-method)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Results Summary](#results-summary)
- [Team](#team)
- [License](#license)

## Overview

This project implements two approaches:

1. **Phase I - PINNs**: Direct enforcement of PDE residuals and boundary conditions through the loss function 
2. **Phase II - Deep Ritz Method (DRM)**: Variational energy minimization (see [DRMFinal/README.md](DRMFinal/README.md))

Both leverage automatic differentiation to compute high-order derivatives for fourth-order PDEs.

## Problem Formulation

We solve the biharmonic equation on the unit square domain $\Omega = [0,1] \times [0,1]$:

$$\Delta^2 u = f \quad \text{in } \Omega$$

with boundary conditions:
- **Dirichlet**: $u = g_1$ on $\partial\Omega$
- **Neumann (Second Normal Derivative)**: $\frac{\partial^2 u}{\partial n^2} = g_2$ on $\partial\Omega$

where $\Delta^2 u = \frac{\partial^4 u}{\partial x^4} + 2\frac{\partial^4 u}{\partial x^2 \partial y^2} + \frac{\partial^4 u}{\partial y^4}$ is the biharmonic (fourth-order) operator.

## Test Examples

### Example 3.1:
$$u(x_1, x_2) = \frac{1}{2\pi^2} \sin(\pi x_1) \sin(\pi x_2)$$
- Smooth sinusoidal solution with well-behaved derivatives and functional range.
- Both PINNs and DRM achieve excellent accuracy

### Example 3.2:
$$u(x_1, x_2) = x_1^2 x_2^2 (1-x_1)^2 (1-x_2)^2$$

- Complex polynomial with very small functional range (max ≈ $2.5 \times 10^{-4}$), challenging for neural networks.
- Challenging for neural networks due to scale.
- DRM outperforms PINNs in L² norm, while PINNs excel in higher-order derivatives.


## Phase I: PINNs Implementation

### Methodology

The PINN approach directly minimizes the PDE residual at collocation points:

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \lambda_{\text{dir}} \mathcal{L}_{\text{Dirichlet}} + \lambda_{\text{neu}} \mathcal{L}_{\text{Neumann}}$$

### Network Architecture

**Feed-Forward Neural Network**:
- **Input Layer**    : 2 neurons $(x, y)$ coordinates
- **Hidden Layers**: 4 layers × 50 neurons each
- **Output Layer**: 1 neuron (solution $u$)
- **Activation Function**: Tanh (smooth derivatives for high-order PDEs)
- **Total Parameters**: 7,851 trainable parameters
- **Weight Initialization**: Xavier uniform

### Training Strategy

**Two-Phase Optimization**:
1. **Adam Optimizer** (30,000 epochs):
   - Learning rate: $10^{-4}$ with ReduceLROnPlateau scheduler
   - Broad exploration of solution space
   - Fast convergence to approximate solution

2. **L-BFGS Optimizer** (up to 10,000 iterations):
   - Quasi-Newton method for precision refinement
   - Strong Wolfe line search
   - Fine-tuning near optimum

### Hyperparameters

- Interior collocation points: 5,000
- Boundary collocation points: 4,000
- Dirichlet penalty weight: $\lambda_{\text{dir}} = 5000$
- Neumann penalty weight: $\lambda_{\text{neu}} = 5000$

### Implementation Details

**Key Files** (`final/` directory):
- `run_solver.py` - Main training loop with evaluation and visualization
- `model.py` - Neural network architecture and weight initialization
- `pde.py` - PDE residual computation and loss assembly using automatic differentiation
- `generate_data.py` - Collocation point sampling and ground truth generation
- `g_tr_ex3_1.py`, `g_tr_ex3_2.py` - Exact solutions using SymPy for symbolic differentiation
- `tools.py` - Utility functions for tensor conversions

### Phase I Results

| Example | L² Error | H¹ Error | H² Error | Final Loss |
|---------|----------|----------|----------|------------|
| 3.1     | 1.93e-03 | 3.69e-03 | 6.55e-03 | 9.90e-05 	|
| 3.2     | 2.07e-01 | 3.50e-01 | 3.55e-01 | 2.88e-03 	|

**Note**: Example 3.2 shows higher errors due to its complex polynomial structure $u \sim x^2 y^2 (1-x)^2 (1-y)^2$ with a very small functional range near zero, making it challenging for neural network approximation.

### Outputs

Training generates the following in `results_Example*/` directories:
- `solution_comparison.png` - Side-by-side comparison: exact solution, PINN prediction, absolute error
- `solution_3d_surface.png` - 3D visualization of the solution
- `loss_history.png` - Training loss convergence plot (Adam + L-BFGS phases)
- `results-*.txt` - Detailed training logs and error metrics

## Phase II: Deep Ritz Method

Phase II implements a variational energy minimization approach with U-shaped neural networks and uncertainty-weighted multi-task learning.

**📂 For complete DRM documentation, methodology, and results, see [DRMFinal/README.md](DRMFinal/README.md)**

**Quick Facts**:
- U-shaped FCN architecture: [2]→[64]→[128]→[256]→[256]→[128]→[64]→[1]
- Energy functional minimization with Dirichlet penalty
- Jupyter notebooks with pre-trained checkpoints available
- Hard constraint at center point $(0.5, 0.5)$ was imposed to help the network understand the functional range
- DRM achieved superior L² accuracy on Example 3.2 compared to PINNs
- Training time: 3-4 hours on GPU

## Project Structure

```
pinns/
├── final/                          # Phase I: PINNs (Final Implementation)
│   ├── run_solver.py              # Main training script
│   ├── model.py                   # Neural network architecture
│   ├── pde.py                     # PDE residual and loss computation
│   ├── generate_data.py           # Dataset generation
│   ├── g_tr_ex3_1.py              # Exact solution for Example 3.1
│   ├── g_tr_ex3_2.py              # Exact solution for Example 3.2
│   ├── tools.py                   # Utility functions
│   ├── dataset/                   # Generated collocation points
│   └── results/                   # Training outputs and plots
│       ├── results_Example3_1/
│       └── results_Example3_2/
│
├── DRMFinal/                      # Phase II: Deep Ritz Method
│   ├── README.md                  # Detailed DRM documentation
│   ├── notebooks/                 # Jupyter notebooks
│   │   ├── Example3.1DRM.ipynb
│   │   └── Example3.2DRM.ipynb
│   ├── checkpoints/               # Pre-trained models
│   └── DRM_old/                   # Experimental code
│
├── Notebooks/                     # Additional experiments
│   ├── P2example1.ipynb           # Example 1 notebook
│   ├── P2example2.ipynb           # Example 2 notebook
│   ├── biharmonic_pinns_version2/ # Checkpoints
│   └── Q2biharmonic_pinns/        # Additional checkpoints
│
├── attempt-1/                     # Earlier PINN attempts
├── attempt-2/                     # Earlier PINN attempts
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Installation & Usage

### Quick Start

```powershell
# Clone and setup
git clone https://github.com/AshutoshKumar1007/pinns.git
cd pinns
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Install PyTorch (GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run PINNs (Phase I)
cd final
python generate_data.py          # Generate collocation points
python run_solver.py             # Train & evaluate

# Run DRM (Phase II)
cd ../DRMFinal/notebooks
jupyter notebook Example3.1DRM.ipynb
```

**Configuration**: Edit `Config` class in `final/run_solver.py` to select example and adjust hyperparameters.

**Dependencies**: `torch`, `numpy`, `scipy`, `matplotlib`, `sympy` (see `requirements.txt`)



**Generated Outputs** (in `final/results_Example*/` and `DRMFinal/checkpoints/`):
- Solution comparison plots, 3D visualizations, loss history, training logs, pre-trained models

## Team

**Group 2** - Deep Learning for Differential Equations Course

**Contributors**:
- [Dr. Ramesh Chandra Sau](https://github.com/rcs1994) - Course Instructor & Project Mentor
- [Sadu Varshini](https://github.com/varshini1782006)
- [Ashutosh Kumar](https://github.com/AshutoshKumar1007)
- [Sarthak Jain](https://github.com/jainsarthak0205)
- [Suraj Kumar](https://github.com/c0mpl1cat3d1)
- [Gowrav Sharma](https://github.com/Gowravsharma)
- [Patan Gouse](https://github.com/Gouse2005)


## License

MIT License - see [LICENSE](LICENSE) file. Copyright (c) 2025 Group 2.

---

**Note**: `attempt-1/` and `attempt-2/` document development iterations. Final implementations: `final/` (PINNs) and `DRMFinal/` (DRM with detailed README).

