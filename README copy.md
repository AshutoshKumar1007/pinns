# Physics-Informed Neural Networks (PINNs) for Biharmonic PDEs

A comprehensive implementation of Physics-Informed Neural Networks (PINNs) and Deep Ritz Method (DRM) for solving fourth-order biharmonic partial differential equations. This project demonstrates two neural network-based approaches to approximate solutions of biharmonic problems with mixed boundary conditions.

## Table of Contents
- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Phase I: PINNs Implementation](#phase-i-pinns-implementation)
- [Phase II: Deep Ritz Method (DRM)](#phase-ii-deep-ritz-method-drm)
- [Test Examples](#test-examples)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Team](#team)
- [License](#license)

## Overview

This project implements two neural network methods for solving biharmonic PDEs:

1. **Physics-Informed Neural Networks (PINNs)** - Phase I: Direct enforcement of PDE residuals and boundary conditions through loss functions
2. **Deep Ritz Method (DRM)** - Phase II: Variational formulation minimizing energy functionals with penalty methods

Both methods leverage automatic differentiation in PyTorch to compute high-order derivatives required for fourth-order PDEs without explicit derivative formulations.

## Problem Formulation

We solve the biharmonic equation on the unit square domain $\Omega = [0,1] \times [0,1]$:

$$\Delta^2 u = f \quad \text{in } \Omega$$

with boundary conditions:
- **Dirichlet**: $u = g_1$ on $\partial\Omega$
- **Neumann (Second Normal Derivative)**: $\frac{\partial^2 u}{\partial n^2} = g_2$ on $\partial\Omega$

where $\Delta^2 u = (u_{xx} + u_{yy})_{xx} + (u_{xx} + u_{yy})_{yy}$ is the biharmonic operator.

## Phase I: PINNs Implementation

### Methodology

The PINN approach directly minimizes the PDE residual at collocation points:

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \lambda_{\text{dir}} \mathcal{L}_{\text{Dirichlet}} + \lambda_{\text{neu}} \mathcal{L}_{\text{Neumann}}$$

where:
- $\mathcal{L}_{\text{PDE}} = \text{MSE}(\Delta^2 u_\theta - f)$ at interior points
- $\mathcal{L}_{\text{Dirichlet}} = \text{MSE}(u_\theta - g_1)$ at boundary points
- $\mathcal{L}_{\text{Neumann}} = \text{MSE}(\frac{\partial^2 u_\theta}{\partial n^2} - g_2)$ at boundary points

### Network Architecture

**Feed-Forward Neural Network**:
- **Input Layer**: 2 neurons $(x, y)$ coordinates
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

| Example | L² Error | H¹ Error | H² Error | Training Time | Final Loss |
|---------|----------|----------|----------|---------------|------------|
| 3.1     | 1.93e-03 | 3.69e-03 | 6.55e-03 | 3,466 s (~58 min) | 9.90e-05 |
| 3.2     | 2.07e-01 | 3.50e-01 | 3.55e-01 | 3,677 s (~61 min) | 2.88e-03 |

**Note**: Example 3.2 shows higher errors due to its complex polynomial structure $u \sim x^2 y^2 (1-x)^2 (1-y)^2$ with a very small functional range near zero, making it challenging for neural network approximation.

### Outputs

Training generates the following in `results_Example*/` directories:
- `solution_comparison.png` - Side-by-side comparison: exact solution, PINN prediction, absolute error
- `solution_3d_surface.png` - 3D visualization of the solution
- `loss_history.png` - Training loss convergence plot (Adam + L-BFGS phases)
- `results-*.txt` - Detailed training logs and error metrics

## Phase II: Deep Ritz Method (DRM)

### Methodology

The DRM formulates the PDE as an energy minimization problem:

$$\mathcal{L}_\lambda(v) = \frac{1}{2}\int_{\Omega} |D^2 v|^2 \, dx - \int_{\Omega} f \cdot v \, dx - \int_{\partial\Omega} g_2 \frac{\partial v}{\partial n} \, ds + \frac{\lambda}{2}\int_{\partial\Omega} (v - g_1)^2 \, ds$$

where $|D^2 v|^2 = v_{xx}^2 + 2v_{xy}^2 + v_{yy}^2$ is the squared Frobenius norm of the Hessian.

**Uncertainty-Weighted Multi-Task Learning**:

$$\mathcal{L} = \frac{1}{2} e^{-s} \mathcal{L}_0 + \frac{1}{2} s$$

where $s$ is a learnable parameter balancing interior and boundary contributions.

### Network Architecture

**U-Shaped Fully Connected Network (U_FCN)**:
- **Architecture**: [2] → [64] → [128] → [256] → [256] → [128] → [64] → [1]
- **Activation**: Tanh throughout
- **Rationale**: U-shape provides better representation capacity for complex solutions

*Note: ResNet architecture (DRM_Model) was also experimented with but U_FCN provided better final results.*

### Training Configuration

- **Optimizer**: Adam with learning rate $10^{-6}$
- **Epochs**: 10,000
- **Sampling**: 10,000 interior + 4,000 boundary points per epoch (resampled each epoch)
- **Dirichlet Penalty**: $\lambda = 1000$
- **Precision**: Double precision (`torch.float64`) for numerical stability

### Phase II Results

| Example | L² Error | H¹ Error | H² Error | Training Time |
|---------|----------|----------|----------|---------------|
| 3.1     | 2.13e-03 | 9.16e-02 | 1.11e-01 | ~4 hours      |
| 3.2     | 5.97e-04 | 5.05e-01 | 8.46e-01 | ~3 hours      |

**Key Observations**:
- Example 3.2 shows higher H¹ and H² errors due to its complex polynomial structure
- Hard constraint at center point $(0.5, 0.5)$ was imposed to help the network understand the functional range
- DRM achieves better L² error for Example 3.2 compared to PINNs

### Implementation Details

**Key Files** (`DRMFinal/notebooks/`):
- `Example3.1DRM.ipynb` - Complete DRM implementation for Example 3.1
- `Example3.2DRM.ipynb` - Complete DRM implementation for Example 3.2

**Checkpoints** (`DRMFinal/checkpoints/`):
- Pre-trained models saved at epoch 10,000 for both examples
- Can be loaded for inference without retraining

## Test Examples

### Example 3.1: Trigonometric Solution

$$u(x_1, x_2) = \frac{1}{2\pi^2} \sin(\pi x_1) \sin(\pi x_2)$$

**Characteristics**:
- Smooth sinusoidal solution
- Well-behaved derivatives
- Easier for neural network approximation
- Both PINNs and DRM achieve excellent accuracy

### Example 3.2: Polynomial Solution

$$u(x_1, x_2) = x_1^2 x_2^2 (1-x_1)^2 (1-x_2)^2$$

**Characteristics**:
- Complex polynomial with 8th-degree terms
- Very small functional range (max ≈ $2.4 \times 10^{-4}$ at center)
- Vanishes at all boundaries naturally
- Challenging for neural networks due to scale

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

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Setup

1. **Clone the repository**:
```powershell
git clone https://github.com/AshutoshKumar1007/pinns.git
cd pinns
```

2. **Create virtual environment**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

4. **Install PyTorch** (choose appropriate version):

   For **CUDA-enabled GPU**:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

   For **CPU-only**:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

### Dependencies

Core packages (from `requirements.txt`):
- `torch>=1.8.0` - Deep learning framework with automatic differentiation
- `numpy>=1.19` - Numerical computations
- `scipy>=1.5` - Scientific computing utilities
- `matplotlib>=3.1` - Plotting and visualization
- `sympy` - Symbolic mathematics for exact solutions

## Usage

### Phase I: PINNs

**Generate datasets**:
```powershell
cd final
python generate_data.py
```

**Train and evaluate**:
```powershell
# Example 3.1
python run_solver.py  # (Set example_name="Example3.1" in script)

# Example 3.2
python run_solver.py  # (Set example_name="Example3.2" in script)
```

**Modify configuration**: Edit the `Config` class in `run_solver.py`:
```python
class Config:
    def __init__(self, example_name):
        self.example_name = example_name  # "Example3.1" or "Example3.2"
        self.adam_epochs = 30000          # Adam training epochs
        self.lbfgs_max_iter = 10000       # L-BFGS iterations
        self.lr = 1e-4                    # Learning rate
        self.bw_dir = 5000.0              # Dirichlet boundary weight
        self.bw_neu = 5000.0              # Neumann boundary weight
```

### Phase II: DRM

**Run Jupyter notebooks**:
```powershell
cd DRMFinal/notebooks
jupyter notebook Example3.1DRM.ipynb
# or
jupyter notebook Example3.2DRM.ipynb
```

**Load pre-trained models**:
```python
import torch
checkpoint = torch.load('checkpoints/Example31_biharmonic_model_Adam_epoch10000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Experiments and Notebooks

Explore additional implementations in `Notebooks/`:
```powershell
cd Notebooks
jupyter lab
# Open P2example1.ipynb or P2example2.ipynb
```

## Results

### Comparative Analysis

| Method | Example | L² Error | H¹ Error | H² Error | Time (GPU) |
|--------|---------|----------|----------|----------|------------|
| **PINNs** | 3.1 | **1.93e-03** | **3.69e-03** | **6.55e-03** | 58 min |
| **DRM**   | 3.1 | 2.13e-03 | 9.16e-02 | 1.11e-01 | 240 min |
| **PINNs** | 3.2 | 2.07e-01 | 3.50e-01 | 3.55e-01 | 61 min |
| **DRM**   | 3.2 | **5.97e-04** | 5.05e-01 | 8.46e-01 | 180 min |

### Key Findings

1. **PINNs advantages**:
   - Faster training convergence (4-5× speedup)
   - Superior performance on smooth solutions (Example 3.1)
   - Better higher-order derivative approximation (H² errors)

2. **DRM advantages**:
   - Better L² accuracy for complex polynomials (Example 3.2)
   - Variational formulation ensures energy minimization
   - More principled mathematical framework

3. **Example 3.2 challenges**:
   - Extremely small functional range poses difficulty for both methods
   - Higher-order errors reflect challenges in approximating complex derivatives
   - Hard constraints at specific points can improve convergence

### Visualizations

Both methods generate comprehensive visualizations:
- **Solution comparison plots**: Exact vs. predicted vs. pointwise error
- **3D surface plots**: Interactive visualization of solution fields
- **Loss convergence**: Training dynamics over epochs/iterations

## Team

**Group 2** - Deep Learning for Differential Equations Course

### Contributors
- [AshutoshKumar1007](https://github.com/AshutoshKumar1007)
- Collaborator 2 (Please add GitHub profile)
- Collaborator 3 (Please add GitHub profile)
- Collaborator 4 (Please add GitHub profile)
- Collaborator 5 (Please add GitHub profile)

*For detailed methodology, theoretical analysis, and ablation studies, please refer to the final project report.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Deep Learning for Differential Equations course staff and instructors
- PyTorch team for the excellent automatic differentiation framework
- Research community for PINNs and Deep Ritz Method foundations

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*.

2. E, W., & Yu, B. (2018). The Deep Ritz Method: A deep learning-based numerical algorithm for solving variational problems. *Communications in Mathematics and Statistics*.

3. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*.

---

**Note**: This repository contains multiple implementation attempts (`attempt-1/`, `attempt-2/`) documenting the development process. The final implementations are in `final/` (PINNs) and `DRMFinal/` (DRM).

