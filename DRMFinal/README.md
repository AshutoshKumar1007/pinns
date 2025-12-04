# Deep Ritz Method (DRM) for Biharmonic PDEs

Implementation of the Deep Ritz Method to solve biharmonic partial differential equations with Dirichlet boundary conditions on the unit square domain $\Omega = [0,1] \times [0,1]$.

##  Overview
This project implements a neural network-based approach (Deep Ritz Method) to approximate solutions of fourth-order biharmonic PDEs:

$$\Delta^2 u = f \quad \text{in } \Omega$$

with boundary conditions:
- $u = g_1$ on $\partial\Omega$ (Dirichlet)
- $\frac{\partial^2 u}{\partial n^2} = g_2$ on $\partial\Omega$ (Second normal derivative)


## Test Problems

**Example 3.1:** $u(x_1, x_2) = \frac{1}{2\pi^2} \sin(\pi x_1) \sin(\pi x_2)$

**Example 3.2:** $u(x_1, x_2) = x_1^2 x_2^2 (1-x_1)^2 (1-x_2)^2$

## Repository Structure

```
DRMFinal/
├── notebooks/
│   ├── Example3.1DRM.ipynb            # Complete implementation for Example 3.1
│   └── Example3.2DRM.ipynb            # Complete implementation for Example 3.2
├── checkpoints/
│   ├── Example31_biharmonic_model_Adam_epoch10000.pt
│   └── Example32_biharmonic_model_Adam_epoch10000.pt
└── DRM_old/                           # Experimental code (not used in final)
```

## Implementation

### Neural Network Architecture

**U_FCN** - Fully connected U-shaped network:
- Hidden dimensions: [64, 128, 256, 256, 128, 64]
- Activation: Tanh (smooth derivatives for 4th-order PDE)
- Input: 2D coordinates → Output: scalar function value

*Note: ResNet architecture (DRM_Model) was experimented with in Example 3.1 but final results use U_FCN for both examples.*

### Loss Formulation

Penalized energy functional:

$$\mathcal{L}_\lambda(v) = \frac{1}{2}\int_{\Omega} |D^2 v|^2 dx - \int_{\Omega} f \cdot v \, dx - \int_{\partial\Omega} g_2 \frac{\partial v}{\partial n} ds + \frac{\lambda}{2}\int_{\partial\Omega} (v - g_1)^2 ds$$

With uncertainty-weighted multi-task learning for balancing interior and boundary terms.
<!-- uncertainty weighting -->
\[
\mathcal{L} = \frac{1}{2} e^{-s} \, \mathcal{L}_0 + \frac{1}{2} s.
\]



### Training Configuration

- **Optimizer:** Adam (lr=1e-6)
- **Epochs:** 10,000
- **Sampling:** 10,000 interior + 4,000 boundary points per epoch
- **Dirichlet Penalty:** λ = 1e3

*Note: Our loss implementation also takes a argument for LBFGS optimizer as well*

## Results

| Example | L2 Error | H1 Error | H2 Error | Training Time |
|---------|----------|----------|----------|---------------|
| 3.1     | 2.13e-03 | 9.16e-02 | 1.11e-01 | ~4 Hr         |
| 3.2     | 5.97e-04 | 5.05e-01 | 8.46e-01 | ~3 Hr         |

**Note:** Example 3.2 has higher errors due it more complex plolynomial structure and lower functional range.
To overcome this we imposed hard constraint on the central point (0.5,0.5), which helped the neural network understand the functional range. 
## Usage

**Run notebooks:**
```bash
jupyter notebook notebooks/Example3.1DRM.ipynb
jupyter notebook notebooks/Example3.2DRM.ipynb
```

**Load trained models:**
```python
import torch
checkpoint = torch.load('checkpoints/Example31_biharmonic_model_Adam_epoch10000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```


**Note:** The `DRM_old/` folder contains experimental code and previous attempts. The main implementations are in the notebooks.


## Dependencies
```
torch>=1.8.0
numpy>=1.19
matplotlib>=3.1
tqdm
```
Install via:
```bash
pip install torch numpy matplotlib tqdm
```

---
## Team
**Group 2** - Final Project Phase II 
*Deep Learning for Differential Equations Course*

## License

MIT License - See repository root for details.


---

**For detailed methodology, ablation studies, and theoretical analysis, please refer to `Group2_FinalReport.pdf`.**

