# Physics-Informed Neural Networks Framework

A comprehensive PyTorch implementation of Physics-Informed Neural Networks (PINNs) for solving ordinary and partial differential equations with automatic differentiation.

## Overview

This repository contains a complete PINN framework that embeds physical laws directly into neural network training through the loss function. The implementation covers both ordinary differential equations (ODEs) and partial differential equations (PDEs) with systematic comparative studies.

**Key Features:**
- Modular PINN architecture supporting multiple equation types
- Automatic differentiation for computing physics residuals
- GPU acceleration with automatic device detection
- Comprehensive test suite and validation framework
- Educational progression from simple to complex problems

## Installation

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (optional)

**Dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run all experiments:
```bash
cd code
python thesis_runner.py --all
```

Run specific experiment categories:
```bash
python thesis_runner.py --ode       # Ordinary differential equations
python thesis_runner.py --pde       # Partial differential equations
python thesis_runner.py --experiments  # Advanced comparative studies
python thesis_runner.py --tests     # Test suite
```

### Code Structure

The `/code` directory contains the core implementation:

#### Core Framework (`pinn_base.py`)
```python
from pinn_base import BasePINN, PINNNet

# Create neural network for PINNs
network = PINNNet(input_dim=2, hidden_dim=64, hidden_layers=4)

# Extend BasePINN for custom physics problems
```

#### ODE Solvers (`ode_solver.py`)
```python
from ode_solver import LinearODEPINN, LogisticODEPINN, run_ode_experiment

# Method 1: Pre-built experiments
solver, history = run_ode_experiment('linear', t_max=3.0, u0=1.0)

# Method 2: Direct instantiation
solver = LinearODEPINN(t_max=5.0, u0=2.0, lambda_coeff=0.5)
history = solver.train(n_iterations=5000, learning_rate=1e-3)

# Generate predictions
import numpy as np
t_test = np.linspace(0, 5, 100).reshape(-1, 1)
solution = solver.predict(t_test)
```

#### PDE Solvers (`pde_solver.py`)
```python
from pde_solver import HeatEquationPINN, WaveEquationPINN, run_pde_experiment

# Quick experiment
solver, history = run_pde_experiment('heat', x_max=1.0, t_max=0.5, alpha=0.1)

# Custom configuration
heat_solver = HeatEquationPINN(x_max=2.0, t_max=1.0, alpha=0.05)
history = heat_solver.train(
    n_iterations=10000,
    learning_rate=1e-3,
    n_collocation=2000,
    lambda_ic=10.0,
    lambda_bc=10.0,
    optimizer_type='adam'
)
```

#### Advanced Studies (`experiments.py`)
```python
from experiments import OptimizerComparison, ArchitectureStudy, NoiseRobustnessStudy

# Optimizer comparison
study = OptimizerComparison(problem_type='linear_ode')
results = study.run_comparison(['adam', 'lbfgs'], n_iterations=5000)

# Architecture sensitivity analysis
arch_study = ArchitectureStudy('heat_pde')
width_results = arch_study.width_study([16, 32, 64, 128])
```

### Custom PINN Implementation

```python
from pinn_base import BasePINN, PINNNet
import torch
import numpy as np

class CustomPINN(BasePINN):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
        self.model = PINNNet(input_dim=1, hidden_dim=32, 
                           hidden_layers=3, output_dim=1).to(self.device)
    
    def compute_pde_residual(self, x):
        u = self.model(x)
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        residual = du_dx + self.custom_param * u
        return residual
    
    def compute_ic_loss(self):
        x0 = torch.zeros(1, 1, device=self.device)
        u0_pred = self.model(x0)
        return torch.mean((u0_pred - 1.0) ** 2)
    
    def generate_collocation_points(self, n_points):
        x = np.random.uniform(0, 5, (n_points, 1))
        return self.to_tensor(x, requires_grad=True)
```

## Implemented Problems

### Ordinary Differential Equations
- **Linear ODE**: `u'(t) = -λu(t)` with analytical solution validation
- **Logistic Growth**: `u'(t) = ru(1-u)` for population dynamics modeling
- **Van der Pol Oscillator**: Second-order nonlinear system

### Partial Differential Equations
- **Heat Equation**: `u_t = αu_xx` for diffusion processes
- **Wave Equation**: `u_tt = c²u_xx` for wave propagation
- **Burgers' Equation**: `u_t + uu_x = νu_xx` for nonlinear convection-diffusion

### Advanced Studies
- Optimizer comparison (Adam vs L-BFGS)
- Network architecture sensitivity analysis
- Noise robustness evaluation
- Hyperparameter optimization studies

## Performance Metrics

| Problem Type | Relative L2 Error | Training Time | Convergence |
|--------------|------------------|---------------|-------------|
| Linear ODE   | 10⁻⁴ to 10⁻⁶   | 10-30 seconds | Excellent   |
| Logistic ODE | 10⁻³ to 10⁻⁵   | 15-45 seconds | Good        |
| Heat PDE     | 10⁻² to 10⁻⁴   | 1-3 minutes   | Good        |
| Wave PDE     | 10⁻² to 10⁻³   | 1.5-5 minutes | Moderate    |
| Burgers PDE  | 10⁻² to 10⁻¹   | 3-10 minutes  | Challenging |

## Testing

Run the test suite:
```bash
python thesis_runner.py --tests
```

The test suite includes:
- Core PINN framework validation
- Solver implementation verification
- Training convergence testing
- Error computation accuracy
- End-to-end experiment validation

## Technical Implementation

**PINN Methodology:**
1. Neural network approximation of solution functions
2. Automatic differentiation for physics residual computation
3. Physics-informed loss function combining PDE, initial, and boundary conditions
4. Gradient-based optimization (Adam/L-BFGS)

**Loss Function:**
```
Total Loss = λ₁ × PDE_Loss + λ₂ × IC_Loss + λ₃ × BC_Loss
```

**Key Features:**
- Modular architecture for easy problem extension
- GPU acceleration with automatic device detection
- Comprehensive random seed management for reproducibility
- Automated visualization and error analysis

## Project Structure

```
PINN/
├── code/
│   ├── pinn_base.py         # Core PINN framework
│   ├── ode_solver.py        # ODE implementations
│   ├── pde_solver.py        # PDE implementations
│   ├── experiments.py       # Advanced studies
│   └── thesis_runner.py     # Main execution script
├── tests/
│   └── test_pinn_thesis.py  # Test suite
├── figures/                 # Generated visualizations
├── docs/                    # Documentation
└── README.md               # This file
```

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Karniadakis, G. E., et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422-440.

3. Cuomo, S., et al. (2022). Scientific machine learning through physics–informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92(3), 1-62.
