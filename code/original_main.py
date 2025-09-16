"""
Title: Differential Equations and Neural Networks
Date: 2025

Abstract
--------
This single Python file contains a compact bachelor-level "thesis" on Physics-Informed Neural
Networks (PINNs). The goal is pedagogical: present a very small thesis (motivation, math,
implementation, experiments, discussion) *and* runnable code that demonstrates the method
on a simple ODE:

    u'(t) = -u(t),  u(0)=1  (analytical solution: u(t)=exp(-t))

The file is structured so you can read the short thesis text in comments and run the code
sections to reproduce the experiment and figures.

Requirements
------------
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install with:
    pip install torch numpy matplotlib

How to run
----------
Run this file with python. It will train a small PINN for the ODE and save two images:
- pinn_solution.png  -> predicted vs exact solution
- pinn_loss.png      -> loss vs iteration

If you can't install PyTorch, you can still read the thesis text in the comments.

--- Thesis Text (shortened for a bachelor level) ---

Introduction
------------
Physics-Informed Neural Networks (PINNs) are function-approximators that are trained to
simultaneously fit observed data and satisfy differential equations. For supervised learning
we usually minimize prediction loss on data; for PINNs we add a PDE/ODE residual term
computed using automatic differentiation. This allows the network to "learn" solutions
consistent with known physical laws.

Problem statement
-----------------
We solve the initial value ODE:
    u'(t) = -u(t),  u(0)=1,  t\in[0,3]
Analytical solution: u(t) = exp(-t).

PINN formulation
----------------
Let u_\theta(t) be a neural network with parameters \theta. The PDE (here ODE) residual is
r(t;\theta) = d/dt u_\theta(t) + u_\theta(t)
We train by minimizing mean squared residual on collocation points plus a mean squared
initial condition loss term:
    L(\theta) = MSE_collocation(r) + MSE_IC(u_\theta(0) - 1)

Experiments & deliverables
--------------------------
- Implement a tiny MLP with tanh activations.
- Use Adam optimizer to minimize L(\theta).
- Visualize final prediction vs analytic solution and plot loss curve.

Discussion
----------
This minimal example shows the core ideas: (1) compute derivatives via autograd, (2) build a
loss from the residual of the differential equation, and (3) train with standard optimizers.

Extensions for a full thesis: try a 1D PDE (heat equation), compare optimizers (Adam vs
L-BFGS), do inverse parameter estimation, and study collocation sampling effects.

"""

# ------------------------------
# Begin runnable PINN code (ODE example)
# ------------------------------

import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise RuntimeError(
        "PyTorch is required to run this script. Install with `pip install torch`.\nOriginal error: %s" % e
    )

# Reproducibility
SEED = 12345
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
# Problem definition
# ------------------------------

T_MAX = 3.0
N_COLLOCATION = 2000  # collocation points for residual
LR = 1e-3
N_ITERS = 8000

# Analytical solution for comparison
def exact_solution(t):
    return np.exp(-t)

# ------------------------------
# Neural network (MLP) definition
# ------------------------------
class PINNNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, hidden_layers=3, output_dim=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        # t: tensor shape (N,1)
        return self.net(t)

# ------------------------------
# Utilities
# ------------------------------

def to_tensor(x, dtype=torch.float32, device=DEVICE):
    return torch.tensor(x, dtype=dtype, device=device)

# Create collocation points (uniform in [0, T_MAX])
def create_collocation_points(n):
    t = np.random.rand(n) * T_MAX
    t = np.sort(t)
    return t.reshape(-1, 1)

# ------------------------------
# Loss components
# ------------------------------

def pinn_loss(model, t_collocation, t_ic, u_ic):
    # Ensure requires_grad for collocation points to compute derivatives
    t_coll = to_tensor(t_collocation, device=DEVICE)
    t_coll.requires_grad = True

    u_pred = model(t_coll)

    # compute du/dt using autograd
    grad_u = torch.autograd.grad(
        outputs=u_pred,
        inputs=t_coll,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual = grad_u + u_pred  # u'(t) + u(t) should be zero
    mse_pde = torch.mean(residual**2)

    # initial condition loss
    t0 = to_tensor(t_ic.reshape(-1, 1), device=DEVICE)
    u0_pred = model(t0)
    mse_ic = torch.mean((u0_pred - to_tensor(u_ic, device=DEVICE))**2)

    LAMBDA_IC = 10.0
    loss = mse_pde + LAMBDA_IC * mse_ic
    return loss, mse_pde.detach().cpu().item(), mse_ic.detach().cpu().item()

# ------------------------------
# Training routine
# ------------------------------

def train_pinn(seed=SEED,
               hidden_dim=32,
               hidden_layers=3,
               n_collocation=N_COLLOCATION,
               lr=LR,
               n_iters=N_ITERS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINNNet(hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # collocation points for PDE residual
    t_collocation = create_collocation_points(n_collocation)

    # initial condition point
    t_ic = np.array([0.0])
    u_ic = np.array([1.0])

    losses = []
    pde_losses = []
    ic_losses = []

    start_time = time.time()
    for it in range(1, n_iters + 1):
        optimizer.zero_grad()
        loss, mse_pde, mse_ic = pinn_loss(model, t_collocation, t_ic, u_ic)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pde_losses.append(mse_pde)
        ic_losses.append(mse_ic)

        if it % 500 == 0 or it == 1:
            print(f"Iter {it:5d} | Loss {loss.item():.6e} | PDE {mse_pde:.6e} | IC {mse_ic:.6e}")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f}s")

    return model, np.array(losses), np.array(pde_losses), np.array(ic_losses)

# ------------------------------
# Run training and visualize
# ------------------------------
if __name__ == '__main__':
    # Train
    model, losses, pde_losses, ic_losses = train_pinn()

    # Prepare test points
    t_test = np.linspace(0.0, T_MAX, 300).reshape(-1, 1)
    t_test_t = to_tensor(t_test, device=DEVICE)
    with torch.no_grad():
        u_pred = model(t_test_t).cpu().numpy().flatten()

    u_exact = exact_solution(t_test.flatten())

    # Save solution plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_test, u_exact, label='Exact (exp(-t))', linewidth=2)
    ax.plot(t_test, u_pred, '--', label='PINN prediction')
    ax.set_xlabel('t')
    ax.set_ylabel('u(t)')
    ax.set_title('PINN solution for u\'(t) = -u(t)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('pinn_solution.png', dpi=200)
    print('Saved pinn_solution.png')

    # Save loss plot
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(losses, label='Total loss')
    ax2.plot(pde_losses, label='PDE loss')
    ax2.plot(ic_losses, label='IC loss')
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE (log scale)')
    ax2.set_title('Training loss (log scale)')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig('pinn_loss.png', dpi=200)
    print('Saved pinn_loss.png')

    # Print final error
    rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f'Relative L2 error on test set: {rel_l2:.3e}')

    # Show plots if running interactively
    try:
        plt.show()
    except Exception:
        pass

# End of file
