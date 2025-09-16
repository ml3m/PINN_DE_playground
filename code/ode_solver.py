"""
PINN-Powered ODE Solving: Where It All Begins
=============================================

Think of ODEs as the training wheels for PINNs. They're simpler than PDEs (only time, 
no space), which makes them perfect for learning how physics-informed neural networks work.

We start with basic exponential decay, move to population dynamics, and finish with 
some oscillatory action. Each one teaches us something different about how to make 
neural networks respect differential equations.

Built by: Someone who learned ODEs the hard way and wants to make it easier for you
Year: 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

from pinn_base import BasePINN, PINNNet


class LinearODEPINN(BasePINN):
    """
    The Classic: Linear Decay (Your First PINN)
    
    This solves u'(t) = -λu(t) with u(0) = u₀
    Think: radioactive decay, cooling coffee, dying enthusiasm for homework.
    
    The beautiful thing? We know the exact answer: u(t) = u₀e^(-λt)
    So we can check if our PINN is actually learning correctly!
    """
    
    def __init__(self, t_max: float = 3.0, u0: float = 1.0, 
                 lambda_coeff: float = 1.0, **kwargs):
        """
        Set up our linear decay problem.
        
        Args:
            t_max: How far in time should we go? (3.0 is usually enough to see the decay)
            u0: Where do we start? (1.0 is a nice round number)
            lambda_coeff: How fast does it decay? (1.0 gives a nice exponential curve)
            **kwargs: Any other settings from the parent class
        """
        super().__init__(**kwargs)
        
        # Store our problem parameters
        self.t_max = t_max
        self.u0 = u0
        self.lambda_coeff = lambda_coeff
        
        # Create a simple neural network: time in, solution out
        # For ODEs, we don't need anything fancy - small networks work great
        self.model = PINNNet(input_dim=1, hidden_dim=32, 
                           hidden_layers=3, output_dim=1).to(self.device)
    
    def compute_pde_residual(self, t: torch.Tensor) -> torch.Tensor:
        """
        Check how well our network satisfies u'(t) + λu(t) = 0.
        
        This is the core PINN magic! We use automatic differentiation to compute
        the derivative of our neural network, then check if it satisfies the ODE.
        
        Args:
            t: Time points where we're checking the physics
            
        Returns:
            How much we're violating the differential equation (should be ~0)
        """
        # Get the neural network's prediction at these time points
        u = self.model(t)
        
        # Here's the magic: compute du/dt automatically!
        # PyTorch tracks how u depends on t, so it can compute derivatives
        du_dt = torch.autograd.grad(
            outputs=u,              # What we want the derivative of
            inputs=t,               # What we're taking the derivative with respect to
            grad_outputs=torch.ones_like(u),  # Standard boilerplate
            create_graph=True,      # Keep the graph for higher-order derivatives
            retain_graph=True       # Don't destroy the graph (we need it for backprop)
        )[0]
        
        # The ODE says: u'(t) = -λu(t), so u'(t) + λu(t) should equal zero
        residual = du_dt + self.lambda_coeff * u
        return residual
    
    def compute_ic_loss(self) -> torch.Tensor:
        """Make sure we start from the right place: u(0) = u₀."""
        t0 = torch.zeros(1, 1, device=self.device)  # Time = 0
        u0_pred = self.model(t0)  # What does our network think u(0) is?
        
        # Penalize the difference between predicted and actual initial condition
        ic_loss = torch.mean((u0_pred - self.u0) ** 2)
        return ic_loss
    
    def generate_collocation_points(self, n_points: int) -> torch.Tensor:
        """Scatter some random time points where we'll check the physics."""
        t = np.random.uniform(0, self.t_max, (n_points, 1))
        t_tensor = self.to_tensor(t, requires_grad=True)  # Don't forget gradients!
        return t_tensor
    
    def exact_solution(self, t: np.ndarray) -> np.ndarray:
        """The cheat sheet: what the solution should actually be."""
        return self.u0 * np.exp(-self.lambda_coeff * t)


class LogisticODEPINN(BasePINN):
    """
    Population Dynamics: The S-Curve Story
    
    This solves u'(t) = r·u(t)·(1-u(t)) with u(0) = u₀
    
    Think of it as population growth with limited resources:
    - When u is small: grows like exponential (lots of space)
    - When u approaches 1: growth slows (getting crowded)
    - Results in the classic S-shaped curve
    
    Also applies to: viral spread, technology adoption, your learning curve!
    """
    
    def __init__(self, t_max: float = 5.0, u0: float = 0.1, 
                 r: float = 1.0, **kwargs):
        """
        Initialize the logistic ODE PINN.
        
        Args:
            t_max: Maximum time for the domain [0, t_max]
            u0: Initial condition value
            r: Growth rate parameter
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        
        self.t_max = t_max
        self.u0 = u0
        self.r = r
        
        # Initialize neural network
        self.model = PINNNet(input_dim=1, hidden_dim=64, 
                           hidden_layers=4, output_dim=1).to(self.device)
    
    def compute_pde_residual(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for u'(t) - r * u(t) * (1 - u(t)) = 0.
        
        Args:
            t: Time points tensor with requires_grad=True
            
        Returns:
            PDE residual tensor
        """
        u = self.model(t)
        
        # Compute du/dt using automatic differentiation
        du_dt = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Residual: u'(t) - r * u(t) * (1 - u(t)) = 0
        residual = du_dt - self.r * u * (1.0 - u)
        return residual
    
    def compute_ic_loss(self) -> torch.Tensor:
        """Compute initial condition loss: u(0) = u0."""
        t0 = torch.zeros(1, 1, device=self.device)
        u0_pred = self.model(t0)
        ic_loss = torch.mean((u0_pred - self.u0) ** 2)
        return ic_loss
    
    def generate_collocation_points(self, n_points: int) -> torch.Tensor:
        """Generate random collocation points in [0, t_max]."""
        t = np.random.uniform(0, self.t_max, (n_points, 1))
        t_tensor = self.to_tensor(t, requires_grad=True)
        return t_tensor
    
    def exact_solution(self, t: np.ndarray) -> np.ndarray:
        """Compute exact analytical solution."""
        if self.u0 == 0:
            return np.zeros_like(t)
        elif self.u0 == 1:
            return np.ones_like(t)
        else:
            exp_term = np.exp(-self.r * t)
            denominator = 1 + ((1 - self.u0) / self.u0) * exp_term
            return 1.0 / denominator


class VanDerPolODEPINN(BasePINN):
    """
    PINN solver for Van der Pol oscillator (second-order ODE):
    u''(t) - mu * (1 - u(t)^2) * u'(t) + u(t) = 0
    
    This is converted to a system of first-order ODEs:
    u'(t) = v(t)
    v'(t) = mu * (1 - u(t)^2) * v(t) - u(t)
    """
    
    def __init__(self, t_max: float = 10.0, u0: float = 2.0, 
                 v0: float = 0.0, mu: float = 1.0, **kwargs):
        """
        Initialize the Van der Pol ODE PINN.
        
        Args:
            t_max: Maximum time for the domain [0, t_max]
            u0: Initial condition for u(0)
            v0: Initial condition for v(0) = u'(0)
            mu: Van der Pol parameter
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        
        self.t_max = t_max
        self.u0 = u0
        self.v0 = v0
        self.mu = mu
        
        # Initialize neural network (2 outputs: u and v)
        self.model = PINNNet(input_dim=1, hidden_dim=64, 
                           hidden_layers=4, output_dim=2).to(self.device)
    
    def compute_pde_residual(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for the Van der Pol system.
        
        Args:
            t: Time points tensor with requires_grad=True
            
        Returns:
            PDE residual tensor
        """
        uv = self.model(t)  # Shape: (n_points, 2)
        u = uv[:, 0:1]      # Shape: (n_points, 1)
        v = uv[:, 1:2]      # Shape: (n_points, 1)
        
        # Compute derivatives
        du_dt = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dv_dt = torch.autograd.grad(
            outputs=v,
            inputs=t,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Residuals
        residual1 = du_dt - v  # u'(t) - v(t) = 0
        residual2 = dv_dt - self.mu * (1.0 - u**2) * v + u  # v'(t) - mu*(1-u^2)*v + u = 0
        
        # Combine residuals
        residual = torch.cat([residual1, residual2], dim=1)
        return residual.flatten()
    
    def compute_ic_loss(self) -> torch.Tensor:
        """Compute initial condition loss: u(0) = u0, v(0) = v0."""
        t0 = torch.zeros(1, 1, device=self.device)
        uv0_pred = self.model(t0)
        u0_pred = uv0_pred[:, 0:1]
        v0_pred = uv0_pred[:, 1:2]
        
        ic_loss_u = torch.mean((u0_pred - self.u0) ** 2)
        ic_loss_v = torch.mean((v0_pred - self.v0) ** 2)
        
        return ic_loss_u + ic_loss_v
    
    def generate_collocation_points(self, n_points: int) -> torch.Tensor:
        """Generate random collocation points in [0, t_max]."""
        t = np.random.uniform(0, self.t_max, (n_points, 1))
        t_tensor = self.to_tensor(t, requires_grad=True)
        return t_tensor
    
    def predict_uv(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both u and v at given time points.
        
        Args:
            t: Time points
            
        Returns:
            Tuple of (u_predictions, v_predictions)
        """
        self.model.eval()
        with torch.no_grad():
            t_tensor = self.to_tensor(t.reshape(-1, 1))
            uv_pred = self.model(t_tensor).cpu().numpy()
            u_pred = uv_pred[:, 0]
            v_pred = uv_pred[:, 1]
            return u_pred, v_pred


def plot_ode_solution(pinn_solver, t_test: np.ndarray, 
                     save_path: str = "ode_solution.png", 
                     title: str = "PINN ODE Solution"):
    """
    Plot PINN solution against exact solution for ODE problems.
    
    Args:
        pinn_solver: Trained PINN solver instance
        t_test: Test time points
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get PINN predictions
    if hasattr(pinn_solver, 'predict_uv'):
        # Van der Pol oscillator (2D system)
        u_pred, v_pred = pinn_solver.predict_uv(t_test)
        
        ax.plot(t_test, u_pred, '--', label='PINN u(t)', linewidth=2)
        ax.plot(t_test, v_pred, ':', label='PINN v(t)', linewidth=2)
        ax.set_ylabel('u(t), v(t)')
        
    else:
        # Single ODE
        u_pred = pinn_solver.predict(t_test.reshape(-1, 1)).flatten()
        u_exact = pinn_solver.exact_solution(t_test)
        
        ax.plot(t_test, u_exact, label='Exact solution', linewidth=2, color='blue')
        ax.plot(t_test, u_pred, '--', label='PINN prediction', linewidth=2, color='red')
        
        # Compute and display error
        rel_error = pinn_solver.compute_relative_l2_error(t_test.reshape(-1, 1), u_exact)
        ax.text(0.02, 0.98, f'Relative L2 error: {rel_error:.3e}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylabel('u(t)')
    
    ax.set_xlabel('t')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Solution plot saved to {save_path}")


def run_ode_experiment(ode_type: str = 'linear', **kwargs):
    """
    Run a complete ODE experiment with training and visualization.
    
    Args:
        ode_type: Type of ODE ('linear', 'logistic', 'vanderpol')
        **kwargs: Additional parameters for the specific ODE
    """
    print(f"\n{'='*50}")
    print(f"Running {ode_type.upper()} ODE Experiment")
    print(f"{'='*50}")
    
    # Extract training parameters
    training_params = {
        'n_iterations': kwargs.pop('n_iterations', 5000),
        'learning_rate': kwargs.pop('learning_rate', 1e-3),
        'n_collocation': kwargs.pop('n_collocation', 1000),
        'print_frequency': kwargs.pop('print_frequency', 500)
    }
    
    # Initialize solver based on type
    if ode_type.lower() == 'linear':
        solver = LinearODEPINN(**kwargs)
        title = f"Linear ODE: u'(t) = -{solver.lambda_coeff}*u(t)"
        
    elif ode_type.lower() == 'logistic':
        solver = LogisticODEPINN(**kwargs)
        title = f"Logistic ODE: u'(t) = {solver.r}*u(t)*(1-u(t))"
        
    elif ode_type.lower() == 'vanderpol':
        solver = VanDerPolODEPINN(**kwargs)
        title = f"Van der Pol Oscillator: μ = {solver.mu}"
        
    else:
        raise ValueError(f"Unknown ODE type: {ode_type}")
    
    # Train the model
    print("\nTraining PINN...")
    history = solver.train(**training_params)
    
    # Generate test points
    t_test = np.linspace(0, solver.t_max, 300)
    
    # Plot solution
    plot_ode_solution(solver, t_test, 
                     save_path=f"figures/{ode_type}_solution.png",
                     title=title)
    
    # Save training history
    from pinn_base import save_training_plots
    save_training_plots(history, save_path=f"figures/{ode_type}_training.png")
    
    # Compute final error (for single ODEs)
    if hasattr(solver, 'exact_solution'):
        u_exact = solver.exact_solution(t_test)
        rel_error = solver.compute_relative_l2_error(t_test.reshape(-1, 1), u_exact)
        print(f"\nFinal relative L2 error: {rel_error:.3e}")
    
    print(f"\n{ode_type.upper()} experiment completed successfully!")
    return solver, history
