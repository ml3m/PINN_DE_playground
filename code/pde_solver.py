"""
PDE Mastery: Where Space and Time Collide
==========================================

Now we're getting to the real deal! PDEs involve both space AND time, which makes
things way more interesting (and complex). We're dealing with heat spreading through
metal bars, waves bouncing around, and fluid dynamics that would make your head spin.

Each PDE teaches us something different:
- Heat equation: How things spread and smooth out over time (like gossip or heat)
- Wave equation: How energy bounces around while conserving itself  
- Burgers' equation: How simple nonlinear effects can create complex phenomena

Warning: This is where PINNs start showing their true power (and their quirks).

Built by: Someone who spent way too much time debugging gradient computations
Year: 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Dict
from mpl_toolkits.mplot3d import Axes3D

from pinn_base import BasePINN, PINNNet


class HeatEquationPINN(BasePINN):
    """
    The Heat Spreader: Why Your Coffee Gets Cold
    
    Solves: u_t = α·u_xx (heat diffusion in 1D)
    Domain: x ∈ [0,L] (space), t ∈ [0,T] (time)
    
    Physical meaning: Heat flows from hot to cold, smoothing out temperature differences.
    - α (alpha) controls how fast heat spreads (thermal diffusivity)
    - Boundary conditions: fixed temperature at both ends
    - Initial condition: starting temperature profile
    
    Real examples: cooling coffee, heat in metal rods, temperature in the ground
    """
    
    def __init__(self, x_max: float = 1.0, t_max: float = 1.0, 
                 alpha: float = 0.1, initial_condition: str = 'sine', **kwargs):
        """
        Set up our heat diffusion problem.
        
        Args:
            x_max: How long is our rod? (1.0 is a nice unit length)
            t_max: How long do we watch the heat spread? (1.0 usually shows the action)
            alpha: How fast does heat move? (0.1 gives nice smooth spreading)
            initial_condition: How do we start? ('sine' is classic, 'gaussian' is a blob, 'step' is a sharp edge)
            **kwargs: Parent class settings
        """
        super().__init__(**kwargs)
        
        # Store our physics parameters
        self.x_max = x_max
        self.t_max = t_max
        self.alpha = alpha
        self.initial_condition_type = initial_condition
        
        # PDEs need bigger networks than ODEs (more complexity in 2D)
        # Input: (x,t) coordinates, Output: temperature u(x,t)
        self.model = PINNNet(input_dim=2, hidden_dim=64, 
                           hidden_layers=4, output_dim=1).to(self.device)
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Define initial condition u(x,0)."""
        if self.initial_condition_type == 'sine':
            return torch.sin(np.pi * x / self.x_max)
        elif self.initial_condition_type == 'gaussian':
            return torch.exp(-50 * (x - self.x_max/2)**2)
        elif self.initial_condition_type == 'step':
            return torch.where(
                torch.abs(x - self.x_max/2) < self.x_max/4,
                torch.ones_like(x),
                torch.zeros_like(x)
            )
        else:
            raise ValueError(f"Unknown initial condition: {self.initial_condition_type}")
    
    def compute_pde_residual(self, xt: torch.Tensor) -> torch.Tensor:
        """
        The PDE checker: does our network satisfy u_t = α·u_xx?
        
        This is where the real magic happens for PDEs. We need:
        - First derivative in time: ∂u/∂t
        - Second derivative in space: ∂²u/∂x²
        Then check if u_t = α·u_xx (heat equation)
        
        Args:
            xt: Points in space-time where we check physics, shape (n_points, 2)
            
        Returns:
            How badly we violate the heat equation (should be ~0)
        """
        x = xt[:, 0:1]  # Spatial coordinates
        t = xt[:, 1:2]  # Time coordinates
        
        # Enable gradient tracking (this is crucial!)
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get network prediction at these (x,t) points
        u = self.model(torch.cat([x, t], dim=1))
        
        # Compute first derivatives using automatic differentiation
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), 
                                 create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), 
                                 create_graph=True, retain_graph=True)[0]
        
        # Second derivative in space (derivative of derivative!)
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), 
                                  create_graph=True, retain_graph=True)[0]
        
        # Heat equation: u_t = α·u_xx, so u_t - α·u_xx should be zero
        residual = u_t - self.alpha * u_xx
        return residual.flatten()
    
    def compute_ic_loss(self) -> torch.Tensor:
        """Compute initial condition loss: u(x,0) = f(x)."""
        # Generate points on initial boundary
        n_ic = 100
        x_ic = torch.linspace(0, self.x_max, n_ic, device=self.device).reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic_pred = self.model(xt_ic)
        u_ic_exact = self.initial_condition(x_ic)
        
        ic_loss = torch.mean((u_ic_pred - u_ic_exact) ** 2)
        return ic_loss
    
    def compute_bc_loss(self) -> torch.Tensor:
        """Compute boundary condition loss: u(0,t) = u(L,t) = 0."""
        # Generate points on spatial boundaries
        n_bc = 100
        t_bc = torch.linspace(0, self.t_max, n_bc, device=self.device).reshape(-1, 1)
        
        # Left boundary: x = 0
        x_left = torch.zeros_like(t_bc)
        xt_left = torch.cat([x_left, t_bc], dim=1)
        u_left_pred = self.model(xt_left)
        
        # Right boundary: x = x_max
        x_right = self.x_max * torch.ones_like(t_bc)
        xt_right = torch.cat([x_right, t_bc], dim=1)
        u_right_pred = self.model(xt_right)
        
        # Boundary conditions: u = 0 at both boundaries
        bc_loss = torch.mean(u_left_pred ** 2) + torch.mean(u_right_pred ** 2)
        return bc_loss
    
    def generate_collocation_points(self, n_points: int) -> torch.Tensor:
        """Generate random collocation points in domain [0,x_max] × [0,t_max]."""
        x = np.random.uniform(0, self.x_max, (n_points, 1))
        t = np.random.uniform(0, self.t_max, (n_points, 1))
        xt = np.hstack([x, t])
        xt_tensor = self.to_tensor(xt, requires_grad=True)
        return xt_tensor
    
    def exact_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compute exact solution for sine initial condition.
        Only available for sine initial condition.
        """
        if self.initial_condition_type == 'sine':
            return np.sin(np.pi * x / self.x_max) * np.exp(-self.alpha * (np.pi / self.x_max)**2 * t)
        else:
            raise NotImplementedError(f"Exact solution not available for {self.initial_condition_type}")


class WaveEquationPINN(BasePINN):
    """
    PINN solver for 1D Wave Equation: u_tt = c^2 * u_xx
    Domain: x ∈ [0, L], t ∈ [0, T]
    Boundary conditions: u(0,t) = u(L,t) = 0
    Initial conditions: u(x,0) = f(x), u_t(x,0) = g(x)
    """
    
    def __init__(self, x_max: float = 1.0, t_max: float = 2.0, 
                 c: float = 1.0, **kwargs):
        """
        Initialize the wave equation PINN.
        
        Args:
            x_max: Spatial domain length [0, x_max]
            t_max: Time domain length [0, t_max]
            c: Wave speed
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        
        self.x_max = x_max
        self.t_max = t_max
        self.c = c
        
        # Initialize neural network
        self.model = PINNNet(input_dim=2, hidden_dim=64, 
                           hidden_layers=5, output_dim=1).to(self.device)
    
    def initial_displacement(self, x: torch.Tensor) -> torch.Tensor:
        """Initial displacement: u(x,0) = sin(pi*x/L)."""
        return torch.sin(np.pi * x / self.x_max)
    
    def initial_velocity(self, x: torch.Tensor) -> torch.Tensor:
        """Initial velocity: u_t(x,0) = 0."""
        return torch.zeros_like(x)
    
    def compute_pde_residual(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for u_tt - c^2 * u_xx = 0.
        
        Args:
            xt: Input tensor of shape (n_points, 2) with columns [x, t]
            
        Returns:
            PDE residual tensor
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.model(torch.cat([x, t], dim=1))
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True, retain_graph=True)[0]
        
        # PDE residual: u_tt - c^2 * u_xx = 0
        residual = u_tt - self.c**2 * u_xx
        return residual.flatten()
    
    def compute_ic_loss(self) -> torch.Tensor:
        """Compute initial condition losses."""
        n_ic = 100
        x_ic = torch.linspace(0, self.x_max, n_ic, device=self.device).reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        
        # For initial velocity, we need to compute u_t at t=0
        x_ic.requires_grad_(True)
        t_ic.requires_grad_(True)
        
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic_pred = self.model(xt_ic)
        
        # Initial displacement loss
        u_ic_exact = self.initial_displacement(x_ic)
        ic_displacement_loss = torch.mean((u_ic_pred - u_ic_exact) ** 2)
        
        # Initial velocity loss
        u_t_ic_pred = torch.autograd.grad(
            u_ic_pred, t_ic, torch.ones_like(u_ic_pred),
            create_graph=True, retain_graph=True
        )[0]
        u_t_ic_exact = self.initial_velocity(x_ic)
        ic_velocity_loss = torch.mean((u_t_ic_pred - u_t_ic_exact) ** 2)
        
        return ic_displacement_loss + ic_velocity_loss
    
    def compute_bc_loss(self) -> torch.Tensor:
        """Compute boundary condition loss: u(0,t) = u(L,t) = 0."""
        n_bc = 100
        t_bc = torch.linspace(0, self.t_max, n_bc, device=self.device).reshape(-1, 1)
        
        # Left boundary
        x_left = torch.zeros_like(t_bc)
        xt_left = torch.cat([x_left, t_bc], dim=1)
        u_left_pred = self.model(xt_left)
        
        # Right boundary
        x_right = self.x_max * torch.ones_like(t_bc)
        xt_right = torch.cat([x_right, t_bc], dim=1)
        u_right_pred = self.model(xt_right)
        
        bc_loss = torch.mean(u_left_pred ** 2) + torch.mean(u_right_pred ** 2)
        return bc_loss
    
    def generate_collocation_points(self, n_points: int) -> torch.Tensor:
        """Generate random collocation points in domain."""
        x = np.random.uniform(0, self.x_max, (n_points, 1))
        t = np.random.uniform(0, self.t_max, (n_points, 1))
        xt = np.hstack([x, t])
        return self.to_tensor(xt, requires_grad=True)
    
    def exact_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Exact solution for sine initial condition."""
        return np.sin(np.pi * x / self.x_max) * np.cos(np.pi * self.c * t / self.x_max)


class BurgersEquationPINN(BasePINN):
    """
    PINN solver for 1D Burgers' Equation: u_t + u * u_x = nu * u_xx
    Domain: x ∈ [x_min, x_max], t ∈ [0, T]
    """
    
    def __init__(self, x_min: float = -1.0, x_max: float = 1.0, 
                 t_max: float = 1.0, nu: float = 0.01, **kwargs):
        """
        Initialize the Burgers' equation PINN.
        
        Args:
            x_min, x_max: Spatial domain bounds
            t_max: Time domain length
            nu: Viscosity parameter
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        
        self.x_min = x_min
        self.x_max = x_max
        self.t_max = t_max
        self.nu = nu
        
        # Initialize neural network
        self.model = PINNNet(input_dim=2, hidden_dim=128, 
                           hidden_layers=6, output_dim=1).to(self.device)
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Initial condition: u(x,0) = -sin(pi*x)."""
        return -torch.sin(np.pi * x)
    
    def compute_pde_residual(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for u_t + u * u_x - nu * u_xx = 0.
        
        Args:
            xt: Input tensor of shape (n_points, 2) with columns [x, t]
            
        Returns:
            PDE residual tensor
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.model(torch.cat([x, t], dim=1))
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        
        # Second derivative u_xx
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        
        # PDE residual: u_t + u * u_x - nu * u_xx = 0
        residual = u_t + u * u_x - self.nu * u_xx
        return residual.flatten()
    
    def compute_ic_loss(self) -> torch.Tensor:
        """Compute initial condition loss."""
        n_ic = 100
        x_ic = torch.linspace(self.x_min, self.x_max, n_ic, device=self.device).reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)
        
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic_pred = self.model(xt_ic)
        u_ic_exact = self.initial_condition(x_ic)
        
        return torch.mean((u_ic_pred - u_ic_exact) ** 2)
    
    def compute_bc_loss(self) -> torch.Tensor:
        """Periodic boundary conditions: u(-1,t) = u(1,t)."""
        n_bc = 100
        t_bc = torch.linspace(0, self.t_max, n_bc, device=self.device).reshape(-1, 1)
        
        # Left boundary
        x_left = self.x_min * torch.ones_like(t_bc)
        xt_left = torch.cat([x_left, t_bc], dim=1)
        u_left_pred = self.model(xt_left)
        
        # Right boundary
        x_right = self.x_max * torch.ones_like(t_bc)
        xt_right = torch.cat([x_right, t_bc], dim=1)
        u_right_pred = self.model(xt_right)
        
        # Periodic boundary condition
        bc_loss = torch.mean((u_left_pred - u_right_pred) ** 2)
        return bc_loss
    
    def generate_collocation_points(self, n_points: int) -> torch.Tensor:
        """Generate random collocation points in domain."""
        x = np.random.uniform(self.x_min, self.x_max, (n_points, 1))
        t = np.random.uniform(0, self.t_max, (n_points, 1))
        xt = np.hstack([x, t])
        return self.to_tensor(xt, requires_grad=True)


def plot_pde_solution_2d(pinn_solver, save_path: str = "pde_solution_2d.png",
                        title: str = "PINN PDE Solution", n_points: int = 100):
    """
    Plot 2D heatmap of PDE solution u(x,t).
    
    Args:
        pinn_solver: Trained PINN solver
        save_path: Path to save the plot
        title: Plot title
        n_points: Number of points in each dimension for plotting
    """
    # Create mesh
    x_plot = np.linspace(getattr(pinn_solver, 'x_min', 0), 
                        getattr(pinn_solver, 'x_max', pinn_solver.x_max), n_points)
    t_plot = np.linspace(0, pinn_solver.t_max, n_points)
    X, T = np.meshgrid(x_plot, t_plot)
    
    # Flatten for prediction
    xt_plot = np.column_stack([X.flatten(), T.flatten()])
    
    # Get predictions
    u_pred = pinn_solver.predict(xt_plot).reshape(n_points, n_points)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    im1 = ax1.contourf(X, T, u_pred, levels=50, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title(f'{title} - Solution u(x,t)')
    plt.colorbar(im1, ax=ax1)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, T, u_pred, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x,t)')
    ax2.set_title(f'{title} - 3D View')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"2D solution plot saved to {save_path}")


def plot_pde_solution_comparison(pinn_solver, save_path: str = "pde_comparison.png",
                               title: str = "PINN vs Exact Solution"):
    """
    Plot comparison between PINN and exact solution (if available).
    """
    if not hasattr(pinn_solver, 'exact_solution'):
        print("Exact solution not available for comparison")
        return
    
    n_points = 100
    x_plot = np.linspace(getattr(pinn_solver, 'x_min', 0), 
                        getattr(pinn_solver, 'x_max', pinn_solver.x_max), n_points)
    t_plot = np.linspace(0, pinn_solver.t_max, n_points)
    X, T = np.meshgrid(x_plot, t_plot)
    
    # Get solutions
    xt_plot = np.column_stack([X.flatten(), T.flatten()])
    u_pred = pinn_solver.predict(xt_plot).reshape(n_points, n_points)
    u_exact = pinn_solver.exact_solution(X, T)
    u_error = np.abs(u_pred - u_exact)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PINN solution
    im1 = axes[0].contourf(X, T, u_pred, levels=50, cmap='viridis')
    axes[0].set_title('PINN Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0])
    
    # Exact solution
    im2 = axes[1].contourf(X, T, u_exact, levels=50, cmap='viridis')
    axes[1].set_title('Exact Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    im3 = axes[2].contourf(X, T, u_error, levels=50, cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def run_pde_experiment(pde_type: str = 'heat', **kwargs):
    """
    Run a complete PDE experiment.
    
    Args:
        pde_type: Type of PDE ('heat', 'wave', 'burgers')
        **kwargs: Additional parameters
    """
    print(f"\n{'='*50}")
    print(f"Running {pde_type.upper()} PDE Experiment")
    print(f"{'='*50}")
    
    # Extract training parameters
    training_params = {
        'n_iterations': kwargs.pop('n_iterations', 10000),
        'learning_rate': kwargs.pop('learning_rate', 1e-3),
        'n_collocation': kwargs.pop('n_collocation', 2000),
        'lambda_ic': kwargs.pop('lambda_ic', 10.0),
        'lambda_bc': kwargs.pop('lambda_bc', 10.0),
        'print_frequency': kwargs.pop('print_frequency', 1000)
    }
    
    # Initialize solver
    if pde_type.lower() == 'heat':
        solver = HeatEquationPINN(**kwargs)
        title = f"Heat Equation (α={solver.alpha})"
        
    elif pde_type.lower() == 'wave':
        solver = WaveEquationPINN(**kwargs)
        title = f"Wave Equation (c={solver.c})"
        
    elif pde_type.lower() == 'burgers':
        solver = BurgersEquationPINN(**kwargs)
        title = f"Burgers' Equation (ν={solver.nu})"
        
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    # Train the model
    print("\nTraining PINN...")
    history = solver.train(**training_params)
    
    # Plot solutions
    plot_pde_solution_2d(solver, 
                        save_path=f"figures/{pde_type}_solution_2d.png",
                        title=title)
    
    # Plot comparison if exact solution available
    if hasattr(solver, 'exact_solution'):
        plot_pde_solution_comparison(solver,
                                   save_path=f"figures/{pde_type}_comparison.png",
                                   title=title)
    
    # Save training history
    from pinn_base import save_training_plots
    save_training_plots(history, save_path=f"figures/{pde_type}_training.png")
    
    print(f"\n{pde_type.upper()} PDE experiment completed successfully!")
    return solver, history
