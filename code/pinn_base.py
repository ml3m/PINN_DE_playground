"""
PINN Foundation: Where Physics Meets Neural Networks
====================================================

Think of this as the blueprint for teaching neural networks to respect the laws of physics.
Everything else in this project builds on what's defined here.

The magic happens when we combine automatic differentiation with physics equations.
Instead of just fitting data, we're making sure our neural network actually follows 
differential equations. Pretty cool, right?

Built by: A student who got way too excited about PINNs
Year: 2025
"""

import math
import time
from typing import Tuple, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class PINNNet(nn.Module):
    """
    The Neural Network That Learns Physics
    
    This is your standard multi-layer perceptron, but with a twist: it's going to learn
    to solve differential equations! It's basically a universal function approximator
    that we'll teach to respect the laws of physics.
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, 
                 hidden_layers: int = 3, output_dim: int = 1, 
                 activation: str = 'tanh'):
        """
        Build your neural network architecture.
        
        Args:
            input_dim: How many inputs? (1 for ODEs, 2 for PDEs usually)
            hidden_dim: How many neurons per hidden layer? (32-64 is often sweet spot)
            hidden_layers: How deep should we go? (3-4 layers work well for most problems)
            output_dim: How many outputs? (usually just 1 for the solution u)
            activation: Which squashing function? ('tanh' is PINN-friendly, 'relu' can work too)
        """
        super().__init__()
        
        # Pick your activation function (like choosing your fighter in a video game)
        if activation == 'tanh':
            act_fn = nn.Tanh  # Smooth, well-behaved, PINN's favorite
        elif activation == 'relu':
            act_fn = nn.ReLU  # Simple, fast, but can be choppy for derivatives
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid  # Classic but can saturate easily
        else:
            raise ValueError(f"Unknown activation: {activation}. Stick to the classics!")
        
        # Time to build our neural network layer by layer
        layers = []
        
        # Input layer: transform your input to hidden dimension
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_fn())
        
        # Hidden layers: this is where the magic happens
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
            
        # Output layer: no activation here, we want raw values
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Wrap it all up in a nice Sequential container
        self.net = nn.Sequential(*layers)
        self._initialize_weights()  # Start with good guesses
    
    def _initialize_weights(self):
        """Give our network a good starting point instead of random chaos."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier knows what he's doing
                nn.init.zeros_(module.bias)  # Start biases at zero (simple and effective)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The actual forward pass - input goes in, prediction comes out.
        
        Args:
            x: Your input points (could be time, space, or both)
            
        Returns:
            The network's best guess at the solution
        """
        return self.net(x)  # That's it! PyTorch does the heavy lifting


class BasePINN(ABC):
    """
    The PINN Blueprint: Where Neural Networks Learn Physics
    
    This is the master template that all our PINN solvers inherit from. Think of it as
    the shared DNA that gives every PINN the ability to:
    - Handle physics equations
    - Train with automatic differentiation  
    - Keep track of losses
    - Work on both CPU and GPU
    
    Every specific problem (like heat equation, wave equation) builds on this foundation.
    """
    
    def __init__(self, device: str = 'auto', seed: int = 12345):
        """
        Set up the PINN foundation.
        
        Args:
            device: Where to run computations ('cpu', 'cuda', or 'auto' to let us decide)
            seed: Random seed (because reproducible results are important!)
        """
        # Figure out where to run our computations
        if device == 'auto':
            # Use GPU if available, otherwise fall back to CPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
        else:
            self.device = torch.device(device)
        
        # Set random seeds so we get the same results every time
        self.seed = seed
        self._set_seeds()
        
        # Set up containers to track how training goes
        self.training_history = {
            'total_loss': [],      # Overall loss (the big picture)
            'pde_loss': [],        # How well we satisfy the physics
            'ic_loss': [],         # How well we match initial conditions
            'bc_loss': []          # How well we respect boundary conditions
        }
        
        # These will be set up by the specific PINN implementations
        self.model = None      # The neural network itself
        self.optimizer = None  # The thing that updates weights
    
    def _set_seeds(self):
        """Lock down randomness so experiments are repeatable."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)  # GPU randomness too
    
    def to_tensor(self, x: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        """
        Convert numpy arrays to PyTorch tensors (with the right settings).
        
        Args:
            x: Your data as a numpy array
            requires_grad: Set to True if you need gradients (spoiler: you usually do for PINNs)
            
        Returns:
            A shiny new tensor ready for neural network action
        """
        tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        if requires_grad:
            tensor.requires_grad = True  # This is how PyTorch knows to track gradients
        return tensor
    
    @abstractmethod
    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        The heart of PINN: how well does our network satisfy the physics?
        
        This is where you define your differential equation. Each specific problem
        (heat, wave, etc.) will implement this differently. The goal is to compute
        how much the neural network's output violates the PDE at each point.
        
        Args:
            x: Points where we're checking the physics (collocation points)
            
        Returns:
            Residual values - ideally these should be close to zero if physics is satisfied
        """
        pass
    
    @abstractmethod
    def compute_ic_loss(self) -> torch.Tensor:
        """
        Make sure we start from the right place.
        
        Initial conditions are like the starting point of our story. 
        "At time t=0, the temperature was..." or "At the beginning, u(0) = 1"
        This method checks how well our neural network respects those starting conditions.
        
        Returns:
            How far off we are from the correct initial conditions
        """
        pass
    
    def compute_bc_loss(self) -> torch.Tensor:
        """
        Compute the boundary condition loss.
        
        Default implementation returns zero (for problems without boundaries).
        Override in subclasses that need boundary conditions.
        
        Returns:
            Boundary condition loss tensor
        """
        return torch.tensor(0.0, device=self.device)
    
    def compute_total_loss(self, x_collocation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total PINN loss.
        
        Args:
            x_collocation: Collocation points for PDE residual evaluation
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # PDE residual loss
        residual = self.compute_pde_residual(x_collocation)
        pde_loss = torch.mean(residual ** 2)
        
        # Initial condition loss
        ic_loss = self.compute_ic_loss()
        
        # Boundary condition loss
        bc_loss = self.compute_bc_loss()
        
        # Combine losses with weights
        total_loss = pde_loss + self.lambda_ic * ic_loss + self.lambda_bc * bc_loss
        
        # Return loss components for monitoring
        loss_components = {
            'pde_loss': pde_loss.detach().cpu().item(),
            'ic_loss': ic_loss.detach().cpu().item(),
            'bc_loss': bc_loss.detach().cpu().item(),
            'total_loss': total_loss.detach().cpu().item()
        }
        
        return total_loss, loss_components
    
    def train(self, n_iterations: int = 5000, 
              learning_rate: float = 1e-3,
              n_collocation: int = 1000,
              lambda_ic: float = 10.0,
              lambda_bc: float = 1.0,
              print_frequency: int = 500,
              optimizer_type: str = 'adam') -> Dict[str, np.ndarray]:
        """
        Train the PINN model.
        
        Args:
            n_iterations: Number of training iterations
            learning_rate: Learning rate for optimizer
            n_collocation: Number of collocation points
            lambda_ic: Weight for initial condition loss
            lambda_bc: Weight for boundary condition loss
            print_frequency: How often to print training progress
            optimizer_type: Type of optimizer ('adam' or 'lbfgs')
            
        Returns:
            Training history dictionary
        """
        # Store loss weights
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'lbfgs':
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Generate collocation points
        x_collocation = self.generate_collocation_points(n_collocation)
        
        # Training loop
        start_time = time.time()
        
        for iteration in range(1, n_iterations + 1):
            if optimizer_type.lower() == 'adam':
                self._train_step_adam(x_collocation)
            else:
                self._train_step_lbfgs(x_collocation)
            
            # Print progress
            if iteration % print_frequency == 0 or iteration == 1:
                self._print_progress(iteration)
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        # Convert training history to numpy arrays
        history = {key: np.array(values) for key, values in self.training_history.items()}
        return history
    
    def _train_step_adam(self, x_collocation: torch.Tensor):
        """Single training step for Adam optimizer."""
        self.optimizer.zero_grad()
        loss, loss_components = self.compute_total_loss(x_collocation)
        loss.backward()
        self.optimizer.step()
        
        # Store training history
        for key, value in loss_components.items():
            self.training_history[key].append(value)
    
    def _train_step_lbfgs(self, x_collocation: torch.Tensor):
        """Single training step for L-BFGS optimizer."""
        def closure():
            self.optimizer.zero_grad()
            loss, loss_components = self.compute_total_loss(x_collocation)
            loss.backward()
            
            # Store training history
            for key, value in loss_components.items():
                if len(self.training_history[key]) == 0 or \
                   value != self.training_history[key][-1]:  # Avoid duplicates
                    self.training_history[key].append(value)
            
            return loss
        
        self.optimizer.step(closure)
    
    def _print_progress(self, iteration: int):
        """Print training progress."""
        if len(self.training_history['total_loss']) > 0:
            total_loss = self.training_history['total_loss'][-1]
            pde_loss = self.training_history['pde_loss'][-1]
            ic_loss = self.training_history['ic_loss'][-1]
            bc_loss = self.training_history['bc_loss'][-1]
            
            print(f"Iter {iteration:5d} | Total: {total_loss:.6e} | "
                  f"PDE: {pde_loss:.6e} | IC: {ic_loss:.6e} | BC: {bc_loss:.6e}")
    
    @abstractmethod
    def generate_collocation_points(self, n_points: int) -> torch.Tensor:
        """
        Generate collocation points for PDE residual evaluation.
        
        Args:
            n_points: Number of collocation points to generate
            
        Returns:
            Tensor of collocation points
        """
        pass
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions at given points.
        
        Args:
            x: Input points for prediction
            
        Returns:
            Predicted values as numpy array
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = self.to_tensor(x)
            predictions = self.model(x_tensor)
            return predictions.cpu().numpy()
    
    def compute_relative_l2_error(self, x_test: np.ndarray, 
                                 u_exact: np.ndarray) -> float:
        """
        Compute relative L2 error against exact solution.
        
        Args:
            x_test: Test points
            u_exact: Exact solution values
            
        Returns:
            Relative L2 error
        """
        u_pred = self.predict(x_test)
        if u_pred.ndim > 1:
            u_pred = u_pred.flatten()
        if u_exact.ndim > 1:
            u_exact = u_exact.flatten()
            
        numerator = np.linalg.norm(u_pred - u_exact)
        denominator = np.linalg.norm(u_exact)
        
        if denominator == 0:
            return float('inf') if numerator > 0 else 0.0
        
        return numerator / denominator


def save_training_plots(history: Dict[str, np.ndarray], 
                       save_path: str = "training_history.png"):
    """
    Save training history plots.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot total loss
    ax1.plot(history['total_loss'], label='Total Loss', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Total Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss components
    ax2.plot(history['pde_loss'], label='PDE Loss', alpha=0.8)
    ax2.plot(history['ic_loss'], label='IC Loss', alpha=0.8)
    if np.any(np.array(history['bc_loss']) > 0):
        ax2.plot(history['bc_loss'], label='BC Loss', alpha=0.8)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {save_path}")
