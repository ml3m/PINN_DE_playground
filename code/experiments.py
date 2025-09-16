"""
Advanced Experiments and Comparisons for PINN Thesis
====================================================

This module contains advanced experiments including optimizer comparisons,
architecture studies, inverse problems, and noise robustness analysis.

Author: Bachelor Thesis Student
Date: 2025
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
import seaborn as sns

from pinn_base import BasePINN, PINNNet, save_training_plots
from ode_solver import LinearODEPINN, LogisticODEPINN
from pde_solver import HeatEquationPINN


class OptimizerComparison:
    """
    Compare different optimizers for PINN training.
    """
    
    def __init__(self, problem_type: str = 'linear_ode', **problem_kwargs):
        """
        Initialize optimizer comparison study.
        
        Args:
            problem_type: Type of problem ('linear_ode', 'logistic_ode', 'heat_pde')
            **problem_kwargs: Arguments for the specific problem
        """
        self.problem_type = problem_type
        self.problem_kwargs = problem_kwargs
        self.results = {}
    
    def run_comparison(self, optimizers: List[str] = None, 
                      n_iterations: int = 5000, **train_kwargs) -> Dict[str, Any]:
        """
        Run optimizer comparison experiment.
        
        Args:
            optimizers: List of optimizers to compare ['adam', 'lbfgs', 'sgd']
            n_iterations: Number of training iterations
            **train_kwargs: Additional training arguments
            
        Returns:
            Dictionary containing comparison results
        """
        if optimizers is None:
            optimizers = ['adam', 'lbfgs']
        
        print(f"\n{'='*60}")
        print(f"Optimizer Comparison for {self.problem_type.upper()}")
        print(f"{'='*60}")
        
        for optimizer in optimizers:
            print(f"\nTraining with {optimizer.upper()} optimizer...")
            
            # Create fresh solver instance
            solver = self._create_solver()
            
            # Special handling for different optimizers
            if optimizer.lower() == 'lbfgs':
                # L-BFGS typically needs fewer iterations
                iterations = min(n_iterations // 5, 1000)
                lr = 1.0  # L-BFGS handles its own line search
            else:
                iterations = n_iterations
                lr = train_kwargs.get('learning_rate', 1e-3)
            
            # Train the model
            start_time = time.time()
            train_kwargs_copy = train_kwargs.copy()
            # Remove print_frequency from kwargs to avoid duplicates
            train_kwargs_copy.pop('print_frequency', None)
            history = solver.train(
                n_iterations=iterations,
                learning_rate=lr,
                optimizer_type=optimizer,
                print_frequency=max(iterations // 10, 100),
                **train_kwargs_copy
            )
            training_time = time.time() - start_time
            
            # Evaluate final error
            final_error = self._compute_final_error(solver)
            
            # Store results
            self.results[optimizer] = {
                'history': history,
                'training_time': training_time,
                'final_error': final_error,
                'final_loss': history['total_loss'][-1],
                'solver': solver
            }
            
            print(f"{optimizer.upper()} Results:")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Final loss: {history['total_loss'][-1]:.6e}")
            print(f"  Final error: {final_error:.6e}")
        
        self._plot_optimizer_comparison()
        return self.results
    
    def _create_solver(self) -> BasePINN:
        """Create a fresh solver instance."""
        if self.problem_type == 'linear_ode':
            return LinearODEPINN(**self.problem_kwargs)
        elif self.problem_type == 'logistic_ode':
            return LogisticODEPINN(**self.problem_kwargs)
        elif self.problem_type == 'heat_pde':
            return HeatEquationPINN(**self.problem_kwargs)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")
    
    def _compute_final_error(self, solver: BasePINN) -> float:
        """Compute final relative L2 error."""
        if hasattr(solver, 'exact_solution'):
            if hasattr(solver, 't_max'):  # ODE case
                t_test = np.linspace(0, solver.t_max, 300).reshape(-1, 1)
                u_exact = solver.exact_solution(t_test.flatten())
                return solver.compute_relative_l2_error(t_test, u_exact)
            else:  # PDE case
                # Use a subset for efficiency
                x_test = np.linspace(0, solver.x_max, 50)
                t_test = np.linspace(0, solver.t_max, 50)
                X, T = np.meshgrid(x_test, t_test)
                xt_test = np.column_stack([X.flatten(), T.flatten()])
                u_exact = solver.exact_solution(X, T).flatten()
                return solver.compute_relative_l2_error(xt_test, u_exact)
        return float('nan')
    
    def _plot_optimizer_comparison(self):
        """Plot optimizer comparison results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss convergence
        for optimizer, results in self.results.items():
            history = results['history']
            ax1.plot(history['total_loss'], label=f'{optimizer.upper()}', linewidth=2)
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss (log scale)')
        ax1.set_title('Loss Convergence Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot PDE loss
        for optimizer, results in self.results.items():
            history = results['history']
            ax2.plot(history['pde_loss'], label=f'{optimizer.upper()}', linewidth=2)
        
        ax2.set_yscale('log')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('PDE Loss (log scale)')
        ax2.set_title('PDE Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bar plot of final metrics
        optimizers = list(self.results.keys())
        final_losses = [self.results[opt]['final_loss'] for opt in optimizers]
        training_times = [self.results[opt]['training_time'] for opt in optimizers]
        
        ax3.bar(optimizers, final_losses)
        ax3.set_yscale('log')
        ax3.set_ylabel('Final Loss (log scale)')
        ax3.set_title('Final Loss Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        ax4.bar(optimizers, training_times)
        ax4.set_ylabel('Training Time (s)')
        ax4.set_title('Training Time Comparison')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'figures/optimizer_comparison_{self.problem_type}.png', 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Optimizer comparison plot saved to figures/optimizer_comparison_{self.problem_type}.png")


class ArchitectureStudy:
    """
    Study the effect of network architecture on PINN performance.
    """
    
    def __init__(self, problem_type: str = 'linear_ode', **problem_kwargs):
        """
        Initialize architecture study.
        
        Args:
            problem_type: Type of problem to study
            **problem_kwargs: Arguments for the specific problem
        """
        self.problem_type = problem_type
        self.problem_kwargs = problem_kwargs
        self.results = {}
    
    def run_width_study(self, widths: List[int] = None, **train_kwargs) -> Dict[str, Any]:
        """
        Study effect of network width (hidden units).
        
        Args:
            widths: List of hidden dimensions to test
            **train_kwargs: Training arguments
            
        Returns:
            Study results
        """
        if widths is None:
            widths = [16, 32, 64, 128, 256]
        
        print(f"\n{'='*50}")
        print("Network Width Study")
        print(f"{'='*50}")
        
        results = {}
        
        for width in widths:
            print(f"\nTesting width: {width}")
            
            # Create solver with specified width
            solver = self._create_solver_with_architecture(hidden_dim=width)
            
            # Train and evaluate
            start_time = time.time()
            train_kwargs_copy = train_kwargs.copy()
            train_kwargs_copy.setdefault('print_frequency', 1000)
            history = solver.train(**train_kwargs_copy)
            training_time = time.time() - start_time
            
            final_error = self._compute_final_error(solver)
            
            results[width] = {
                'history': history,
                'training_time': training_time,
                'final_error': final_error,
                'final_loss': history['total_loss'][-1]
            }
            
            print(f"  Final loss: {history['total_loss'][-1]:.6e}")
            print(f"  Final error: {final_error:.6e}")
            print(f"  Training time: {training_time:.2f}s")
        
        self.results['width_study'] = results
        self._plot_width_study(results)
        return results
    
    def run_depth_study(self, depths: List[int] = None, **train_kwargs) -> Dict[str, Any]:
        """
        Study effect of network depth (number of layers).
        
        Args:
            depths: List of hidden layer counts to test
            **train_kwargs: Training arguments
            
        Returns:
            Study results
        """
        if depths is None:
            depths = [2, 3, 4, 5, 6, 8]
        
        print(f"\n{'='*50}")
        print("Network Depth Study")
        print(f"{'='*50}")
        
        results = {}
        
        for depth in depths:
            print(f"\nTesting depth: {depth}")
            
            # Create solver with specified depth
            solver = self._create_solver_with_architecture(hidden_layers=depth)
            
            # Train and evaluate
            start_time = time.time()
            train_kwargs_copy = train_kwargs.copy()
            train_kwargs_copy.setdefault('print_frequency', 1000)
            history = solver.train(**train_kwargs_copy)
            training_time = time.time() - start_time
            
            final_error = self._compute_final_error(solver)
            
            results[depth] = {
                'history': history,
                'training_time': training_time,
                'final_error': final_error,
                'final_loss': history['total_loss'][-1]
            }
            
            print(f"  Final loss: {history['total_loss'][-1]:.6e}")
            print(f"  Final error: {final_error:.6e}")
            print(f"  Training time: {training_time:.2f}s")
        
        self.results['depth_study'] = results
        self._plot_depth_study(results)
        return results
    
    def run_activation_study(self, activations: List[str] = None, **train_kwargs) -> Dict[str, Any]:
        """
        Study effect of activation functions.
        
        Args:
            activations: List of activation functions to test
            **train_kwargs: Training arguments
            
        Returns:
            Study results
        """
        if activations is None:
            activations = ['tanh', 'relu', 'sigmoid']
        
        print(f"\n{'='*50}")
        print("Activation Function Study")
        print(f"{'='*50}")
        
        results = {}
        
        for activation in activations:
            print(f"\nTesting activation: {activation}")
            
            # Create solver with specified activation
            solver = self._create_solver_with_architecture(activation=activation)
            
            # Train and evaluate
            start_time = time.time()
            train_kwargs_copy = train_kwargs.copy()
            train_kwargs_copy.setdefault('print_frequency', 1000)
            history = solver.train(**train_kwargs_copy)
            training_time = time.time() - start_time
            
            final_error = self._compute_final_error(solver)
            
            results[activation] = {
                'history': history,
                'training_time': training_time,
                'final_error': final_error,
                'final_loss': history['total_loss'][-1]
            }
            
            print(f"  Final loss: {history['total_loss'][-1]:.6e}")
            print(f"  Final error: {final_error:.6e}")
            print(f"  Training time: {training_time:.2f}s")
        
        self.results['activation_study'] = results
        self._plot_activation_study(results)
        return results
    
    def _create_solver_with_architecture(self, **arch_kwargs) -> BasePINN:
        """Create solver with specified architecture."""
        # Default architecture parameters
        arch_params = {
            'hidden_dim': 32,
            'hidden_layers': 3,
            'activation': 'tanh'
        }
        arch_params.update(arch_kwargs)
        
        # Create solver and replace its model
        if self.problem_type == 'linear_ode':
            solver = LinearODEPINN(**self.problem_kwargs)
        elif self.problem_type == 'logistic_ode':
            solver = LogisticODEPINN(**self.problem_kwargs)
        elif self.problem_type == 'heat_pde':
            solver = HeatEquationPINN(**self.problem_kwargs)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")
        
        # Replace model with new architecture
        input_dim = 1 if 'ode' in self.problem_type else 2
        output_dim = 1
        
        solver.model = PINNNet(
            input_dim=input_dim,
            output_dim=output_dim,
            **arch_params
        ).to(solver.device)
        
        return solver
    
    def _compute_final_error(self, solver: BasePINN) -> float:
        """Compute final relative L2 error."""
        if hasattr(solver, 'exact_solution'):
            if hasattr(solver, 't_max'):  # ODE case
                t_test = np.linspace(0, solver.t_max, 300).reshape(-1, 1)
                u_exact = solver.exact_solution(t_test.flatten())
                return solver.compute_relative_l2_error(t_test, u_exact)
            else:  # PDE case
                x_test = np.linspace(0, solver.x_max, 50)
                t_test = np.linspace(0, solver.t_max, 50)
                X, T = np.meshgrid(x_test, t_test)
                xt_test = np.column_stack([X.flatten(), T.flatten()])
                u_exact = solver.exact_solution(X, T).flatten()
                return solver.compute_relative_l2_error(xt_test, u_exact)
        return float('nan')
    
    def _plot_width_study(self, results: Dict):
        """Plot width study results."""
        widths = sorted(results.keys())
        final_losses = [results[w]['final_loss'] for w in widths]
        final_errors = [results[w]['final_error'] for w in widths]
        training_times = [results[w]['training_time'] for w in widths]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        ax1.plot(widths, final_losses, 'o-', linewidth=2, markersize=8)
        ax1.set_yscale('log')
        ax1.set_xlabel('Network Width (Hidden Units)')
        ax1.set_ylabel('Final Loss (log scale)')
        ax1.set_title('Loss vs Network Width')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(widths, final_errors, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_yscale('log')
        ax2.set_xlabel('Network Width (Hidden Units)')
        ax2.set_ylabel('Final Error (log scale)')
        ax2.set_title('Error vs Network Width')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(widths, training_times, 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Network Width (Hidden Units)')
        ax3.set_ylabel('Training Time (s)')
        ax3.set_title('Training Time vs Network Width')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figures/width_study_{self.problem_type}.png', 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Width study plot saved to figures/width_study_{self.problem_type}.png")
    
    def _plot_depth_study(self, results: Dict):
        """Plot depth study results."""
        depths = sorted(results.keys())
        final_losses = [results[d]['final_loss'] for d in depths]
        final_errors = [results[d]['final_error'] for d in depths]
        training_times = [results[d]['training_time'] for d in depths]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        ax1.plot(depths, final_losses, 'o-', linewidth=2, markersize=8)
        ax1.set_yscale('log')
        ax1.set_xlabel('Network Depth (Hidden Layers)')
        ax1.set_ylabel('Final Loss (log scale)')
        ax1.set_title('Loss vs Network Depth')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(depths, final_errors, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_yscale('log')
        ax2.set_xlabel('Network Depth (Hidden Layers)')
        ax2.set_ylabel('Final Error (log scale)')
        ax2.set_title('Error vs Network Depth')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(depths, training_times, 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Network Depth (Hidden Layers)')
        ax3.set_ylabel('Training Time (s)')
        ax3.set_title('Training Time vs Network Depth')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figures/depth_study_{self.problem_type}.png', 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Depth study plot saved to figures/depth_study_{self.problem_type}.png")
    
    def _plot_activation_study(self, results: Dict):
        """Plot activation function study results."""
        activations = list(results.keys())
        final_losses = [results[a]['final_loss'] for a in activations]
        final_errors = [results[a]['final_error'] for a in activations]
        training_times = [results[a]['training_time'] for a in activations]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        ax1.bar(activations, final_losses)
        ax1.set_yscale('log')
        ax1.set_ylabel('Final Loss (log scale)')
        ax1.set_title('Loss vs Activation Function')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(activations, final_errors, color='orange')
        ax2.set_yscale('log')
        ax2.set_ylabel('Final Error (log scale)')
        ax2.set_title('Error vs Activation Function')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.bar(activations, training_times, color='green')
        ax3.set_ylabel('Training Time (s)')
        ax3.set_title('Training Time vs Activation Function')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'figures/activation_study_{self.problem_type}.png', 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Activation study plot saved to figures/activation_study_{self.problem_type}.png")


class NoiseRobustnessStudy:
    """
    Study robustness of PINNs to noisy data.
    """
    
    def __init__(self, problem_type: str = 'linear_ode', **problem_kwargs):
        """
        Initialize noise robustness study.
        
        Args:
            problem_type: Type of problem to study
            **problem_kwargs: Arguments for the specific problem
        """
        self.problem_type = problem_type
        self.problem_kwargs = problem_kwargs
        self.results = {}
    
    def run_noise_study(self, noise_levels: List[float] = None, 
                       n_data_points: int = 50, **train_kwargs) -> Dict[str, Any]:
        """
        Study effect of noise in initial/boundary condition data.
        
        Args:
            noise_levels: List of noise standard deviations to test
            n_data_points: Number of data points to use
            **train_kwargs: Training arguments
            
        Returns:
            Study results
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
        
        print(f"\n{'='*50}")
        print("Noise Robustness Study")
        print(f"{'='*50}")
        
        results = {}
        
        for noise_level in noise_levels:
            print(f"\nTesting noise level: {noise_level}")
            
            # Create solver
            solver = self._create_solver()
            
            # Add noise to initial/boundary conditions (modify loss computation)
            solver.noise_level = noise_level
            solver.n_noisy_points = n_data_points
            
            # Override IC loss to include noisy data
            original_ic_loss = solver.compute_ic_loss
            solver.compute_ic_loss = lambda: self._compute_noisy_ic_loss(solver, original_ic_loss)
            
            # Train and evaluate
            start_time = time.time()
            train_kwargs_copy = train_kwargs.copy()
            train_kwargs_copy.setdefault('print_frequency', 1000)
            history = solver.train(**train_kwargs_copy)
            training_time = time.time() - start_time
            
            final_error = self._compute_final_error(solver)
            
            results[noise_level] = {
                'history': history,
                'training_time': training_time,
                'final_error': final_error,
                'final_loss': history['total_loss'][-1]
            }
            
            print(f"  Final loss: {history['total_loss'][-1]:.6e}")
            print(f"  Final error: {final_error:.6e}")
        
        self.results = results
        self._plot_noise_study(results)
        return results
    
    def _create_solver(self) -> BasePINN:
        """Create a fresh solver instance."""
        if self.problem_type == 'linear_ode':
            return LinearODEPINN(**self.problem_kwargs)
        elif self.problem_type == 'logistic_ode':
            return LogisticODEPINN(**self.problem_kwargs)
        elif self.problem_type == 'heat_pde':
            return HeatEquationPINN(**self.problem_kwargs)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")
    
    def _compute_noisy_ic_loss(self, solver: BasePINN, original_ic_loss: Callable) -> torch.Tensor:
        """Compute IC loss with added noisy data points."""
        # Original IC loss
        ic_loss = original_ic_loss()
        
        if hasattr(solver, 'noise_level') and solver.noise_level > 0:
            # Add noisy data points
            if hasattr(solver, 't_max'):  # ODE case
                # Generate noisy initial condition data
                t_data = torch.zeros(solver.n_noisy_points, 1, device=solver.device)
                u_exact = solver.u0 * torch.ones_like(t_data)
                noise = torch.normal(0, solver.noise_level, size=u_exact.shape, device=solver.device)
                u_noisy = u_exact + noise
                
                u_pred = solver.model(t_data)
                data_loss = torch.mean((u_pred - u_noisy) ** 2)
                
            else:  # PDE case
                # Generate noisy initial condition data
                x_data = torch.linspace(0, solver.x_max, solver.n_noisy_points, device=solver.device).reshape(-1, 1)
                t_data = torch.zeros_like(x_data)
                xt_data = torch.cat([x_data, t_data], dim=1)
                
                u_exact = solver.initial_condition(x_data)
                noise = torch.normal(0, solver.noise_level, size=u_exact.shape, device=solver.device)
                u_noisy = u_exact + noise
                
                u_pred = solver.model(xt_data)
                data_loss = torch.mean((u_pred - u_noisy) ** 2)
            
            # Combine original IC loss with noisy data loss
            ic_loss = ic_loss + 10.0 * data_loss  # Weight noisy data more
        
        return ic_loss
    
    def _compute_final_error(self, solver: BasePINN) -> float:
        """Compute final relative L2 error."""
        if hasattr(solver, 'exact_solution'):
            if hasattr(solver, 't_max'):  # ODE case
                t_test = np.linspace(0, solver.t_max, 300).reshape(-1, 1)
                u_exact = solver.exact_solution(t_test.flatten())
                return solver.compute_relative_l2_error(t_test, u_exact)
            else:  # PDE case
                x_test = np.linspace(0, solver.x_max, 50)
                t_test = np.linspace(0, solver.t_max, 50)
                X, T = np.meshgrid(x_test, t_test)
                xt_test = np.column_stack([X.flatten(), T.flatten()])
                u_exact = solver.exact_solution(X, T).flatten()
                return solver.compute_relative_l2_error(xt_test, u_exact)
        return float('nan')
    
    def _plot_noise_study(self, results: Dict):
        """Plot noise robustness study results."""
        noise_levels = sorted(results.keys())
        final_losses = [results[n]['final_loss'] for n in noise_levels]
        final_errors = [results[n]['final_error'] for n in noise_levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(noise_levels, final_losses, 'o-', linewidth=2, markersize=8)
        ax1.set_yscale('log')
        ax1.set_xlabel('Noise Level (σ)')
        ax1.set_ylabel('Final Loss (log scale)')
        ax1.set_title('Loss vs Noise Level')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(noise_levels, final_errors, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_yscale('log')
        ax2.set_xlabel('Noise Level (σ)')
        ax2.set_ylabel('Final Error (log scale)')
        ax2.set_title('Error vs Noise Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figures/noise_study_{self.problem_type}.png', 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Noise study plot saved to figures/noise_study_{self.problem_type}.png")


def run_comprehensive_study(problem_type: str = 'linear_ode', **problem_kwargs):
    """
    Run a comprehensive study including all experiments.
    
    Args:
        problem_type: Type of problem to study
        **problem_kwargs: Problem-specific arguments
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PINN STUDY: {problem_type.upper()}")
    print(f"{'='*80}")
    
    # Common training parameters
    train_kwargs = {
        'n_iterations': 3000,
        'learning_rate': 1e-3,
        'n_collocation': 1000
    }
    
    results = {}
    
    # 1. Optimizer Comparison
    print("\n1. Running Optimizer Comparison...")
    opt_study = OptimizerComparison(problem_type, **problem_kwargs)
    results['optimizer'] = opt_study.run_comparison(**train_kwargs)
    
    # 2. Architecture Studies
    print("\n2. Running Architecture Studies...")
    arch_study = ArchitectureStudy(problem_type, **problem_kwargs)
    
    # Width study
    results['width'] = arch_study.run_width_study(**train_kwargs)
    
    # Depth study
    results['depth'] = arch_study.run_depth_study(**train_kwargs)
    
    # Activation study
    results['activation'] = arch_study.run_activation_study(**train_kwargs)
    
    # 3. Noise Robustness Study
    print("\n3. Running Noise Robustness Study...")
    noise_study = NoiseRobustnessStudy(problem_type, **problem_kwargs)
    results['noise'] = noise_study.run_noise_study(**train_kwargs)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE STUDY COMPLETED!")
    print(f"{'='*80}")
    print("Results saved in figures/ directory:")
    print(f"  - optimizer_comparison_{problem_type}.png")
    print(f"  - width_study_{problem_type}.png")
    print(f"  - depth_study_{problem_type}.png")
    print(f"  - activation_study_{problem_type}.png")
    print(f"  - noise_study_{problem_type}.png")
    
    return results
