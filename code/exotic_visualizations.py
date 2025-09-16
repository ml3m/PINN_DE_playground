"""
Exotic and Satisfying 3D Visualizations for PINN Data
=====================================================

This module provides advanced 3D visualization techniques for Physics-Informed Neural Networks,
including interactive plots, animated surfaces, particle systems, and artistic renderings.

Author: Bachelor Thesis Student
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Tuple, List, Optional
import torch

# Import your PINN classes
from pde_solver import HeatEquationPINN, WaveEquationPINN, BurgersEquationPINN
from ode_solver import VanDerPolODEPINN


class ExoticPINNVisualizer:
    """Advanced 3D visualization class for PINN solutions."""
    
    def __init__(self, figsize=(12, 8), dpi=150):
        self.figsize = figsize
        self.dpi = dpi
        
        # Create custom colormaps
        self.plasma_map = self._create_plasma_colormap()
        self.fire_map = self._create_fire_colormap()
        self.ocean_map = self._create_ocean_colormap()
    
    def _create_plasma_colormap(self):
        """Create a plasma-like colormap."""
        colors = ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921']
        return LinearSegmentedColormap.from_list('plasma_custom', colors)
    
    def _create_fire_colormap(self):
        """Create a fire-like colormap."""
        colors = ['#000000', '#330000', '#660000', '#990000', '#cc3300', 
                 '#ff6600', '#ff9900', '#ffcc00', '#ffff00', '#ffffff']
        return LinearSegmentedColormap.from_list('fire', colors)
    
    def _create_ocean_colormap(self):
        """Create an ocean-like colormap."""
        colors = ['#000428', '#004e92', '#009ffd', '#00d2ff', '#ffffff']
        return LinearSegmentedColormap.from_list('ocean', colors)
    
    def visualize_heat_equation_3d_artistic(self, solver: HeatEquationPINN, 
                                          save_path: str = "figures/heat_3d_artistic.png"):
        """
        Create an artistic 3D visualization of heat equation with multiple visual effects.
        """
        # Generate solution data
        x = np.linspace(0, solver.x_max, 80)
        t = np.linspace(0, solver.t_max, 80)
        X, T = np.meshgrid(x, t)
        
        # Get PINN predictions
        xt_flat = np.column_stack([X.flatten(), T.flatten()])
        xt_tensor = solver.to_tensor(xt_flat)
        
        with torch.no_grad():
            u_pred = solver.model(xt_tensor).cpu().numpy()
        U = u_pred.reshape(X.shape)
        
        # Create the main 3D plot
        fig = plt.figure(figsize=(16, 12))
        
        # Main 3D surface
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(X, T, U, cmap=self.plasma_map, alpha=0.8,
                               linewidth=0, antialiased=True, shade=True)
        
        # Add wireframe overlay
        ax1.plot_wireframe(X, T, U, colors='white', alpha=0.3, linewidth=0.5)
        
        # Add contour projections
        ax1.contour(X, T, U, zdir='z', offset=U.min()-0.1, cmap=self.plasma_map, alpha=0.6)
        ax1.contour(X, T, U, zdir='x', offset=-0.1, cmap=self.plasma_map, alpha=0.6)
        ax1.contour(X, T, U, zdir='y', offset=solver.t_max+0.1, cmap=self.plasma_map, alpha=0.6)
        
        ax1.set_xlabel('Space (x)', fontsize=12, labelpad=10)
        ax1.set_ylabel('Time (t)', fontsize=12, labelpad=10)
        ax1.set_zlabel('Temperature (u)', fontsize=12, labelpad=10)
        ax1.set_title('Heat Equation - Artistic 3D Surface', fontsize=14, fontweight='bold')
        
        # Top-down heatmap with contours
        ax2 = fig.add_subplot(222)
        im = ax2.imshow(U, extent=[0, solver.x_max, 0, solver.t_max], 
                       cmap=self.fire_map, aspect='auto', origin='lower')
        contours = ax2.contour(X, T, U, levels=15, colors='white', alpha=0.7, linewidths=1.5)
        ax2.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        ax2.set_xlabel('Space (x)')
        ax2.set_ylabel('Time (t)')
        ax2.set_title('Temperature Field with Isotherms')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # Cross-sections at different times
        ax3 = fig.add_subplot(223)
        time_slices = [0.1, 0.3, 0.5, 0.7, 0.9]
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_slices)))
        
        for i, t_slice in enumerate(time_slices):
            t_idx = int(t_slice * (len(t) - 1))
            ax3.plot(x, U[t_idx, :], color=colors[i], linewidth=2.5, 
                    label=f't={t_slice:.1f}', alpha=0.8)
            ax3.fill_between(x, 0, U[t_idx, :], color=colors[i], alpha=0.2)
        
        ax3.set_xlabel('Space (x)')
        ax3.set_ylabel('Temperature (u)')
        ax3.set_title('Temperature Evolution Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gradient magnitude visualization
        ax4 = fig.add_subplot(224)
        grad_u = np.gradient(U)
        grad_magnitude = np.sqrt(grad_u[0]**2 + grad_u[1]**2)
        
        im2 = ax4.imshow(grad_magnitude, extent=[0, solver.x_max, 0, solver.t_max],
                        cmap=self.ocean_map, aspect='auto', origin='lower')
        ax4.set_xlabel('Space (x)')
        ax4.set_ylabel('Time (t)')
        ax4.set_title('Temperature Gradient Magnitude')
        plt.colorbar(im2, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Artistic 3D heat visualization saved to {save_path}")
        return X, T, U
    
    def create_interactive_3d_plotly(self, X, T, U, equation_name="PINN Solution"):
        """
        Create an interactive 3D plot using Plotly with advanced features.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "surface", "colspan": 2}, None],
                   [{"type": "heatmap"}, {"type": "scatter"}]],
            subplot_titles=('Interactive 3D Surface', 'Top View Heatmap', 'Cross Sections'),
            vertical_spacing=0.08
        )
        
        # Main 3D surface
        surface = go.Surface(
            x=X, y=T, z=U,
            colorscale='Viridis',
            name='PINN Solution',
            showscale=True,
            colorbar=dict(x=0.9, len=0.7),
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                roughness=0.1,
                specular=0.6,
                fresnel=0.02
            ),
            contours=dict(
                x=dict(show=True, usecolormap=True, project_x=True),
                y=dict(show=True, usecolormap=True, project_y=True),
                z=dict(show=True, usecolormap=True, project_z=True)
            )
        )
        
        fig.add_trace(surface, row=1, col=1)
        
        # Heatmap
        heatmap = go.Heatmap(
            x=X[0, :], y=T[:, 0], z=U,
            colorscale='Plasma',
            showscale=False
        )
        fig.add_trace(heatmap, row=2, col=1)
        
        # Cross sections
        time_indices = [0, len(T)//4, len(T)//2, 3*len(T)//4, -1]
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        for i, t_idx in enumerate(time_indices):
            fig.add_trace(
                go.Scatter(
                    x=X[0, :], y=U[t_idx, :],
                    mode='lines+markers',
                    name=f't={T[t_idx, 0]:.2f}',
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=4)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Interactive {equation_name} Visualization",
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title="Space (x)",
                yaxis_title="Time (t)",
                zaxis_title="Solution (u)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=800
        )
        
        # Save as HTML
        fig.write_html("figures/interactive_3d_visualization.html")
        print("Interactive 3D visualization saved to figures/interactive_3d_visualization.html")
        
        return fig
    
    def create_particle_system_visualization(self, solver: HeatEquationPINN,
                                           save_path: str = "figures/particle_heat.gif"):
        """
        Create a particle system visualization where particles represent heat flow.
        """
        # Generate solution data
        x = np.linspace(0, solver.x_max, 50)
        t = np.linspace(0, solver.t_max, 50)
        X, T = np.meshgrid(x, t)
        
        # Get PINN predictions
        xt_flat = np.column_stack([X.flatten(), T.flatten()])
        xt_tensor = solver.to_tensor(xt_flat)
        
        with torch.no_grad():
            u_pred = solver.model(xt_tensor).cpu().numpy()
        U = u_pred.reshape(X.shape)
        
        # Create particle positions
        n_particles = 200
        particle_x = np.random.uniform(0, solver.x_max, n_particles)
        particle_y = np.zeros(n_particles)
        
        # Animation function
        def animate_particles(frame):
            ax.clear()
            
            # Current time
            current_t = frame * solver.t_max / 100
            t_idx = int(frame * len(t) / 100)
            if t_idx >= len(t):
                t_idx = len(t) - 1
            
            # Update particle positions based on temperature gradient
            current_temp = U[t_idx, :]
            temp_interp = np.interp(particle_x, x, current_temp)
            
            # Color particles based on temperature
            colors = plt.cm.hot(temp_interp / temp_interp.max())
            sizes = 50 + 200 * temp_interp / temp_interp.max()
            
            # Plot background temperature field
            ax.imshow(U[:t_idx+1, :], extent=[0, solver.x_max, 0, current_t],
                     cmap='hot', aspect='auto', origin='lower', alpha=0.7)
            
            # Plot particles
            ax.scatter(particle_x, np.full(n_particles, current_t), 
                      c=colors, s=sizes, alpha=0.8, edgecolors='white', linewidth=0.5)
            
            # Add streamlines for flow visualization
            if t_idx > 0:
                Y, X_grid = np.meshgrid(np.linspace(0, current_t, t_idx+1), x)
                DX = np.gradient(U[:t_idx+1, :], axis=1)
                DT = np.gradient(U[:t_idx+1, :], axis=0)
                ax.streamplot(X_grid, Y, DX.T, DT.T, color='cyan', alpha=0.6, density=0.8)
            
            ax.set_xlim(0, solver.x_max)
            ax.set_ylim(0, solver.t_max)
            ax.set_xlabel('Space (x)')
            ax.set_ylabel('Time (t)')
            ax.set_title(f'Heat Flow Particle System - t={current_t:.3f}')
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 8))
        anim = FuncAnimation(fig, animate_particles, frames=100, interval=100, repeat=True)
        
        # Save animation
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        plt.close()
        
        print(f"Particle system animation saved to {save_path}")
    
    def create_van_der_pol_phase_space_3d(self, solver: VanDerPolODEPINN,
                                         save_path: str = "figures/van_der_pol_3d.png"):
        """
        Create exotic 3D phase space visualization for Van der Pol oscillator.
        """
        # Generate time points
        t = np.linspace(0, solver.t_max, 1000)
        t_tensor = solver.to_tensor(t.reshape(-1, 1))
        
        with torch.no_grad():
            solution = solver.model(t_tensor).cpu().numpy()
        
        u = solution[:, 0]  # Position
        v = solution[:, 1]  # Velocity
        
        fig = plt.figure(figsize=(16, 12))
        
        # 3D phase space with time as third dimension
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Color by time
        colors = plt.cm.viridis(t / t.max())
        
        # Plot trajectory as a tube
        for i in range(len(t)-1):
            ax1.plot([u[i], u[i+1]], [v[i], v[i+1]], [t[i], t[i+1]], 
                    color=colors[i], linewidth=2, alpha=0.8)
        
        # Add scatter points for emphasis
        ax1.scatter(u[::50], v[::50], t[::50], c=colors[::50], s=30, alpha=0.9)
        
        ax1.set_xlabel('Position (u)')
        ax1.set_ylabel('Velocity (v)')
        ax1.set_zlabel('Time (t)')
        ax1.set_title('Van der Pol Phase Space Evolution')
        
        # Traditional 2D phase portrait
        ax2 = fig.add_subplot(222)
        
        # Plot trajectory with varying line width based on speed
        speed = np.sqrt(np.gradient(u)**2 + np.gradient(v)**2)
        for i in range(len(u)-1):
            width = 0.5 + 3 * speed[i] / speed.max()
            ax2.plot([u[i], u[i+1]], [v[i], v[i+1]], 
                    color=colors[i], linewidth=width, alpha=0.7)
        
        # Add arrows to show direction
        arrow_indices = range(0, len(u), len(u)//20)
        for i in arrow_indices:
            if i < len(u) - 1:
                du = u[i+1] - u[i]
                dv = v[i+1] - v[i]
                ax2.arrow(u[i], v[i], du*10, dv*10, head_width=0.1, 
                         head_length=0.1, fc=colors[i], ec=colors[i])
        
        ax2.set_xlabel('Position (u)')
        ax2.set_ylabel('Velocity (v)')
        ax2.set_title('Phase Portrait with Speed Encoding')
        ax2.grid(True, alpha=0.3)
        
        # Energy surface visualization
        ax3 = fig.add_subplot(223, projection='3d')
        
        # Create energy landscape
        u_grid = np.linspace(u.min()-1, u.max()+1, 50)
        v_grid = np.linspace(v.min()-1, v.max()+1, 50)
        U_grid, V_grid = np.meshgrid(u_grid, v_grid)
        
        # Van der Pol energy-like function (not conserved, but interesting)
        E_grid = 0.5 * V_grid**2 + 0.5 * U_grid**2 - solver.mu/3 * U_grid**3
        
        surf = ax3.plot_surface(U_grid, V_grid, E_grid, alpha=0.3, cmap='coolwarm')
        
        # Project trajectory onto surface
        E_traj = 0.5 * v**2 + 0.5 * u**2 - solver.mu/3 * u**3
        ax3.plot(u, v, E_traj, color='red', linewidth=3, alpha=0.9, label='Trajectory')
        
        ax3.set_xlabel('Position (u)')
        ax3.set_ylabel('Velocity (v)')
        ax3.set_zlabel('Energy-like Function')
        ax3.set_title('Trajectory on Energy Surface')
        
        # Poincaré map (if limit cycle exists)
        ax4 = fig.add_subplot(224)
        
        # Find crossings of u=0 with v>0
        crossings_u = []
        crossings_v = []
        crossings_t = []
        
        for i in range(len(u)-1):
            if u[i] <= 0 <= u[i+1] and v[i] > 0:
                # Linear interpolation to find exact crossing
                alpha = -u[i] / (u[i+1] - u[i])
                v_cross = v[i] + alpha * (v[i+1] - v[i])
                t_cross = t[i] + alpha * (t[i+1] - t[i])
                crossings_u.append(0)
                crossings_v.append(v_cross)
                crossings_t.append(t_cross)
        
        if len(crossings_v) > 1:
            ax4.plot(crossings_t, crossings_v, 'ro-', markersize=8, linewidth=2)
            ax4.set_xlabel('Time at Crossing')
            ax4.set_ylabel('Velocity at u=0')
            ax4.set_title('Poincaré Map (u=0, v>0)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No limit cycle detected\nor insufficient data', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Poincaré Analysis')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Van der Pol 3D visualization saved to {save_path}")
    
    def create_wave_interference_visualization(self, solver: WaveEquationPINN,
                                             save_path: str = "figures/wave_interference.gif"):
        """
        Create wave interference visualization with multiple wave packets.
        """
        # Generate solution data
        x = np.linspace(0, solver.x_max, 100)
        t = np.linspace(0, solver.t_max, 100)
        X, T = np.meshgrid(x, t)
        
        # Get PINN predictions
        xt_flat = np.column_stack([X.flatten(), T.flatten()])
        xt_tensor = solver.to_tensor(xt_flat)
        
        with torch.no_grad():
            u_pred = solver.model(xt_tensor).cpu().numpy()
        U = u_pred.reshape(X.shape)
        
        # Animation function
        def animate_wave(frame):
            ax.clear()
            
            # Current time slice
            t_idx = frame
            if t_idx >= len(t):
                t_idx = len(t) - 1
            
            current_u = U[t_idx, :]
            
            # Main wave plot with filled area
            ax.fill_between(x, 0, current_u, alpha=0.6, color='steelblue', label='Wave')
            ax.plot(x, current_u, 'navy', linewidth=3)
            
            # Add wave crests and troughs
            peaks = []
            troughs = []
            for i in range(1, len(current_u)-1):
                if current_u[i] > current_u[i-1] and current_u[i] > current_u[i+1]:
                    peaks.append((x[i], current_u[i]))
                elif current_u[i] < current_u[i-1] and current_u[i] < current_u[i+1]:
                    troughs.append((x[i], current_u[i]))
            
            # Plot peaks and troughs
            if peaks:
                peak_x, peak_y = zip(*peaks)
                ax.scatter(peak_x, peak_y, color='red', s=100, zorder=5, 
                          marker='^', label='Peaks')
            
            if troughs:
                trough_x, trough_y = zip(*troughs)
                ax.scatter(trough_x, trough_y, color='blue', s=100, zorder=5, 
                          marker='v', label='Troughs')
            
            # Add energy visualization
            kinetic_energy = np.gradient(current_u)**2
            ax2 = ax.twinx()
            ax2.plot(x, kinetic_energy, 'orange', alpha=0.7, linewidth=2, 
                    label='Kinetic Energy')
            ax2.set_ylabel('Kinetic Energy', color='orange')
            
            ax.set_xlim(0, solver.x_max)
            ax.set_ylim(U.min()*1.2, U.max()*1.2)
            ax.set_xlabel('Position (x)')
            ax.set_ylabel('Displacement (u)')
            ax.set_title(f'Wave Propagation - t={t[t_idx]:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 6))
        anim = FuncAnimation(fig, animate_wave, frames=len(t), interval=100, repeat=True)
        
        # Save animation
        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer)
        plt.close()
        
        print(f"Wave interference animation saved to {save_path}")


def demo_exotic_visualizations():
    """Demonstrate all exotic visualization techniques."""
    print("Creating Exotic PINN Visualizations...")
    
    # Initialize visualizer
    viz = ExoticPINNVisualizer()
    
    # Create and train a heat equation PINN
    print("\nSetting up Heat Equation PINN...")
    heat_solver = HeatEquationPINN(x_max=1.0, t_max=0.5, alpha=0.1)
    heat_solver.train(n_iterations=1000, print_frequency=200)
    
    # Create artistic 3D visualization
    print("\nCreating artistic 3D heat visualization...")
    X, T, U = viz.visualize_heat_equation_3d_artistic(heat_solver)
    
    # Create interactive Plotly visualization
    print("\nCreating interactive 3D visualization...")
    plotly_fig = viz.create_interactive_3d_plotly(X, T, U, "Heat Equation")
    
    # Create particle system (this might take a while)
    print("\nCreating particle system animation...")
    viz.create_particle_system_visualization(heat_solver)
    
    # Van der Pol phase space
    print("\nSetting up Van der Pol oscillator...")
    vdp_solver = VanDerPolODEPINN(t_max=20.0, mu=1.0)
    vdp_solver.train(n_iterations=2000, print_frequency=400)
    
    print("\nCreating Van der Pol 3D phase space...")
    viz.create_van_der_pol_phase_space_3d(vdp_solver)
    
    # Wave equation
    print("\nSetting up Wave Equation...")
    wave_solver = WaveEquationPINN(x_max=1.0, t_max=2.0, c=1.0)
    wave_solver.train(n_iterations=1500, print_frequency=300)
    
    print("\nCreating wave interference animation...")
    viz.create_wave_interference_visualization(wave_solver)
    
    print("\nAll exotic visualizations created!")
    print("\nGenerated files:")
    print("- figures/heat_3d_artistic.png")
    print("- figures/interactive_3d_visualization.html")
    print("- figures/particle_heat.gif")
    print("- figures/van_der_pol_3d.png")
    print("- figures/wave_interference.gif")


if __name__ == "__main__":
    demo_exotic_visualizations()
