"""
Advanced 3D Coordinate System Visualizations for PINN Data
==========================================================

This module provides cutting-edge 3D visualization techniques including:
- Volumetric rendering
- Isosurface extraction
- Vector field visualizations
- Morphing geometries
- Holographic-style projections

Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import Tuple, List, Optional

# Import PINN classes
from pde_solver import HeatEquationPINN, WaveEquationPINN, BurgersEquationPINN


class Advanced3DVisualizer:
    """Advanced 3D visualization techniques for PINN solutions."""
    
    def __init__(self):
        self.golden_ratio = 1.618033988749
        
    def create_volumetric_heat_rendering(self, solver: HeatEquationPINN, 
                                       save_path: str = "figures/volumetric_heat.html"):
        """
        Create volumetric rendering of heat equation solution using Plotly.
        """
        # Generate high-resolution 3D data
        x = np.linspace(0, solver.x_max, 50)
        t = np.linspace(0, solver.t_max, 50)
        z = np.linspace(0, 1, 30)  # Artificial third dimension for volume
        
        X, T, Z = np.meshgrid(x, t, z)
        
        # Get PINN solution for x-t plane
        xt_flat = np.column_stack([X[:,:,0].flatten(), T[:,:,0].flatten()])
        xt_tensor = solver.to_tensor(xt_flat)
        
        with torch.no_grad():
            u_pred = solver.model(xt_tensor).cpu().numpy()
        U_2d = u_pred.reshape(X[:,:,0].shape)
        
        # Extend to 3D volume with decay in z-direction
        U_3d = np.zeros_like(X)
        for k in range(len(z)):
            decay_factor = np.exp(-2 * z[k])  # Exponential decay in z
            U_3d[:,:,k] = U_2d * decay_factor
        
        # Create volumetric plot
        fig = go.Figure()
        
        # Add volume trace
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=T.flatten(), 
            z=Z.flatten(),
            value=U_3d.flatten(),
            isomin=0.01,
            isomax=U_3d.max(),
            opacity=0.1,
            surface_count=15,
            colorscale='Viridis',
            name='Heat Volume'
        ))
        
        # Add some isosurfaces for key temperature levels
        for iso_val in [0.2, 0.5, 0.8]:
            if iso_val <= U_3d.max():
                fig.add_trace(go.Isosurface(
                    x=X.flatten(),
                    y=T.flatten(),
                    z=Z.flatten(),
                    value=U_3d.flatten(),
                    isomin=iso_val,
                    isomax=iso_val,
                    opacity=0.6,
                    surface_fill=0.7,
                    colorscale='Hot',
                    name=f'Isosurface T={iso_val:.1f}'
                ))
        
        fig.update_layout(
            title='Volumetric Heat Distribution',
            scene=dict(
                xaxis_title='Space (x)',
                yaxis_title='Time (t)',
                zaxis_title='Depth (z)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        fig.write_html(save_path)
        print(f"Volumetric rendering saved to {save_path}")
        return fig
    
    def create_holographic_projection(self, solver: HeatEquationPINN,
                                    save_path: str = "figures/holographic_heat.png"):
        """
        Create holographic-style projection with multiple viewing angles.
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
        
        # Create figure with multiple projections
        fig = plt.figure(figsize=(20, 15))
        
        # Central 3D plot
        ax_main = fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        # Main surface with transparency and lighting effects
        surf = ax_main.plot_surface(X, T, U, cmap='plasma', alpha=0.8, 
                                   linewidth=0, antialiased=True)
        
        # Add wireframe for structure
        ax_main.plot_wireframe(X, T, U, color='white', alpha=0.3, linewidth=0.5)
        
        # Holographic projections on coordinate planes
        # XY projection (bottom)
        ax_main.contourf(X, T, U, zdir='z', offset=U.min()-0.2, 
                        cmap='plasma', alpha=0.7, levels=20)
        
        # XZ projection (side)
        # Create averaged profile for projection
        U_avg_t = np.mean(U, axis=0)
        X_proj = X[0, :]
        Z_proj = np.linspace(U.min()-0.2, U.max()+0.2, len(U_avg_t))
        X_proj_mesh, Z_proj_mesh = np.meshgrid(X_proj, Z_proj)
        U_proj_xz = np.outer(Z_proj, U_avg_t) / np.max(U_avg_t)
        
        ax_main.contourf(X_proj_mesh, solver.t_max + 0.2, U_proj_xz, 
                        zdir='y', cmap='plasma', alpha=0.6, levels=15)
        
        # YZ projection (front)
        U_avg_x = np.mean(U, axis=1)
        T_proj = T[:, 0]
        Z_proj_yz = np.linspace(U.min()-0.2, U.max()+0.2, len(U_avg_x))
        T_proj_mesh, Z_proj_mesh = np.meshgrid(T_proj, Z_proj_yz)
        U_proj_yz = np.outer(Z_proj_yz, U_avg_x) / np.max(U_avg_x)
        
        ax_main.contourf(-0.2, T_proj_mesh, U_proj_yz, 
                        zdir='x', cmap='plasma', alpha=0.6, levels=15)
        
        ax_main.set_xlabel('Space (x)')
        ax_main.set_ylabel('Time (t)')
        ax_main.set_zlabel('Temperature (u)')
        ax_main.set_title('Holographic Heat Distribution', fontsize=16, fontweight='bold')
        
        # Side projection views
        ax_xy = fig.add_subplot(2, 3, 2)
        im1 = ax_xy.imshow(U, extent=[0, solver.x_max, 0, solver.t_max], 
                          cmap='plasma', aspect='auto', origin='lower')
        ax_xy.set_title('XY Projection (Top View)')
        ax_xy.set_xlabel('Space (x)')
        ax_xy.set_ylabel('Time (t)')
        plt.colorbar(im1, ax=ax_xy)
        
        # XZ projection
        ax_xz = fig.add_subplot(2, 3, 3)
        U_xz = np.outer(np.linspace(0, 1, 50), U_avg_t)
        im2 = ax_xz.imshow(U_xz, extent=[0, solver.x_max, U.min(), U.max()], 
                          cmap='plasma', aspect='auto', origin='lower')
        ax_xz.set_title('XZ Projection (Side View)')
        ax_xz.set_xlabel('Space (x)')
        ax_xz.set_ylabel('Temperature (u)')
        plt.colorbar(im2, ax=ax_xz)
        
        # YZ projection
        ax_yz = fig.add_subplot(2, 3, 5)
        U_yz = np.outer(np.linspace(0, 1, 50), U_avg_x)
        im3 = ax_yz.imshow(U_yz, extent=[0, solver.t_max, U.min(), U.max()], 
                          cmap='plasma', aspect='auto', origin='lower')
        ax_yz.set_title('YZ Projection (Front View)')
        ax_yz.set_xlabel('Time (t)')
        ax_yz.set_ylabel('Temperature (u)')
        plt.colorbar(im3, ax=ax_yz)
        
        # 3D trajectory visualization
        ax_traj = fig.add_subplot(2, 3, 6, projection='3d')
        
        # Create temperature gradient flow lines
        n_lines = 10
        for i in range(n_lines):
            x_start = i * solver.x_max / n_lines
            x_idx = int(i * len(x) / n_lines)
            
            # Create trajectory following temperature gradient
            traj_x = [x_start]
            traj_t = [0]
            traj_u = [U[0, x_idx]]
            
            for j in range(1, len(t)):
                # Simple gradient following
                if x_idx > 0 and x_idx < len(x)-1:
                    grad_x = (U[j, x_idx+1] - U[j, x_idx-1]) / (2 * (x[1] - x[0]))
                    new_x = traj_x[-1] + 0.01 * grad_x
                    new_x = np.clip(new_x, 0, solver.x_max)
                    new_x_idx = int(new_x * len(x) / solver.x_max)
                    new_x_idx = np.clip(new_x_idx, 0, len(x)-1)
                    
                    traj_x.append(new_x)
                    traj_t.append(t[j])
                    traj_u.append(U[j, new_x_idx])
            
            # Plot trajectory
            colors = plt.cm.viridis(np.array(traj_u) / np.max(U))
            for k in range(len(traj_x)-1):
                ax_traj.plot([traj_x[k], traj_x[k+1]], 
                           [traj_t[k], traj_t[k+1]],
                           [traj_u[k], traj_u[k+1]], 
                           color=colors[k], linewidth=2, alpha=0.8)
        
        ax_traj.set_xlabel('Space (x)')
        ax_traj.set_ylabel('Time (t)')
        ax_traj.set_zlabel('Temperature (u)')
        ax_traj.set_title('Temperature Flow Trajectories')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Holographic projection saved to {save_path}")
    
    def create_morphing_geometry_animation(self, solver: WaveEquationPINN,
                                         save_path: str = "figures/morphing_wave.gif"):
        """
        Create morphing geometry animation where the wave solution deforms a 3D object.
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
        
        # Create base geometry (torus)
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, 2*np.pi, 20)
        THETA, PHI = np.meshgrid(theta, phi)
        
        R = 2  # Major radius
        r = 0.5  # Minor radius
        
        # Base torus coordinates
        X_torus = (R + r * np.cos(PHI)) * np.cos(THETA)
        Y_torus = (R + r * np.cos(PHI)) * np.sin(THETA)
        Z_torus = r * np.sin(PHI)
        
        def animate_morphing(frame):
            ax.clear()
            
            # Current time
            t_idx = frame % len(t)
            current_wave = U[t_idx, :]
            
            # Interpolate wave to match torus resolution
            wave_interp = np.interp(np.linspace(0, 1, len(THETA[0])), 
                                   np.linspace(0, 1, len(current_wave)), 
                                   current_wave)
            
            # Deform torus based on wave
            deformation = 1 + 0.5 * np.outer(np.ones(len(phi)), wave_interp)
            
            X_deformed = X_torus * deformation
            Y_deformed = Y_torus * deformation
            Z_deformed = Z_torus + 0.3 * np.outer(wave_interp[::-1], np.ones(len(theta)))
            
            # Color based on deformation
            colors = plt.cm.viridis(deformation / deformation.max())
            
            # Plot morphed surface
            surf = ax.plot_surface(X_deformed, Y_deformed, Z_deformed, 
                                  facecolors=colors, alpha=0.8, 
                                  linewidth=0, antialiased=True)
            
            # Add wireframe
            ax.plot_wireframe(X_deformed, Y_deformed, Z_deformed, 
                            color='white', alpha=0.3, linewidth=0.5)
            
            # Add wave profile as a separate curve
            wave_x = np.linspace(-3, 3, len(current_wave))
            wave_y = np.zeros_like(wave_x)
            wave_z = current_wave * 2
            
            ax.plot(wave_x, wave_y, wave_z, 'red', linewidth=4, alpha=0.9, label='Wave Profile')
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_zlim(-2, 2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Morphing Torus - Wave Time: {t[t_idx]:.3f}')
            ax.legend()
            
            # Rotate view
            ax.view_init(elev=20, azim=frame*2)
        
        # Create animation
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        anim = FuncAnimation(fig, animate_morphing, frames=len(t), 
                           interval=150, repeat=True)
        
        # Save animation
        writer = PillowWriter(fps=8)
        anim.save(save_path, writer=writer)
        plt.close()
        
        print(f"Morphing geometry animation saved to {save_path}")
    
    def create_vector_field_3d(self, solver: BurgersEquationPINN,
                              save_path: str = "figures/burgers_vector_field.html"):
        """
        Create 3D vector field visualization for Burgers equation.
        """
        # Generate solution data
        x = np.linspace(0, solver.x_max, 30)
        t = np.linspace(0, solver.t_max, 30)
        X, T = np.meshgrid(x, t)
        
        # Get PINN predictions
        xt_flat = np.column_stack([X.flatten(), T.flatten()])
        xt_tensor = solver.to_tensor(xt_flat)
        
        with torch.no_grad():
            u_pred = solver.model(xt_tensor).cpu().numpy()
        U = u_pred.reshape(X.shape)
        
        # Compute gradients for vector field
        du_dx, du_dt = np.gradient(U)
        
        # Create artificial Z dimension
        Z = np.zeros_like(X)
        
        # Vector components
        Vx = du_dx  # Spatial gradient
        Vy = du_dt  # Temporal gradient  
        Vz = U      # Solution magnitude
        
        # Normalize vectors
        magnitude = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        magnitude[magnitude == 0] = 1  # Avoid division by zero
        
        Vx_norm = Vx / magnitude
        Vy_norm = Vy / magnitude
        Vz_norm = Vz / magnitude
        
        # Create Plotly 3D vector field
        fig = go.Figure()
        
        # Sample points for vectors (reduce density for clarity)
        step = 3
        X_sample = X[::step, ::step]
        T_sample = T[::step, ::step]
        Z_sample = Z[::step, ::step]
        Vx_sample = Vx_norm[::step, ::step]
        Vy_sample = Vy_norm[::step, ::step]
        Vz_sample = Vz_norm[::step, ::step]
        magnitude_sample = magnitude[::step, ::step]
        
        # Add surface
        fig.add_trace(go.Surface(
            x=X, y=T, z=U,
            colorscale='Viridis',
            opacity=0.7,
            name='Burgers Solution'
        ))
        
        # Add vector field using cones
        fig.add_trace(go.Cone(
            x=X_sample.flatten(),
            y=T_sample.flatten(),
            z=U[::step, ::step].flatten(),
            u=Vx_sample.flatten(),
            v=Vy_sample.flatten(),
            w=Vz_sample.flatten(),
            sizemode="absolute",
            sizeref=0.1,
            colorscale='Hot',
            showscale=False,
            name='Gradient Vectors'
        ))
        
        # Add streamlines
        # Create seed points for streamlines
        seed_x = np.linspace(0, solver.x_max, 5)
        seed_t = np.full(5, 0.1)
        
        for i, (sx, st) in enumerate(zip(seed_x, seed_t)):
            # Simple streamline integration
            stream_x = [sx]
            stream_t = [st]
            stream_u = [np.interp(sx, x, U[1, :])]
            
            for step in range(20):
                # Get current gradient
                curr_x_idx = np.argmin(np.abs(x - stream_x[-1]))
                curr_t_idx = np.argmin(np.abs(t - stream_t[-1]))
                
                if curr_x_idx < len(x)-1 and curr_t_idx < len(t)-1:
                    dx = du_dx[curr_t_idx, curr_x_idx] * 0.01
                    dt = du_dt[curr_t_idx, curr_x_idx] * 0.01
                    
                    new_x = stream_x[-1] + dx
                    new_t = stream_t[-1] + dt
                    
                    if 0 <= new_x <= solver.x_max and 0 <= new_t <= solver.t_max:
                        new_u = np.interp(new_x, x, U[min(curr_t_idx+1, len(t)-1), :])
                        stream_x.append(new_x)
                        stream_t.append(new_t)
                        stream_u.append(new_u)
                    else:
                        break
                else:
                    break
            
            # Add streamline
            fig.add_trace(go.Scatter3d(
                x=stream_x,
                y=stream_t,
                z=stream_u,
                mode='lines+markers',
                line=dict(color=f'rgb({255-i*40}, {100+i*30}, {i*50})', width=4),
                marker=dict(size=3),
                name=f'Streamline {i+1}',
                showlegend=(i==0)
            ))
        
        fig.update_layout(
            title='Burgers Equation - 3D Vector Field Analysis',
            scene=dict(
                xaxis_title='Space (x)',
                yaxis_title='Time (t)',
                zaxis_title='Solution (u)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        fig.write_html(save_path)
        print(f"3D vector field saved to {save_path}")
        return fig
    
    def create_crystalline_structure_viz(self, solver: HeatEquationPINN,
                                       save_path: str = "figures/crystalline_heat.png"):
        """
        Create crystalline/lattice structure visualization of the solution.
        """
        # Generate solution data
        x = np.linspace(0, solver.x_max, 40)
        t = np.linspace(0, solver.t_max, 40)
        X, T = np.meshgrid(x, t)
        
        # Get PINN predictions
        xt_flat = np.column_stack([X.flatten(), T.flatten()])
        xt_tensor = solver.to_tensor(xt_flat)
        
        with torch.no_grad():
            u_pred = solver.model(xt_tensor).cpu().numpy()
        U = u_pred.reshape(X.shape)
        
        fig = plt.figure(figsize=(16, 12))
        
        # Main crystalline plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Create lattice points
        for i in range(0, len(x), 2):
            for j in range(0, len(t), 2):
                # Base point
                xi, tj, uj = X[j, i], T[j, i], U[j, i]
                
                # Create crystalline structure around each point
                size = 0.02 + 0.05 * abs(uj) / np.max(np.abs(U))
                color = plt.cm.viridis(uj / np.max(U))
                
                # Draw crystal as octahedron
                vertices = np.array([
                    [xi, tj, uj + size],     # top
                    [xi, tj, uj - size],     # bottom
                    [xi + size, tj, uj],     # right
                    [xi - size, tj, uj],     # left
                    [xi, tj + size, uj],     # front
                    [xi, tj - size, uj]      # back
                ])
                
                # Connect vertices to form octahedron
                faces = [
                    [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],  # top pyramid
                    [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]   # bottom pyramid
                ]
                
                for face in faces:
                    triangle = vertices[face]
                    ax1.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                                   color=color, alpha=0.7, shade=True)
                
                # Add connecting bonds to neighbors
                if i < len(x) - 2:
                    xi_next = X[j, i+2]
                    uj_next = U[j, i+2]
                    ax1.plot([xi, xi_next], [tj, tj], [uj, uj_next], 
                           'white', alpha=0.5, linewidth=1)
                
                if j < len(t) - 2:
                    tj_next = T[j+2, i]
                    uj_next = U[j+2, i]
                    ax1.plot([xi, xi], [tj, tj_next], [uj, uj_next], 
                           'white', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Space (x)')
        ax1.set_ylabel('Time (t)')
        ax1.set_zlabel('Temperature (u)')
        ax1.set_title('Crystalline Lattice Structure')
        
        # Molecular-style bonds visualization
        ax2 = fig.add_subplot(222, projection='3d')
        
        # Sample points for molecular visualization
        n_molecules = 100
        indices = np.random.choice(len(xt_flat), n_molecules, replace=False)
        
        for idx in indices:
            xi, ti = xt_flat[idx]
            ui = u_pred[idx]
            
            # Draw atom
            color = plt.cm.plasma(ui / np.max(u_pred))
            size = 50 + 200 * abs(ui) / np.max(np.abs(u_pred))
            
            ax2.scatter([xi], [ti], [ui], c=[color], s=size, alpha=0.8)
            
            # Find nearby atoms for bonds
            distances = cdist([[xi, ti]], xt_flat[:, :2])[0]
            nearby_indices = np.where((distances < 0.1) & (distances > 0))[0]
            
            for near_idx in nearby_indices[:3]:  # Limit bonds per atom
                xi_near, ti_near = xt_flat[near_idx]
                ui_near = u_pred[near_idx]
                
                # Draw bond
                bond_strength = 1 - distances[near_idx] / 0.1
                ax2.plot([xi, xi_near], [ti, ti_near], [ui, ui_near],
                        'silver', alpha=bond_strength, linewidth=bond_strength*3)
        
        ax2.set_xlabel('Space (x)')
        ax2.set_ylabel('Time (t)')
        ax2.set_zlabel('Temperature (u)')
        ax2.set_title('Molecular Bond Network')
        
        # Topological landscape
        ax3 = fig.add_subplot(223, projection='3d')
        
        # Create landscape with peaks and valleys
        landscape = U.copy()
        
        # Add noise for interesting topology
        noise = 0.05 * np.random.randn(*landscape.shape)
        landscape += noise
        
        # Smooth the landscape
        from scipy.ndimage import gaussian_filter
        landscape = gaussian_filter(landscape, sigma=1)
        
        # Plot as contour landscape
        levels = np.linspace(landscape.min(), landscape.max(), 20)
        for level in levels[::2]:
            contour_set = ax3.contour(X, T, landscape, levels=[level], 
                                    colors=[plt.cm.terrain(level/landscape.max())],
                                    alpha=0.7)
        
        # Add surface
        surf = ax3.plot_surface(X, T, landscape, cmap='terrain', alpha=0.6)
        
        ax3.set_xlabel('Space (x)')
        ax3.set_ylabel('Time (t)')
        ax3.set_zlabel('Temperature (u)')
        ax3.set_title('Topological Landscape')
        
        # Fractal-inspired visualization
        ax4 = fig.add_subplot(224, projection='3d')
        
        # Create fractal-like structure based on solution
        def fractal_branch(x_start, t_start, u_start, depth, direction):
            if depth <= 0:
                return
            
            # Branch parameters
            length = 0.1 / (depth + 1)
            angle_x = direction[0] + 0.3 * (np.random.rand() - 0.5)
            angle_t = direction[1] + 0.3 * (np.random.rand() - 0.5)
            angle_u = u_start * 0.5
            
            # End point
            x_end = x_start + length * np.cos(angle_x)
            t_end = t_start + length * np.sin(angle_t)
            u_end = u_start + angle_u * 0.1
            
            # Color based on temperature
            color = plt.cm.hot(abs(u_start) / np.max(np.abs(U)))
            
            # Draw branch
            ax4.plot([x_start, x_end], [t_start, t_end], [u_start, u_end],
                    color=color, linewidth=3-depth, alpha=0.8)
            
            # Recursive branches
            for _ in range(2):
                new_direction = [angle_x + np.random.randn()*0.5, 
                               angle_t + np.random.randn()*0.5]
                fractal_branch(x_end, t_end, u_end, depth-1, new_direction)
        
        # Start fractal trees from high-temperature points
        high_temp_indices = np.where(U > 0.7 * np.max(U))
        
        for i in range(min(10, len(high_temp_indices[0]))):
            t_idx, x_idx = high_temp_indices[0][i], high_temp_indices[1][i]
            x_start, t_start, u_start = X[t_idx, x_idx], T[t_idx, x_idx], U[t_idx, x_idx]
            
            # Initial direction
            if x_idx > 0 and x_idx < len(x)-1:
                grad_x = U[t_idx, x_idx+1] - U[t_idx, x_idx-1]
                grad_t = U[min(t_idx+1, len(t)-1), x_idx] - U[max(t_idx-1, 0), x_idx]
                direction = [grad_x, grad_t]
            else:
                direction = [1, 1]
            
            fractal_branch(x_start, t_start, u_start, 3, direction)
        
        ax4.set_xlabel('Space (x)')
        ax4.set_ylabel('Time (t)')
        ax4.set_zlabel('Temperature (u)')
        ax4.set_title('Fractal Temperature Trees')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Crystalline structure visualization saved to {save_path}")


def demo_advanced_3d():
    """Demonstrate advanced 3D visualization techniques."""
    print("Creating Advanced 3D PINN Visualizations...")
    
    # Initialize visualizer
    viz = Advanced3DVisualizer()
    
    # Create and train solvers
    print("\nSetting up Heat Equation PINN...")
    heat_solver = HeatEquationPINN(x_max=1.0, t_max=0.5, alpha=0.1)
    heat_solver.train(n_iterations=1000, print_frequency=500)
    
    print("\nSetting up Wave Equation PINN...")
    wave_solver = WaveEquationPINN(x_max=1.0, t_max=2.0, c=1.0)
    wave_solver.train(n_iterations=1000, print_frequency=500)
    
    print("\nSetting up Burgers Equation PINN...")
    burgers_solver = BurgersEquationPINN(x_max=1.0, t_max=0.5, nu=0.01)
    burgers_solver.train(n_iterations=1500, print_frequency=500)
    
    # Create visualizations
    print("\nCreating volumetric rendering...")
    viz.create_volumetric_heat_rendering(heat_solver)
    
    print("\nCreating holographic projection...")
    viz.create_holographic_projection(heat_solver)
    
    print("\nCreating morphing geometry animation...")
    viz.create_morphing_geometry_animation(wave_solver)
    
    print("\nCreating 3D vector field...")
    viz.create_vector_field_3d(burgers_solver)
    
    print("\nCreating crystalline structure...")
    viz.create_crystalline_structure_viz(heat_solver)
    
    print("\nAll advanced 3D visualizations created!")
    print("\nGenerated files:")
    print("- figures/volumetric_heat.html")
    print("- figures/holographic_heat.png")
    print("- figures/morphing_wave.gif")
    print("- figures/burgers_vector_field.html")
    print("- figures/crystalline_heat.png")


if __name__ == "__main__":
    demo_advanced_3d()
