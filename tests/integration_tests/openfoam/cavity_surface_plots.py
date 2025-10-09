#!/usr/bin/env python3
"""
    Create 3D surface plots for cavity surrogate model:
    1. lid_velocity vs viscosity vs Ux
    2. lid_velocity vs viscosity vs Uy  
    3. lid_velocity vs viscosity vs pressure

    Create 2D viscosity sensitivity plots
    Create 2D lid velocity sensitivity plots
    Debug training data relationships
    Compare training vs predicted data in 2D
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Ensure QUEENS source directory is on sys.path (needed for unpickling model)
queens_src_path = Path('/home/a11evina/queens/src').resolve()
if str(queens_src_path) not in sys.path:
    sys.path.insert(0, str(queens_src_path))

def load_trained_model(model_path="trained_models/cavity_surrogate_model.pkl"):
    """Load the trained surrogate model."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['surrogate_model'], model_data['parameter_ranges']
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

def load_training_data(data_path="surrogate_data_output/cavity_flow_surrogate_surrogate_data.pkl"):
    """Load training data for comparison."""
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['X_train'], data['Y_train']
    except FileNotFoundError:
        print(f"❌ Training data file not found: {data_path}")
        return None, None

def extract_velocity_pressure_components(Y_pred, probe_idx=0):
    """
    Extract Ux, Uy, pressure from probe data.
    Assumes Y_pred has shape (n_points, n_probes*4) where each probe has [Ux, Uy, Uz, p]
    """
    # Based on your probe structure: [Ux, Uy, Uz, p] for each probe
    start_idx = probe_idx * 4
    
    ux = Y_pred[:, start_idx]      # Ux component
    uy = Y_pred[:, start_idx + 1]  # Uy component  
    pressure = Y_pred[:, start_idx + 3]  # Pressure component
    
    return ux, uy, pressure

def create_training_vs_predicted_plot_2d(surrogate_model, param_ranges, probe_idx=0):
    """Create 2D scatter plot comparing training data vs predicted data (like your image)."""
    
    # Load training data
    X_train, Y_train = load_training_data()
    if X_train is None:
        print("❌ Cannot create comparison plot - training data not available")
        return None
    
    # Extract training components for selected probe
    ux_train, uy_train, pressure_train = extract_velocity_pressure_components(Y_train, probe_idx)
    
    # Create dense prediction grid for smooth visualization
    resolution = 50
    vel_vals = np.linspace(param_ranges['min'][0], param_ranges['max'][0], resolution)
    visc_vals = np.linspace(param_ranges['min'][1], param_ranges['max'][1], resolution)
    VEL, VISC = np.meshgrid(vel_vals, visc_vals)
    X_grid = np.column_stack([VEL.ravel(), VISC.ravel()])
    
    # Get surrogate predictions
    pred_dict = surrogate_model.predict(X_grid)
    Y_pred = pred_dict['result']
    ux_pred, uy_pred, pressure_pred = extract_velocity_pressure_components(Y_pred, probe_idx)
    
    # Create three 2D comparison plots
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    probe_name = probe_names[probe_idx]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Ux vs viscosity
    # Predicted data (colored by lid velocity, small points)
    scatter1_pred = ax1.scatter(X_grid[:, 1], ux_pred, c=X_grid[:, 0], 
                               cmap='viridis', alpha=0.4, s=8, label='Predicted')
    # Training data (distinct diamond markers with black edges)
    scatter1_train = ax1.scatter(X_train[:, 1], ux_train, c=X_train[:, 0], 
                                cmap='viridis', s=80, marker='D', 
                                edgecolors='black', linewidth=1, label='Training')
    ax1.set_xlabel('Viscosity (m²/s)')
    ax1.set_ylabel('Ux (m/s)')
    ax1.set_title(f'Ux vs viscosity - {probe_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1_pred, ax=ax1, label='Lid Velocity')
    
    # Plot 2: Uy vs viscosity
    scatter2_pred = ax2.scatter(X_grid[:, 1], uy_pred, c=X_grid[:, 0], 
                               cmap='plasma', alpha=0.4, s=8, label='Predicted')
    scatter2_train = ax2.scatter(X_train[:, 1], uy_train, c=X_train[:, 0], 
                                cmap='plasma', s=80, marker='s', 
                                edgecolors='black', linewidth=1, label='Training')
    ax2.set_xlabel('Viscosity (m²/s)')
    ax2.set_ylabel('Uy (m/s)')
    ax2.set_title(f'Uy vs viscosity - {probe_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2_pred, ax=ax2, label='Lid Velocity')
    
    # Plot 3: Pressure vs viscosity
    scatter3_pred = ax3.scatter(X_grid[:, 1], pressure_pred, c=X_grid[:, 0], 
                               cmap='coolwarm', alpha=0.4, s=8, label='Predicted')
    scatter3_train = ax3.scatter(X_train[:, 1], pressure_train, c=X_train[:, 0], 
                                cmap='coolwarm', s=80, marker='^', 
                                edgecolors='black', linewidth=1, label='Training')
    ax3.set_xlabel('Viscosity (m²/s)')
    ax3.set_ylabel('Output Pressure (Pa)')
    ax3.set_title(f'Pressure vs viscosity - {probe_name}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3_pred, ax=ax3, label='Lid Velocity')
    
    plt.suptitle(f'Training vs Predicted Data - {probe_name.replace("_", " ").title()} Probe', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    fig.savefig(f"plots/training_vs_predicted_2d_{probe_name}_probe_{probe_idx}.png", 
               dpi=300, bbox_inches='tight')
    print(f"✅ 2D training vs predicted plot saved for {probe_name}")
    
    return fig

def create_3d_surface_plots(surrogate_model, param_ranges, probe_idx=0, resolution=30):
    """Create 3 surface plots: Ux, Uy, and pressure."""
    
    # Create parameter grid
    vel_vals = np.linspace(param_ranges['min'][0], param_ranges['max'][0], resolution)
    visc_vals = np.linspace(param_ranges['min'][1], param_ranges['max'][1], resolution)
    VEL, VISC = np.meshgrid(vel_vals, visc_vals)
    
    # Flatten for prediction
    X_grid = np.column_stack([VEL.ravel(), VISC.ravel()])
    
    # Get surrogate predictions
    print(f"Making {len(X_grid)} predictions...")
    pred_dict = surrogate_model.predict(X_grid)
    Y_pred = pred_dict['result']
    
    # Extract velocity components and pressure
    ux, uy, pressure = extract_velocity_pressure_components(Y_pred, probe_idx)
    
    # Reshape back to grid
    UX = ux.reshape(VEL.shape)
    UY = uy.reshape(VEL.shape)
    P = pressure.reshape(VEL.shape)
    
    # Create the 3 surface plots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Ux surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(VEL, VISC, UX, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Lid Velocity')
    ax1.set_ylabel('Viscosity (m²/s)')
    ax1.set_zlabel('Ux (m/s)')
    ax1.set_title(f'Ux Component - Probe {probe_idx}')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot 2: Uy surface  
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(VEL, VISC, UY, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Lid Velocity')
    ax2.set_ylabel('Viscosity (m²/s)')
    ax2.set_zlabel('Uy (m/s)')
    ax2.set_title(f'Uy Component - Probe {probe_idx}')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Plot 3: Pressure surface
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(VEL, VISC, P, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('Lid Velocity')
    ax3.set_ylabel('Viscosity (m²/s)')
    ax3.set_zlabel('Pressure (Pa)')
    ax3.set_title(f'Pressure - Probe {probe_idx}')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    return fig

def create_individual_surface_plots(surrogate_model, param_ranges, probe_idx=0, resolution=40):
    """Create 3 separate high-quality surface plots."""
    
    # Create parameter grid
    vel_vals = np.linspace(param_ranges['min'][0], param_ranges['max'][0], resolution)
    visc_vals = np.linspace(param_ranges['min'][1], param_ranges['max'][1], resolution)
    VEL, VISC = np.meshgrid(vel_vals, visc_vals)
    
    X_grid = np.column_stack([VEL.ravel(), VISC.ravel()])
    
    # Get predictions
    pred_dict = surrogate_model.predict(X_grid)
    Y_pred = pred_dict['result']
    
    # Extract components
    ux, uy, pressure = extract_velocity_pressure_components(Y_pred, probe_idx)
    
    UX = ux.reshape(VEL.shape)
    UY = uy.reshape(VEL.shape)
    P = pressure.reshape(VEL.shape)
    
    figures = []
    
    # Individual Ux plot
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(VEL, VISC, UX, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Lid Velocity (m/s)', fontsize=12)
    ax1.set_ylabel('Viscosity (m²/s)', fontsize=12)
    ax1.set_zlabel('Ux Velocity (m/s)', fontsize=12)
    ax1.set_title(f'X-Velocity Component - Probe {probe_idx}', fontsize=14, pad=20)
    fig1.colorbar(surf1, ax=ax1, shrink=0.6, label='Ux (m/s)')
    figures.append(fig1)
    
    # Individual Uy plot
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(VEL, VISC, UY, cmap='plasma', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('Lid Velocity (m/s)', fontsize=12)
    ax2.set_ylabel('Viscosity (m²/s)', fontsize=12)
    ax2.set_zlabel('Uy Velocity (m/s)', fontsize=12)
    ax2.set_title(f'Y-Velocity Component - Probe {probe_idx}', fontsize=14, pad=20)
    fig2.colorbar(surf2, ax=ax2, shrink=0.6, label='Uy (m/s)')
    figures.append(fig2)
    
    # Individual Pressure plot
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf3 = ax3.plot_surface(VEL, VISC, P, cmap='coolwarm', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('Lid Velocity (m/s)', fontsize=12)
    ax3.set_ylabel('Viscosity (m²/s)', fontsize=12)
    ax3.set_zlabel('Pressure (Pa)', fontsize=12)
    ax3.set_title(f'Pressure Field - Probe {probe_idx}', fontsize=14, pad=20)
    fig3.colorbar(surf3, ax=ax3, shrink=0.6, label='Pressure (Pa)')
    figures.append(fig3)
    
    return figures

def plot_viscosity_sensitivity_2d(surrogate_model, param_ranges):
    """Create 2D plots showing viscosity sensitivity for each probe and variable."""
    
    # Fixed lid velocity at middle of range
    fixed_lid_vel = (param_ranges['min'][0] + param_ranges['max'][0]) / 2
    
    # Viscosity range
    visc_vals = np.linspace(param_ranges['min'][1], param_ranges['max'][1], 100)
    X_viscosity_sweep = np.column_stack([np.full(len(visc_vals), fixed_lid_vel), visc_vals])
    
    # Get predictions
    pred_dict = surrogate_model.predict(X_viscosity_sweep)
    Y_pred = pred_dict['result']
    
    # Probe names and locations
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    
    # Create plots for each probe
    for probe_idx in range(5):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract components for this probe
        ux, uy, pressure_out = extract_velocity_pressure_components(Y_pred, probe_idx)
        
        # Ux vs viscosity
        ax1.plot(visc_vals, ux, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Viscosity (m²/s)')
        ax1.set_ylabel('Ux (m/s)')
        ax1.set_title(f'Ux vs Viscosity - {probe_names[probe_idx]}')
        ax1.grid(True, alpha=0.3)
        
        # Uy vs viscosity  
        ax2.plot(visc_vals, uy, 'r-', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Viscosity (m²/s)')
        ax2.set_ylabel('Uy (m/s)')
        ax2.set_title(f'Uy vs Viscosity - {probe_names[probe_idx]}')
        ax2.grid(True, alpha=0.3)
        
        # Pressure vs viscosity
        ax3.plot(visc_vals, pressure_out, 'g-', linewidth=2, marker='o', markersize=3)
        ax3.set_xlabel('Viscosity (m²/s)')
        ax3.set_ylabel('Output Pressure (Pa)')
        ax3.set_title(f'Pressure vs Viscosity - {probe_names[probe_idx]}')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Viscosity Sensitivity - {probe_names[probe_idx].replace("_", " ").title()} Probe (lid_vel={fixed_lid_vel:.2f})')
        plt.tight_layout()
        
        fig.savefig(f"plots/viscosity_sensitivity_{probe_names[probe_idx]}_probe_{probe_idx}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Viscosity sensitivity plot saved for {probe_names[probe_idx]}")
    
    print("✅ All viscosity sensitivity plots created!")

def plot_lid_velocity_sensitivity_2d(surrogate_model, param_ranges):
   """Create 2D plots showing lid velocity sensitivity for each probe and variable."""
   
   # Fixed viscosity at middle of range
   fixed_viscosity = (param_ranges['min'][1] + param_ranges['max'][1]) / 2
   
   # Lid velocity range
   velocity_vals = np.linspace(param_ranges['min'][0], param_ranges['max'][0], 100)
   X_velocity_sweep = np.column_stack([velocity_vals, np.full(len(velocity_vals), fixed_viscosity)])
   
   # Get predictions
   pred_dict = surrogate_model.predict(X_velocity_sweep)
   Y_pred = pred_dict['result']
   
   # Probe names
   probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
   
   # Create plots for each probe
   for probe_idx in range(5):
       fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
       
       # Extract components for this probe
       ux, uy, pressure_out = extract_velocity_pressure_components(Y_pred, probe_idx)
       
       # Ux vs lid velocity
       ax1.plot(velocity_vals, ux, 'b-', linewidth=2, marker='o', markersize=3)
       ax1.set_xlabel('Lid Velocity')
       ax1.set_ylabel('Ux (m/s)')
       ax1.set_title(f'Ux vs Lid Velocity - {probe_names[probe_idx]}')
       ax1.grid(True, alpha=0.3)
       
       # Uy vs lid velocity
       ax2.plot(velocity_vals, uy, 'r-', linewidth=2, marker='o', markersize=3)
       ax2.set_xlabel('Lid Velocity')
       ax2.set_ylabel('Uy (m/s)')
       ax2.set_title(f'Uy vs Lid Velocity - {probe_names[probe_idx]}')
       ax2.grid(True, alpha=0.3)
       
       # Pressure vs lid velocity
       ax3.plot(velocity_vals, pressure_out, 'g-', linewidth=2, marker='o', markersize=3)
       ax3.set_xlabel('Lid Velocity')
       ax3.set_ylabel('Output Pressure (Pa)')
       ax3.set_title(f'Pressure vs Lid Velocity - {probe_names[probe_idx]}')
       ax3.grid(True, alpha=0.3)
       
       plt.suptitle(f'Lid Velocity Sensitivity - {probe_names[probe_idx].replace("_", " ").title()} Probe (viscosity={fixed_viscosity:.3f})')
       plt.tight_layout()
       
       fig.savefig(f"plots/lid_velocity_sensitivity_{probe_names[probe_idx]}_probe_{probe_idx}.png", 
                  dpi=300, bbox_inches='tight')
       print(f"✅ Lid velocity sensitivity plot saved for {probe_names[probe_idx]}")
   
   print("✅ All lid velocity sensitivity plots created!")

def debug_training_data():
    """Check if training data shows viscosity-Ux relationship."""
    try:
        with open("surrogate_data_output/cavity_flow_surrogate_surrogate_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        X_train = data['X_train']  # [lid_vel, viscosity]
        Y_train = data['Y_train']  # probe outputs
        
        # Plot Ux vs viscosity colored by lid velocity for center probe (probe 0)
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_train[:, 1], Y_train[:, 0], c=X_train[:, 0], cmap='viridis', alpha=0.7)
        ax.set_xlabel('Viscosity (m²/s)')
        ax.set_ylabel('Ux at Center Probe (training data)')
        ax.set_title('Training Data: Ux vs Viscosity (colored by lid velocity)')
        plt.colorbar(scatter, label='Lid Velocity')
        
        fig.savefig("plots/debug_training_data_ux_viscosity.png", dpi=300, bbox_inches='tight')
        print("✅ Debug plot saved: debug_training_data_ux_viscosity.png")
        
        # Print some statistics
        print(f"\nTraining data statistics:")
        print(f"Lid velocity range: {X_train[:, 0].min():.3f} to {X_train[:, 0].max():.3f}")
        print(f"Viscosity range: {X_train[:, 1].min():.3f} to {X_train[:, 1].max():.3f}")
        print(f"Ux range at center: {Y_train[:, 0].min():.6f} to {Y_train[:, 0].max():.6f}")
        
    except FileNotFoundError:
        print("❌ Training data file not found")

def create_all_training_vs_predicted_plots(surrogate_model, param_ranges):
    """Create training vs predicted comparison plots for all 5 probes."""
    print("🔬 CREATING TRAINING VS PREDICTED COMPARISON PLOTS...")
    print("=" * 55)
    
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    
    for probe_idx in range(5):
        print(f"📈 Creating comparison plot for {probe_names[probe_idx]} probe...")
        fig = create_training_vs_predicted_plot_2d(surrogate_model, param_ranges, probe_idx)
        if fig is not None:
            # Don't call plt.show() in headless environment
            pass  
    
    print("🎉 All training vs predicted plots completed!")

def create_3d_surface_with_training_points(surrogate_model, param_ranges, probe_idx=0, resolution=40):
    """
    Create 3D surface plots showing both the surrogate model predictions (as smooth surfaces) 
    and the training data points overlaid on top, similar to the provided image.
    """
    # Load training data
    X_train, Y_train = load_training_data()
    if X_train is None:
        print("❌ Cannot create surface+training plot - training data not available")
        return None
    
    # Extract training components for selected probe
    ux_train, uy_train, pressure_train = extract_velocity_pressure_components(Y_train, probe_idx)
    
    # Create parameter grid for smooth surface
    vel_vals = np.linspace(param_ranges['min'][0], param_ranges['max'][0], resolution)
    visc_vals = np.linspace(param_ranges['min'][1], param_ranges['max'][1], resolution)
    VEL, VISC = np.meshgrid(vel_vals, visc_vals)
    
    # Flatten for prediction
    X_grid = np.column_stack([VEL.ravel(), VISC.ravel()])
    
    # Get surrogate predictions
    pred_dict = surrogate_model.predict(X_grid)
    Y_pred = pred_dict['result']
    
    # Extract velocity components and pressure
    ux_pred, uy_pred, pressure_pred = extract_velocity_pressure_components(Y_pred, probe_idx)
    
    # Reshape back to grid for surface plots
    UX_SURF = ux_pred.reshape(VEL.shape)
    UY_SURF = uy_pred.reshape(VEL.shape)
    P_SURF = pressure_pred.reshape(VEL.shape)
    
    # Probe names
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    probe_name = probe_names[probe_idx]
    
    # Create the figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Ux surface + training points
    ax1 = fig.add_subplot(131, projection='3d')
    # Surface plot (semi-transparent)
    surf1 = ax1.plot_surface(VEL, VISC, UX_SURF, cmap='viridis', alpha=0.7, 
                            linewidth=0, antialiased=True)
    # Training points (distinct markers)
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], ux_train, 
                          c='red', s=60, marker='o', edgecolors='black', 
                          linewidth=1, label='Training Data', alpha=0.9)
    ax1.set_xlabel('Lid Velocity (m/s)', fontsize=10)
    ax1.set_ylabel('Viscosity (m²/s)', fontsize=10)
    ax1.set_zlabel('Ux (m/s)', fontsize=10)
    ax1.set_title(f'Ux: Training vs Predicted\n{probe_name}', fontsize=12)
    ax1.legend()
    
    # Plot 2: Uy surface + training points
    ax2 = fig.add_subplot(132, projection='3d')
    # Surface plot (semi-transparent)
    surf2 = ax2.plot_surface(VEL, VISC, UY_SURF, cmap='plasma', alpha=0.7,
                            linewidth=0, antialiased=True)
    # Training points (distinct markers)
    scatter2 = ax2.scatter(X_train[:, 0], X_train[:, 1], uy_train,
                          c='blue', s=60, marker='s', edgecolors='black',
                          linewidth=1, label='Training Data', alpha=0.9)
    ax2.set_xlabel('Lid Velocity (m/s)', fontsize=10)
    ax2.set_ylabel('Viscosity (m²/s)', fontsize=10)
    ax2.set_zlabel('Uy (m/s)', fontsize=10)
    ax2.set_title(f'Uy: Training vs Predicted\n{probe_name}', fontsize=12)
    ax2.legend()
    
    # Plot 3: Pressure surface + training points
    ax3 = fig.add_subplot(133, projection='3d')
    # Surface plot (semi-transparent)
    surf3 = ax3.plot_surface(VEL, VISC, P_SURF, cmap='coolwarm', alpha=0.7,
                            linewidth=0, antialiased=True)
    # Training points (distinct markers)
    scatter3 = ax3.scatter(X_train[:, 0], X_train[:, 1], pressure_train,
                          c='black', s=60, marker='^', edgecolors='white',
                          linewidth=1, label='Training Data', alpha=0.9)
    ax3.set_xlabel('Lid Velocity (m/s)', fontsize=10)
    ax3.set_ylabel('Viscosity (m²/s)', fontsize=10)
    ax3.set_zlabel('Pressure (Pa)', fontsize=10)
    ax3.set_title(f'Pressure: Training vs Predicted\n{probe_name}', fontsize=12)
    ax3.legend()
    
    # # Adjust layout and add overall title
    # plt.suptitle(f'3D Surface + Training Data - {probe_name.replace("_", " ").title()} Probe', 
    #              fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig

def create_individual_3d_surfaces_with_training_points(surrogate_model, param_ranges, probe_idx=0, resolution=50):
    """
    Create 3 separate high-quality 3D surface plots with training points overlaid,
    similar to the provided image style.
    """
    # Load training data
    X_train, Y_train = load_training_data()
    if X_train is None:
        print("❌ Cannot create surface+training plot - training data not available")
        return []
    
    # Extract training components
    ux_train, uy_train, pressure_train = extract_velocity_pressure_components(Y_train, probe_idx)
    
    # Create parameter grid
    vel_vals = np.linspace(param_ranges['min'][0], param_ranges['max'][0], resolution)
    visc_vals = np.linspace(param_ranges['min'][1], param_ranges['max'][1], resolution)
    VEL, VISC = np.meshgrid(vel_vals, visc_vals)
    
    X_grid = np.column_stack([VEL.ravel(), VISC.ravel()])
    
    # Get predictions
    pred_dict = surrogate_model.predict(X_grid)
    Y_pred = pred_dict['result']
    
    # Extract components
    ux_pred, uy_pred, pressure_pred = extract_velocity_pressure_components(Y_pred, probe_idx)
    
    UX_SURF = ux_pred.reshape(VEL.shape)
    UY_SURF = uy_pred.reshape(VEL.shape)
    P_SURF = pressure_pred.reshape(VEL.shape)
    
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    probe_name = probe_names[probe_idx]
    
    figures = []
    
    # Individual Ux plot with training points
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Surface (semi-transparent)
    surf1 = ax1.plot_surface(VEL, VISC, UX_SURF, cmap='viridis', alpha=0.8, 
                            linewidth=0, antialiased=True, edgecolor='none')
    
    # Training points overlaid
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], ux_train, 
                          c='red', s=80, marker='o', edgecolors='black', 
                          linewidth=1.5, label='Training Data', alpha=1.0, zorder=10)
    
    ax1.set_xlabel('Lid Velocity (m/s)', fontsize=14, labelpad=10)
    ax1.set_ylabel('Viscosity (m²/s)', fontsize=14, labelpad=10)
    ax1.set_zlabel('Ux Velocity (m/s)', fontsize=14, labelpad=10)
    ax1.set_title(f'X-Velocity: Training vs Predicted\n{probe_name.replace("_", " ").title()} Probe', 
                  fontsize=16, pad=20)
    
    # Colorbar and legend
    cbar1 = fig1.colorbar(surf1, ax=ax1, shrink=0.6, label='Ux (m/s)', pad=0.1)
    ax1.legend(loc='upper left')
    
    # Improve viewing angle
    ax1.view_init(elev=20, azim=45)
    figures.append(fig1)
    
    # Individual Uy plot with training points  
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    surf2 = ax2.plot_surface(VEL, VISC, UY_SURF, cmap='plasma', alpha=0.8,
                            linewidth=0, antialiased=True, edgecolor='none')
    
    scatter2 = ax2.scatter(X_train[:, 0], X_train[:, 1], uy_train,
                          c='blue', s=80, marker='s', edgecolors='black',
                          linewidth=1.5, label='Training Data', alpha=1.0, zorder=10)
    
    ax2.set_xlabel('Lid Velocity (m/s)', fontsize=14, labelpad=10)
    ax2.set_ylabel('Viscosity (m²/s)', fontsize=14, labelpad=10)
    ax2.set_zlabel('Uy Velocity (m/s)', fontsize=14, labelpad=10)
    ax2.set_title(f'Y-Velocity: Training vs Predicted\n{probe_name.replace("_", " ").title()} Probe', 
                  fontsize=16, pad=20)
    
    cbar2 = fig2.colorbar(surf2, ax=ax2, shrink=0.6, label='Uy (m/s)', pad=0.1)
    ax2.legend(loc='upper left')
    ax2.view_init(elev=20, azim=45)
    figures.append(fig2)
    
    # Individual Pressure plot with training points
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    surf3 = ax3.plot_surface(VEL, VISC, P_SURF, cmap='coolwarm', alpha=0.8,
                            linewidth=0, antialiased=True, edgecolor='none')
    
    scatter3 = ax3.scatter(X_train[:, 0], X_train[:, 1], pressure_train,
                          c='black', s=80, marker='^', edgecolors='white',
                          linewidth=1.5, label='Training Data', alpha=1.0, zorder=10)
    
    ax3.set_xlabel('Lid Velocity (m/s)', fontsize=14, labelpad=10)
    ax3.set_ylabel('Viscosity (m²/s)', fontsize=14, labelpad=10)
    ax3.set_zlabel('Pressure (Pa)', fontsize=14, labelpad=10)
    ax3.set_title(f'Pressure: Training vs Predicted\n{probe_name.replace("_", " ").title()} Probe', 
                  fontsize=16, pad=20)
    
    cbar3 = fig3.colorbar(surf3, ax=ax3, shrink=0.6, label='Pressure (Pa)', pad=0.1)
    ax3.legend(loc='upper left')
    ax3.view_init(elev=20, azim=45)
    figures.append(fig3)
    
    return figures

def create_all_3d_surfaces_with_training_data(surrogate_model, param_ranges):
    """Create 3D surface plots with training data overlay for all 5 probes."""
    print("🎯 CREATING 3D SURFACES WITH TRAINING DATA OVERLAY...")
    print("=" * 55)
    
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    
    # Create combined plots for all 5 probes
    for probe_idx in range(5):
        probe_name = probe_names[probe_idx]
        print(f"📊 Creating 3D surface+training plot for probe {probe_idx} ({probe_name})...")
        
        # Combined plot (3 subplots in one figure)
        fig_combined = create_3d_surface_with_training_points(
            surrogate_model, param_ranges, probe_idx
        )
        if fig_combined:
            fig_combined.savefig(f"plots/cavity_surface_training_{probe_name}_probe_{probe_idx}.png", 
                                dpi=300, bbox_inches='tight')
            print(f"   ✅ Combined 3D surface+training plot saved for {probe_name}")
        
        # Individual high-quality plots  
        fig_individual = create_individual_3d_surfaces_with_training_points(
            surrogate_model, param_ranges, probe_idx
        )
        
        if fig_individual:
            variable_names = ['ux', 'uy', 'pressure']
            for i, fig in enumerate(fig_individual):
                fig.savefig(f"plots/individual_surface_training_{variable_names[i]}_{probe_name}_probe_{probe_idx}.png", 
                           dpi=300, bbox_inches='tight')
                print(f"   ✅ Individual {variable_names[i]} surface+training plot saved for {probe_name}")
    
    print("🎉 All 3D surface plots with training data completed!")




def main():
    """Main function to create surface plots for all 5 probes."""
    print("🎯 CREATING 3D SURFACE PLOTS FOR ALL 5 PROBES")
    print("=" * 55)
    
    # Load trained model
    surrogate_model, param_ranges = load_trained_model()
    print("✅ Surrogate model loaded")
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Probe names and locations
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    probe_coords = [
        '(0.05, 0.05, 0.005)',  # center
        '(0.01, 0.01, 0.005)',  # bottom_left
        '(0.09, 0.01, 0.005)',  # bottom_right
        '(0.01, 0.09, 0.005)',  # top_left
        '(0.09, 0.09, 0.005)',  # top_right
    ]
    
    # Create combined plots for all 5 probes
    for probe_idx in range(5):
        probe_name = probe_names[probe_idx]
        probe_coord = probe_coords[probe_idx]
        
        print(f"📊 Creating plots for probe {probe_idx} ({probe_name}) at {probe_coord}...")
        
        fig_combined = create_3d_surface_plots(surrogate_model, param_ranges, probe_idx)
        fig_combined.suptitle(f'Cavity Flow - {probe_name.replace("_", " ").title()} Probe', fontsize=16)
        
        fig_combined.savefig(f"plots/cavity_surfaces_{probe_name}_probe_{probe_idx}.png", 
                            dpi=300, bbox_inches='tight')
        print(f"   ✅ {probe_name} surface plots saved")
    
    print(f"\n🎉 All surface plots created for 5 probes!")
    print("Files saved in 'plots/' directory:")
    for i, name in enumerate(probe_names):
        print(f"- cavity_surfaces_{name}_probe_{i}.png")
    
    # Don't show plots in headless environment
    # plt.show()

if __name__ == "__main__":
    import os
    
    # Create main 3D surface plots for all 5 probes
    main()
    
    # Create 2D viscosity sensitivity plots
    print("\n🔍 CREATING 2D VISCOSITY SENSITIVITY PLOTS...")
    surrogate_model, param_ranges = load_trained_model()
    plot_viscosity_sensitivity_2d(surrogate_model, param_ranges)

    # Create 2D lid velocity sensitivity plots
    print("\n🔍 CREATING 2D LID VELOCITY SENSITIVITY PLOTS...")
    surrogate_model, param_ranges = load_trained_model()
    plot_lid_velocity_sensitivity_2d(surrogate_model, param_ranges)
    
    # Add debug analysis
    print("\n🔍 DEBUGGING TRAINING DATA RELATIONSHIPS...")
    debug_training_data()
    
    # NEW: Create training vs predicted comparison plots
    print("\n🔬 CREATING TRAINING VS PREDICTED COMPARISON...")
    create_all_training_vs_predicted_plots(surrogate_model, param_ranges)

    # Create 3D surfaces with training data overlay
    print("\n🎯 CREATING 3D SURFACES WITH TRAINING DATA OVERLAY...")
    surrogate_model, param_ranges = load_trained_model()
    create_all_3d_surfaces_with_training_data(surrogate_model, param_ranges)