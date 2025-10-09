#!/usr/bin/env python3
"""
Generate plots with uncertainties for cavity flow simulation results.
This script creates 2 figures:
1. Lid velocity sensitivity (varying lid velocity vs fixed viscosity)  
2. Viscosity sensitivity (varying viscosity vs fixed lid velocity)

Each figure contains 3 subfigures: ux, uy, and pressure
All plots include uncertainty bands/error bars from the surrogate model predictions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add QUEENS source directory to Python path
queens_src_path = Path('/home/a11evina/queens/src').resolve()
if str(queens_src_path) not in sys.path:
    sys.path.insert(0, str(queens_src_path))

def load_trained_model():
    """Load the trained surrogate model and parameter ranges."""
    model_path = "trained_models/cavity_surrogate_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        print("✅ Trained surrogate model loaded successfully")
        
        surrogate_model = saved_data['surrogate_model']
        param_ranges = saved_data['parameter_ranges']
        
        return surrogate_model, param_ranges
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Make sure you've run the training script first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def extract_velocity_pressure_components(Y_data, probe_idx=0):
    # CORRECT - 4 components per probe: [Ux, Uy, Uz, pressure]
    start_idx = probe_idx * 4
    ux = Y_data[:, start_idx]      
    uy = Y_data[:, start_idx + 1]  
    pressure = Y_data[:, start_idx + 3]  # Skip Uz (index +2), get pressure (+3)
    return ux, uy, pressure

def create_lid_velocity_sensitivity_plots(surrogate_model, param_ranges, probe_idx=0, n_points=50):
    """
    Create plots showing sensitivity to lid velocity while keeping viscosity fixed.
    Returns figure with 3 subplots: ux, uy, pressure vs lid velocity.
    """
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    probe_name = probe_names[probe_idx]
    
    # Fixed viscosity at middle of range
    fixed_viscosity = (param_ranges['min'][1] + param_ranges['max'][1]) / 2
    
    # Varying lid velocity across its range
    lid_velocities = np.linspace(param_ranges['min'][0], param_ranges['max'][0], n_points)
    
    # Create input grid
    X_test = np.column_stack([lid_velocities, np.full(n_points, fixed_viscosity)])
    
    print(f"📊 Creating lid velocity sensitivity plots for {probe_name} probe...")
    print(f"   Fixed viscosity: {fixed_viscosity:.6f} m²/s")
    print(f"   Lid velocity range: {param_ranges['min'][0]:.2f} - {param_ranges['max'][0]:.2f} m/s")
    
    # Get predictions and uncertainties
    pred_dict = surrogate_model.predict(X_test)
    Y_pred = pred_dict['result']
    Y_std = np.sqrt(pred_dict['variance'])
    
    # Extract components
    ux_pred, uy_pred, pressure_pred = extract_velocity_pressure_components(Y_pred, probe_idx)
    ux_std, uy_std, pressure_std = extract_velocity_pressure_components(Y_std, probe_idx)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Ux vs lid velocity
    ax1.plot(lid_velocities, ux_pred, 'b-', linewidth=2, label='Mean prediction')
    ax1.fill_between(lid_velocities, 
                     ux_pred - 1.96*ux_std, 
                     ux_pred + 1.96*ux_std, 
                     alpha=0.3, color='blue', label='95% confidence interval')
    ax1.set_xlabel('Lid Velocity (m/s)')
    ax1.set_ylabel('Ux (m/s)')
    ax1.set_title(f'Ux vs Lid Velocity - {probe_name.replace("_", " ").title()}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Uy vs lid velocity
    ax2.plot(lid_velocities, uy_pred, 'r-', linewidth=2, label='Mean prediction')
    ax2.fill_between(lid_velocities, 
                     uy_pred - 1.96*uy_std, 
                     uy_pred + 1.96*uy_std, 
                     alpha=0.3, color='red', label='95% confidence interval')
    ax2.set_xlabel('Lid Velocity (m/s)')
    ax2.set_ylabel('Uy (m/s)')
    ax2.set_title(f'Uy vs Lid Velocity - {probe_name.replace("_", " ").title()}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Pressure vs lid velocity
    ax3.plot(lid_velocities, pressure_pred, 'g-', linewidth=2, label='Mean prediction')
    ax3.fill_between(lid_velocities, 
                     pressure_pred - 1.96*pressure_std, 
                     pressure_pred + 1.96*pressure_std, 
                     alpha=0.3, color='green', label='95% confidence interval')
    ax3.set_xlabel('Lid Velocity (m/s)')
    ax3.set_ylabel('Pressure (Pa)')
    ax3.set_title(f'Pressure vs Lid Velocity - {probe_name.replace("_", " ").title()}')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(f'Lid Velocity Sensitivity Analysis - {probe_name.replace("_", " ").title()} Probe\n'
                 f'(Fixed Viscosity: {fixed_viscosity:.6f} m²/s)', fontsize=16)
    plt.tight_layout()
    
    return fig

def create_viscosity_sensitivity_plots(surrogate_model, param_ranges, probe_idx=0, n_points=50):
    """
    Create plots showing sensitivity to viscosity while keeping lid velocity fixed.
    Returns figure with 3 subplots: ux, uy, pressure vs viscosity.
    """
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    probe_name = probe_names[probe_idx]
    
    # Fixed lid velocity at middle of range
    fixed_lid_velocity = (param_ranges['min'][0] + param_ranges['max'][0]) / 2
    
    # Varying viscosity across its range
    viscosities = np.linspace(param_ranges['min'][1], param_ranges['max'][1], n_points)
    
    # Create input grid
    X_test = np.column_stack([np.full(n_points, fixed_lid_velocity), viscosities])
    
    print(f"📊 Creating viscosity sensitivity plots for {probe_name} probe...")
    print(f"   Fixed lid velocity: {fixed_lid_velocity:.2f} m/s")
    print(f"   Viscosity range: {param_ranges['min'][1]:.6f} - {param_ranges['max'][1]:.6f} m²/s")
    
    # Get predictions and uncertainties
    pred_dict = surrogate_model.predict(X_test)
    Y_pred = pred_dict['result']
    Y_std = np.sqrt(pred_dict['variance'])
    
    # Extract components
    ux_pred, uy_pred, pressure_pred = extract_velocity_pressure_components(Y_pred, probe_idx)
    ux_std, uy_std, pressure_std = extract_velocity_pressure_components(Y_std, probe_idx)
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Ux vs viscosity
    ax1.plot(viscosities, ux_pred, 'b-', linewidth=2, label='Mean prediction')
    ax1.fill_between(viscosities, 
                     ux_pred - 1.96*ux_std, 
                     ux_pred + 1.96*ux_std, 
                     alpha=0.3, color='blue', label='95% confidence interval')
    ax1.set_xlabel('Viscosity (m²/s)')
    ax1.set_ylabel('Ux (m/s)')
    ax1.set_title(f'Ux vs Viscosity - {probe_name.replace("_", " ").title()}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Uy vs viscosity
    ax2.plot(viscosities, uy_pred, 'r-', linewidth=2, label='Mean prediction')
    ax2.fill_between(viscosities, 
                     uy_pred - 1.96*uy_std, 
                     uy_pred + 1.96*uy_std, 
                     alpha=0.3, color='red', label='95% confidence interval')
    ax2.set_xlabel('Viscosity (m²/s)')
    ax2.set_ylabel('Uy (m/s)')
    ax2.set_title(f'Uy vs Viscosity - {probe_name.replace("_", " ").title()}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Pressure vs viscosity
    ax3.plot(viscosities, pressure_pred, 'g-', linewidth=2, label='Mean prediction')
    ax3.fill_between(viscosities, 
                     pressure_pred - 1.96*pressure_std, 
                     pressure_pred + 1.96*pressure_std, 
                     alpha=0.3, color='green', label='95% confidence interval')
    ax3.set_xlabel('Viscosity (m²/s)')
    ax3.set_ylabel('Pressure (Pa)')
    ax3.set_title(f'Pressure vs Viscosity - {probe_name.replace("_", " ").title()}')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(f'Viscosity Sensitivity Analysis - {probe_name.replace("_", " ").title()} Probe\n'
                 f'(Fixed Lid Velocity: {fixed_lid_velocity:.2f} m/s)', fontsize=16)
    plt.tight_layout()
    
    return fig

def create_all_uncertainty_plots():
    """Create uncertainty plots for all probes and both parameter sensitivities."""
    # Load trained model
    surrogate_model, param_ranges = load_trained_model()
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    
    print("🎯 CREATING UNCERTAINTY PLOTS FOR CAVITY FLOW")
    print("=" * 55)
    
    # Create plots for each probe
    for probe_idx in range(5):
        probe_name = probe_names[probe_idx]
        
        # Lid velocity sensitivity plots
        print(f"\n📊 Creating lid velocity sensitivity plots for {probe_name}...")
        fig_lid = create_lid_velocity_sensitivity_plots(surrogate_model, param_ranges, probe_idx)
        fig_lid.savefig(f"plots/lid_velocity_sensitivity_{probe_name}_probe_{probe_idx}_uncertainty.png", 
                       dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: lid_velocity_sensitivity_{probe_name}_probe_{probe_idx}_uncertainty.png")
        
        # Viscosity sensitivity plots
        print(f"📊 Creating viscosity sensitivity plots for {probe_name}...")
        fig_visc = create_viscosity_sensitivity_plots(surrogate_model, param_ranges, probe_idx)
        fig_visc.savefig(f"plots/viscosity_sensitivity_{probe_name}_probe_{probe_idx}_uncertainty.png", 
                        dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: viscosity_sensitivity_{probe_name}_probe_{probe_idx}_uncertainty.png")
        
        plt.close(fig_lid)
        plt.close(fig_visc)
    
    print(f"\n🎉 All uncertainty plots created!")
    print("Files saved in 'plots/' directory:")
    print("Lid velocity sensitivity plots:")
    for i, name in enumerate(probe_names):
        print(f"- lid_velocity_sensitivity_{name}_probe_{i}_uncertainty.png")
    print("Viscosity sensitivity plots:")
    for i, name in enumerate(probe_names):
        print(f"- viscosity_sensitivity_{name}_probe_{i}_uncertainty.png")

def create_sample_uncertainty_plots_with_synthetic_data():
    """
    Create sample plots with synthetic data to demonstrate the functionality.
    Use this if the trained model is not available.
    """
    print("🔧 Creating sample plots with synthetic data...")
    
    # Create synthetic parameter ranges
    param_ranges = {
        'min': [0.5, 0.005],    # [min_lid_velocity, min_viscosity]
        'max': [2.0, 0.015]     # [max_lid_velocity, max_viscosity]
    }
    
    n_points = 5
    probe_names = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
    
    # Create output directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    for probe_idx in range(1):  # Just create for center probe as example
        probe_name = probe_names[probe_idx]
        
        # Lid velocity sensitivity with synthetic data
        fixed_viscosity = (param_ranges['min'][1] + param_ranges['max'][1]) / 2
        lid_velocities = np.linspace(param_ranges['min'][0], param_ranges['max'][0], n_points)
        
        # Generate synthetic data with realistic behavior
        ux_pred = lid_velocities * 0.8 + 0.1 * np.sin(lid_velocities * 5) + np.random.normal(0, 0.02, n_points)
        uy_pred = -0.2 * lid_velocities + 0.05 * np.cos(lid_velocities * 3) + np.random.normal(0, 0.01, n_points)
        pressure_pred = lid_velocities**2 * 0.5 + np.random.normal(0, 0.1, n_points)
        
        # Generate uncertainties (typically smaller for higher velocities due to more data)
        ux_std = 0.05 + 0.02 / (lid_velocities + 0.5)
        uy_std = 0.03 + 0.01 / (lid_velocities + 0.5)
        pressure_std = 0.1 + 0.05 / (lid_velocities + 0.5)
        
        # Create lid velocity plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.plot(lid_velocities, ux_pred, 'b-', linewidth=2, label='Mean prediction')
        ax1.fill_between(lid_velocities, ux_pred - 1.96*ux_std, ux_pred + 1.96*ux_std, 
                        alpha=0.3, color='blue', label='95% confidence interval')
        ax1.set_xlabel('Lid Velocity (m/s)')
        ax1.set_ylabel('Ux (m/s)')
        ax1.set_title(f'Ux vs Lid Velocity - {probe_name.replace("_", " ").title()}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(lid_velocities, uy_pred, 'r-', linewidth=2, label='Mean prediction')
        ax2.fill_between(lid_velocities, uy_pred - 1.96*uy_std, uy_pred + 1.96*uy_std, 
                        alpha=0.3, color='red', label='95% confidence interval')
        ax2.set_xlabel('Lid Velocity (m/s)')
        ax2.set_ylabel('Uy (m/s)')
        ax2.set_title(f'Uy vs Lid Velocity - {probe_name.replace("_", " ").title()}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.plot(lid_velocities, pressure_pred, 'g-', linewidth=2, label='Mean prediction')
        ax3.fill_between(lid_velocities, pressure_pred - 1.96*pressure_std, pressure_pred + 1.96*pressure_std, 
                        alpha=0.3, color='green', label='95% confidence interval')
        ax3.set_xlabel('Lid Velocity (m/s)')
        ax3.set_ylabel('Pressure (Pa)')
        ax3.set_title(f'Pressure vs Lid Velocity - {probe_name.replace("_", " ").title()}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.suptitle(f'Lid Velocity Sensitivity Analysis - {probe_name.replace("_", " ").title()} Probe\n'
                     f'(Fixed Viscosity: {fixed_viscosity:.6f} m²/s) - SYNTHETIC DATA', fontsize=16)
        plt.tight_layout()
        
        fig.savefig(f"plots/sample_lid_velocity_sensitivity_{probe_name}_uncertainty.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Sample lid velocity plot created: sample_lid_velocity_sensitivity_{probe_name}_uncertainty.png")

if __name__ == "__main__":
    """
    Main execution function.
    
    Usage:
    python cavity_uncertainty_plots.py                    # Create plots with trained model
    python cavity_uncertainty_plots.py --sample          # Create sample plots with synthetic data
    """
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        # Create sample plots with synthetic data
        create_sample_uncertainty_plots_with_synthetic_data()
    else:
        # Create plots with trained model (normal usage)
        try:
            create_all_uncertainty_plots()
        except SystemExit:
            print("\n💡 TIP: If you don't have a trained model, you can run:")
            print("python cavity_uncertainty_plots.py --sample")
            print("to create sample plots with synthetic data.")