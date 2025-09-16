#!/usr/bin/env python3
"""
Comprehensive mesh convergence validation plotting script
Compares OpenFOAM results across multiple mesh resolutions with ACE Numerics benchmark data

Usage: python mesh_convergence_plots.py

Expected directory structure:
mesh_convergence/
├── mesh_64x64/validation_data/
├── mesh_128x128/validation_data/
├── mesh_256x256/validation_data/
├── mesh_512x512/validation_data/
├── mesh_1000x1000/validation_data/
└── mesh_convergence_plots.py  (this file)
"""

import csv
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_ace_benchmark():
    """ACE Numerics benchmark data for Re=1000"""
    
    # ACE VERTICAL BENCHMARK: v-velocity along horizontal line y=0.5 (Table 2)
    # Format: (x_position, v_velocity) - sampling across x-direction at y=0.5
    ace_vertical = [
        [0.0000, 0.0000000], [0.0100, 0.0709203], [0.0200, 0.1306210], [0.0300, 0.1796062],
        [0.0400, 0.2189314], [0.0625, 0.2807056], [0.0703, 0.2962703], [0.0781, 0.3099097],
        [0.0938, 0.3330442], [0.1563, 0.3769189], [0.2266, 0.3339924], [0.2344, 0.3253592],
        [0.3500, 0.1882246], [0.5000, 0.0257995], [0.6800, -0.1730887], [0.8047, -0.3202137],
        [0.8594, -0.4264545], [0.9063, -0.5264392], [0.9453, -0.4103754], [0.9531, -0.3553213],
        [0.9609, -0.2936869], [0.9688, -0.2279225], [0.9700, -0.2178657], [0.9800, -0.1359277],
        [0.9900, -0.0617528], [0.9950, -0.0291290], [1.0000, 0.0000000]
    ]
    
    # ACE HORIZONTAL BENCHMARK: u-velocity along vertical line x=0.5 (Table 4) 
    # Format: (y_position, u_velocity) - sampling across y-direction at x=0.5
    ace_horizontal = [
        [0.0000, 0.0000000], [0.0100, -0.0397486], [0.0200, -0.0759628], [0.0300, -0.1090870],
        [0.0400, -0.1396601], [0.0547, -0.1812881], [0.0625, -0.2023300], [0.0703, -0.2228955],
        [0.1016, -0.3004561], [0.1719, -0.3885690], [0.2813, -0.2803696], [0.4531, -0.1081999],
        [0.5000, -0.0620561], [0.6172, 0.0570178], [0.7344, 0.1886747], [0.8516, 0.3372212],
        [0.9531, 0.4723329], [0.9609, 0.5169277], [0.9688, 0.5808359], [0.9766, 0.6644227],
        [0.9800, 0.7070189], [0.9900, 0.8489396], [1.0000, 1.0000000]
    ]
    
    # Key benchmarks
    ace_extrema = {
        'umin': -0.3885698, 'ymin': 0.1716968,
        'vmax': 0.3769447, 'xmax': 0.1578365,
        'vmin': -0.5270773, 'xmin': 0.9092470
    }
    
    ace_vortex_center = {'x': 0.53079011, 'y': 0.56524055}
    
    return ace_vertical, ace_horizontal, ace_extrema, ace_vortex_center

def read_csv_file(filename):
    """Simple CSV reader without pandas"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for row in reader:
                data.append(row)
        return data, headers
    except Exception as e:
        print(f"⚠️  Warning: Could not read {filename}: {e}")
        return None, None

def interpolate_linear(x_data, y_data, x_target):
    """Simple linear interpolation"""
    if x_target <= x_data[0]:
        return y_data[0]
    if x_target >= x_data[-1]:
        return y_data[-1]
    
    for i in range(len(x_data) - 1):
        if x_data[i] <= x_target <= x_data[i + 1]:
            # Linear interpolation
            t = (x_target - x_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + t * (y_data[i + 1] - y_data[i])
    
    return None

def load_mesh_data(mesh_dir):
    """Load all validation data for a single mesh case"""
    validation_dir = os.path.join(mesh_dir, 'validation_data')
    
    if not os.path.exists(validation_dir):
        print(f"❌ Validation data directory not found: {validation_dir}")
        return None
    
    # Load vertical centerline (x=0.5, varying y) for u-velocity
    vertical_file = os.path.join(validation_dir, 'vertical_centerline_fixed.csv')
    vertical_data, _ = read_csv_file(vertical_file)
    
    # Load horizontal centerline (y=0.5, varying x) for v-velocity  
    horizontal_file = os.path.join(validation_dir, 'horizontal_centerline_fixed.csv')
    horizontal_data, _ = read_csv_file(horizontal_file)
    
    if not vertical_data or not horizontal_data:
        return None
        
    return {
        'vertical': vertical_data,
        'horizontal': horizontal_data,
        'validation_dir': validation_dir
    }

def calculate_rms_errors(mesh_data, ace_vertical, ace_horizontal):
    """Calculate RMS errors for centerline comparisons"""
    
    # Vertical centerline (u-velocity vs y)
    foam_y_vert = []
    foam_u_vert = []
    for row in mesh_data['vertical']:
        y = float(row['Points:1'])  # y-coordinate
        u = float(row['U:0'])       # u-velocity
        foam_y_vert.append(y)
        foam_u_vert.append(u)
    
    # Sort by y-coordinate
    sorted_indices_v = sorted(range(len(foam_y_vert)), key=lambda k: foam_y_vert[k])
    foam_y_vert_sorted = [foam_y_vert[i] for i in sorted_indices_v]
    foam_u_vert_sorted = [foam_u_vert[i] for i in sorted_indices_v]
    
    # Horizontal centerline (v-velocity vs x)
    foam_x_horiz = []
    foam_v_horiz = []
    for row in mesh_data['horizontal']:
        x = float(row['Points:0'])  # x-coordinate
        v = float(row['U:1'])       # v-velocity
        foam_x_horiz.append(x)
        foam_v_horiz.append(v)
    
    # Sort by x-coordinate
    sorted_indices_h = sorted(range(len(foam_x_horiz)), key=lambda k: foam_x_horiz[k])
    foam_x_horiz_sorted = [foam_x_horiz[i] for i in sorted_indices_h]
    foam_v_horiz_sorted = [foam_v_horiz[i] for i in sorted_indices_h]
    
    # Calculate errors for u-velocity (vertical centerline)
    u_errors = []
    for y_bench, u_bench in ace_horizontal:
        u_foam = interpolate_linear(foam_y_vert_sorted, foam_u_vert_sorted, y_bench)
        if u_foam is not None:
            error = abs(u_foam - u_bench)
            u_errors.append(error)
    
    # Calculate errors for v-velocity (horizontal centerline)
    v_errors = []
    for x_bench, v_bench in ace_vertical:
        v_foam = interpolate_linear(foam_x_horiz_sorted, foam_v_horiz_sorted, x_bench)
        if v_foam is not None:
            error = abs(v_foam - v_bench)
            v_errors.append(error)
    
    # Calculate RMS errors
    rms_u = math.sqrt(sum(e**2 for e in u_errors) / len(u_errors)) if u_errors else float('inf')
    rms_v = math.sqrt(sum(e**2 for e in v_errors) / len(v_errors)) if v_errors else float('inf')
    rms_overall = math.sqrt((rms_u**2 + rms_v**2) / 2)
    
    return {
        'rms_u': rms_u,
        'rms_v': rms_v,
        'rms_overall': rms_overall,
        'foam_y_vert': foam_y_vert_sorted,
        'foam_u_vert': foam_u_vert_sorted,
        'foam_x_horiz': foam_x_horiz_sorted,
        'foam_v_horiz': foam_v_horiz_sorted
    }

def plot_centerline_comparisons(all_mesh_data, ace_vertical, ace_horizontal):
    """Create centerline velocity profile comparisons"""
    
    # Check for scipy availability
    try:
        from scipy.interpolate import PchipInterpolator
        HAS_SCIPY = True
        interp_method = "PCHIP"
    except ImportError:
        HAS_SCIPY = False
        interp_method = "Linear"
        print("⚠️  Using linear interpolation for ACE curves (scipy not available)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different mesh resolutions
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_mesh_data)))
    
    # Plot 1: U-velocity along vertical centerline (x=0.5)
    ax1.set_title('U-velocity along Vertical Centerline (x=0.5)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('y-position')
    ax1.set_ylabel('u-velocity')
    ax1.grid(True, alpha=0.3)
    
    # Create smooth ACE benchmark curve
    ace_y = np.array([point[0] for point in ace_horizontal])
    ace_u = np.array([point[1] for point in ace_horizontal])
    # Sort by y-coordinate to ensure monotonicity
    sort_indices = np.argsort(ace_y)
    ace_y_sorted = ace_y[sort_indices]
    ace_u_sorted = ace_u[sort_indices]
    
    # Create dense interpolation points for smooth curve
    y_smooth = np.linspace(ace_y_sorted.min(), ace_y_sorted.max(), 1000)
    
    if HAS_SCIPY:
        # Use PCHIP for truly smooth curves
        u_smooth = PchipInterpolator(ace_y_sorted, ace_u_sorted)(y_smooth)
    else:
        # Fallback to linear interpolation
        u_smooth = np.interp(y_smooth, ace_y_sorted, ace_u_sorted)
    
    # Plot smooth ACE benchmark curve
    ax1.plot(y_smooth, u_smooth, 'k-', linewidth=3, label=f'ACE Benchmark ({interp_method})', zorder=10)
    # Show original ACE points
    ax1.plot(ace_y_sorted, ace_u_sorted, 'ko', markersize=5, alpha=0.8, 
         markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, zorder=11)
    
    # Plot OpenFOAM results for each mesh
    for i, (mesh_name, results) in enumerate(all_mesh_data.items()):
        if results is not None:
            ax1.plot(results['foam_y_vert'], results['foam_u_vert'], 
                    color=colors[i], linewidth=1.5, label=f'OpenFOAM {mesh_name}', alpha=0.8)
    
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 1.1)
    
    # Plot 2: V-velocity along horizontal centerline (y=0.5)
    ax2.set_title('V-velocity along Horizontal Centerline (y=0.5)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x-position')
    ax2.set_ylabel('v-velocity')
    ax2.grid(True, alpha=0.3)
    
    # Create smooth ACE benchmark curve
    ace_x = np.array([point[0] for point in ace_vertical])
    ace_v = np.array([point[1] for point in ace_vertical])
    # Sort by x-coordinate to ensure monotonicity
    sort_indices = np.argsort(ace_x)
    ace_x_sorted = ace_x[sort_indices]
    ace_v_sorted = ace_v[sort_indices]
    
    # Create dense interpolation points for smooth curve
    x_smooth = np.linspace(ace_x_sorted.min(), ace_x_sorted.max(), 1000)
    
    if HAS_SCIPY:
        # Use PCHIP for truly smooth curves
        v_smooth = PchipInterpolator(ace_x_sorted, ace_v_sorted)(x_smooth)
    else:
        # Fallback to linear interpolation
        v_smooth = np.interp(x_smooth, ace_x_sorted, ace_v_sorted)
    
    # Plot smooth ACE benchmark curve
    ax2.plot(x_smooth, v_smooth, 'k-', linewidth=3, label=f'ACE Benchmark ({interp_method})', zorder=10)
    # Show original ACE points
    ax2.plot(ace_x_sorted, ace_v_sorted, 'ko', markersize=5, alpha=0.8, markerfacecolor='white', 
             markeredgecolor='black', markeredgewidth=1.5, zorder=11)
    
    # Plot OpenFOAM results for each mesh
    for i, (mesh_name, results) in enumerate(all_mesh_data.items()):
        if results is not None:
            ax2.plot(results['foam_x_horiz'], results['foam_v_horiz'], 
                    color=colors[i], linewidth=1.5, label=f'OpenFOAM {mesh_name}', alpha=0.8)
    
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.6, 0.4)
    
    plt.tight_layout()
    plt.savefig('plots/centerline_comparisons.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: plots/centerline_comparisons.png")
    return fig

def plot_mesh_convergence(all_mesh_data):
    """Plot RMS error convergence with mesh refinement"""
    
    mesh_sizes = []
    rms_u_errors = []
    rms_v_errors = []
    rms_overall_errors = []
    
    # Extract mesh sizes and errors
    for mesh_name, results in all_mesh_data.items():
        if results is not None:
            # Extract mesh size from name (e.g., "mesh_256x256" -> 256)
            mesh_size = int(mesh_name.split('_')[1].split('x')[0])
            mesh_sizes.append(mesh_size)
            rms_u_errors.append(results['rms_u'])
            rms_v_errors.append(results['rms_v'])
            rms_overall_errors.append(results['rms_overall'])
    
    # Sort by mesh size
    sorted_indices = sorted(range(len(mesh_sizes)), key=lambda k: mesh_sizes[k])
    mesh_sizes = [mesh_sizes[i] for i in sorted_indices]
    rms_u_errors = [rms_u_errors[i] for i in sorted_indices]
    rms_v_errors = [rms_v_errors[i] for i in sorted_indices]
    rms_overall_errors = [rms_overall_errors[i] for i in sorted_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: RMS errors vs mesh size
    ax1.loglog(mesh_sizes, rms_u_errors, 'bo-', linewidth=2, markersize=6, label='U-velocity RMS Error')
    ax1.loglog(mesh_sizes, rms_v_errors, 'ro-', linewidth=2, markersize=6, label='V-velocity RMS Error')
    ax1.loglog(mesh_sizes, rms_overall_errors, 'go-', linewidth=2, markersize=6, label='Overall RMS Error')
    
    # # Add theoretical convergence rates
    # if len(mesh_sizes) >= 2:
    #     # Second-order convergence reference
    #     h = [1/size for size in mesh_sizes]
    #     second_order_ref = [rms_overall_errors[0] * (h[0]/hi)**2 for hi in h]
    #     ax1.loglog(mesh_sizes, second_order_ref, 'k--', alpha=0.5, label='2nd Order Reference')
    
    ax1.set_xlabel('Mesh Size (NxN)')
    ax1.set_ylabel('RMS Error')
    ax1.set_title('Mesh Convergence Study', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Grid convergence index (if enough data)
    if len(mesh_sizes) >= 3:
        ax2.semilogx(mesh_sizes, rms_overall_errors, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Mesh Size (NxN)')
        ax2.set_ylabel('Overall RMS Error')
        ax2.set_title('Grid Convergence Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # # Add convergence rate annotation
        # if len(mesh_sizes) >= 2:
        #     p = math.log(rms_overall_errors[-2]/rms_overall_errors[-1]) / math.log(mesh_sizes[-1]/mesh_sizes[-2])
        #     ax2.text(0.05, 0.95, f'Apparent order of accuracy: {p:.2f}', 
        #             transform=ax2.transAxes, verticalalignment='top',
        #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'Need at least 3 mesh\nresolutions for\nconvergence analysis', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Grid Convergence Analysis (Insufficient Data)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('plots/mesh_convergence.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: plots/mesh_convergence.png")
    return fig

def plot_validation_dashboard(all_mesh_data):
    """Create a comprehensive validation dashboard"""
    
    # Count successful cases
    successful_cases = sum(1 for results in all_mesh_data.values() if results is not None)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('OpenFOAM Cavity Flow Validation Dashboard (Re=1000)', fontsize=16, fontweight='bold')
    
    # Case status summary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.7, f'Mesh Cases Analyzed: {successful_cases}', ha='center', va='center',
             transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.5, f'Total Expected: {len(all_mesh_data)}', ha='center', va='center',
             transform=ax1.transAxes, fontsize=12)
    ax1.text(0.5, 0.3, f'Success Rate: {successful_cases/len(all_mesh_data)*100:.0f}%', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Case Status', fontweight='bold')
    
    # RMS error summary (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    mesh_names = []
    rms_values = []
    colors_status = []
    
    for mesh_name, results in all_mesh_data.items():
        if results is not None:
            mesh_names.append(mesh_name.replace('mesh_', ''))
            rms_values.append(results['rms_overall'])
            # Color coding: green < 0.01, yellow < 0.05, red >= 0.05
            if results['rms_overall'] < 0.01:
                colors_status.append('green')
            elif results['rms_overall'] < 0.05:
                colors_status.append('orange')
            else:
                colors_status.append('red')
    
    if mesh_names:
        bars = ax2.bar(mesh_names, rms_values, color=colors_status, alpha=0.7)
        ax2.set_ylabel('Overall RMS Error')
        ax2.set_title('RMS Error by Mesh', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Excellent (< 0.01)')
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Good (< 0.05)')
        ax2.legend(fontsize=8)
    
    # Mesh size distribution (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if mesh_names:
        mesh_sizes = [int(name.split('x')[0]) for name in mesh_names]
        ax3.bar(mesh_names, mesh_sizes, alpha=0.7, color='skyblue')
        ax3.set_ylabel('Mesh Size (N)')
        ax3.set_title('Mesh Resolutions', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Detailed convergence plot (middle row, full width)
    ax4 = fig.add_subplot(gs[1, :])
    if len([r for r in all_mesh_data.values() if r is not None]) >= 2:
        mesh_sizes_conv = []
        rms_overall_conv = []
        
        for mesh_name, results in all_mesh_data.items():
            if results is not None:
                mesh_size = int(mesh_name.split('_')[1].split('x')[0])
                mesh_sizes_conv.append(mesh_size)
                rms_overall_conv.append(results['rms_overall'])
        
        # Sort by mesh size
        sorted_data = sorted(zip(mesh_sizes_conv, rms_overall_conv))
        mesh_sizes_conv, rms_overall_conv = zip(*sorted_data)
        
        ax4.loglog(mesh_sizes_conv, rms_overall_conv, 'bo-', linewidth=2, markersize=8, label='Overall RMS Error')
        ax4.set_xlabel('Mesh Size (NxN)')
        ax4.set_ylabel('Overall RMS Error')
        ax4.set_title('Mesh Convergence Analysis', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # # Add convergence rate if possible
        # if len(mesh_sizes_conv) >= 2:
        #     p = math.log(rms_overall_conv[-2]/rms_overall_conv[-1]) / math.log(mesh_sizes_conv[-1]/mesh_sizes_conv[-2])
        #     ax4.text(0.05, 0.95, f'Apparent Order of Accuracy: {p:.2f}', 
        #             transform=ax4.transAxes, verticalalignment='top',
        #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Summary statistics table (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    if any(results is not None for results in all_mesh_data.values()):
        table_data = []
        headers = ['Mesh', 'RMS U-Error', 'RMS V-Error', 'Overall RMS', 'Status']
        
        for mesh_name, results in all_mesh_data.items():
            if results is not None:
                status = 'Excellent' if results['rms_overall'] < 0.01 else \
                        'Good' if results['rms_overall'] < 0.05 else 'Needs Improvement'
                table_data.append([
                    mesh_name.replace('mesh_', ''),
                    f"{results['rms_u']:.6f}",
                    f"{results['rms_v']:.6f}",
                    f"{results['rms_overall']:.6f}",
                    status
                ])
        
        # Create table
        table = ax5.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         bbox=[0, 0.2, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Color code the status column
        for i in range(1, len(table_data) + 1):
            status = table_data[i-1][4]
            if status == 'Excellent':
                table[(i, 4)].set_facecolor('lightgreen')
            elif status == 'Good':
                table[(i, 4)].set_facecolor('lightyellow')
            else:
                table[(i, 4)].set_facecolor('lightcoral')
    
    plt.savefig('plots/validation_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: plots/validation_dashboard.png")
    return fig

def main():
    """Main function to create all validation plots"""
    
    print("🎯 OpenFOAM Mesh Convergence Validation Analysis")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load ACE benchmark data
    ace_vertical, ace_horizontal, ace_extrema, ace_vortex = load_ace_benchmark()
    print("✅ Loaded ACE Numerics benchmark data")
    
    # Define expected mesh cases
    mesh_cases = ['mesh_64x64', 'mesh_128x128', 'mesh_256x256', 'mesh_512x512', 'mesh_1000x1000']
    
    # Load all mesh data
    all_mesh_data = {}
    print(f"\n📊 Loading mesh data...")
    
    for mesh_case in mesh_cases:
        print(f"   Loading {mesh_case}...", end=' ')
        mesh_data = load_mesh_data(mesh_case)
        
        if mesh_data is not None:
            # Calculate RMS errors
            results = calculate_rms_errors(mesh_data, ace_vertical, ace_horizontal)
            all_mesh_data[mesh_case] = results
            print(f"✅ RMS: {results['rms_overall']:.6f}")
        else:
            all_mesh_data[mesh_case] = None
            print("❌ No data found")
    
    successful_cases = sum(1 for results in all_mesh_data.values() if results is not None)
    print(f"\n📈 Successfully loaded {successful_cases}/{len(mesh_cases)} mesh cases")
    
    if successful_cases == 0:
        print("❌ No validation data found. Please check directory structure.")
        return
    
    print("\n🎨 Creating validation plots...")
    
    # Create all plots
    try:
        plot_centerline_comparisons(all_mesh_data, ace_vertical, ace_horizontal)
        plot_mesh_convergence(all_mesh_data)
        plot_validation_dashboard(all_mesh_data)
        
        print("\n🎉 All plots created successfully!")
        print("📁 Check the 'plots/' directory for output files:")
        print("   • centerline_comparisons.png - Velocity profile comparisons")
        print("   • mesh_convergence.png - Grid convergence analysis") 
        print("   • validation_dashboard.png - Comprehensive summary")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()