#!/usr/bin/env python3
"""
OpenFOAM Benchmark Validation Plotting Script
Compares OpenFOAM cavity flow results with ACE Numerics benchmark data for multiple Reynolds numbers

Usage: python benchmark_validation_plots.py

Expected directory structure:
benchmark_validation/
├── re_1000_mesh_256x256/validation_data/
├── re_2500_mesh_256x256/validation_data/
└── benchmark_validation_plots.py  (this file)

Validation data files (CSV format):
- vertical_centerline_fixed.csv: U-velocity along vertical centerline (x=0.5)
- horizontal_centerline_fixed.csv: V-velocity along horizontal centerline (y=0.5)
"""

import csv
import os
import sys
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Check for scipy availability for smooth interpolation
try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
    INTERP_METHOD = "PCHIP"
except ImportError:
    HAS_SCIPY = False
    INTERP_METHOD = "Linear"

def load_ace_benchmark_re1000():
    """Load ACE Numerics benchmark data for Re=1000"""
    
    # Vertical velocity (v) along horizontal line y=0.5 (Table 2)
    ace_vertical = np.array([
        [0.0000, 0.0000000], [0.0100, 0.0709203], [0.0200, 0.1306210], [0.0300, 0.1796062],
        [0.0400, 0.2189314], [0.0625, 0.2807056], [0.0703, 0.2962703], [0.0781, 0.3099097],
        [0.0938, 0.3330442], [0.1563, 0.3769189], [0.2266, 0.3339924], [0.2344, 0.3253592],
        [0.3500, 0.1882246], [0.5000, 0.0257995], [0.6800, -0.1730887], [0.8047, -0.3202137],
        [0.8594, -0.4264545], [0.9063, -0.5264392], [0.9453, -0.4103754], [0.9531, -0.3553213],
        [0.9609, -0.2936869], [0.9688, -0.2279225], [0.9700, -0.2178657], [0.9800, -0.1359277],
        [0.9900, -0.0617528], [0.9950, -0.0291290], [1.0000, 0.0000000]
    ])
    
    # Horizontal velocity (u) along vertical line x=0.5 (Table 4)
    ace_horizontal = np.array([
        [0.0000, 0.0000000], [0.0100, -0.0397486], [0.0200, -0.0759628], [0.0300, -0.1090870],
        [0.0400, -0.1396601], [0.0547, -0.1812881], [0.0625, -0.2023300], [0.0703, -0.2228955],
        [0.1016, -0.3004561], [0.1719, -0.3885690], [0.2813, -0.2803696], [0.4531, -0.1081999],
        [0.5000, -0.0620561], [0.6172, 0.0570178], [0.7344, 0.1886747], [0.8516, 0.3372212],
        [0.9531, 0.4723329], [0.9609, 0.5169277], [0.9688, 0.5808359], [0.9766, 0.6644227],
        [0.9800, 0.7070189], [0.9900, 0.8489396], [1.0000, 1.0000000]
    ])
    
    # Velocity extrema (Table 6)
    ace_extrema = {
        'umin': -0.3885698, 'ymin': 0.1716968,
        'vmax': 0.3769447, 'xmax': 0.1578365,
        'vmin': -0.5270773, 'xmin': 0.9092470
    }
    
    # Primary vortex center (Table 1)
    ace_vortex = {'x': 0.53079011, 'y': 0.56524055, 'strength': -0.118936603}
    
    return ace_vertical, ace_horizontal, ace_extrema, ace_vortex

def load_ace_benchmark_re2500():
    """Load ACE Numerics benchmark data for Re=2500"""
    
    # Vertical velocity (v) along horizontal line y=0.5
    ace_vertical = np.array([
        [0.0000, 0.0000000], [0.0100, 0.1143948], [0.0200, 0.2023102], [0.0300, 0.2647949],
        [0.0400, 0.3083658], [0.0625, 0.3725036], [0.0703, 0.3886347], [0.0781, 0.4018733],
        [0.0938, 0.4190397], [0.1563, 0.3855220], [0.2266, 0.3013197], [0.2344, 0.2927453],
        [0.3500, 0.1699611], [0.5000, 0.0159682], [0.6800, -0.1723257], [0.8047, -0.3155607],
        [0.8594, -0.3790584], [0.9063, -0.4633788], [0.9453, -0.5594828], [0.9531, -0.5328728],
        [0.9609, -0.4760871], [0.9688, -0.3899442], [0.9700, -0.3748432], [0.9800, -0.2379860],
        [0.9900, -0.1039195], [0.9950, -0.0469896], [1.0000, 0.0000000]
    ])
    
    # Horizontal velocity (u) along vertical line x=0.5
    ace_horizontal = np.array([
        [0.0000, 0.0000000], [0.0100, -0.0842063], [0.0200, -0.1527752], [0.0300, -0.2085355],
        [0.0400, -0.2563321], [0.0547, -0.3188175], [0.0625, -0.3483073], [0.0703, -0.3742030],
        [0.1016, -0.4274977], [0.1719, -0.3545135], [0.2813, -0.2472328], [0.4531, -0.0849425],
        [0.5000, -0.0403805], [0.6172, 0.0743445], [0.7344, 0.1998386], [0.8516, 0.3481987],
        [0.9531, 0.4543291], [0.9609, 0.4644843], [0.9688, 0.4923424], [0.9766, 0.5532739],
        [0.9800, 0.5943851], [0.9900, 0.7717888], [1.0000, 1.0000000]
    ])
    
    # Velocity extrema
    ace_extrema = {
        'umin': -0.4281529, 'ymin': 0.1057214,
        'vmax': 0.4236136, 'xmax': 0.1075504,
        'vmin': -0.5626456, 'xmin': 0.9412528
    }
    
    # Primary vortex center
    ace_vortex = {'x': 0.5197769, 'y': 0.5439244, 'strength': -0.1214689}
    
    return ace_vertical, ace_horizontal, ace_extrema, ace_vortex

def load_reynolds_data(re_case):
    """Load validation data for a specific Reynolds number case"""
    
    validation_dir = Path(re_case) / "validation_data"
    
    if not validation_dir.exists():
        print(f"⚠️  Validation data directory not found: {validation_dir}")
        return None
    
    data = {}
    
    try:
        # Load centerline u-velocity (vertical line at x=0.5)
        u_file = validation_dir / "vertical_centerline_fixed.csv"
        if u_file.exists():
            with open(u_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                u_data = [[float(row[0]), float(row[1])] for row in reader]
                data['foam_y_vert'] = np.array([point[0] for point in u_data])
                data['foam_u_vert'] = np.array([point[1] for point in u_data])
        
        # Load centerline v-velocity (horizontal line at y=0.5)
        v_file = validation_dir / "horizontal_centerline_fixed.csv"
        if v_file.exists():
            with open(v_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                v_data = [[float(row[0]), float(row[1])] for row in reader]
                data['foam_x_horiz'] = np.array([point[0] for point in v_data])
                data['foam_v_horiz'] = np.array([point[1] for point in v_data])
        
        # Load extrema data (optional - not required)
        # extrema_file = validation_dir / "extrema.csv"
        # if extrema_file.exists():
        #     with open(extrema_file, 'r') as f:
        #         reader = csv.reader(f)
        #         next(reader)  # Skip header
        #         for row in reader:
        #             if len(row) >= 3:
        #                 data['foam_umin'] = float(row[0])
        #                 data['foam_vmax'] = float(row[1])
        #                 data['foam_vmin'] = float(row[2])
        
        return data if data else None
        
    except Exception as e:
        print(f"❌ Error loading data for {re_case}: {e}")
        return None

def calculate_rms_errors(foam_data, ace_vertical, ace_horizontal):
    """Calculate RMS errors between OpenFOAM and ACE benchmark data"""
    
    errors = {}
    
    # U-velocity RMS error (vertical centerline)
    if 'foam_u_vert' in foam_data:
        # Interpolate ACE data to OpenFOAM y-coordinates
        ace_y = ace_horizontal[:, 0]
        ace_u = ace_horizontal[:, 1]
        ace_u_interp = np.interp(foam_data['foam_y_vert'], ace_y, ace_u)
        
        u_diff = foam_data['foam_u_vert'] - ace_u_interp
        errors['rms_u'] = np.sqrt(np.mean(u_diff**2))
    else:
        errors['rms_u'] = float('inf')
    
    # V-velocity RMS error (horizontal centerline)
    if 'foam_v_horiz' in foam_data:
        # Interpolate ACE data to OpenFOAM x-coordinates
        ace_x = ace_vertical[:, 0]
        ace_v = ace_vertical[:, 1]
        ace_v_interp = np.interp(foam_data['foam_x_horiz'], ace_x, ace_v)
        
        v_diff = foam_data['foam_v_horiz'] - ace_v_interp
        errors['rms_v'] = np.sqrt(np.mean(v_diff**2))
    else:
        errors['rms_v'] = float('inf')
    
    # Overall RMS error
    if errors['rms_u'] != float('inf') and errors['rms_v'] != float('inf'):
        errors['rms_overall'] = np.sqrt((errors['rms_u']**2 + errors['rms_v']**2) / 2)
    else:
        errors['rms_overall'] = float('inf')
    
    return errors

def plot_reynolds_validation(all_re_data, ace_data_dict):
    """Create comprehensive Reynolds number validation plots"""
    
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    
    # Color scheme for different Reynolds numbers
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    re_numbers = sorted([int(re.split('_')[1]) for re in all_re_data.keys()])
    
    # Plot 1: U-velocity profiles comparison
    ax1 = plt.subplot(2, 3, 1)
    
    for i, re in enumerate(re_numbers):
        re_case = f"Re_{re}"
        if re_case in all_re_data and all_re_data[re_case] is not None:
            foam_data = all_re_data[re_case]
            if 'foam_u_vert' in foam_data:
                ax1.plot(foam_data['foam_u_vert'], foam_data['foam_y_vert'], 
                        color=colors[i], linewidth=2, label=f'OpenFOAM Re={re}', alpha=0.8)
        
        # Plot corresponding ACE benchmark
        if re in ace_data_dict:
            ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re]
            ace_y = ace_horizontal[:, 0]
            ace_u = ace_horizontal[:, 1]
            
            # Create smooth interpolation for ACE data
            y_smooth = np.linspace(ace_y.min(), ace_y.max(), 500)
            if HAS_SCIPY:
                u_smooth = PchipInterpolator(ace_y, ace_u)(y_smooth)
            else:
                u_smooth = np.interp(y_smooth, ace_y, ace_u)
            
            ax1.plot(u_smooth, y_smooth, color=colors[i], linestyle='--', linewidth=2.5, 
                    label=f'ACE Re={re}', alpha=0.9)
            ax1.plot(ace_u, ace_y, 'o', color=colors[i], markersize=4, alpha=0.7,
                    markerfacecolor='white', markeredgewidth=1.5)
    
    ax1.set_xlabel('U-velocity')
    ax1.set_ylabel('Y-coordinate')
    ax1.set_title('U-velocity at Vertical Centerline (x=0.5)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim(-0.5, 1.1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: V-velocity profiles comparison
    ax2 = plt.subplot(2, 3, 2)
    
    for i, re in enumerate(re_numbers):
        re_case = f"Re_{re}"
        if re_case in all_re_data and all_re_data[re_case] is not None:
            foam_data = all_re_data[re_case]
            if 'foam_v_horiz' in foam_data:
                ax2.plot(foam_data['foam_x_horiz'], foam_data['foam_v_horiz'], 
                        color=colors[i], linewidth=2, label=f'OpenFOAM Re={re}', alpha=0.8)
        
        # Plot corresponding ACE benchmark
        if re in ace_data_dict:
            ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re]
            ace_x = ace_vertical[:, 0]
            ace_v = ace_vertical[:, 1]
            
            # Create smooth interpolation for ACE data
            x_smooth = np.linspace(ace_x.min(), ace_x.max(), 500)
            if HAS_SCIPY:
                v_smooth = PchipInterpolator(ace_x, ace_v)(x_smooth)
            else:
                v_smooth = np.interp(x_smooth, ace_x, ace_v)
            
            ax2.plot(x_smooth, v_smooth, color=colors[i], linestyle='--', linewidth=2.5, 
                    label=f'ACE Re={re}', alpha=0.9)
            ax2.plot(ace_x, ace_v, 'o', color=colors[i], markersize=4, alpha=0.7,
                    markerfacecolor='white', markeredgewidth=1.5)
    
    ax2.set_xlabel('X-coordinate')
    ax2.set_ylabel('V-velocity')
    ax2.set_title('V-velocity at Horizontal Centerline (y=0.5)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.6, 0.5)
    
    # Plot 3: RMS Error Analysis
    ax3 = plt.subplot(2, 3, 3)
    
    re_vals = []
    rms_u_errors = []
    rms_v_errors = []
    rms_overall_errors = []
    
    for re in re_numbers:
        re_case = f"Re_{re}"
        if re_case in all_re_data and all_re_data[re_case] is not None and re in ace_data_dict:
            foam_data = all_re_data[re_case]
            ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re]
            
            errors = calculate_rms_errors(foam_data, ace_vertical, ace_horizontal)
            
            re_vals.append(re)
            rms_u_errors.append(errors['rms_u'])
            rms_v_errors.append(errors['rms_v'])
            rms_overall_errors.append(errors['rms_overall'])
    
    if re_vals:
        ax3.semilogy(re_vals, rms_u_errors, 'bo-', linewidth=2, markersize=6, label='U-velocity RMS')
        ax3.semilogy(re_vals, rms_v_errors, 'ro-', linewidth=2, markersize=6, label='V-velocity RMS')
        ax3.semilogy(re_vals, rms_overall_errors, 'go-', linewidth=2, markersize=6, label='Overall RMS')
        
        # Add accuracy threshold line
        ax3.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='2% Threshold')
        
        ax3.set_xlabel('Reynolds Number')
        ax3.set_ylabel('RMS Error')
        ax3.set_title('Validation Accuracy vs Reynolds Number', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # Plot 4: Reynolds Number vs RMS Error (simplified without extrema)
    ax4 = plt.subplot(2, 3, 4)
    
    if re_vals:
        # Plot RMS errors vs Reynolds number
        ax4.plot(re_vals, rms_u_errors, 'bo-', linewidth=2, markersize=8, label='U-velocity RMS', alpha=0.8)
        ax4.plot(re_vals, rms_v_errors, 'ro-', linewidth=2, markersize=8, label='V-velocity RMS', alpha=0.8)
        ax4.plot(re_vals, rms_overall_errors, 'go-', linewidth=2, markersize=8, label='Overall RMS', alpha=0.8)
        
        # Add accuracy threshold
        ax4.axhline(y=0.02, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='2% Threshold')
        ax4.axhline(y=0.01, color='green', linestyle=':', linewidth=2, alpha=0.7, label='1% Threshold')
        
        ax4.set_xlabel('Reynolds Number')
        ax4.set_ylabel('RMS Error')
        ax4.set_title('Accuracy vs Reynolds Number', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, 'No validation data available', ha='center', va='center', transform=ax4.transAxes)
    
    # Plot 5: Validation Summary Table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    if re_vals:
        headers = ['Reynolds', 'RMS U', 'RMS V', 'RMS Overall', 'Status']
        table_data = []
        
        for i, re in enumerate(re_vals):
            status = "Excellent" if rms_overall_errors[i] < 0.01 else \
                     "Good" if rms_overall_errors[i] < 0.02 else \
                     "Marginal" if rms_overall_errors[i] < 0.05 else "Poor"
            
            table_data.append([
                f"Re = {re}",
                f"{rms_u_errors[i]:.6f}",
                f"{rms_v_errors[i]:.6f}",
                f"{rms_overall_errors[i]:.6f}",
                status
            ])
        
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
    
    ax5.set_title('Validation Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Plot 6: Overall Assessment
    ax6 = plt.subplot(2, 3, 6)
    
    if re_vals:
        # Calculate overall validation metrics
        avg_rms = np.mean(rms_overall_errors)
        max_rms = np.max(rms_overall_errors)
        min_rms = np.min(rms_overall_errors)
        successful_cases = sum(1 for rms in rms_overall_errors if rms < 0.02)
        
        metrics_text = f"""BENCHMARK VALIDATION SUMMARY
        
📊 Cases Analyzed: {len(re_vals)}
✅ Successful Validations: {successful_cases}/{len(re_vals)} (< 2% RMS)
📈 Average RMS Error: {avg_rms:.4f}
📉 Minimum RMS Error: {min_rms:.4f}
📈 Maximum RMS Error: {max_rms:.4f}

🎯 INTERPOLATION METHOD: {INTERP_METHOD}

STATUS: {"PASSED" if successful_cases == len(re_vals) else f"PARTIAL ({successful_cases}/{len(re_vals)} passed)"}
        """
        
        ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
    
    # Ensure consistent background and layout without cropping subplots
    fig.patch.set_facecolor('white')
    plt.savefig('plots/reynolds_validation.png', dpi=300, facecolor=fig.get_facecolor())
    print("✅ Saved: plots/reynolds_validation.png")
    return fig

def create_validation_report(all_re_data, ace_data_dict):
    """Generate a detailed validation report"""
    
    report_lines = [
        "OPENFOAM BENCHMARK VALIDATION REPORT",
        "=" * 50,
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Interpolation method: {INTERP_METHOD}",
        "",
        "VALIDATION CASES:",
    ]
    
    re_numbers = sorted([int(re.split('_')[1]) for re in all_re_data.keys()])
    total_cases = len(re_numbers)
    successful_cases = 0
    
    for re in re_numbers:
        re_case = f"Re_{re}"
        if re_case in all_re_data and all_re_data[re_case] is not None and re in ace_data_dict:
            foam_data = all_re_data[re_case]
            ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re]
            
            errors = calculate_rms_errors(foam_data, ace_vertical, ace_horizontal)
            
            status = "PASS" if errors['rms_overall'] < 0.02 else "FAIL"
            if status == "PASS":
                successful_cases += 1
            
            report_lines.extend([
                f"",
                f"Reynolds Number: {re}",
                f"  RMS U-velocity error: {errors['rms_u']:.6f}",
                f"  RMS V-velocity error: {errors['rms_v']:.6f}",
                f"  Overall RMS error:    {errors['rms_overall']:.6f}",
                f"  Validation status:    {status} ({'<2% threshold' if status == 'PASS' else '≥2% threshold'})"
            ])
    
    report_lines.extend([
        "",
        "SUMMARY:",
        f"  Total cases:        {total_cases}",
        f"  Successful cases:   {successful_cases}",
        f"  Success rate:       {successful_cases/total_cases*100:.1f}%",
        f"  Overall status:     {'PASSED' if successful_cases == total_cases else 'FAILED'}",
        "",
        "REFERENCE: ACE Numerics lid-driven cavity benchmark data",
        "VALIDATION CRITERIA: RMS error < 2% for acceptance"
    ])
    
    # Save report to file
    os.makedirs('plots', exist_ok=True)
    with open('plots/validation_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("✅ Saved: plots/validation_report.txt")
    return '\n'.join(report_lines)

def main():
    """Main function to create benchmark validation plots"""
    
    print("🎯 OpenFOAM Benchmark Validation Analysis")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load ACE benchmark data for different Reynolds numbers
    ace_data_dict = {}
    
    try:
        ace_data_dict[1000] = load_ace_benchmark_re1000()
        print("✅ Loaded ACE benchmark data for Re=1000")
    except Exception as e:
        print(f"❌ Error loading Re=1000 benchmark: {e}")
    
    try:
        ace_data_dict[2500] = load_ace_benchmark_re2500()
        print("✅ Loaded ACE benchmark data for Re=2500")
    except Exception as e:
        print(f"❌ Error loading Re=2500 benchmark: {e}")
    
    # Define expected Reynolds cases
    re_cases = ['re_1000_mesh_256x256', 're_2500_mesh_256x256']
    
    # Load all Reynolds data
    all_re_data = {}
    print(f"\n📊 Loading Reynolds number validation data...")
    
    for re_case in re_cases:
        print(f"   Loading {re_case}...", end=' ')
        re_data = load_reynolds_data(re_case)
        
        if re_data is not None:
            re_number = int(re_case.split('_')[1])
            if re_number in ace_data_dict:
                ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re_number]
                errors = calculate_rms_errors(re_data, ace_vertical, ace_horizontal)
                re_data.update(errors)
                all_re_data[re_case] = re_data
                print(f"✅ RMS: {errors['rms_overall']:.6f}")
            else:
                all_re_data[re_case] = None
                print("❌ No ACE benchmark data available")
        else:
            all_re_data[re_case] = None
            print("❌ No validation data found")
    
    successful_cases = sum(1 for results in all_re_data.values() if results is not None)
    print(f"\n📈 Successfully loaded {successful_cases}/{len(re_cases)} Reynolds cases")
    
    if successful_cases == 0:
        print("❌ No validation data found. Please check directory structure.")
        return
    
    print("\n🎨 Creating validation plots...")
    
    # Create all plots
    try:
        plot_reynolds_validation(all_re_data, ace_data_dict)
        create_validation_report(all_re_data, ace_data_dict)
        
        print("\n🎉 Benchmark validation analysis completed successfully!")
        print("📁 Check the 'plots/' directory for output files:")
        print("   • reynolds_validation.png - Comprehensive validation dashboard")
        print("   • validation_report.txt - Detailed validation report")
        
        # Print summary to console
        print("\n📋 VALIDATION SUMMARY:")
        total_passed = 0
        analyzed_cases = 0
        
        for case_key, results in sorted(all_re_data.items(), key=lambda kv: int(kv[0].split('_')[1])):
            if results is not None:
                analyzed_cases += 1
                re_val = int(case_key.split('_')[1])
                rms_error = results['rms_overall']
                status = "PASS" if rms_error < 0.02 else "FAIL"
                if status == "PASS":
                    total_passed += 1
                print(f"   Re={re_val}: RMS={rms_error:.6f} - {status}")
        
        print(f"\n🏆 Overall: {total_passed}/{analyzed_cases} cases passed (RMS < 2%)")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()