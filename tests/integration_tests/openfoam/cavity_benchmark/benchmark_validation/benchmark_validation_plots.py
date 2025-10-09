#!/usr/bin/env python3
"""
OpenFOAM Benchmark Validation Plotting Script - FIXED VERSION
Compares OpenFOAM cavity flow results with ACE Numerics benchmark data for multiple Reynolds numbers

Key fixes:
1. Corrected directory structure mapping (re_1000_mesh_256x256 -> Re_{re})
2. Fixed CSV data extraction (proper column indexing)
3. Improved RMS calculation with proper data alignment
4. Added debug output for troubleshooting
5. Fixed plotting logic to handle actual data

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
        [0.9844, 0.7391114], [0.9922, 0.8183289], [1.0000, 1.0000000]
    ])
    
    # Extrema values for comparison (optional)
    ace_extrema = {
        'umin': -0.38859,
        'vmax': 0.37692,
        'vmin': -0.49533
    }
    
    # Primary vortex center (optional)
    ace_vortex = [0.6172, 0.7344]
    
    return ace_vertical, ace_horizontal, ace_extrema, ace_vortex

def load_ace_benchmark_re2500():
    """Load ACE Numerics benchmark data for Re=2500"""
    
    # Vertical velocity (v) along horizontal line y=0.5
    ace_vertical = np.array([
        [0.0000, 0.0000000], [0.0100, 0.1198044], [0.0200, 0.2201260], [0.0300, 0.3020124],
        [0.0400, 0.3667248], [0.0625, 0.4604728], [0.0703, 0.4820656], [0.0781, 0.5011156],
        [0.0938, 0.5313080], [0.1563, 0.5516260], [0.2266, 0.4188516], [0.2344, 0.4024532],
        [0.3500, 0.2046800], [0.5000, 0.0176060], [0.6800, -0.2527160], [0.8047, -0.4467640],
        [0.8594, -0.5574520], [0.9063, -0.6351880], [0.9453, -0.5108880], [0.9531, -0.4418920],
        [0.9609, -0.3637840], [0.9688, -0.2809080], [0.9700, -0.2679400], [0.9800, -0.1666920],
        [0.9900, -0.0756960], [0.9950, -0.0356460], [1.0000, 0.0000000]
    ])
    
    # Horizontal velocity (u) along vertical line x=0.5
    ace_horizontal = np.array([
        [0.0000, 0.0000000], [0.0100, -0.0533912], [0.0200, -0.1016928], [0.0300, -0.1453208],
        [0.0400, -0.1852248], [0.0547, -0.2385544], [0.0625, -0.2653328], [0.0703, -0.2912448],
        [0.1016, -0.3880776], [0.1719, -0.4883520], [0.2813, -0.3224888], [0.4531, -0.1165200],
        [0.5000, -0.0649032], [0.6172, 0.0869736], [0.7344, 0.2894112], [0.8516, 0.5301704],
        [0.9531, 0.7550728], [0.9609, 0.8271592], [0.9688, 0.9330448], [0.9766, 1.0881032],
        [0.9844, 1.2946920], [0.9922, 1.6022600], [1.0000, 1.0000000]
    ])
    
    # Extrema values
    ace_extrema = {
        'umin': -0.48835,
        'vmax': 0.55163,
        'vmin': -0.63519
    }
    
    # Primary vortex center
    ace_vortex = [0.5625, 0.6094]
    
    return ace_vertical, ace_horizontal, ace_extrema, ace_vortex

def load_reynolds_data(re_case):
    """Load OpenFOAM validation data for a Reynolds number case"""
    
    validation_dir = Path(re_case) / "validation_data"
    
    if not validation_dir.exists():
        print(f"⚠️  Validation data directory not found: {validation_dir}")
        return None
    
    data = {}
    
    try:
        # Load centerline u-velocity (vertical line at x=0.5)
        u_file = validation_dir / "vertical_centerline_fixed.csv"
        if u_file.exists():
            print(f"    Reading {u_file}")
            with open(u_file, 'r') as f:
                reader = csv.DictReader(f)
                
                # Debug: print headers to understand CSV structure
                headers = reader.fieldnames
                print(f"    CSV headers: {headers}")
                
                rows = list(reader)
                print(f"    CSV rows: {len(rows)}")
                
                if len(rows) > 0:
                    # Extract coordinate and velocity data
                    # Vertical centerline: y-coordinate and u-velocity
                    if 'Points:1' in headers and 'U:0' in headers:
                        data['foam_y_vert'] = np.array([float(row['Points:1']) for row in rows])
                        data['foam_u_vert'] = np.array([float(row['U:0']) for row in rows])
                        print(f"    Extracted vertical u-velocity: {len(data['foam_u_vert'])} points")
                        print(f"    U range: [{np.min(data['foam_u_vert']):.6f}, {np.max(data['foam_u_vert']):.6f}]")
                    else:
                        print(f"    Expected columns not found: Points:1, U:0")
        else:
            print(f"    File not found: {u_file}")
        
        # Load centerline v-velocity (horizontal line at y=0.5)
        v_file = validation_dir / "horizontal_centerline_fixed.csv"
        if v_file.exists():
            print(f"    Reading {v_file}")
            with open(v_file, 'r') as f:
                reader = csv.DictReader(f)
                
                headers = reader.fieldnames
                print(f"    CSV headers: {headers}")
                
                rows = list(reader)
                print(f"    CSV rows: {len(rows)}")
                
                if len(rows) > 0:
                    # Extract coordinate and velocity data
                    # Horizontal centerline: x-coordinate and v-velocity
                    if 'Points:0' in headers and 'U:1' in headers:
                        data['foam_x_horiz'] = np.array([float(row['Points:0']) for row in rows])
                        data['foam_v_horiz'] = np.array([float(row['U:1']) for row in rows])
                        print(f"    Extracted horizontal v-velocity: {len(data['foam_v_horiz'])} points")
                        print(f"    V range: [{np.min(data['foam_v_horiz']):.6f}, {np.max(data['foam_v_horiz']):.6f}]")
                    else:
                        print(f"    Expected columns not found: Points:0, U:1")
        else:
            print(f"    File not found: {v_file}")
        
        # Check if we have both datasets
        if 'foam_u_vert' in data and 'foam_v_horiz' in data:
            print(f"    ✅ Successfully loaded both velocity profiles")
            return data
        else:
            print(f"    ❌ Missing velocity profiles")
            return None
        
    except Exception as e:
        print(f"❌ Error loading data for {re_case}: {e}")
        return None

def calculate_rms_errors(foam_data, ace_vertical, ace_horizontal):
    """Calculate RMS errors between OpenFOAM and ACE benchmark data"""
    
    errors = {}
    
    # U-velocity RMS error (vertical centerline)
    if 'foam_u_vert' in foam_data:
        print(f"    Calculating RMS for U-velocity...")
        
        # ACE data: ace_horizontal contains [y, u] pairs
        ace_y = ace_horizontal[:, 0]
        ace_u = ace_horizontal[:, 1]
        
        # OpenFOAM data
        foam_y = foam_data['foam_y_vert']
        foam_u = foam_data['foam_u_vert']
        
        print(f"    ACE y range: [{np.min(ace_y):.3f}, {np.max(ace_y):.3f}]")
        print(f"    FOAM y range: [{np.min(foam_y):.3f}, {np.max(foam_y):.3f}]")
        print(f"    ACE u range: [{np.min(ace_u):.6f}, {np.max(ace_u):.6f}]")
        print(f"    FOAM u range: [{np.min(foam_u):.6f}, {np.max(foam_u):.6f}]")
        
        # Interpolate ACE data to OpenFOAM y-coordinates
        ace_u_interp = np.interp(foam_y, ace_y, ace_u)
        
        # Calculate differences and RMS
        u_diff = foam_u - ace_u_interp
        errors['rms_u'] = np.sqrt(np.mean(u_diff**2))
        
        print(f"    U-velocity RMS: {errors['rms_u']:.6f}")
    else:
        errors['rms_u'] = float('inf')
        print(f"    No U-velocity data available")
    
    # V-velocity RMS error (horizontal centerline)
    if 'foam_v_horiz' in foam_data:
        print(f"    Calculating RMS for V-velocity...")
        
        # ACE data: ace_vertical contains [x, v] pairs  
        ace_x = ace_vertical[:, 0]
        ace_v = ace_vertical[:, 1]
        
        # OpenFOAM data
        foam_x = foam_data['foam_x_horiz']
        foam_v = foam_data['foam_v_horiz']
        
        print(f"    ACE x range: [{np.min(ace_x):.3f}, {np.max(ace_x):.3f}]")
        print(f"    FOAM x range: [{np.min(foam_x):.3f}, {np.max(foam_x):.3f}]")
        print(f"    ACE v range: [{np.min(ace_v):.6f}, {np.max(ace_v):.6f}]")
        print(f"    FOAM v range: [{np.min(foam_v):.6f}, {np.max(foam_v):.6f}]")
        
        # Interpolate ACE data to OpenFOAM x-coordinates
        ace_v_interp = np.interp(foam_x, ace_x, ace_v)
        
        # Calculate differences and RMS
        v_diff = foam_v - ace_v_interp
        errors['rms_v'] = np.sqrt(np.mean(v_diff**2))
        
        print(f"    V-velocity RMS: {errors['rms_v']:.6f}")
    else:
        errors['rms_v'] = float('inf')
        print(f"    No V-velocity data available")
    
    # Overall RMS error
    if errors['rms_u'] != float('inf') and errors['rms_v'] != float('inf'):
        errors['rms_overall'] = np.sqrt((errors['rms_u']**2 + errors['rms_v']**2) / 2)
        print(f"    Overall RMS: {errors['rms_overall']:.6f}")
    else:
        errors['rms_overall'] = float('inf')
        print(f"    Overall RMS: infinite (missing data)")
    
    return errors

def plot_reynolds_validation(all_re_data, ace_data_dict):
    """Create comprehensive Reynolds number validation plots"""
    
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    
    # Color scheme for different Reynolds numbers
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Extract Reynolds numbers and data
    re_vals = []
    rms_u_errors = []
    rms_v_errors = []
    rms_overall_errors = []
    
    # Process data from all_re_data keys
    for re_case, data in all_re_data.items():
        if data is not None and 'rms_overall' in data:
            re_number = int(re_case.split('_')[1])
            re_vals.append(re_number)
            rms_u_errors.append(data['rms_u'])
            rms_v_errors.append(data['rms_v'])
            rms_overall_errors.append(data['rms_overall'])
    
    # Sort by Reynolds number
    if re_vals:
        sorted_indices = np.argsort(re_vals)
        re_vals = [re_vals[i] for i in sorted_indices]
        rms_u_errors = [rms_u_errors[i] for i in sorted_indices]
        rms_v_errors = [rms_v_errors[i] for i in sorted_indices]
        rms_overall_errors = [rms_overall_errors[i] for i in sorted_indices]
    
    # Plot 1: U-velocity profiles comparison
    ax1 = plt.subplot(2, 3, 1)
    
    for i, re in enumerate(re_vals):
        re_case = f"re_{re}_mesh_256x256"
        if re_case in all_re_data and all_re_data[re_case] is not None:
            foam_data = all_re_data[re_case]
            if 'foam_u_vert' in foam_data:
                ax1.plot(foam_data['foam_u_vert'], foam_data['foam_y_vert'], 
                        color=colors[i % len(colors)], linewidth=2, label=f'OpenFOAM Re={re}', alpha=0.8)
        
        # Plot corresponding ACE benchmark
        if re in ace_data_dict:
            ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re]
            ace_y = ace_horizontal[:, 0]
            ace_u = ace_horizontal[:, 1]
            
            ax1.plot(ace_u, ace_y, color=colors[i % len(colors)], linestyle='--', 
                    linewidth=2, label=f'ACE Re={re}', alpha=0.6)
    
    ax1.set_xlabel('U-velocity')
    ax1.set_ylabel('Y-coordinate')
    ax1.set_title('U-velocity at Vertical Centerline (x=0.5)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: V-velocity profiles comparison
    ax2 = plt.subplot(2, 3, 2)
    
    for i, re in enumerate(re_vals):
        re_case = f"re_{re}_mesh_256x256"
        if re_case in all_re_data and all_re_data[re_case] is not None:
            foam_data = all_re_data[re_case]
            if 'foam_v_horiz' in foam_data:
                ax2.plot(foam_data['foam_x_horiz'], foam_data['foam_v_horiz'], 
                        color=colors[i % len(colors)], linewidth=2, label=f'OpenFOAM Re={re}', alpha=0.8)
        
        # Plot corresponding ACE benchmark
        if re in ace_data_dict:
            ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re]
            ace_x = ace_vertical[:, 0]
            ace_v = ace_vertical[:, 1]
            
            ax2.plot(ace_x, ace_v, color=colors[i % len(colors)], linestyle='--',
                    linewidth=2, label=f'ACE Re={re}', alpha=0.6)
    
    ax2.set_xlabel('X-coordinate')
    ax2.set_ylabel('V-velocity')
    ax2.set_title('V-velocity at Horizontal Centerline (y=0.5)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: RMS errors (log scale)
    ax3 = plt.subplot(2, 3, 3)
    
    if re_vals:
        ax3.semilogy(re_vals, rms_u_errors, 'bo-', linewidth=2, markersize=6, label='U-velocity RMS')
        ax3.semilogy(re_vals, rms_v_errors, 'ro-', linewidth=2, markersize=6, label='V-velocity RMS')
        ax3.semilogy(re_vals, rms_overall_errors, 'go-', linewidth=2, markersize=6, label='Overall RMS')
        
        # Add accuracy threshold lines
        ax3.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='10% Threshold')
        ax3.axhline(y=0.05, color='green', linestyle=':', alpha=0.7, label='5% Threshold')
        
        ax3.set_xlabel('Reynolds Number')
        ax3.set_ylabel('RMS Error')
        ax3.set_title('Validation Accuracy vs Reynolds Number', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No validation data available', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Linear scale RMS errors
    ax4 = plt.subplot(2, 3, 4)
    
    if re_vals:
        ax4.plot(re_vals, rms_u_errors, 'bo-', linewidth=2, markersize=8, label='U-velocity RMS', alpha=0.8)
        ax4.plot(re_vals, rms_v_errors, 'ro-', linewidth=2, markersize=8, label='V-velocity RMS', alpha=0.8)
        ax4.plot(re_vals, rms_overall_errors, 'go-', linewidth=2, markersize=8, label='Overall RMS', alpha=0.8)
        
        # Add accuracy thresholds
        ax4.axhline(y=0.10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10% Threshold')
        ax4.axhline(y=0.05, color='green', linestyle=':', linewidth=2, alpha=0.7, label='5% Threshold')
        
        ax4.set_xlabel('Reynolds Number')
        ax4.set_ylabel('RMS Error')
        ax4.set_title('Accuracy vs Reynolds Number (Linear)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
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
                     "Good" if rms_overall_errors[i] < 0.1 else \
                     "Marginal" if rms_overall_errors[i] < 0.05 else "Poor"
            
            table_data.append([
                f"Re = {re}",
                f"{rms_u_errors[i]:.4f}",
                f"{rms_v_errors[i]:.4f}",
                f"{rms_overall_errors[i]:.4f}",
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
        successful_cases = sum(1 for rms in rms_overall_errors if rms < 0.1)
        
        metrics_text = f"""BENCHMARK VALIDATION SUMMARY
        
📊 Cases Analyzed: {len(re_vals)}
✅ Successful Validations: {successful_cases}/{len(re_vals)} (< 10% RMS)
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
    else:
        ax6.text(0.5, 0.5, 'No validation data available\n\nCheck directory structure:\n\nre_1000_mesh_256x256/validation_data/\nre_2500_mesh_256x256/validation_data/',
                ha='center', va='center', transform=ax6.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
    
    # Ensure consistent background and layout
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
    
    # Extract case information 
    total_cases = 0
    successful_cases = 0
    
    for re_case, data in sorted(all_re_data.items(), key=lambda kv: int(kv[0].split('_')[1])):
        if data is not None:
            total_cases += 1
            re_number = int(re_case.split('_')[1])
            
            status = "PASS" if data['rms_overall'] < 0.1 else "FAIL"
            if status == "PASS":
                successful_cases += 1
            
            report_lines.extend([
                f"",
                f"Reynolds Number: {re_number}",
                f"  RMS U-velocity error: {data['rms_u']:.6f}",
                f"  RMS V-velocity error: {data['rms_v']:.6f}",
                f"  Overall RMS error:    {data['rms_overall']:.6f}",
                f"  Validation status:    {status} ({'<10% threshold' if status == 'PASS' else '≥10% threshold'})"
            ])
    
    report_lines.extend([
        "",
        "SUMMARY:",
        f"  Total cases:        {total_cases}",
        f"  Successful cases:   {successful_cases}",
        f"  Success rate:       {successful_cases/total_cases*100:.1f}%" if total_cases > 0 else "  Success rate:       N/A",
        f"  Overall status:     {'PASSED' if successful_cases == total_cases and total_cases > 0 else 'FAILED'}",
        "",
        "REFERENCE: ACE Numerics lid-driven cavity benchmark data",
        "VALIDATION CRITERIA: RMS error < 10% for acceptance"
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
    
    # Define expected Reynolds cases (FIXED: correct directory names)
    re_cases = ['re_1000_mesh_256x256', 're_2500_mesh_256x256']
    
    # Load all Reynolds data
    all_re_data = {}
    print(f"\n📊 Loading Reynolds number validation data...")
    
    for re_case in re_cases:
        print(f"   Loading {re_case}...")
        re_data = load_reynolds_data(re_case)
        
        if re_data is not None:
            re_number = int(re_case.split('_')[1])
            if re_number in ace_data_dict:
                print(f"   Calculating RMS errors for Re={re_number}...")
                ace_vertical, ace_horizontal, ace_extrema, ace_vortex = ace_data_dict[re_number]
                errors = calculate_rms_errors(re_data, ace_vertical, ace_horizontal)
                re_data.update(errors)
                all_re_data[re_case] = re_data
                print(f"   ✅ RMS: {errors['rms_overall']:.6f}")
            else:
                all_re_data[re_case] = None
                print(f"   ❌ No ACE benchmark data available for Re={re_number}")
        else:
            all_re_data[re_case] = None
            print(f"   ❌ No validation data found for {re_case}")
    
    successful_cases = sum(1 for results in all_re_data.values() if results is not None)
    print(f"\n📈 Successfully loaded {successful_cases}/{len(re_cases)} Reynolds cases")
    
    if successful_cases == 0:
        print("❌ No validation data found. Please check directory structure.")
        print("Expected structure:")
        print("  benchmark_validation/")
        print("  ├── re_1000_mesh_256x256/validation_data/")
        print("  │   ├── vertical_centerline_fixed.csv")
        print("  │   └── horizontal_centerline_fixed.csv")
        print("  └── re_2500_mesh_256x256/validation_data/")
        print("      ├── vertical_centerline_fixed.csv")
        print("      └── horizontal_centerline_fixed.csv")
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
                status = "PASS" if rms_error < 0.1 else "FAIL"
                if status == "PASS":
                    total_passed += 1
                print(f"   Re={re_val}: RMS={rms_error:.6f} - {status}")
        
        print(f"\n🏆 Overall: {total_passed}/{analyzed_cases} cases passed (RMS < 10%)")
        
        # Debug information
        print(f"\n🔧 DEBUG INFORMATION:")
        print(f"   Interpolation method: {INTERP_METHOD}")
        print(f"   Working directory: {os.getcwd()}")
        print(f"   Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()