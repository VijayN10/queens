#!/usr/bin/env python3
"""
Simple validation comparison without external dependencies
Compares OpenFOAM results with ACE Numerics benchmark data (Re=1000)

Usage: python simple_validation.py ./validation_data/
"""

import csv
import sys
import os
import math

def load_ace_benchmark():
    """ACE Numerics benchmark data for Re=1000"""
    
    # Vertical velocity along y=0.5 (Table 2)
    ace_vertical = [
        [0.0000, 0.0000000], [0.0100, 0.0709203], [0.0200, 0.1306210], [0.0300, 0.1796062],
        [0.0400, 0.2189314], [0.0625, 0.2807056], [0.0703, 0.2962703], [0.0781, 0.3099097],
        [0.0938, 0.3330442], [0.1563, 0.3769189], [0.2266, 0.3339924], [0.2344, 0.3253592],
        [0.3500, 0.1882246], [0.5000, 0.0257995], [0.6800, -0.1730887], [0.8047, -0.3202137],
        [0.8594, -0.4264545], [0.9063, -0.5264392], [0.9453, -0.4103754], [0.9531, -0.3553213],
        [0.9609, -0.2936869], [0.9688, -0.2279225], [0.9700, -0.2178657], [0.9800, -0.1359277],
        [0.9900, -0.0617528], [0.9950, -0.0291290], [1.0000, 0.0000000]
    ]
    
    # Horizontal velocity along x=0.5 (Table 4) 
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
        print(f"Error reading {filename}: {e}")
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

def validate_results(validation_dir):
    """Validate OpenFOAM results against ACE Numerics benchmark"""
    
    print("🔍 Loading benchmark data...")
    ace_v, ace_h, ace_extrema, ace_vortex = load_ace_benchmark()
    
    print("📊 Loading OpenFOAM data...")
    
    # Load vertical centerline
    vertical_file = os.path.join(validation_dir, 'vertical_centerline_fixed.csv')
    vertical_data, v_headers = read_csv_file(vertical_file)
    
    # Load horizontal centerline
    horizontal_file = os.path.join(validation_dir, 'horizontal_centerline_fixed.csv')
    horizontal_data, h_headers = read_csv_file(horizontal_file)
    
    # Load probe data
    probe_file = os.path.join(validation_dir, 'probe_data.csv')
    probe_data, p_headers = read_csv_file(probe_file)
    
    if not vertical_data or not horizontal_data:
        print("❌ Could not load OpenFOAM data files")
        return
    
    print(f"✅ Loaded vertical centerline: {len(vertical_data)} points")
    print(f"✅ Loaded horizontal centerline: {len(horizontal_data)} points")
    if probe_data:
        print(f"✅ Loaded probe data: {len(probe_data)} points")
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS - Re = 1000")
    print("="*60)
    
    # ========== VERTICAL CENTERLINE VALIDATION ==========
    print("\n📈 VERTICAL CENTERLINE VALIDATION (v-velocity along y=0.5)")
    print("-" * 50)
    
    # Extract OpenFOAM vertical data (sorted by y-position)
    foam_y = []
    foam_v = []
    for row in vertical_data:
        y = float(row['Points:1'])  # y-coordinate
        v = float(row['U:1'])       # v-velocity
        foam_y.append(y)
        foam_v.append(v)
    
    # Sort by y-coordinate
    sorted_indices = sorted(range(len(foam_y)), key=lambda k: foam_y[k])
    foam_y_sorted = [foam_y[i] for i in sorted_indices]
    foam_v_sorted = [foam_v[i] for i in sorted_indices]
    
    # Compare with benchmark at key points
    v_errors = []
    print(f"{'x':<8} {'ACE v':<12} {'OpenFOAM v':<12} {'Error':<12} {'% Error'}")
    print("-" * 50)
    
    for x_bench, v_bench in ace_v[:15]:  # First 15 points for readability
        v_foam = interpolate_linear(foam_y_sorted, foam_v_sorted, x_bench)
        if v_foam is not None:
            error = abs(v_foam - v_bench)
            pct_error = (error / (abs(v_bench) + 1e-10)) * 100
            v_errors.append(error)
            print(f"{x_bench:<8.4f} {v_bench:<12.6f} {v_foam:<12.6f} {error:<12.6f} {pct_error:<8.2f}%")
    
    # Calculate RMS error
    if v_errors:
        rms_v = math.sqrt(sum(e**2 for e in v_errors) / len(v_errors))
        print(f"\n🎯 Vertical RMS Error: {rms_v:.8f}")
    
    # ========== HORIZONTAL CENTERLINE VALIDATION ==========
    print("\n📈 HORIZONTAL CENTERLINE VALIDATION (u-velocity along x=0.5)")
    print("-" * 50)
    
    # Extract OpenFOAM horizontal data
    foam_x = []
    foam_u = []
    for row in horizontal_data:
        x = float(row['Points:0'])  # x-coordinate  
        u = float(row['U:0'])       # u-velocity
        foam_x.append(x)
        foam_u.append(u)
    
    # Sort by x-coordinate
    sorted_indices = sorted(range(len(foam_x)), key=lambda k: foam_x[k])
    foam_x_sorted = [foam_x[i] for i in sorted_indices]
    foam_u_sorted = [foam_u[i] for i in sorted_indices]
    
    # Compare with benchmark
    u_errors = []
    print(f"{'y':<8} {'ACE u':<12} {'OpenFOAM u':<12} {'Error':<12} {'% Error'}")
    print("-" * 50)
    
    for y_bench, u_bench in ace_h[:15]:  # First 15 points
        u_foam = interpolate_linear(foam_x_sorted, foam_u_sorted, y_bench)
        if u_foam is not None:
            error = abs(u_foam - u_bench)
            pct_error = (error / (abs(u_bench) + 1e-10)) * 100
            u_errors.append(error)
            print(f"{y_bench:<8.4f} {u_bench:<12.6f} {u_foam:<12.6f} {error:<12.6f} {pct_error:<8.2f}%")
    
    # Calculate RMS error
    if u_errors:
        rms_u = math.sqrt(sum(e**2 for e in u_errors) / len(u_errors))
        print(f"\n🎯 Horizontal RMS Error: {rms_u:.8f}")
    
    # ========== EXTREMA VALIDATION ==========
    print("\n📊 EXTREMA VALIDATION")
    print("-" * 30)
    
    # Find extrema in OpenFOAM data
    foam_u_min = min(foam_u_sorted)
    foam_u_max = max(foam_u_sorted)
    foam_v_min = min(foam_v_sorted)
    foam_v_max = max(foam_v_sorted)
    
    print(f"Quantity          ACE Benchmark    OpenFOAM         Error")
    print("-" * 55)
    print(f"u_min            {ace_extrema['umin']:<12.6f} {foam_u_min:<12.6f} {abs(foam_u_min - ace_extrema['umin']):<12.6f}")
    print(f"v_max            {ace_extrema['vmax']:<12.6f} {foam_v_max:<12.6f} {abs(foam_v_max - ace_extrema['vmax']):<12.6f}")
    print(f"v_min            {ace_extrema['vmin']:<12.6f} {foam_v_min:<12.6f} {abs(foam_v_min - ace_extrema['vmin']):<12.6f}")
    
    # ========== PROBE VALIDATION ==========
    if probe_data:
        print("\n🎯 PROBE POINT VALIDATION")
        print("-" * 30)
        
        print(f"{'Point':<8} {'x':<8} {'y':<8} {'u':<12} {'v':<12} {'p':<12}")
        print("-" * 60)
        
        for i, row in enumerate(probe_data):
            x = float(row['Points:0'])
            y = float(row['Points:1'])
            u = float(row['U:0'])
            v = float(row['U:1'])
            p = float(row['p'])
            print(f"{i+1:<8} {x:<8.3f} {y:<8.3f} {u:<12.6f} {v:<12.6f} {p:<12.6f}")
        
        # Find center point closest to (0.5, 0.5)
        center_distances = []
        for row in probe_data:
            x = float(row['Points:0'])
            y = float(row['Points:1'])
            dist = math.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            center_distances.append(dist)
        
        center_idx = center_distances.index(min(center_distances))
        center_row = probe_data[center_idx]
        
        print(f"\n🎯 Center point comparison:")
        print(f"ACE vortex center: ({ace_vortex['x']:.6f}, {ace_vortex['y']:.6f})")
        print(f"OpenFOAM nearest:  ({float(center_row['Points:0']):.6f}, {float(center_row['Points:1']):.6f})")
        print(f"Distance error: {min(center_distances):.6f}")
    
    # ========== OVERALL ASSESSMENT ==========
    print("\n" + "="*60)
    print("🎉 VALIDATION SUMMARY")
    print("="*60)
    
    if v_errors and u_errors:
        overall_rms = math.sqrt((rms_v**2 + rms_u**2) / 2)
        print(f"Overall RMS Error: {overall_rms:.8f}")
        
        if overall_rms < 0.01:
            status = "✅ EXCELLENT"
        elif overall_rms < 0.05:
            status = "✅ GOOD"
        elif overall_rms < 0.1:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ NEEDS IMPROVEMENT"
        
        print(f"Validation Status: {status}")
        
        print(f"\nBenchmark: ACE Numerics Re=1000")
        print(f"OpenFOAM mesh: {len(vertical_data)} x {len(horizontal_data)} (estimated)")
        print(f"Data points validated: {len(v_errors) + len(u_errors)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_validation.py ./validation_data/")
        sys.exit(1)
    
    validation_dir = sys.argv[1]
    
    if not os.path.exists(validation_dir):
        print(f"❌ Directory {validation_dir} not found")
        sys.exit(1)
    
    validate_results(validation_dir)

if __name__ == "__main__":
    main()