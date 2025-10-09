#!/usr/bin/env python3
"""
Simple ParaView script for cavity flow - just probe points
Usage: pvpython simple_cavity_probes.py /path/to/case/
"""

import sys
import numpy as np
import json
from pathlib import Path

# Import ParaView modules
try:
    import paraview
    paraview.compatibility.major = 5
    paraview.compatibility.minor = 13
    from paraview.simple import *
    from paraview import servermanager
except ImportError as e:
    print(f"❌ Error importing ParaView: {e}")
    sys.exit(1)

def extract_simple_probes(case_path, output_file="cavity_probes.json"):
    """Extract velocity and pressure at probe locations only."""
    
    case_path = Path(case_path)
    print(f"Processing case: {case_path}")
    
    # Physical domain (0.1m × 0.1m × 0.01m)
    X_CENTER, Y_CENTER, Z_CENTER = 0.05, 0.05, 0.005
    
    # Create/find .foam file
    foam_file = str(case_path / f"{case_path.name}.foam")
    if not Path(foam_file).exists():
        with open(foam_file, 'w') as f:
            f.write("")
    
    # Load case
    reader = OpenFOAMReader(FileName=foam_file)
    reader.MeshRegions = ['internalMesh']
    reader.CellArrays = ['p', 'U']
    reader.UpdatePipeline()
    
    # Go to latest time
    time_values = reader.TimestepValues
    if time_values:
        latest_time = time_values[-1]
        view = GetRenderView()
        view.ViewTime = latest_time
        reader.UpdatePipeline()
    
    # Define probe points
    probe_points = {
        'center': [0.05, 0.05, 0.005],
        'bottom_left': [0.01, 0.01, 0.005],
        'bottom_right': [0.09, 0.01, 0.005],
        'top_left': [0.01, 0.09, 0.005],
        'top_right': [0.09, 0.09, 0.005],
    }
    
    results = {}
    
    print("Extracting probe data...")
    for probe_name, coords in probe_points.items():
        try:
            probe = ProbeLocation(Input=reader)
            probe.ProbeType = 'Fixed Radius Point Source'
            probe.ProbeType.Center = coords
            probe.ProbeType.Radius = 0.001
            probe.UpdatePipeline()
            
            probe_data = servermanager.Fetch(probe)
            if probe_data.GetNumberOfPoints() > 0:
                point_data = probe_data.GetPointData()
                velocity = point_data.GetArray('U').GetTuple(0)
                pressure = point_data.GetArray('p').GetValue(0)
                
                results[probe_name] = {
                    'u_velocity': velocity[0],
                    'v_velocity': velocity[1],
                    'pressure': pressure
                }
                print(f"✅ {probe_name}: U={velocity[0]:.6f}, V={velocity[1]:.6f}, p={pressure:.6f}")
            else:
                print(f"❌ {probe_name}: No data")
                results[probe_name] = None
        except Exception as e:
            print(f"❌ {probe_name}: Error - {e}")
            results[probe_name] = None
    
    # Save results
    output_path = case_path / output_file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: pvpython simple_cavity_probes.py /path/to/case/")
        sys.exit(1)
    
    case_path = sys.argv[1]
    
    try:
        results = extract_simple_probes(case_path)
        print("\n🎉 Extraction completed!")
        
        # Print summary
        successful = sum(1 for v in results.values() if v is not None)
        print(f"Successfully extracted {successful}/{len(results)} probe points")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()