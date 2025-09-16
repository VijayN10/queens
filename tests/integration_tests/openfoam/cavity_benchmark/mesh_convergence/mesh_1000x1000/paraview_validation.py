#!/usr/bin/env python3
"""
Quick fix for centerline extraction in ParaView 5.13+
Run this after the main script to get centerline data

Usage: pvpython fix_centerlines.py /path/to/case/directory
"""

import sys
import os
from paraview.simple import *

def extract_centerlines(case_dir):
    """Extract centerline profiles using ParaView 5.13+ syntax"""
    
    print(f"Extracting centerlines for case: {case_dir}")
    
    # Load OpenFOAM case
    foam_file = os.path.join(case_dir, f"{os.path.basename(case_dir)}.foam")
    reader = OpenFOAMReader(FileName=foam_file)
    reader.MeshRegions = ['internalMesh']
    reader.CellArrays = ['p', 'U']
    
    # Get final time
    times = reader.TimestepValues
    final_time = times[-1] if times else 0.0
    reader.UpdatePipeline(time=final_time)
    
    output_dir = os.path.join(case_dir, 'validation_data')
    
    print("Extracting centerlines using Plot Over Line...")
    
    # ========== VERTICAL CENTERLINE (ALTERNATIVE METHOD) ==========
    try:
        # Use Plot Over Line with correct syntax for ParaView 5.13+
        vertical_line = PlotOverLine(Input=reader)
        # Set the line points directly
        vertical_line.Point1 = [0.5, 0.0, 0.05]
        vertical_line.Point2 = [0.5, 1.0, 0.05]
        vertical_line.Resolution = 100
        vertical_line.UpdatePipeline(time=final_time)
        
        vertical_output = os.path.join(output_dir, 'vertical_centerline_fixed.csv')
        CreateWriter(vertical_output, vertical_line).UpdatePipeline()
        print(f"   ✅ Saved: vertical_centerline_fixed.csv ({os.path.getsize(vertical_output)} bytes)")
        
    except Exception as e:
        print(f"   ❌ Vertical centerline failed: {e}")
        
        # Alternative: Manual sampling using multiple probe points
        try:
            print("   Trying manual sampling...")
            
            # Create 101 probe points along vertical line
            vertical_probes = ProgrammableSource()
            vertical_probes.OutputDataSetType = 'vtkPolyData'
            vertical_probes.Script = """
import vtk
output = self.GetPolyDataOutput()
points = vtk.vtkPoints()

# 101 points from y=0 to y=1 at x=0.5
for i in range(101):
    y = i / 100.0
    points.InsertNextPoint([0.5, y, 0.05])

output.SetPoints(points)
vertices = vtk.vtkCellArray()
for i in range(points.GetNumberOfPoints()):
    vertices.InsertNextCell(1, [i])
output.SetVerts(vertices)
"""
            
            vertical_probe_filter = ProbeLocation(Input=reader)
            vertical_probe_filter.ProbeType = vertical_probes
            vertical_probe_filter.UpdatePipeline(time=final_time)
            
            vertical_output = os.path.join(output_dir, 'vertical_centerline_manual.csv')
            CreateWriter(vertical_output, vertical_probe_filter).UpdatePipeline()
            print(f"   ✅ Saved: vertical_centerline_manual.csv ({os.path.getsize(vertical_output)} bytes)")
            
        except Exception as e2:
            print(f"   ❌ Manual vertical sampling failed: {e2}")
    
    # ========== HORIZONTAL CENTERLINE ==========
    try:
        horizontal_line = PlotOverLine(Input=reader)
        horizontal_line.Point1 = [0.0, 0.5, 0.05]
        horizontal_line.Point2 = [1.0, 0.5, 0.05]
        horizontal_line.Resolution = 100
        horizontal_line.UpdatePipeline(time=final_time)
        
        horizontal_output = os.path.join(output_dir, 'horizontal_centerline_fixed.csv')
        CreateWriter(horizontal_output, horizontal_line).UpdatePipeline()
        print(f"   ✅ Saved: horizontal_centerline_fixed.csv ({os.path.getsize(horizontal_output)} bytes)")
        
    except Exception as e:
        print(f"   ❌ Horizontal centerline failed: {e}")
        
        # Alternative: Manual sampling
        try:
            print("   Trying manual sampling...")
            
            horizontal_probes = ProgrammableSource()
            horizontal_probes.OutputDataSetType = 'vtkPolyData'
            horizontal_probes.Script = """
import vtk
output = self.GetPolyDataOutput()
points = vtk.vtkPoints()

# 101 points from x=0 to x=1 at y=0.5
for i in range(101):
    x = i / 100.0
    points.InsertNextPoint([x, 0.5, 0.05])

output.SetPoints(points)
vertices = vtk.vtkCellArray()
for i in range(points.GetNumberOfPoints()):
    vertices.InsertNextCell(1, [i])
output.SetVerts(vertices)
"""
            
            horizontal_probe_filter = ProbeLocation(Input=reader)
            horizontal_probe_filter.ProbeType = horizontal_probes
            horizontal_probe_filter.UpdatePipeline(time=final_time)
            
            horizontal_output = os.path.join(output_dir, 'horizontal_centerline_manual.csv')
            CreateWriter(horizontal_output, horizontal_probe_filter).UpdatePipeline()
            print(f"   ✅ Saved: horizontal_centerline_manual.csv ({os.path.getsize(horizontal_output)} bytes)")
            
        except Exception as e2:
            print(f"   ❌ Manual horizontal sampling failed: {e2}")
    
    print("\n🎉 Centerline extraction complete!")
    
    # List all files in validation_data
    try:
        files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        print(f"📄 CSV files in {output_dir}:")
        for filename in sorted(files):
            filepath = os.path.join(output_dir, filename)
            size = os.path.getsize(filepath)
            print(f"   • {filename}: {size} bytes")
    except Exception as e:
        print(f"Error listing files: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: pvpython fix_centerlines.py /path/to/case/directory")
        sys.exit(1)
    
    extract_centerlines(sys.argv[1])