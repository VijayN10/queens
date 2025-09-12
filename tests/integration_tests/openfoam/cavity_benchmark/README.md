# OpenFOAM Cavity Flow Benchmark Cases

## Directory Structure

```
cavity_benchmark/
├── mesh_convergence/          # Mesh independence study
│   ├── coarse_40x40/         # 40×40 mesh, Re=1000
│   ├── medium_80x80/         # 80×80 mesh, Re=1000  
│   ├── fine_120x120/         # 120×120 mesh, Re=1000
│   └── extra_fine_160x160/   # 160×160 mesh, Re=1000
├── benchmark_validation/      # Reynolds number validation
│   ├── Re_1000/              # Re=1000, 120×120 mesh
│   ├── Re_1500/              # Re=1500, 120×120 mesh
│   └── Re_2000/              # Re=2000, 120×120 mesh
├── results/                   # Extracted data and plots
├── analysis_scripts/          # Python analysis tools
└── README.md                  # This file
```

## Usage

1. **Run individual cases:**
   ```bash
   cd cavity_benchmark/mesh_convergence/fine_120x120
   blockMesh
   icoFoam
   ```

2. **Extract centerline profiles:**
   Use the sample utility with appropriate sampleDict

3. **Analysis:**
   Use the provided Python scripts for comparison with ACE Numerics data

## Key Parameters

- **Domain:** Unit square (1×1×0.1)
- **Solver:** icoFoam (transient, incompressible, laminar)
- **Boundary conditions:** Moving lid (U=1 m/s), no-slip walls
- **Mesh:** Graded in y-direction for boundary layer resolution
- **Reynolds numbers:** Re = U×L/ν = 1×1/ν

## Expected Results

- **Mesh convergence:** Should converge by 120×120 grid
- **Benchmark validation:** Match ACE Numerics data within <2% error
