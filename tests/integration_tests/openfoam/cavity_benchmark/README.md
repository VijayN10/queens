# OpenFOAM Cavity Flow Benchmark Cases

## Directory Structure

```
cavity_benchmark/
├── mesh_convergence/          # Mesh independence study
│   ├── mesh_64x64/           # 64×64 mesh, Re=1000
│   ├── mesh_128x128/         # 128×128 mesh, Re=1000
│   ├── mesh_256x256/         # 256×256 mesh, Re=1000
│   ├── mesh_512x512/         # 512×512 mesh, Re=1000
│   └── mesh_1000x1000/       # 1000×1000 mesh, Re=1000
├── benchmark_validation/      # Reynolds number validation
│   ├── Re_1000/              # Re=1000, 256×256 mesh
│   └── Re_2500/              # Re=2500, 256×256 mesh
├── results/                   # Extracted data and plots
├── analysis_scripts/          # Python analysis tools
└── README.md                  # This file
```

## Usage

1. **Run individual cases:**
   ```bash
   cd cavity_benchmark/mesh_convergence/mesh_256x256
   blockMesh
   icoFoam
   ```

2. **Run benchmark validation cases:**
   ```bash
   cd cavity_benchmark/benchmark_validation/Re_1000
   blockMesh
   icoFoam
   ```

3. **Extract centerline profiles:**
   Use the sample utility with appropriate sampleDict for post-processing

4. **Analysis:**
   Use the provided Python scripts for comparison with ACE Numerics data
   ```bash
   cd cavity_benchmark/mesh_convergence
   python mesh_convergence_plots.py
   ```

## Key Parameters

- **Domain:** Unit square (1×1×0.1)
- **Solver:** icoFoam (transient, incompressible, laminar)
- **Boundary conditions:** Moving lid (U=1 m/s), no-slip walls
- **Mesh:** Graded in y-direction for boundary layer resolution
- **Reynolds numbers:** Re = U×L/ν = 1×1/ν

## Test Cases

### Mesh Convergence Study
- **Purpose:** Determine grid-independent solution
- **Reynolds number:** Re = 1000 (fixed)
- **Mesh resolutions:** 64×64, 128×128, 256×256, 512×512, 1000×1000
- **Expected result:** Convergence by 256×256 grid

### Benchmark Validation  
- **Purpose:** Validate against ACE Numerics benchmark data
- **Mesh resolution:** 256×256 (converged grid)
- **Reynolds numbers:** 
  - Re = 1000 (primary validation case)
  - Re = 2500 (additional validation - matches available ACE data)
- **Expected result:** Match ACE Numerics data within <2% RMS error

## Benchmark Data Source

Validation against ACE Numerics benchmark data for driven cavity flow:
- **Re = 1000:** Primary validation with comprehensive centerline velocity profiles
- **Re = 2500:** Additional validation point available in ACE database
- **Comparison metrics:** Centerline u-velocity (vertical line at x=0.5) and v-velocity (horizontal line at y=0.5)

## Expected Results

- **Mesh convergence:** Grid independence demonstrated by 256×256 resolution
- **Validation accuracy:** RMS error < 2% compared to ACE Numerics benchmark
- **Flow features:** Accurate capture of primary vortex structure and boundary layer profiles