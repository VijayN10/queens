# QUEENS-AORTA OpenFOAM Integration

Complete workflow for uncertainty quantification of AAA geometries with OpenFOAM CFD simulations.

## Overview

This integration combines:
- **QUEENS**: Uncertainty quantification framework
- **AORTA**: AAA geometry generation (Method 4)
- **OpenFOAM**: CFD simulations of blood flow

## Architecture

### Static vs Dynamic Components

#### Dynamic (Generated per geometry):
- **Geometry files**: `inlet.stl`, `outlet.stl`, `wall.stl`
- **blockMeshDict**: Bounding box based on geometry size
- **snappyHexMeshDict**: Interior point location
- Complete OpenFOAM case structure

#### Static (Same for all cases):
- **System files**: `controlDict`, `fvSchemes`, `fvSolution`, `decomposeParDict`
- **Constant files**: `transportProperties`, `turbulenceProperties`
- **Initial conditions**: `0/U`, `0/p` (can be parameterized later)

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    QUEENS Framework                         │
│  • Latin Hypercube Sampling                                 │
│  • Parameter distributions                                   │
│  • Uncertainty quantification                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              AORTA Geometry Model                           │
│  • Generate AAA geometries                                  │
│  • Convex hull validation                                   │
│  • Create complete OpenFOAM cases                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           OpenFOAM Case Runner                              │
│  • blockMesh (background mesh)                              │
│  • surfaceFeatures (geometry features)                      │
│  • snappyHexMesh (body-fitted mesh)                         │
│  • Solver (pimpleFoam for blood flow)                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            Post-processing (Future)                         │
│  • Extract WSS, pressure, velocity                          │
│  • Data processors for QUEENS                               │
│  • Uncertainty quantification results                       │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
queens_aaa_workflow/
├── run_full_integration.py       # Main integration script
├── run_openfoam_cases.py          # Standalone OpenFOAM runner
├── aorta_simple_wrapper.py        # AORTA-QUEENS interface
├── queens_aaa_config.py           # Parameter distributions
│
├── geometries/                    # Generated geometries
│   ├── case_000/
│   │   ├── inlet.stl
│   │   ├── outlet.stl
│   │   ├── wall.stl
│   │   └── validation_results.json
│   └── validation_summary.json
│
├── openfoam_cases/                # Complete OpenFOAM cases
│   └── case_000/
│       ├── 0/                     # Initial conditions
│       │   ├── U
│       │   └── p
│       ├── system/                # Simulation settings
│       │   ├── controlDict
│       │   ├── fvSchemes
│       │   ├── fvSolution
│       │   ├── blockMeshDict     # Generated per geometry
│       │   └── snappyHexMeshDict # Generated per geometry
│       ├── constant/              # Physical properties
│       │   ├── transportProperties
│       │   ├── turbulenceProperties
│       │   └── triSurface/       # Geometry files
│       │       ├── inlet.stl
│       │       ├── outlet.stl
│       │       └── wall.stl
│       └── log.*                  # Simulation logs (after running)
│
└── queens_output/                 # QUEENS results
    └── aorta_integration_test/
```

## Usage

### 1. Generate Geometries and OpenFOAM Cases

```bash
cd /home/a11evina/queens/queens_aaa_workflow
python run_full_integration.py
```

This will:
- Generate 5 AAA geometries using Latin Hypercube Sampling
- Validate parameters against convex hull
- Create complete OpenFOAM cases for each geometry
- Save all outputs to respective directories

**Output:**
- `geometries/case_000` to `case_004`: STL files
- `openfoam_cases/case_000` to `case_004`: Complete OpenFOAM cases
- `queens_output/`: QUEENS sampling results

### 2. Run OpenFOAM Simulations

#### Option A: Via Integration Script

Edit `run_full_integration.py`:
```python
run_simulations = True  # Change from False to True
```

Then run:
```bash
python run_full_integration.py
```

#### Option B: Standalone Runner (Recommended)

**Test meshing only (fast):**
```bash
python run_openfoam_cases.py openfoam_cases/ --skip-solver
```

**Run complete simulation (slow):**
```bash
python run_openfoam_cases.py openfoam_cases/
```

**Parallel simulation (for production):**
```bash
python run_openfoam_cases.py openfoam_cases/ --parallel --num-procs 16
```

#### Option C: Manual Execution

```bash
cd openfoam_cases/case_000

# Source OpenFOAM environment
source /opt/openfoam9/etc/bashrc

# Generate background mesh
blockMesh

# Extract surface features
surfaceFeatures

# Generate body-fitted mesh
snappyHexMesh -overwrite

# Check mesh quality
checkMesh

# Run solver
pimpleFoam  # or your preferred solver
```

#### Option D: SLURM Cluster Submission

Create job script `run_case.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=aaa_cfd
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00

# Load OpenFOAM
source /opt/openfoam9/etc/bashrc

# Navigate to case
cd $SLURM_SUBMIT_DIR

# Mesh generation
blockMesh
surfaceFeatures
snappyHexMesh -overwrite
checkMesh

# Decompose for parallel run
decomposePar

# Run solver in parallel
mpirun -np 16 pimpleFoam -parallel

# Reconstruct solution
reconstructPar
```

Submit:
```bash
cd openfoam_cases/case_000
sbatch run_case.sh
```

## Configuration

### Execution Modes

The runner supports different execution modes:

**Serial (for testing):**
```python
runner = OpenFOAMCaseRunner(
    parallel=False,
    num_procs=1,
    run_solver=False  # Skip solver for quick testing
)
```

**Parallel (for production):**
```python
runner = OpenFOAMCaseRunner(
    parallel=True,
    num_procs=16,  # Match decomposeParDict
    run_solver=True
)
```

### OpenFOAM Solver Settings

Edit `openfoam_cases/case_000/system/controlDict` to configure:
- Time step
- End time
- Write frequency
- etc.

Current settings (from base case):
- Solver: `pimpleFoam` (transient, incompressible)
- Time: Cardiac cycle based
- Turbulence: k-omega SST (typical for blood flow)

## Parameters

The workflow uses 4 geometric parameters (in mm):

1. **Neck Diameter 1** (`neck_diameter_1`): Proximal neck diameter
2. **Neck Diameter 2** (`neck_diameter_2`): Distal neck diameter
3. **Max Diameter** (`max_diameter`): Maximum aneurysm diameter
4. **Distal Diameter** (`distal_diameter`): Distal segment diameter

Distributions are loaded from fitted data or use default ranges.

## Convex Hull Validation

Each generated geometry is validated against physiologically realistic parameter ranges:

- **Method**: Point-in-convex-hull check
- **Data**: Based on clinical AAA database
- **Validation**: Checks parameter pairs (neck1-neck2, neck2-max, max-distal)
- **Output**: `validation_results.json` in each case directory

## Performance Considerations

### Meshing Time (per case):
- `blockMesh`: ~1 second
- `surfaceFeatures`: ~5 seconds
- `snappyHexMesh`: ~5-30 minutes (depends on refinement)

### Solver Time (per case):
- Serial: Hours to days (depends on mesh size, time steps)
- Parallel (16 cores): ~6-12 hours for typical AAA case

### Recommendations:
1. **Test first**: Run with `--skip-solver` to verify mesh generation
2. **Use parallel**: Set `parallel=True` and `num_procs=16` for production
3. **Use cluster**: Submit to SLURM for long-running cases
4. **Monitor**: Check log files for convergence issues

## Next Steps

### 1. Post-processing Integration
Convert existing ParaView scripts to QUEENS data processors:
- Wall Shear Stress (WSS)
- Time-Averaged WSS (TAWSS)
- Oscillatory Shear Index (OSI)
- Pressure drop
- Velocity profiles

### 2. Uncertainty Quantification
Once simulations complete:
- Extract quantities of interest (QoI)
- Build surrogate models
- Perform sensitivity analysis
- Quantify uncertainty in hemodynamic indicators

### 3. Parameterize Boundary Conditions
Make inlet velocity/pressure variable:
- Create templates for `0/U` and `0/p`
- Add flow parameters to QUEENS distributions
- Study effect of flow conditions on hemodynamics

### 4. Advanced Analysis
- Multi-fidelity UQ (coarse vs fine meshes)
- Time-dependent analysis
- Rupture risk prediction
- Patient-specific calibration

## Troubleshooting

### OpenFOAM not found
```bash
# Find your OpenFOAM installation
find /opt -name bashrc -path "*/openfoam*/etc/bashrc"

# Update in run_openfoam_cases.py or run_full_integration.py
openfoam_bashrc="/path/to/your/openfoam/etc/bashrc"
```

### Mesh generation fails
- Check `log.blockMesh` and `log.snappyHexMesh`
- Verify STL files are valid: `surfaceCheck wall.stl`
- Adjust mesh parameters in `system/blockMeshDict` or `system/snappyHexMeshDict`

### Solver fails
- Check `log.pimpleFoam` for error messages
- Verify mesh quality: `checkMesh`
- Reduce time step in `system/controlDict`
- Check boundary conditions in `0/U` and `0/p`

### Memory issues
- Reduce mesh refinement in `system/snappyHexMeshDict`
- Use parallel execution to distribute memory load
- Run on cluster with more RAM

## Files

- `run_full_integration.py`: Main workflow script
- `run_openfoam_cases.py`: Standalone OpenFOAM runner
- `aorta_simple_wrapper.py`: AORTA geometry wrapper
- `queens_aaa_config.py`: Parameter configuration
- `README_OPENFOAM_INTEGRATION.md`: This file

## References

- QUEENS: https://github.com/queens-py/queens
- OpenFOAM: https://openfoam.org/
- AORTA Method 4: `../Aorta/ofCaseGen/Method_4/`
