# Combined AORTA CFD Driver for QUEENS

## Overview

This document explains the **combined driver approach** where geometry generation and OpenFOAM simulation are integrated into a single QUEENS driver.

## Architecture Comparison

### Original Approach (Separated)
```
Parameters → Iterator → Scheduler → Driver (AortaGeometryDriver)
                                    └─> Geometry generation only

[Then separately]
OpenFOAMCaseRunner.run_multiple_cases(...)
```

### New Approach (Combined)
```
Parameters → Iterator → Scheduler → Driver (AortaCFDDriver)
                                    ├─> Geometry generation
                                    ├─> OpenFOAM case setup
                                    ├─> Mesh generation
                                    ├─> CFD solver
                                    └─> Result extraction
```

## Key Differences

| Aspect | Separated | Combined |
|--------|-----------|----------|
| **Driver** | AortaGeometryDriver + OpenFOAMCaseRunner | AortaCFDDriver |
| **Workflow** | Two-stage (geometry then simulation) | Single-stage (complete pipeline) |
| **Parallelism** | Can parallelize stages independently | Parallelizes complete pipelines |
| **Flexibility** | Can run geometry without simulation | Must run complete pipeline |
| **Testing** | Easy to test stages independently | Tests entire pipeline at once |
| **Resource Usage** | Optimized per stage | Fixed per job |

## Files

- **`aorta_cfd_driver.py`**: Combined driver implementing full CFD workflow
- **`run_combined_cfd_workflow.py`**: Example workflow script using combined driver
- **`aorta_queens_driver.py`**: Original geometry-only driver (still available)
- **`run_openfoam_cases.py`**: Standalone OpenFOAM runner (still available)

## Usage

### Basic Usage

```python
from aorta_cfd_driver import AortaCFDDriver
from queens.parameters import Parameters
from queens.models import Simulation
from queens.schedulers import Pool

# Create combined driver
driver = AortaCFDDriver(
    parameters=parameters,
    geometry_output_dir='./geometries',
    openfoam_cases_dir='./openfoam_cases',
    # Geometry settings
    enable_morphing=False,
    # OpenFOAM settings
    solver="pimpleFoam",
    parallel=False,
    run_blockmesh=True,
    run_snappyhexmesh=True,
    run_solver=True,  # Set to True to run CFD solver
    # Result extraction
    extract_results=True
)

# Create QUEENS workflow
scheduler = Pool(experiment_name="aorta_cfd", num_jobs=4)
model = Simulation(scheduler=scheduler, driver=driver)

# Run with iterator
iterator = LatinHypercubeSampling(model=model, parameters=parameters, ...)
iterator.run()
```

### Running the Example

```bash
# Quick test (meshing only, no solver)
python run_combined_cfd_workflow.py

# Full CFD simulation (edit script to set run_solver=True)
# Then run:
python run_combined_cfd_workflow.py
```

## Combined Driver Features

### Pipeline Stages

The `AortaCFDDriver.run()` method executes these stages sequentially:

1. **Geometry Generation**: Creates parametric AAA geometry from input parameters
2. **Case Setup**: Creates complete OpenFOAM case structure
3. **Meshing**: Runs blockMesh, surfaceFeatures, snappyHexMesh, checkMesh
4. **Solver**: Runs CFD solver (serial or parallel)
5. **Result Extraction**: Extracts quantities of interest (QoI)

### Configuration Options

#### Geometry Settings
- `geometry_output_dir`: Where to save geometry STL files
- `openfoam_cases_dir`: Where to create OpenFOAM cases
- `enable_morphing`: Apply morphological perturbations
- `convex_hull_metadata_path`: Enable parameter validation

#### OpenFOAM Settings
- `openfoam_bashrc`: Path to OpenFOAM environment
- `solver`: CFD solver to use (default: `pimpleFoam`)
- `parallel`: Run solver in parallel with MPI
- `num_procs_sim`: Number of processors for parallel runs
- `run_blockmesh`: Enable/disable blockMesh
- `run_snappyhexmesh`: Enable/disable snappyHexMesh
- `run_solver`: Enable/disable CFD solver execution

#### Result Extraction
- `extract_results`: Extract results after simulation
- `result_fields`: Fields to extract (e.g., `['U', 'p']`)
- `result_time`: Time step to extract (e.g., `'latestTime'`)

### Customizing Result Extraction

The `_extract_results()` method is a **placeholder** that you should customize based on your quantities of interest (QoI):

```python
def _extract_results(self, of_case_dir):
    """Extract your QoI here."""
    results = {}

    # Example: Extract pressure drop
    # Use OpenFOAM postProcess utilities or parse field files

    # Option 1: Use OpenFOAM postProcess
    # self._run_openfoam_command("postProcess -func 'patchAverage(name=inlet,field=p)'")

    # Option 2: Parse field files directly
    # p_inlet = self._read_field_value(of_case_dir / 'postProcessing/...')

    # Option 3: Use QUEENS DataProcessor (ParaView)
    # results = self.data_processor.process(of_case_dir)

    results['pressure_drop'] = 0.0  # Replace with actual calculation
    results['max_wall_shear_stress'] = 0.0

    return results
```

### Output Files

Each job creates:
- `geometries/case_XXX/`: Geometry STL files (inlet.stl, wall.stl, outlet.stl)
- `openfoam_cases/case_XXX/`: Complete OpenFOAM case with mesh and results
- `<experiment_dir>/XXX/pipeline_results.json`: Complete pipeline status and results
- `<experiment_dir>/XXX/validation_results.json`: Parameter validation (if enabled)

## When to Use Combined vs Separated

### Use Combined Driver When:
✅ You want a **single entry point** for the complete workflow
✅ Your supervisor/team prefers **unified interface**
✅ You always want to run the **complete pipeline** together
✅ You're setting up **production workflows** with fixed stages
✅ You want **automated end-to-end** execution

### Use Separated Drivers When:
✅ You need **stage-wise optimization** (different parallelism for geometry vs simulation)
✅ You want to **test/debug** geometry generation without running simulations
✅ You need **checkpointing** (restart from meshing if solver fails)
✅ You want **flexible workflows** (generate 1000 geometries, simulate best 10)
✅ You're in **development/research** mode with frequent changes

## Performance Considerations

### Combined Approach
- **Pro**: Simple to use, automatic dependency handling
- **Pro**: All outputs in one place
- **Con**: Long-running jobs (geometry + meshing + solver)
- **Con**: Cannot optimize parallelism per stage
- **Con**: Wastes resources if geometry generation fails

### Separated Approach
- **Pro**: Fast geometry generation phase (100s of geometries in minutes)
- **Pro**: Can run meshing with different resource requirements
- **Pro**: Can checkpoint and restart from failed stages
- **Con**: More complex workflow scripts
- **Con**: Need to manage intermediate outputs

## HPC Cluster Usage

### Combined Driver with SLURM

```python
from queens.schedulers import SlurmScheduler

scheduler = SlurmScheduler(
    experiment_name="aorta_cfd",
    num_jobs=50,
    num_procs=4,  # Processors per job
    queue="normal",
    walltime="04:00:00",  # 4 hours per complete pipeline
    verbose=True
)

model = Simulation(scheduler=scheduler, driver=driver)
```

### Resource Estimation
- Geometry generation: ~30 seconds (CPU-light)
- blockMesh: ~1 minute
- snappyHexMesh: ~10-30 minutes (CPU-intensive)
- Solver (100 time steps): ~1-4 hours (CPU-intensive)

**Total per job**: ~1.5-5 hours depending on mesh size and solver settings

## Recommendations

1. **For Development**: Use separated approach ([run_full_integration_queens_proper.py](run_full_integration_queens_proper.py))
   - Faster iteration
   - Easy debugging
   - Flexible testing

2. **For Production**: Use combined approach ([run_combined_cfd_workflow.py](run_combined_cfd_workflow.py))
   - Cleaner interface
   - Automated workflow
   - Better for sharing with collaborators

3. **For HPC**: Use combined approach with SlurmScheduler
   - One job = one complete CFD pipeline
   - Easy resource allocation
   - Simple job monitoring

## Example: Comparison Side-by-Side

### Separated Workflow
```python
# Phase 1: Generate geometries (fast, parallel)
driver1 = AortaGeometryDriver(...)
model1 = Simulation(scheduler=Pool(num_jobs=20), driver=driver1)
iterator1 = LatinHypercubeSampling(model=model1, num_samples=100)
iterator1.run()  # Generates 100 geometries in parallel

# Phase 2: Run simulations (slow, limited parallelism)
runner = OpenFOAMCaseRunner(...)
runner.run_multiple_cases('./openfoam_cases')  # Run sequentially or batch
```

### Combined Workflow
```python
# Single phase: Complete pipeline (parallel)
driver = AortaCFDDriver(...)
model = Simulation(scheduler=Pool(num_jobs=4), driver=driver)
iterator = LatinHypercubeSampling(model=model, num_samples=100)
iterator.run()  # Runs 4 complete pipelines at a time until all 100 are done
```

## Conclusion

Both approaches are valid QUEENS patterns:
- **Combined**: Matches your supervisor's request for "one driver/wrapper"
- **Separated**: Matches QUEENS best practices for flexibility

Choose based on your use case and team preferences.
