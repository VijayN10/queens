#!/bin/bash
#SBATCH --job-name=aaa_openfoam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Job script to run a single OpenFOAM AAA case
# Usage: sbatch run_single_case.sh case_000

echo "========================================"
echo "AAA OpenFOAM Simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Get case directory from argument or use default
CASE_DIR=${1:-case_000}

# OpenFOAM installation path (from cavity example)
OPENFOAM_BASHRC="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc"

echo "Case directory: $CASE_DIR"
echo "OpenFOAM bashrc: $OPENFOAM_BASHRC"
echo ""

# Check if OpenFOAM bashrc exists
if [ ! -f "$OPENFOAM_BASHRC" ]; then
    echo "ERROR: OpenFOAM bashrc not found at $OPENFOAM_BASHRC"
    exit 1
fi

# Source OpenFOAM environment
echo "Loading OpenFOAM environment..."
source "$OPENFOAM_BASHRC"

# Navigate to case directory
cd "$SLURM_SUBMIT_DIR/openfoam_cases/$CASE_DIR" || exit 1
echo "Working directory: $(pwd)"
echo ""

# Step 1: blockMesh
echo "========================================"
echo "Step 1: Running blockMesh..."
echo "========================================"
blockMesh > log.blockMesh 2>&1
if [ $? -eq 0 ]; then
    echo "✓ blockMesh completed successfully"
else
    echo "✗ blockMesh failed - check log.blockMesh"
    exit 1
fi
echo ""

# Step 2: surfaceFeatures
echo "========================================"
echo "Step 2: Running surfaceFeatures..."
echo "========================================"
surfaceFeatures > log.surfaceFeatures 2>&1
if [ $? -eq 0 ]; then
    echo "✓ surfaceFeatures completed successfully"
else
    echo "⚠ surfaceFeatures failed - may not be critical"
fi
echo ""

# Step 3: snappyHexMesh
echo "========================================"
echo "Step 3: Running snappyHexMesh..."
echo "========================================"
echo "This may take 5-30 minutes..."
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
if [ $? -eq 0 ]; then
    echo "✓ snappyHexMesh completed successfully"
else
    echo "✗ snappyHexMesh failed - check log.snappyHexMesh"
    exit 1
fi
echo ""

# Step 4: checkMesh
echo "========================================"
echo "Step 4: Running checkMesh..."
echo "========================================"
checkMesh > log.checkMesh 2>&1
if [ $? -eq 0 ]; then
    echo "✓ checkMesh completed successfully"
else
    echo "⚠ checkMesh reported issues - check log.checkMesh"
fi
echo ""

# Step 5: Solver (optional - comment out if too slow for testing)
# Uncomment the following lines to run the solver
echo "========================================"
echo "Step 5: Solver (SKIPPED for testing)"
echo "========================================"
echo "To run solver, uncomment lines in job script"
# echo "Running pimpleFoam..."
# pimpleFoam > log.pimpleFoam 2>&1
# if [ $? -eq 0 ]; then
#     echo "✓ pimpleFoam completed successfully"
# else
#     echo "✗ pimpleFoam failed - check log.pimpleFoam"
#     exit 1
# fi
echo ""

echo "========================================"
echo "Job completed successfully!"
echo "========================================"
echo "Results in: $SLURM_SUBMIT_DIR/openfoam_cases/$CASE_DIR"
echo "Logs: log.blockMesh, log.snappyHexMesh, log.checkMesh"
