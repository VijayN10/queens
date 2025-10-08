#!/bin/bash
# Submit all OpenFOAM cases to SLURM

echo "Submitting OpenFOAM cases to SLURM..."
echo ""

# Find all case directories
CASES_DIR="openfoam_cases"
CASES=$(ls -d ${CASES_DIR}/case_* 2>/dev/null | xargs -n 1 basename)

if [ -z "$CASES" ]; then
    echo "ERROR: No cases found in $CASES_DIR"
    exit 1
fi

# Count cases
NUM_CASES=$(echo "$CASES" | wc -l)
echo "Found $NUM_CASES cases to submit"
echo ""

# Submit each case
JOB_IDS=""
for CASE in $CASES; do
    echo "Submitting $CASE..."
    JOB_ID=$(sbatch --parsable run_single_case.sh $CASE)
    if [ $? -eq 0 ]; then
        echo "  ✓ Job ID: $JOB_ID"
        JOB_IDS="$JOB_IDS $JOB_ID"
    else
        echo "  ✗ Failed to submit"
    fi
done

echo ""
echo "========================================"
echo "Submitted $NUM_CASES jobs"
echo "========================================"
echo "Job IDs: $JOB_IDS"
echo ""
echo "To check job status:"
echo "  squeue -u $USER"
echo ""
echo "To check specific job:"
echo "  squeue -j <JOB_ID>"
echo ""
echo "To cancel all jobs:"
echo "  scancel $JOB_IDS"
echo ""
echo "Logs will be in: slurm_<JOB_ID>.out and slurm_<JOB_ID>.err"
