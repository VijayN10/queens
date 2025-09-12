#!/bin/bash
# OpenFOAM Cavity Flow Case Setup (Files Only - No Execution)
# Creates all case directories and configuration files

echo "🏗️  Setting up OpenFOAM cavity flow benchmark cases..."
echo "📁 Creating directory structure and files only"
echo ""

# Function to create a complete case
create_case() {
    local case_dir=$1
    local mesh_size=$2
    local reynolds_number=$3
    local case_name=$(basename "$case_dir")
    
    echo "📂 Creating case: $case_name (${mesh_size}×${mesh_size}, Re=$reynolds_number)"
    
    # Create directory structure
    mkdir -p "$case_dir"/{0,constant,system}
    
    # Calculate viscosity for Reynolds number
    # Re = U*L/nu, where U=1, L=1, so nu = 1/Re
    local viscosity=$(echo "scale=8; 1.0 / $reynolds_number" | bc -l 2>/dev/null || echo "0.001")
    
    # === blockMeshDict ===
    cat > "$case_dir/system/blockMeshDict" << EOF
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Cavity dimensions (unit square)
convertToMeters 1.0;

// Mesh resolution: ${mesh_size}×${mesh_size} (Re=$reynolds_number)
vertices
(
    (0 0 0)    // 0
    (1 0 0)    // 1  
    (1 1 0)    // 2
    (0 1 0)    // 3
    (0 0 0.1)  // 4 (front face)
    (1 0 0.1)  // 5
    (1 1 0.1)  // 6
    (0 1 0.1)  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ($mesh_size $mesh_size 1) 
    simpleGrading 
    (
        1                               // x-direction (uniform)
        (                               // y-direction (graded for boundary layers)
            (0.25 0.3 4)               // bottom 25%: 30% cells, expansion 4
            (0.50 0.4 1)               // middle 50%: 40% cells, uniform  
            (0.25 0.3 0.25)            // top 25%: 30% cells, compression 0.25
        )
        1                               // z-direction (uniform)
    )
);

edges
(
);

boundary
(
    movingWall
    {
        type wall;
        faces
        (
            (3 7 6 2)  // top wall (moving lid)
        );
    }
    fixedWalls
    {
        type wall;
        faces
        (
            (0 4 7 3)  // left wall
            (2 6 5 1)  // right wall
            (1 5 4 0)  // bottom wall
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 3 2 1)  // back face
            (4 5 6 7)  // front face
        );
    }
);

mergePatchPairs
(
);

// ************************************************************************* //
EOF

    # === transportProperties ===
    cat > "$case_dir/constant/transportProperties" << EOF
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Transport model
transportModel  Newtonian;

// Kinematic viscosity for Re = $reynolds_number
// Re = U*L/nu, where U=1 m/s, L=1 m
// Therefore: nu = U*L/Re = 1/$reynolds_number = $viscosity
nu              [0 2 -1 0 0 0 0] $viscosity;

// ************************************************************************* //
EOF

    # === controlDict ===
    cat > "$case_dir/system/controlDict" << EOF
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     icoFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         3.0;        // Sufficient time for convergence

deltaT          0.005;      // Small time step for stability

writeControl    timeStep;
writeInterval   200;        // Write every 1.0s
purgeWrite      3;          // Keep only last 3 time directories

writeFormat     ascii;
writePrecision  8;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
EOF

    # === fvSchemes ===
    cat > "$case_dir/system/fvSchemes" << EOF
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss upwind;    // Upwind for stability
}

laplacianSchemes
{
    default         Gauss linear orthogonal;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         orthogonal;
}

// ************************************************************************* //
EOF

    # === fvSolution ===
    cat > "$case_dir/system/fvSolution" << EOF
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-08;
        relTol          0.05;
    }

    pFinal
    {
        \$p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}

// ************************************************************************* //
EOF

    # === Initial condition: U field ===
    cat > "$case_dir/0/U" << EOF
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);  // Lid velocity = 1 m/s
    }
    
    fixedWalls
    {
        type            noSlip;
    }
    
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
EOF

    # === Initial condition: p field ===
    cat > "$case_dir/0/p" << EOF
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }
    
    fixedWalls
    {
        type            zeroGradient;
    }
    
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
EOF

    echo "   ✅ $case_name complete (Re=$reynolds_number, ${mesh_size}×${mesh_size})"
}

# Main execution
echo "📁 Creating base directory structure..."
mkdir -p cavity_benchmark/{mesh_convergence,benchmark_validation,results,analysis_scripts}

echo ""
echo "🔬 Creating mesh convergence study cases..."
echo "============================================"

# Mesh convergence cases (all at Re=1000)
create_case "cavity_benchmark/mesh_convergence/coarse_40x40" 40 1000
create_case "cavity_benchmark/mesh_convergence/medium_80x80" 80 1000  
create_case "cavity_benchmark/mesh_convergence/fine_120x120" 120 1000
create_case "cavity_benchmark/mesh_convergence/extra_fine_160x160" 160 1000

echo ""
echo "🎯 Creating Reynolds number validation cases..."
echo "=============================================="

# Reynolds validation cases (all at converged mesh 120×120)
create_case "cavity_benchmark/benchmark_validation/Re_1000" 120 1000
create_case "cavity_benchmark/benchmark_validation/Re_1500" 120 1500
create_case "cavity_benchmark/benchmark_validation/Re_2000" 120 2000

echo ""
echo "📊 Creating analysis directory structure..."
mkdir -p cavity_benchmark/analysis_scripts
mkdir -p cavity_benchmark/results

# Create a README file
cat > cavity_benchmark/README.md << EOF
# OpenFOAM Cavity Flow Benchmark Cases

## Directory Structure

\`\`\`
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
\`\`\`

## Usage

1. **Run individual cases:**
   \`\`\`bash
   cd cavity_benchmark/mesh_convergence/fine_120x120
   blockMesh
   icoFoam
   \`\`\`

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
EOF

echo ""
echo "🎉 Case setup completed successfully!"
echo ""
echo "📋 Summary:"
echo "   • 7 complete OpenFOAM cases created"
echo "   • 4 mesh convergence cases (40×40 to 160×160)"  
echo "   • 3 Reynolds validation cases (Re=1000,1500,2000)"
echo "   • All configuration files ready"
echo ""
echo "📁 Location: $(pwd)/cavity_benchmark/"
echo ""
echo "🚀 Next steps:"
echo "   1. Navigate to any case directory"
echo "   2. Run: blockMesh && icoFoam"
echo "   3. Use analysis scripts for comparison"
echo ""
echo "💡 Tips:"
echo "   • Check mesh quality with: checkMesh"
echo "   • Monitor convergence with: tail -f log.icoFoam"
echo "   • Extract profiles with: postProcess -func sample"