#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 32

module load openfoam/9-foss-2021a
source $FOAM_BASH



blockMesh
#foamToVTK -time 0 -constant -noZero
#mv VTK VTK_blockMesh

foamDictionary -entry "method" -set "scotch" system/decomposeParDict
foamDictionary -entry 'numberOfSubdomains' -set "32" system/decomposeParDict
decomposePar -noZero -force > decomposePar.log 2>&1

checkMesh > checkMesh.output 2>&1

decomposePar -force
mpirun -np 32 icoFoam -parallel > pisoFoam.log 2>&1
reconstructPar

rm -rf processor*