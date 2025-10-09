#!/usr/bin/env python3
"""
Combined QUEENS Driver for AORTA AAA: Geometry Generation + OpenFOAM Simulation

This driver integrates the complete CFD workflow:
1. Generates parametric AAA geometry
2. Creates OpenFOAM case
3. Runs OpenFOAM simulation (blockMesh, snappyHexMesh, solver)
4. Returns simulation results

Each call to run() executes the FULL pipeline for one parameter sample.
"""

import sys
import os
import numpy as np
from pathlib import Path
import json
import subprocess
import shutil

# QUEENS imports
sys.path.insert(0, '/home/a11evina/queens/src')
from queens.drivers._driver import Driver
from queens.utils.logger_settings import log_init_args

# AORTA imports
repo_root = Path(__file__).parent.parent
aorta_dir = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'

# Save current directory
original_dir = os.getcwd()
os.chdir(str(aorta_dir))

sys.path.insert(0, str(aorta_dir))
from config import ConfigParams
from main import generate_base_geometry_without_perturbation, apply_perturbation
from src.vesselGen.save_stl_from_patches import save_stl_from_patches

# Convex hull validation
sys.path.insert(0, str(aorta_dir / 'data'))
from data_bound_with_morphed_data import is_point_inside_hull

# Return to original directory
os.chdir(original_dir)


class AortaCFDDriver(Driver):
    """
    Combined QUEENS driver for complete AORTA CFD workflow.

    This driver handles the FULL pipeline:
    1. Geometry generation from AAA parameters
    2. OpenFOAM case setup
    3. Mesh generation (blockMesh, snappyHexMesh)
    4. CFD simulation (solver)
    5. Result extraction

    Each run() call processes ONE sample through the complete workflow.
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        geometry_output_dir="geometries",
        openfoam_cases_dir="openfoam_cases",
        enable_morphing=False,
        convex_hull_metadata_path=None,
        gender=None,
        age_group=None,
        outlier_method='manual',
        # OpenFOAM simulation settings
        openfoam_bashrc="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc",
        solver="pimpleFoam",
        parallel=False,
        num_procs_sim=1,
        run_blockmesh=True,
        run_snappyhexmesh=True,
        run_solver=True,
        # Result extraction settings
        extract_results=True,
        result_fields=['U', 'p'],
        result_time='latestTime'
    ):
        """
        Initialize combined AORTA CFD driver.

        Args:
            parameters (Parameters): QUEENS Parameters object
            geometry_output_dir (str): Directory for geometry STL files
            openfoam_cases_dir (str): Directory for OpenFOAM cases
            enable_morphing (bool): Whether to apply morphological perturbations
            convex_hull_metadata_path (str): Path to convex hull validation metadata
            gender (str): Gender for validation ('M', 'F', 'All')
            age_group (str): Age group for validation (e.g., '70-79')
            outlier_method (str): Outlier detection method ('manual', 'zscore', etc.)

            OpenFOAM simulation settings:
            openfoam_bashrc (str): Path to OpenFOAM bashrc for environment sourcing
            solver (str): OpenFOAM solver to use (default: pimpleFoam)
            parallel (bool): Whether to run solver in parallel with MPI
            num_procs_sim (int): Number of processors for parallel simulation
            run_blockmesh (bool): Whether to run blockMesh
            run_snappyhexmesh (bool): Whether to run snappyHexMesh
            run_solver (bool): Whether to run the CFD solver

            Result extraction settings:
            extract_results (bool): Whether to extract results from simulation
            result_fields (list): Fields to extract (e.g., ['U', 'p'])
            result_time (str): Time step to extract ('latestTime', 'lastTime', or specific time)
        """
        # No input files to copy for geometry generation
        files_to_copy = []

        # Initialize parent Driver class with parameters
        super().__init__(parameters=parameters, files_to_copy=files_to_copy)

        # Geometry settings
        self.geometry_output_dir = Path(geometry_output_dir)
        self.openfoam_cases_dir = Path(openfoam_cases_dir)
        self.enable_morphing = enable_morphing

        # Validation settings
        self.convex_hull_metadata_path = convex_hull_metadata_path
        self.gender = gender
        self.age_group = age_group
        self.outlier_method = outlier_method
        self.convex_hull_data = None

        # OpenFOAM simulation settings
        self.openfoam_bashrc = Path(openfoam_bashrc)
        self.solver = solver
        self.parallel = parallel
        self.num_procs_sim = num_procs_sim
        self.run_blockmesh = run_blockmesh
        self.run_snappyhexmesh = run_snappyhexmesh
        self.run_solver = run_solver

        # Result extraction settings
        self.extract_results = extract_results
        self.result_fields = result_fields
        self.result_time = result_time

        # Create output directories
        self.geometry_output_dir.mkdir(exist_ok=True, parents=True)
        self.openfoam_cases_dir.mkdir(exist_ok=True, parents=True)

        # Load convex hull metadata if provided
        if convex_hull_metadata_path and Path(convex_hull_metadata_path).exists():
            self._load_convex_hull_metadata()

        # Store AORTA directory for context switching
        self.aorta_dir = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'
        self.original_dir = os.getcwd()

        # Verify OpenFOAM installation
        if not self.openfoam_bashrc.exists():
            print(f"⚠️  Warning: OpenFOAM bashrc not found at {self.openfoam_bashrc}")
            self._find_openfoam()

        print(f"✅ AortaCFDDriver initialized (Combined geometry + simulation)")
        print(f"   📁 Geometry output: {self.geometry_output_dir.absolute()}")
        print(f"   📁 OpenFOAM cases: {self.openfoam_cases_dir.absolute()}")
        print(f"   🌊 Solver: {self.solver} (parallel={self.parallel}, procs={self.num_procs_sim})")
        print(f"   🔧 Pipeline: blockMesh={self.run_blockmesh}, snappyHexMesh={self.run_snappyhexmesh}, solver={self.run_solver}")
        if self.convex_hull_data:
            print(f"   ✅ Convex hull validation: {gender}, {age_group}, {outlier_method}")

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """
        Execute FULL CFD workflow for one parameter sample (QUEENS Driver interface).

        Pipeline:
        1. Generate AAA geometry
        2. Create OpenFOAM case
        3. Run blockMesh
        4. Run snappyHexMesh
        5. Run CFD solver
        6. Extract results

        Args:
            sample (np.array): Parameter array [neck_d1, neck_d2, max_d, distal_d] in mm
            job_id (int): Unique job ID from QUEENS scheduler
            num_procs (int): Number of processors (used for parallel simulation if parallel=True)
            experiment_dir (Path): QUEENS experiment directory
            experiment_name (str): QUEENS experiment name

        Returns:
            dict or float: Simulation results (e.g., pressure drop, wall shear stress) or success indicator
        """
        case_id = f"case_{job_id:03d}"

        print(f"\n{'='*70}")
        print(f"🚀 Job {job_id}: FULL CFD Pipeline for {case_id}")
        print(f"   Parameters: neck_d1={sample[0]:.1f}, neck_d2={sample[1]:.1f}, "
              f"max_d={sample[2]:.1f}, distal_d={sample[3]:.1f} mm")
        print(f"{'='*70}")

        # Validate parameters if convex hull is enabled
        if self.convex_hull_data:
            validation_results = self._validate_parameters(sample)
            if validation_results['all_valid']:
                print(f"   ✅ Parameters inside convex hull boundaries")
            else:
                print(f"   ⚠️  Parameters OUTSIDE convex hull boundaries")
                for comparison, result in validation_results.items():
                    if comparison != 'all_valid' and isinstance(result, dict) and not result.get('valid'):
                        print(f"      ❌ {comparison}")

            # Save validation results
            self._save_validation_results(experiment_dir, job_id, sample, validation_results)

        pipeline_results = {
            'case_id': case_id,
            'job_id': int(job_id),
            'parameters': {
                'neck_diameter_1': float(sample[0]),
                'neck_diameter_2': float(sample[1]),
                'max_diameter': float(sample[2]),
                'distal_diameter': float(sample[3])
            },
            'success': False,
            'stages_completed': []
        }

        try:
            # ========================================================================
            # STAGE 1: Generate Geometry
            # ========================================================================
            print(f"\n📐 Stage 1/5: Generating AAA geometry...")
            geometry_dir = self._generate_geometry(sample, case_id)
            pipeline_results['stages_completed'].append('geometry')
            pipeline_results['geometry_dir'] = str(geometry_dir)
            print(f"   ✅ Geometry generated: {geometry_dir}")

            # ========================================================================
            # STAGE 2: Create OpenFOAM Case
            # ========================================================================
            print(f"\n🏗️  Stage 2/5: Creating OpenFOAM case...")
            of_case_dir = self._create_openfoam_case(case_id, geometry_dir)
            pipeline_results['stages_completed'].append('case_setup')
            pipeline_results['openfoam_case_dir'] = str(of_case_dir)
            print(f"   ✅ OpenFOAM case created: {of_case_dir}")

            # ========================================================================
            # STAGE 3: Run Meshing
            # ========================================================================
            print(f"\n🔧 Stage 3/5: Generating mesh...")
            mesh_success = self._run_meshing(of_case_dir)
            if not mesh_success:
                print(f"   ❌ Meshing failed")
                pipeline_results['error'] = 'Meshing failed'
                self._save_pipeline_results(experiment_dir, job_id, pipeline_results)
                return 0.0

            pipeline_results['stages_completed'].append('meshing')
            print(f"   ✅ Mesh generated successfully")

            # ========================================================================
            # STAGE 4: Run CFD Solver
            # ========================================================================
            if self.run_solver:
                print(f"\n🌊 Stage 4/5: Running CFD solver ({self.solver})...")
                print(f"   ⏳ This may take a long time...")

                solver_success = self._run_solver(of_case_dir)
                if not solver_success:
                    print(f"   ❌ Solver failed")
                    pipeline_results['error'] = 'Solver failed'
                    self._save_pipeline_results(experiment_dir, job_id, pipeline_results)
                    return 0.0

                pipeline_results['stages_completed'].append('solver')
                print(f"   ✅ Solver completed successfully")
            else:
                print(f"\n⏭️  Stage 4/5: Solver execution skipped (run_solver=False)")

            # ========================================================================
            # STAGE 5: Extract Results
            # ========================================================================
            if self.extract_results and self.run_solver:
                print(f"\n📊 Stage 5/5: Extracting simulation results...")
                results = self._extract_results(of_case_dir)
                pipeline_results['simulation_results'] = results
                pipeline_results['stages_completed'].append('results')
                print(f"   ✅ Results extracted: {list(results.keys())}")
            else:
                print(f"\n⏭️  Stage 5/5: Result extraction skipped")
                results = {'success': True}

            # ========================================================================
            # Success!
            # ========================================================================
            pipeline_results['success'] = True
            self._save_pipeline_results(experiment_dir, job_id, pipeline_results)

            print(f"\n{'='*70}")
            print(f"✅ Job {job_id}: FULL PIPELINE COMPLETED for {case_id}")
            print(f"   Stages: {' → '.join(pipeline_results['stages_completed'])}")
            print(f"{'='*70}")

            # Return results for QUEENS (can be customized based on QoI)
            if self.extract_results and 'simulation_results' in pipeline_results:
                return pipeline_results['simulation_results']
            else:
                return 1.0  # Success indicator

        except Exception as e:
            print(f"\n❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()

            pipeline_results['error'] = str(e)
            pipeline_results['traceback'] = traceback.format_exc()
            self._save_pipeline_results(experiment_dir, job_id, pipeline_results)

            return 0.0  # Failure indicator

    def _generate_geometry(self, parameters, case_id):
        """Generate AAA geometry from parameters."""
        from src.vesselStats.parameter_sampler import AAAGeometryParams

        # Change to AORTA directory
        os.chdir(str(self.aorta_dir))

        try:
            # Create config
            config = ConfigParams()

            # Override with QUEENS-provided parameters
            config.geometry_params = AAAGeometryParams(
                neck_diameter_1=float(parameters[0]),
                neck_diameter_2=float(parameters[1]),
                max_diameter=float(parameters[2]),
                distal_diameter=float(parameters[3])
            )

            # Regenerate anatomical points
            config.anatomical_points = config.anatomical_template.generate_points(config.geometry_params)

            # Generate base geometry
            geometry = generate_base_geometry_without_perturbation(config)

            # Apply morphing if enabled
            if self.enable_morphing:
                geometry = apply_perturbation(geometry, config)

            # Change back to original directory before saving
            os.chdir(self.original_dir)

            # Create case directory
            case_dir = self.geometry_output_dir / case_id
            case_dir.mkdir(exist_ok=True, parents=True)

            # Save STL files
            inlet_file = case_dir / "inlet.stl"
            wall_file = case_dir / "wall.stl"
            outlet_file = case_dir / "outlet.stl"

            save_stl_from_patches(geometry['inlet_patch'], geometry['vertices'], str(inlet_file))
            save_stl_from_patches(geometry['wall_patch'], geometry['vertices'], str(wall_file))
            save_stl_from_patches(geometry['outlet_patch'], geometry['vertices'], str(outlet_file))

            return case_dir

        finally:
            os.chdir(self.original_dir)

    def _create_openfoam_case(self, case_id, geometry_dir):
        """Create OpenFOAM case from geometry."""
        from src.ofCaseGen.compute_normals import computeNormals
        from src.ofCaseGen.find_point_inside_stl import findPointInsideSTL
        from src.ofCaseGen.correct_normal import correctNormal
        from src.ofCaseGen.clean_and_repeat_cardiac_cycle import cleanAndRepeatCardiacCycle
        from src.ofCaseGen.convert_velocity_components import convertVelocityComponents
        from src.ofCaseGen.generate_u_file import generateUFile
        from src.ofCaseGen.create_snappy_hex_mesh_dict import createSnappyHexMeshDict
        from src.ofCaseGen.create_block_mesh_dict import createBlockMeshDict

        # Setup paths
        of_case_dir = (Path(self.original_dir) / self.openfoam_cases_dir / case_id).resolve()
        geometry_dir = Path(geometry_dir).resolve()

        # Change to AORTA directory
        os.chdir(str(self.aorta_dir))

        try:
            # Copy base case
            base_case = self.aorta_dir / 'data' / 'input' / 'of_base_case'
            of_case_dir.mkdir(parents=True, exist_ok=True)

            for item in base_case.iterdir():
                if item.is_dir():
                    shutil.copytree(item, of_case_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, of_case_dir / item.name)

            # Copy geometry files to constant/triSurface
            tri_surface_dir = of_case_dir / 'constant' / 'triSurface'
            tri_surface_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(geometry_dir / 'inlet.stl', tri_surface_dir / 'inlet.stl')
            shutil.copy2(geometry_dir / 'wall.stl', tri_surface_dir / 'wall.stl')
            shutil.copy2(geometry_dir / 'outlet.stl', tri_surface_dir / 'outlet.stl')

            # Compute normals and interior point
            normals, avgNormal, referencePoint = computeNormals(str(tri_surface_dir / 'inlet.stl'))
            insidePoint = findPointInsideSTL(str(tri_surface_dir / 'wall.stl'))
            correctedNormal = correctNormal(avgNormal, referencePoint, insidePoint)

            # Process velocity data (if available)
            config = ConfigParams()
            velocity_file = self.aorta_dir / 'data' / 'input' / 'U' / 'velocity_time.csv'
            if velocity_file.exists():
                cleanedData = cleanAndRepeatCardiacCycle(str(velocity_file), config.vessel_settings.num_cycles)

                temp_files_dir = Path('data/output/files')
                temp_files_dir.mkdir(parents=True, exist_ok=True)

                outputCSV = temp_files_dir / 'corrected_velocity_time.csv'
                convertVelocityComponents(cleanedData, str(outputCSV), correctedNormal)
                generateUFile(str(outputCSV))

                u_file_src = temp_files_dir / 'U'
                if u_file_src.exists():
                    shutil.copy2(u_file_src, of_case_dir / '0' / 'U')

            # Create mesh dictionaries
            temp_files_dir = Path('data/output/files')
            temp_files_dir.mkdir(parents=True, exist_ok=True)

            # snappyHexMeshDict
            shm_top = self.aorta_dir / 'data' / 'input' / 'shm' / 'shm_top.txt'
            shm_bottom = self.aorta_dir / 'data' / 'input' / 'shm' / 'shm_bottom.txt'

            if shm_top.exists() and shm_bottom.exists():
                snappy_dict_path = temp_files_dir / 'snappyHexMeshDict'
                createSnappyHexMeshDict(insidePoint, str(shm_top), str(shm_bottom), str(snappy_dict_path))
                shutil.copy2(snappy_dict_path, of_case_dir / 'system' / 'snappyHexMeshDict')

            # blockMeshDict
            wall_stl = tri_surface_dir / 'wall.stl'
            block_dict_path = temp_files_dir / 'blockMeshDict'
            createBlockMeshDict(str(wall_stl), str(block_dict_path))
            shutil.copy2(block_dict_path, of_case_dir / 'system' / 'blockMeshDict')

            return of_case_dir

        finally:
            os.chdir(self.original_dir)

    def _run_meshing(self, of_case_dir):
        """Run OpenFOAM meshing pipeline (blockMesh + snappyHexMesh)."""
        original_dir = os.getcwd()
        os.chdir(of_case_dir)

        try:
            # Step 1: blockMesh
            if self.run_blockmesh:
                print(f"      🔧 Running blockMesh...")
                success = self._run_openfoam_command("blockMesh", log_file="log.blockMesh")
                if not success:
                    print(f"      ❌ blockMesh failed")
                    return False

            # Step 2: surfaceFeatures
            print(f"      🔍 Running surfaceFeatures...")
            success = self._run_openfoam_command("surfaceFeatures", log_file="log.surfaceFeatures")
            if not success:
                print(f"      ⚠️  surfaceFeatures failed (may not be critical)")

            # Step 3: snappyHexMesh
            if self.run_snappyhexmesh:
                print(f"      🎯 Running snappyHexMesh (this may take several minutes)...")
                success = self._run_openfoam_command("snappyHexMesh -overwrite", log_file="log.snappyHexMesh")
                if not success:
                    print(f"      ❌ snappyHexMesh failed")
                    return False

            # Step 4: checkMesh
            print(f"      ✓ Running checkMesh...")
            success = self._run_openfoam_command("checkMesh", log_file="log.checkMesh")
            if not success:
                print(f"      ⚠️  checkMesh reported issues")

            return True

        finally:
            os.chdir(original_dir)

    def _run_solver(self, of_case_dir):
        """Run OpenFOAM CFD solver."""
        original_dir = os.getcwd()
        os.chdir(of_case_dir)

        try:
            if self.parallel and self.num_procs_sim > 1:
                # Parallel execution
                print(f"      🔧 Decomposing for {self.num_procs_sim} processors...")
                success = self._run_openfoam_command("decomposePar", log_file="log.decomposePar")
                if not success:
                    print(f"      ❌ decomposePar failed")
                    return False

                # Run solver in parallel
                print(f"      🌊 Running {self.solver} in parallel...")
                solver_cmd = f"mpirun -np {self.num_procs_sim} {self.solver} -parallel"
                success = self._run_openfoam_command(solver_cmd, log_file=f"log.{self.solver}")
                if not success:
                    print(f"      ❌ {self.solver} failed")
                    return False

                # Reconstruct solution
                print(f"      🔧 Reconstructing parallel solution...")
                success = self._run_openfoam_command("reconstructPar", log_file="log.reconstructPar")
                if not success:
                    print(f"      ⚠️  reconstructPar failed")
                    return False

            else:
                # Serial execution
                print(f"      🌊 Running {self.solver} in serial...")
                success = self._run_openfoam_command(self.solver, log_file=f"log.{self.solver}")
                if not success:
                    print(f"      ❌ {self.solver} failed")
                    return False

            return True

        finally:
            os.chdir(original_dir)

    def _run_openfoam_command(self, command, log_file=None):
        """
        Run OpenFOAM command with proper environment.

        Args:
            command: OpenFOAM command to run
            log_file: Optional log file path

        Returns:
            bool: True if command succeeded
        """
        shell_cmd = f"source {self.openfoam_bashrc} && {command}"

        if log_file:
            shell_cmd += f" > {log_file} 2>&1"

        try:
            result = subprocess.run(
                shell_cmd,
                shell=True,
                executable="/bin/bash",
                check=False,
                capture_output=(log_file is None),
                text=True
            )
            return result.returncode == 0

        except Exception as e:
            print(f"      ❌ Error running command: {e}")
            return False

    def _extract_results(self, of_case_dir):
        """
        Extract simulation results (e.g., pressure drop, wall shear stress, velocity).

        This is a placeholder - customize based on your QoI (Quantities of Interest).

        Returns:
            dict: Extracted results
        """
        results = {
            'case_dir': str(of_case_dir),
            'success': True
        }

        # Example: Extract pressure drop (inlet to outlet)
        # You can use OpenFOAM postProcessing utilities or parse field files

        # Placeholder values - replace with actual extraction logic
        results['pressure_drop'] = 0.0  # Pa
        results['max_wall_shear_stress'] = 0.0  # Pa
        results['max_velocity'] = 0.0  # m/s

        # TODO: Implement actual result extraction
        # Options:
        # 1. Use OpenFOAM postProcess utility
        # 2. Parse field files directly
        # 3. Use ParaView Python API
        # 4. Use QUEENS DataProcessor

        return results

    def _find_openfoam(self):
        """Try to find OpenFOAM installation."""
        common_paths = [
            "/opt/openfoam9/etc/bashrc",
            "/opt/openfoam/etc/bashrc",
            "/usr/lib/openfoam/openfoam9/etc/bashrc",
            Path.home() / "OpenFOAM" / "OpenFOAM-9" / "etc" / "bashrc"
        ]

        for path in common_paths:
            if Path(path).exists():
                self.openfoam_bashrc = Path(path)
                print(f"   ✅ Found OpenFOAM at: {self.openfoam_bashrc}")
                return

        print(f"   ❌ Could not find OpenFOAM installation")

    def _load_convex_hull_metadata(self):
        """Load convex hull validation metadata."""
        try:
            with open(self.convex_hull_metadata_path, 'r') as f:
                all_hull_data = json.load(f)

            self.convex_hull_data = {}

            for hull_entry in all_hull_data:
                if (hull_entry['gender'] == self.gender and
                    hull_entry['age_group'] == self.age_group and
                    hull_entry['outlier_method'] == self.outlier_method):

                    comparison = hull_entry['parameter_comparison']
                    self.convex_hull_data[comparison] = {
                        'hull_points': np.array(hull_entry['convex_hull_points']),
                        'group_size': hull_entry['group_size']
                    }

            print(f"   📊 Loaded {len(self.convex_hull_data)} convex hull boundaries")

        except Exception as e:
            print(f"   ⚠️  Failed to load convex hull metadata: {e}")
            self.convex_hull_data = None

    def _validate_parameters(self, parameters):
        """Validate parameters against convex hull boundaries."""
        if not self.convex_hull_data:
            return {'valid': True, 'message': 'No validation configured'}

        neck_d1, neck_d2, max_d, distal_d = parameters

        validations = {}
        param_checks = [
            ('Neck Diameter 1 vs Neck Diameter 2', (neck_d1, neck_d2)),
            ('Neck Diameter 2 vs Maximum Aneurysm Diameter', (neck_d2, max_d)),
            ('Maximum Aneurysm Diameter vs Distal Diameter', (max_d, distal_d))
        ]

        all_valid = True
        for comparison, (param1, param2) in param_checks:
            if comparison in self.convex_hull_data:
                hull_points = self.convex_hull_data[comparison]['hull_points']
                hull_closed = np.vstack((hull_points, hull_points[0]))

                point = np.array([param1, param2])
                is_valid = is_point_inside_hull(point, hull_closed)

                validations[comparison] = {
                    'valid': is_valid,
                    'point': point.tolist()
                }

                if not is_valid:
                    all_valid = False

        validations['all_valid'] = all_valid
        return validations

    def _save_validation_results(self, experiment_dir, job_id, parameters, validation_results):
        """Save validation results to job directory."""
        validation_data = {
            'job_id': int(job_id),
            'parameters': {
                'neck_diameter_1': float(parameters[0]),
                'neck_diameter_2': float(parameters[1]),
                'max_diameter': float(parameters[2]),
                'distal_diameter': float(parameters[3])
            },
            'validation': validation_results,
            'gender': self.gender,
            'age_group': self.age_group,
            'outlier_method': self.outlier_method
        }

        validation_file = experiment_dir / str(job_id) / 'validation_results.json'
        validation_file.parent.mkdir(exist_ok=True, parents=True)
        with open(validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)

    def _save_pipeline_results(self, experiment_dir, job_id, pipeline_results):
        """Save complete pipeline results to job directory."""
        results_file = experiment_dir / str(job_id) / 'pipeline_results.json'
        results_file.parent.mkdir(exist_ok=True, parents=True)
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
