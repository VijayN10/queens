#!/usr/bin/env python3
"""
QUEENS-compatible Driver for AORTA geometry generation and OpenFOAM case setup.

This driver follows the standard QUEENS Driver pattern for AAA geometry generation.
Each call to run() generates one geometry + OpenFOAM case for a given parameter sample.
"""

import sys
import os
import numpy as np
from pathlib import Path
import json

# QUEENS imports
sys.path.insert(0, '/home/a11evina/queens/src')
from queens.drivers._driver import Driver
from queens.utils.logger_settings import log_init_args

# AORTA imports (your existing code)
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


class AortaGeometryDriver(Driver):
    """
    QUEENS driver for AORTA AAA geometry generation.

    This driver handles:
    1. Geometry generation from AAA parameters
    2. OpenFOAM case setup (optional)
    3. Convex hull validation (optional)

    Each run() call processes ONE sample and returns a result.
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        geometry_output_dir="geometries",
        openfoam_cases_dir="openfoam_cases",
        create_openfoam_cases=True,
        enable_morphing=False,
        convex_hull_metadata_path=None,
        gender=None,
        age_group=None,
        outlier_method='manual'
    ):
        """
        Initialize AORTA geometry driver.

        Args:
            parameters (Parameters): QUEENS Parameters object
            geometry_output_dir (str): Directory for geometry STL files
            openfoam_cases_dir (str): Directory for OpenFOAM cases
            create_openfoam_cases (bool): Whether to create full OpenFOAM cases
            enable_morphing (bool): Whether to apply morphological perturbations
            convex_hull_metadata_path (str): Path to convex hull validation metadata
            gender (str): Gender for validation ('M', 'F', 'All')
            age_group (str): Age group for validation (e.g., '70-79')
            outlier_method (str): Outlier detection method ('manual', 'zscore', etc.)
        """
        # No input files to copy for geometry generation
        files_to_copy = []

        # Initialize parent Driver class with parameters
        super().__init__(parameters=parameters, files_to_copy=files_to_copy)

        self.geometry_output_dir = Path(geometry_output_dir)
        self.openfoam_cases_dir = Path(openfoam_cases_dir)
        self.create_openfoam_cases = create_openfoam_cases
        self.enable_morphing = enable_morphing

        # Convex hull validation
        self.convex_hull_metadata_path = convex_hull_metadata_path
        self.gender = gender
        self.age_group = age_group
        self.outlier_method = outlier_method
        self.convex_hull_data = None

        # Create output directories
        self.geometry_output_dir.mkdir(exist_ok=True, parents=True)
        if self.create_openfoam_cases:
            self.openfoam_cases_dir.mkdir(exist_ok=True, parents=True)

        # Load convex hull metadata if provided
        if convex_hull_metadata_path and Path(convex_hull_metadata_path).exists():
            self._load_convex_hull_metadata()

        # Store AORTA directory for context switching
        self.aorta_dir = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'
        self.original_dir = os.getcwd()

        print(f"✅ AortaGeometryDriver initialized")
        print(f"   📁 Geometry output: {self.geometry_output_dir.absolute()}")
        if create_openfoam_cases:
            print(f"   📁 OpenFOAM cases: {self.openfoam_cases_dir.absolute()}")
        if self.convex_hull_data:
            print(f"   ✅ Convex hull validation enabled: {gender}, {age_group}, {outlier_method}")

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """
        Generate one AORTA geometry + OpenFOAM case (QUEENS Driver interface).

        This method is called by the QUEENS scheduler for each sample.

        Args:
            sample (np.array): Parameter array [neck_d1, neck_d2, max_d, distal_d] in mm
            job_id (int): Unique job ID from QUEENS scheduler
            num_procs (int): Number of processors (unused for geometry generation)
            experiment_dir (Path): QUEENS experiment directory
            experiment_name (str): QUEENS experiment name

        Returns:
            float: Success indicator (1.0 for success, 0.0 for failure)
        """
        case_id = f"case_{job_id:03d}"

        print(f"\n{'='*70}")
        print(f"📐 Job {job_id}: Generating {case_id}")
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

            # Save validation results to job directory
            self._save_validation_results(experiment_dir, job_id, sample, validation_results)

        try:
            # Generate geometry
            geometry_file = self._generate_single_geometry(sample, case_id)

            # Create metadata
            metadata = {
                'case_id': case_id,
                'job_id': int(job_id),  # Convert numpy int64 to Python int
                'parameters': {
                    'neck_diameter_1': float(sample[0]),
                    'neck_diameter_2': float(sample[1]),
                    'max_diameter': float(sample[2]),
                    'distal_diameter': float(sample[3])
                },
                'geometry_file': str(geometry_file),
                'success': True
            }

            # Save metadata to experiment directory (QUEENS convention)
            metadata_file = experiment_dir / str(job_id) / 'aorta_metadata.json'
            metadata_file.parent.mkdir(exist_ok=True, parents=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"   ✅ Success: {case_id}")
            print(f"   📁 Geometry: {geometry_file}")
            if self.create_openfoam_cases:
                print(f"   🏗️  OpenFOAM case: {self.openfoam_cases_dir / case_id}")

            return 1.0  # Success indicator

        except Exception as e:
            print(f"   ❌ Error: {e}")

            # Save error metadata
            error_metadata = {
                'case_id': case_id,
                'job_id': int(job_id),  # Convert numpy int64 to Python int
                'parameters': sample.tolist(),
                'success': False,
                'error': str(e)
            }

            metadata_file = experiment_dir / str(job_id) / 'aorta_metadata.json'
            metadata_file.parent.mkdir(exist_ok=True, parents=True)
            with open(metadata_file, 'w') as f:
                json.dump(error_metadata, f, indent=2)

            return 0.0  # Failure indicator

    def _generate_single_geometry(self, parameters, case_id):
        """
        Generate a single AORTA geometry from parameters.

        This is your existing geometry generation logic from AortaGeometryModel.

        Args:
            parameters (np.array): [neck_d1, neck_d2, max_d, distal_d] in mm
            case_id (str): Unique case identifier

        Returns:
            Path: Path to geometry directory containing STL files
        """
        from src.vesselStats.parameter_sampler import AAAGeometryParams

        # Change to AORTA directory (needed for relative paths)
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
            print(f"   ✅ Base geometry: {len(geometry['faces'])} faces")

            # Apply morphing if enabled
            if self.enable_morphing:
                geometry = apply_perturbation(geometry, config)
                print(f"   🔄 Morphing applied")

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

            print(f"   💾 STL files: inlet.stl, wall.stl, outlet.stl")

            # Create OpenFOAM case if enabled
            if self.create_openfoam_cases:
                try:
                    # Convert to absolute path before passing to _create_openfoam_case
                    case_dir_absolute = case_dir.resolve()
                    of_case_dir = self._create_openfoam_case(geometry, case_id, config, case_dir_absolute)
                    print(f"   🏗️  OpenFOAM case: {of_case_dir.name}")
                except Exception as e:
                    print(f"   ⚠️  OpenFOAM case creation failed: {e}")
                    import traceback
                    traceback.print_exc()

            return case_dir

        finally:
            # Always return to original directory
            os.chdir(self.original_dir)

    def _create_openfoam_case(self, geometry, case_id, config, geometry_case_dir):
        """
        Create OpenFOAM case from geometry.

        This is your existing OpenFOAM case setup logic.
        """
        print(f"   🔧 Starting OpenFOAM case setup for {case_id}...")

        import shutil
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
        print(f"   📁 OpenFOAM case dir: {of_case_dir}")

        # Change to AORTA directory
        os.chdir(str(self.aorta_dir))

        try:
            # Copy base case
            print(f"   📦 Copying base case...")
            base_case = self.aorta_dir / 'data' / 'input' / 'of_base_case'
            of_case_dir.mkdir(parents=True, exist_ok=True)

            for item in base_case.iterdir():
                if item.is_dir():
                    shutil.copytree(item, of_case_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, of_case_dir / item.name)

            # Copy geometry files to constant/triSurface
            print(f"   📐 Copying geometry files...")
            tri_surface_dir = of_case_dir / 'constant' / 'triSurface'
            tri_surface_dir.mkdir(parents=True, exist_ok=True)

            print(f"   📂 Geometry source: {geometry_case_dir}")
            print(f"   📂 triSurface dest: {tri_surface_dir}")

            shutil.copy2(geometry_case_dir / 'inlet.stl', tri_surface_dir / 'inlet.stl')
            shutil.copy2(geometry_case_dir / 'wall.stl', tri_surface_dir / 'wall.stl')
            shutil.copy2(geometry_case_dir / 'outlet.stl', tri_surface_dir / 'outlet.stl')
            print(f"   ✅ STL files copied")

            # Compute normals and interior point
            print(f"   🧭 Computing normals and interior point...")
            normals, avgNormal, referencePoint = computeNormals(str(tri_surface_dir / 'inlet.stl'))
            insidePoint = findPointInsideSTL(str(tri_surface_dir / 'wall.stl'))
            correctedNormal = correctNormal(avgNormal, referencePoint, insidePoint)
            print(f"   ✅ Normals computed")

            # Process velocity data (if available)
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
            'job_id': int(job_id),  # Convert numpy int64 to Python int
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
