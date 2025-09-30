# aorta_simple_wrapper.py
"""
UPDATED: Simple AORTA wrapper for QUEENS integration with consolidated repo structure.
This wrapper does NOT inherit from QUEENS Simulation class but is compatible with QUEENS workflows.
Includes convex hull validation from Method 4.
"""

import sys
import numpy as np
from pathlib import Path
import json

# NEW consolidated repo paths
repo_root = Path(__file__).parent.parent  # Go up to queens/ root
aorta_dir = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'

# Change to AORTA directory before imports (needed for relative paths in AORTA code)
import os
original_dir = os.getcwd()
os.chdir(str(aorta_dir))

sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(aorta_dir))

# AORTA imports (updated paths)
from config import ConfigParams
from main import generate_base_geometry_without_perturbation, apply_perturbation
from src.vesselGen.save_stl_from_patches import save_stl_from_patches

# Import convex hull validation from Method 4
sys.path.insert(0, str(aorta_dir / 'data'))
from data_bound_with_morphed_data import is_point_inside_hull

# Change back to original directory
os.chdir(original_dir)

class AortaGeometryModel:
    """
    UPDATED: Simple AORTA geometry model wrapper with consolidated repo structure.

    This class provides a clean interface to AORTA's geometry generation
    functionality for use with QUEENS framework. Now includes full OpenFOAM case generation.
    """

    def __init__(self, output_dir="queens_output", enable_morphing=False, random_seed=42,
                 create_openfoam_cases=True, convex_hull_metadata_path=None,
                 gender=None, age_group=None, outlier_method='manual'):
        """
        Initialize AORTA geometry model.

        Args:
            output_dir: Directory for QUEENS output files
            enable_morphing: Whether to apply morphological perturbations
            random_seed: Seed for reproducible geometry generation
            create_openfoam_cases: Whether to create full OpenFOAM cases (not just geometries)
            convex_hull_metadata_path: Path to convex hull metadata JSON from Method 4
            gender: Gender for convex hull validation ('M', 'F', or 'All')
            age_group: Age group for convex hull validation (e.g., '70-79')
            outlier_method: Outlier detection method to use ('manual', 'zscore', etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create geometry output directory
        self.geometry_dir = Path("geometries")
        self.geometry_dir.mkdir(exist_ok=True)

        # Create OpenFOAM cases output directory
        self.openfoam_cases_dir = Path("openfoam_cases")
        if create_openfoam_cases:
            self.openfoam_cases_dir.mkdir(exist_ok=True)

        self.enable_morphing = enable_morphing
        self.create_openfoam_cases = create_openfoam_cases
        self.random_seed = random_seed
        self.case_counter = 0

        # Store AORTA directory for changing context during geometry generation
        import os
        self.aorta_dir = Path(__file__).parent.parent / 'Aorta' / 'ofCaseGen' / 'Method_4'
        self.original_dir = os.getcwd()

        # Convex hull validation parameters
        self.convex_hull_metadata_path = convex_hull_metadata_path
        self.gender = gender
        self.age_group = age_group
        self.outlier_method = outlier_method
        self.convex_hull_data = None

        # Load convex hull metadata if provided
        if convex_hull_metadata_path and Path(convex_hull_metadata_path).exists():
            self._load_convex_hull_metadata()

        print(f"🎯 AortaGeometryModel initialized (UPDATED with OpenFOAM and Convex Hull)")
        print(f"   📁 Output directory: {self.output_dir.absolute()}")
        print(f"   📁 Geometry directory: {self.geometry_dir.absolute()}")
        if create_openfoam_cases:
            print(f"   📁 OpenFOAM cases directory: {self.openfoam_cases_dir.absolute()}")
        print(f"   🔄 Morphing enabled: {enable_morphing}")
        print(f"   🏗️  Create OpenFOAM cases: {create_openfoam_cases}")
        print(f"   🎲 Random seed: {random_seed}")
        if self.convex_hull_data:
            print(f"   ✅ Convex hull validation enabled:")
            print(f"      Gender: {gender}, Age: {age_group}, Method: {outlier_method}")

    def evaluate(self, samples):
        """
        Generate AORTA geometries for given parameter samples with convex hull validation.

        Args:
            samples: numpy array of shape (n_samples, 4) containing:
                    [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]

        Returns:
            Dictionary with 'result' key containing success indicators (1.0 or 0.0)
        """
        print(f"\n🚀 Starting AORTA geometry generation for {len(samples)} samples...")

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        results = []
        self.geometry_metadata = []  # Store detailed metadata separately
        validation_summary = {'valid': 0, 'invalid': 0, 'not_validated': 0}

        for i, sample in enumerate(samples):
            case_id = f"case_{i:03d}"
            print(f"\n📐 Generating geometry {i+1}/{len(samples)} - {case_id}")

            # Validate parameters first if convex hull validation is enabled
            validation_results = None
            if self.convex_hull_data:
                validation_results = self.validate_parameters(sample)
                if validation_results['all_valid']:
                    print(f"   ✅ Parameters are inside convex hull boundaries")
                    validation_summary['valid'] += 1
                else:
                    print(f"   ⚠️  Parameters are OUTSIDE convex hull boundaries")
                    validation_summary['invalid'] += 1
                    for comparison, result in validation_results.items():
                        if comparison != 'all_valid' and isinstance(result, dict) and not result.get('valid'):
                            print(f"      ❌ {comparison}: point {result['point']}")
            else:
                validation_summary['not_validated'] += 1

            try:
                # Create geometry regardless of validation (for analysis purposes)
                geometry_file = self._generate_single_geometry(sample, case_id)

                # Save validation results if available
                if validation_results:
                    self.save_validation_results(case_id, sample, validation_results)

                # Store successful result
                self.geometry_metadata.append({
                    'case_id': case_id,
                    'geometry_file': str(geometry_file),
                    'parameters': sample.tolist(),
                    'validation': validation_results,
                    'success': True
                })
                results.append(1.0)  # Success indicator

                print(f"   ✅ Success: {Path(geometry_file).name}")

            except Exception as e:
                print(f"   ❌ Error generating geometry: {str(e)}")
                self.geometry_metadata.append({
                    'case_id': case_id,
                    'geometry_file': None,
                    'parameters': sample.tolist(),
                    'validation': validation_results,
                    'success': False,
                    'error': str(e)
                })
                results.append(0.0)  # Failure indicator

        # Summary
        successful = sum(1 for r in self.geometry_metadata if r['success'])
        print(f"\n✅ Geometry generation complete!")
        print(f"   📊 Success rate: {successful}/{len(results)}")

        if self.convex_hull_data:
            print(f"   🔍 Validation summary:")
            print(f"      ✅ Valid (inside hull): {validation_summary['valid']}")
            print(f"      ⚠️  Invalid (outside hull): {validation_summary['invalid']}")
            print(f"      - Not validated: {validation_summary['not_validated']}")

        # Save overall summary
        self._save_overall_summary(validation_summary)

        # Return in QUEENS-compatible format
        return {"result": np.array(results).reshape(-1, 1)}

    def _save_overall_summary(self, validation_summary):
        """Save overall validation summary to file."""
        summary_file = self.geometry_dir / 'validation_summary.json'

        summary_data = {
            'total_cases': len(self.geometry_metadata),
            'validation_summary': validation_summary,
            'convex_hull_config': {
                'enabled': self.convex_hull_data is not None,
                'gender': self.gender,
                'age_group': self.age_group,
                'outlier_method': self.outlier_method
            },
            'cases': self.geometry_metadata
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"   📄 Validation summary saved to: {summary_file}")
    
    def _generate_single_geometry(self, parameters, case_id):
        """
        Generate a single AORTA geometry from parameters and optionally create OpenFOAM case.

        Args:
            parameters: Array of [neck_d1, neck_d2, max_d, distal_d] in mm
            case_id: Unique identifier for this case

        Returns:
            Path to case directory containing 3 STL files (and OpenFOAM case if enabled)
        """
        import os
        from src.vesselStats.parameter_sampler import AAAGeometryParams

        # Change to AORTA directory (needed for relative paths in ConfigParams)
        os.chdir(str(self.aorta_dir))

        try:
            # Create config (this loads distributions and generates default params)
            config = ConfigParams()

            # Override with QUEENS-provided parameters (in mm)
            config.geometry_params = AAAGeometryParams(
                neck_diameter_1=float(parameters[0]),
                neck_diameter_2=float(parameters[1]),
                max_diameter=float(parameters[2]),
                distal_diameter=float(parameters[3])
            )

            # Regenerate anatomical points with updated parameters
            config.anatomical_points = config.anatomical_template.generate_points(config.geometry_params)

            print(f"   Parameters (mm): neck_d1={parameters[0]:.1f}, neck_d2={parameters[1]:.1f}, "
                  f"max_d={parameters[2]:.1f}, distal_d={parameters[3]:.1f}")

            # Generate base geometry
            geometry = generate_base_geometry_without_perturbation(config)
            print(f"   ✅ Base geometry: {len(geometry['faces'])} faces, 3 patches")

            # Apply morphing if enabled
            if self.enable_morphing:
                geometry = apply_perturbation(geometry, config)
                print(f"   🔄 Morphing applied")

            # Change back to original directory before saving
            os.chdir(self.original_dir)

            # Save as 3 separate STL files (inlet, wall, outlet) like Method 4
            case_dir = self.geometry_dir / case_id
            case_dir.mkdir(exist_ok=True)

            inlet_file = case_dir / "inlet.stl"
            wall_file = case_dir / "wall.stl"
            outlet_file = case_dir / "outlet.stl"

            save_stl_from_patches(geometry['inlet_patch'], geometry['vertices'], str(inlet_file))
            save_stl_from_patches(geometry['wall_patch'], geometry['vertices'], str(wall_file))
            save_stl_from_patches(geometry['outlet_patch'], geometry['vertices'], str(outlet_file))

            print(f"   💾 STL files saved: inlet.stl, wall.stl, outlet.stl")

            # Create OpenFOAM case if enabled
            if self.create_openfoam_cases:
                of_case_dir = self._create_openfoam_case(geometry, case_id, config)
                print(f"   🏗️  OpenFOAM case created: {of_case_dir.name}")

            return case_dir

        finally:
            # Ensure we always return to original directory
            os.chdir(self.original_dir)

    def _create_openfoam_case(self, geometry, case_id, config):
        """
        Create a complete OpenFOAM case using Method 4 code.

        Args:
            geometry: Dictionary containing geometry data (vertices, faces, patches)
            case_id: Unique identifier for this case
            config: ConfigParams object from Method 4

        Returns:
            Path to created OpenFOAM case directory
        """
        import os
        import shutil
        from src.ofCaseGen.compute_normals import computeNormals
        from src.ofCaseGen.find_point_inside_stl import findPointInsideSTL
        from src.ofCaseGen.correct_normal import correctNormal
        from src.ofCaseGen.clean_and_repeat_cardiac_cycle import cleanAndRepeatCardiacCycle
        from src.ofCaseGen.convert_velocity_components import convertVelocityComponents
        from src.ofCaseGen.generate_u_file import generateUFile
        from src.ofCaseGen.create_snappy_hex_mesh_dict import createSnappyHexMeshDict
        from src.ofCaseGen.create_block_mesh_dict import createBlockMeshDict

        print(f"   🏗️  Setting up OpenFOAM case for {case_id}...")

        # Setup paths - use absolute paths
        geometry_case_dir = (Path(self.original_dir) / self.geometry_dir / case_id).resolve()
        of_case_dir = (Path(self.original_dir) / self.openfoam_cases_dir / case_id).resolve()

        # Change to AORTA directory (needed for relative paths)
        os.chdir(str(self.aorta_dir))

        try:
            # Step 1: Copy base case
            print(f"   📦 Copying base case...")
            base_case = self.aorta_dir / 'data' / 'input' / 'of_base_case'
            of_case_dir.mkdir(parents=True, exist_ok=True)

            # Copy base case structure
            for item in base_case.iterdir():
                if item.is_dir():
                    shutil.copytree(item, of_case_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, of_case_dir / item.name)

            # Step 2: Copy geometry files to constant/triSurface
            print(f"   📐 Copying geometry files...")
            tri_surface_dir = of_case_dir / 'constant' / 'triSurface'
            tri_surface_dir.mkdir(parents=True, exist_ok=True)

            # Copy geometry files using absolute paths
            shutil.copy2(geometry_case_dir / 'inlet.stl', tri_surface_dir / 'inlet.stl')
            shutil.copy2(geometry_case_dir / 'wall.stl', tri_surface_dir / 'wall.stl')
            shutil.copy2(geometry_case_dir / 'outlet.stl', tri_surface_dir / 'outlet.stl')

            # Step 3: Compute normals and find interior point
            print(f"   🧭 Computing normals and interior point...")
            normals, avgNormal, referencePoint = computeNormals(str(tri_surface_dir / 'inlet.stl'))
            insidePoint = findPointInsideSTL(str(tri_surface_dir / 'wall.stl'))
            correctedNormal = correctNormal(avgNormal, referencePoint, insidePoint)

            # Step 4: Process velocity data
            print(f"   🌊 Processing velocity data...")
            velocity_file = self.aorta_dir / 'data' / 'input' / 'U' / 'velocity_time.csv'

            if not velocity_file.exists():
                print(f"   ⚠️  Warning: velocity file not found at {velocity_file}")
                print(f"   ⚠️  Skipping velocity processing...")
            else:
                cleanedData = cleanAndRepeatCardiacCycle(str(velocity_file), config.vessel_settings.num_cycles)

                # Create temporary output directory for files
                temp_files_dir = Path('data/output/files')
                temp_files_dir.mkdir(parents=True, exist_ok=True)

                outputCSV = temp_files_dir / 'corrected_velocity_time.csv'
                convertVelocityComponents(cleanedData, str(outputCSV), correctedNormal)
                generateUFile(str(outputCSV))

                # Copy generated U file to case
                u_file_src = temp_files_dir / 'U'
                if u_file_src.exists():
                    shutil.copy2(u_file_src, of_case_dir / '0' / 'U')

            # Step 5: Create mesh dictionaries
            print(f"   📝 Creating mesh dictionaries...")
            temp_files_dir = Path('data/output/files')
            temp_files_dir.mkdir(parents=True, exist_ok=True)

            # Create snappyHexMeshDict
            shm_top = self.aorta_dir / 'data' / 'input' / 'shm' / 'shm_top.txt'
            shm_bottom = self.aorta_dir / 'data' / 'input' / 'shm' / 'shm_bottom.txt'

            if shm_top.exists() and shm_bottom.exists():
                snappy_dict_path = temp_files_dir / 'snappyHexMeshDict'
                createSnappyHexMeshDict(insidePoint, str(shm_top), str(shm_bottom), str(snappy_dict_path))

                # Copy to case
                shutil.copy2(snappy_dict_path, of_case_dir / 'system' / 'snappyHexMeshDict')
            else:
                print(f"   ⚠️  Warning: snappyHexMesh templates not found")

            # Create blockMeshDict
            wall_stl_for_blockmesh = tri_surface_dir / 'wall.stl'
            block_dict_path = temp_files_dir / 'blockMeshDict'
            createBlockMeshDict(str(wall_stl_for_blockmesh), str(block_dict_path))

            # Copy to case
            shutil.copy2(block_dict_path, of_case_dir / 'system' / 'blockMeshDict')

            print(f"   ✅ OpenFOAM case complete at {of_case_dir.relative_to(Path(self.original_dir))}")

            return of_case_dir

        finally:
            # Always return to original directory
            os.chdir(self.original_dir)
    
    def _load_convex_hull_metadata(self):
        """Load convex hull metadata from Method 4."""
        try:
            with open(self.convex_hull_metadata_path, 'r') as f:
                all_hull_data = json.load(f)

            # Filter for the specific gender, age group, and outlier method
            self.convex_hull_data = {}

            # Parameter pairs we need to validate
            param_pairs = [
                ('Neck Diameter 1', 'Neck Diameter 2'),
                ('Neck Diameter 2', 'Maximum Aneurysm Diameter'),
                ('Maximum Aneurysm Diameter', 'Distal Diameter')
            ]

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

    def validate_parameters(self, parameters):
        """
        Validate parameters against convex hull boundaries.

        Args:
            parameters: Array of [neck_d1, neck_d2, max_d, distal_d] in mm

        Returns:
            dict: Validation results for each parameter pair
        """
        if not self.convex_hull_data:
            return {'valid': True, 'message': 'No convex hull validation configured'}

        # Extract parameter values
        neck_d1, neck_d2, max_d, distal_d = parameters

        # Check each parameter pair
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
                # Close the hull polygon
                hull_closed = np.vstack((hull_points, hull_points[0]))

                point = np.array([param1, param2])
                is_valid = is_point_inside_hull(point, hull_closed)

                validations[comparison] = {
                    'valid': is_valid,
                    'point': point.tolist()
                }

                if not is_valid:
                    all_valid = False
            else:
                validations[comparison] = {
                    'valid': None,
                    'message': 'No hull data for this comparison'
                }

        validations['all_valid'] = all_valid
        return validations

    def save_validation_results(self, case_id, parameters, validation_results):
        """Save validation results to JSON file."""
        case_dir = self.geometry_dir / case_id
        validation_file = case_dir / 'validation_results.json'

        validation_data = {
            'case_id': case_id,
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

        with open(validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)

    def validate_geometry(self, geometry_file, case_id):
        """
        Validate geometry using AORTA's convex hull methodology.

        Args:
            geometry_file: Path to STL geometry file
            case_id: Case identifier

        Returns:
            bool: True if geometry is valid (interior to convex hull)
        """
        # This is a placeholder - in practice, validation is done on parameters
        # before geometry generation
        print(f"   🔍 Validating {case_id}...")
        return True

# Test function
def test_simple_wrapper():
    """Test the updated simple wrapper."""
    print("🧪 Testing AortaGeometryModel with updated paths...")
    
    # Create model
    model = AortaGeometryModel(
        output_dir="queens_output",
        enable_morphing=False,
        random_seed=42
    )
    
    # Test with sample parameters (all in mm)
    test_samples = np.array([
        [25.0, 28.0, 50.0, 22.0],  # Test case 1: moderate AAA (mm)
        [23.0, 26.0, 60.0, 20.0],  # Test case 2: larger AAA (mm)
    ])
    
    # Generate geometries
    results = model.evaluate(test_samples)
    
    # Check results
    successful = sum(1 for r in results if r['success'])
    print(f"\n🎯 Test Results: {successful}/{len(results)} successful")
    
    return successful == len(results)

if __name__ == "__main__":
    success = test_simple_wrapper()
    
    if success:
        print("\n✅ Simple wrapper test PASSED!")
    else:
        print("\n❌ Simple wrapper test FAILED!")
    
    sys.exit(0 if success else 1)