# aorta_simple_wrapper.py
"""
UPDATED: Simple AORTA wrapper for QUEENS integration with consolidated repo structure.
This wrapper does NOT inherit from QUEENS Simulation class but is compatible with QUEENS workflows.
"""

import sys
import numpy as np
from pathlib import Path

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

# Change back to original directory
os.chdir(original_dir)

class AortaGeometryModel:
    """
    UPDATED: Simple AORTA geometry model wrapper with consolidated repo structure.
    
    This class provides a clean interface to AORTA's geometry generation
    functionality for use with QUEENS framework.
    """
    
    def __init__(self, output_dir="queens_output", enable_morphing=False, random_seed=42):
        """
        Initialize AORTA geometry model.

        Args:
            output_dir: Directory for QUEENS output files
            enable_morphing: Whether to apply morphological perturbations
            random_seed: Seed for reproducible geometry generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create geometry output directory
        self.geometry_dir = Path("geometries")
        self.geometry_dir.mkdir(exist_ok=True)

        self.enable_morphing = enable_morphing
        self.random_seed = random_seed
        self.case_counter = 0

        # Store AORTA directory for changing context during geometry generation
        import os
        self.aorta_dir = Path(__file__).parent.parent / 'Aorta' / 'ofCaseGen' / 'Method_4'
        self.original_dir = os.getcwd()

        print(f"🎯 AortaGeometryModel initialized (UPDATED)")
        print(f"   📁 Output directory: {self.output_dir.absolute()}")
        print(f"   📁 Geometry directory: {self.geometry_dir.absolute()}")
        print(f"   🔄 Morphing enabled: {enable_morphing}")
        print(f"   🎲 Random seed: {random_seed}")

    def evaluate(self, samples):
        """
        Generate AORTA geometries for given parameter samples.
        
        Args:
            samples: numpy array of shape (n_samples, 4) containing:
                    [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]
        
        Returns:
            numpy array of results containing geometry files and metadata
        """
        print(f"\n🚀 Starting AORTA geometry generation for {len(samples)} samples...")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        results = []
        
        for i, sample in enumerate(samples):
            case_id = f"case_{i:03d}"
            print(f"\n📐 Generating geometry {i+1}/{len(samples)} - {case_id}")
            
            try:
                # Create geometry
                geometry_file = self._generate_single_geometry(sample, case_id)
                
                # Store successful result
                results.append({
                    'case_id': case_id,
                    'geometry_file': str(geometry_file),
                    'parameters': sample.tolist(),
                    'success': True
                })
                
                print(f"   ✅ Success: {Path(geometry_file).name}")
                
            except Exception as e:
                print(f"   ❌ Error generating geometry: {str(e)}")
                results.append({
                    'case_id': case_id, 
                    'geometry_file': None,
                    'parameters': sample.tolist(),
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n✅ Geometry generation complete!")
        print(f"   📊 Success rate: {successful}/{len(results)}")
        
        return np.array(results)
    
    def _generate_single_geometry(self, parameters, case_id):
        """
        Generate a single AORTA geometry from parameters.

        Args:
            parameters: Array of [neck_d1, neck_d2, max_d, distal_d] in mm
            case_id: Unique identifier for this case

        Returns:
            Path to case directory containing 3 STL files
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

            return case_dir

        finally:
            # Ensure we always return to original directory
            os.chdir(self.original_dir)
    
    def validate_geometry(self, geometry_file, case_id):
        """
        Validate geometry using AORTA's convex hull methodology.
        
        Args:
            geometry_file: Path to STL geometry file
            case_id: Case identifier
            
        Returns:
            bool: True if geometry is valid (interior to convex hull)
        """
        # TODO: Implement convex hull validation
        # This will use your existing AORTA validation framework
        
        print(f"   🔍 Validating {case_id}... (validation TODO)")
        
        # Placeholder - always return True for now
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
    
    # Test with sample parameters
    test_samples = np.array([
        [25.0, 28.0, 50.0, 22.0],  # Test case 1: moderate AAA
        [23.0, 26.0, 60.0, 20.0],  # Test case 2: larger 
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