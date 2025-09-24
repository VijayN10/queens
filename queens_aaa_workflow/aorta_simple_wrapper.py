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
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'))

# AORTA imports (updated paths)
from config import ConfigParams
from main import generate_base_geometry_without_perturbation, apply_perturbation
from src.vesselGen.save_stl_from_patches import save_stl_from_patches

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
            parameters: Array of [neck_d1, neck_d2, max_d, distal_d] 
            case_id: Unique identifier for this case
            
        Returns:
            Path to generated STL file
        """
        # Create config with parameters
        config = ConfigParams()
        
        # Set demographic parameters (fixed for Phase 1)
        config.gender = 'F'  # Female
        config.age_group = '70-79'
        config.random_seed = self.random_seed
        
        # Update geometric parameters from QUEENS
        config.neck_diameter_1 = float(parameters[0])
        config.neck_diameter_2 = float(parameters[1])
        config.max_diameter = float(parameters[2])
        config.distal_diameter = float(parameters[3])
        
        print(f"   Parameters: neck_d1={parameters[0]:.3f}, neck_d2={parameters[1]:.3f}, "
              f"max_d={parameters[2]:.3f}, distal_d={parameters[3]:.3f}")
        
        # Generate base geometry
        patches = generate_base_geometry_without_perturbation(config)
        print(f"   ✅ Base geometry: {len(patches)} patches")
        
        # Apply morphing if enabled 
        if self.enable_morphing:
            patches = apply_perturbation(patches, config)
            print(f"   🔄 Morphing applied")
        
        # Save as STL
        stl_file = self.geometry_dir / f"{case_id}.stl"
        save_stl_from_patches(patches, str(stl_file))
        print(f"   💾 STL saved: {stl_file.name}")
        
        return stl_file
    
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
        [2.5, 2.8, 5.5, 2.2],  # Test case 1
        [2.3, 2.6, 5.1, 2.0],  # Test case 2
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