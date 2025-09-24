#!/usr/bin/env python3
"""
Simple AORTA wrapper for QUEENS integration.
This wrapper does NOT inherit from QUEENS Simulation class.
Uses the correct 4-parameter system: neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter
"""

import sys
import os
from pathlib import Path
import numpy as np
import json

# Setup paths for new integrated structure
# Assuming we're running from queens/queens_aaa_workflows/
current_dir = Path(__file__).parent
repo_root = current_dir.parent  # Should be queens/ directory

# AORTA path - now inside queens repo
AORTA_PATH = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'

# Add paths to Python path
sys.path.insert(0, str(repo_root / 'src'))  # For QUEENS
sys.path.insert(0, str(AORTA_PATH))         # For AORTA

print(f"🔧 AORTA path: {AORTA_PATH}")
print(f"🔧 AORTA exists: {AORTA_PATH.exists()}")

# AORTA imports
try:
    from config import ConfigParams
    from main import generate_base_geometry_without_perturbation, apply_perturbation
    from src.vesselGen.save_stl_from_patches import save_stl_from_patches
    print("✅ AORTA imports successful")
except ImportError as e:
    print(f"❌ AORTA import error: {e}")
    print("Make sure AORTA is properly installed and paths are correct")
    raise



class AortaGeometryModel:
    """Simple AORTA geometry model wrapper without QUEENS inheritance."""
    
    def __init__(self, output_dir='./queens_geometries', enable_perturbation=True):
        """
        Initialize the AORTA geometry model.
        
        Args:
            output_dir: Directory to save generated geometries
            enable_perturbation: Whether to apply geometry perturbation
        """
        # Save outputs under this script's directory by default
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = current_dir / output_path
        self.output_dir = output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_perturbation = enable_perturbation
        self.geometry_registry = {}
        
        print(f"🔧 AORTA model initialized, output directory: {self.output_dir}")
        
    def evaluate(self, samples):
        """
        Generate geometries for given parameter samples.
        
        Args:
            samples: numpy array of shape (n_samples, 4)
                    Expected parameter order: [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]
        
        Returns:
            results: numpy array of success flags (1.0 for success, 0.0 for failure)
        """
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        
        n_samples, n_params = samples.shape
        print(f"📄 Evaluating {n_samples} geometry samples with {n_params} parameters each")
        
        if n_params != 4:
            raise ValueError(f"AORTA expects exactly 4 parameters [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter], got {n_params}")
        
        results = []
        
        for i, sample in enumerate(samples):
            case_id = f"case_{i:04d}"
            print(f"\n🗿 Generating geometry {case_id}...")
            
            try:
                # Map parameters to the correct 4-parameter system
                params = {
                    'neck_diameter_1': float(sample[0]),
                    'neck_diameter_2': float(sample[1]),
                    'max_diameter': float(sample[2]),
                    'distal_diameter': float(sample[3])
                }
                
                print(f"   📋 Parameters: {params}")
                
                # Generate geometry using AORTA
                success = self._generate_single_geometry(case_id, params)
                results.append(1.0 if success else 0.0)
                
                if success:
                    print(f"✅ Geometry {case_id} generated successfully")
                else:
                    print(f"❌ Failed to generate geometry {case_id}")
                    
            except Exception as e:
                print(f"❌ Error generating geometry {case_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append(0.0)
        
        results_array = np.array(results).reshape(-1, 1)
        success_count = np.sum(results_array)
        print(f"\n📊 Summary: {success_count}/{n_samples} geometries generated successfully")
        
        return results_array
    
    def _generate_single_geometry(self, case_id, params):
        """
        Generate a single geometry with given parameters.
        
        Args:
            case_id: Unique identifier for this geometry
            params: Dictionary containing the 4 AORTA geometry parameters
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Store original working directory
            original_cwd = os.getcwd()
            
            # Ensure we are in the AORTA directory so relative data paths resolve
            os.chdir(str(AORTA_PATH))
            
            # Create AORTA config
            config = ConfigParams()
            
            # Set the 4 core geometry parameters - these are the ONLY ones that matter
            config.geometry_params.neck_diameter_1 = params['neck_diameter_1']
            config.geometry_params.neck_diameter_2 = params['neck_diameter_2']
            config.geometry_params.max_diameter = params['max_diameter']
            config.geometry_params.distal_diameter = params['distal_diameter']
            
            # THIS IS THE CRITICAL FIX: Update anatomical points with new parameters
            print(f"   🔄 Updating anatomical points with new parameters...")
            config.anatomical_points = config.anatomical_template.generate_points(
                config.geometry_params
            )
            
            # Debug: Print the actual anatomical points being used
            print(f"   🔍 Anatomical points:")
            for point in config.anatomical_points:
                print(f"      {point.name}: diameter={point.diameter:.1f}mm at ({point.x:.1f}, {point.y:.1f})")
            
            try:
                # Generate base geometry
                print(f"   🔧 Generating base geometry...")
                base_geometry = generate_base_geometry_without_perturbation(config)
                
                # Apply perturbation if enabled
                if self.enable_perturbation:
                    print(f"   🎲 Applying perturbation...")
                    geometry = apply_perturbation(base_geometry, config)
                else:
                    geometry = base_geometry
                
                # Change back to original directory before saving
                os.chdir(original_cwd)
                
                # Save geometry files
                case_dir = self.output_dir / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                
                # Save STL file
                stl_file = case_dir / 'wall.stl'
                print(f"   💾 Saving STL to: {stl_file}")
                
                # Debug: Print geometry structure
                print(f"   🔍 Geometry keys: {list(geometry.keys())}")
                
                # Try different approaches based on geometry structure
                if 'wall_patch' in geometry and 'vertices' in geometry:
                    print(f"   🔍 Using wall_patch and vertices")
                    save_stl_from_patches(geometry['wall_patch'], geometry['vertices'], str(stl_file))
                    print(f"   ✅ STL saved successfully using wall_patch")
                elif 'patches' in geometry and 'vertices' in geometry:
                    print(f"   🔍 Using patches and vertices")
                    save_stl_from_patches(geometry['patches'], geometry['vertices'], str(stl_file))
                    print(f"   ✅ STL saved successfully using patches")
                elif 'faces' in geometry and 'vertices' in geometry:
                    print(f"   🔍 Using faces and vertices")
                    save_stl_from_patches(geometry['faces'], geometry['vertices'], str(stl_file))
                    print(f"   ✅ STL saved successfully using faces")
                elif 'patches' in geometry:
                    print(f"   🔍 Using patches only")
                    save_stl_from_patches(geometry['patches'], str(stl_file))
                    print(f"   ✅ STL saved successfully using patches only")
                else:
                    print(f"   ⚠️  Warning: Could not find compatible structure for STL save")
                    print(f"   Available keys: {list(geometry.keys())}")
                    # Try to save vertices/faces info for debugging
                    debug_file = case_dir / 'geometry_debug.json'
                    debug_data = {key: f"Type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}" 
                                for key, value in geometry.items()
                                if not key.startswith('_')}
                    with open(debug_file, 'w') as f:
                        json.dump(debug_data, f, indent=4)
                    
                # Save parameters
                param_file = case_dir / 'parameters.json'
                with open(param_file, 'w') as f:
                    json.dump(params, f, indent=4)
                
                # Save centerline if available
                if 'centerline' in geometry:
                    centerline_file = case_dir / 'centerline.json'
                    centerline_data = {
                        'points': geometry['centerline'].tolist() if isinstance(
                            geometry['centerline'], np.ndarray
                        ) else geometry['centerline']
                    }
                    with open(centerline_file, 'w') as f:
                        json.dump(centerline_data, f, indent=4)
                
                # Register successful geometry
                self.geometry_registry[case_id] = {
                    'parameters': params,
                    'stl_file': str(stl_file),
                    'case_dir': str(case_dir),
                    'success': True
                }
                
                return True
                
            finally:
                # Always change back to original directory
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"   ❌ Error in _generate_single_geometry: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_generated_geometries(self):
        """Return information about generated geometries."""
        return self.geometry_registry
    
    def cleanup(self):
        """Clean up any temporary files or resources."""
        pass


# Test function
def test_simple_wrapper():
    """Test the simple wrapper directly with correct 4-parameter system."""
    print("🧪 Testing Simple AORTA Wrapper (4-parameter system)")
    print("=" * 50)
    
    # Create model
    model = AortaGeometryModel(output_dir='test_wrapper_output_4param')
    
    # Test with different samples to verify they create different geometries
    # Parameters: [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]
    test_samples = np.array([
        [22.0, 24.0, 52.0, 18.0],  # Sample 1 - typical AAA
        [25.0, 28.0, 60.0, 20.0],  # Sample 2 - larger AAA
        [18.0, 20.0, 45.0, 15.0],  # Sample 3 - smaller AAA
        [20.0, 20.0, 20.0, 20.0]   # Sample 4 - uniform (should look like tube)
    ])
    
    print(f"Test samples (4-parameter system):")
    print(f"Parameters: [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]")
    for i, sample in enumerate(test_samples):
        print(f"Sample {i+1}: {sample}")
    
    result = model.evaluate(test_samples)
    print(f"Result: {result}")
    
    # Show registry
    print(f"\nGenerated geometries: {len(model.get_generated_geometries())}")
    for case_id, info in model.get_generated_geometries().items():
        params = info['parameters']
        print(f"{case_id}: neck1={params['neck_diameter_1']:.1f}, neck2={params['neck_diameter_2']:.1f}, "
              f"max={params['max_diameter']:.1f}, distal={params['distal_diameter']:.1f}")
    
    return np.all(result == 1.0)


if __name__ == "__main__":
    success = test_simple_wrapper()
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")