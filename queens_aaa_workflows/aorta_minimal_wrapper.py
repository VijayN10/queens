#!/usr/bin/env python3
"""
Minimal AORTA wrapper that bypasses ConfigParams initialization issues.
"""

import sys
import os
from pathlib import Path
import numpy as np
import json

# Setup paths
AORTA_PATH = Path('/home/a11evina/Aorta/ofCaseGen/Method_4').resolve()
sys.path.insert(0, str(AORTA_PATH))

# Change to AORTA directory BEFORE importing anything
original_cwd = os.getcwd()
if AORTA_PATH.exists():
    os.chdir(str(AORTA_PATH))
    print(f"🔧 Working directory changed to: {AORTA_PATH}")

# Now import AORTA modules
from src.vesselStats.parameter_sampler import AAAParameterSampler, AAAGeometryParams
from src.vesselGen.vessel_spline import AnatomicalPoint, AnatomicalTemplate
from main import generate_base_geometry_without_perturbation, apply_perturbation
from src.vesselGen.save_stl_from_patches import save_stl_from_patches


class MinimalAortaModel:
    """Minimal AORTA wrapper that avoids ConfigParams complexity."""
    
    def __init__(self, output_dir='./minimal_geometries'):
        """Initialize with minimal setup."""
        # Change back to original directory for output
        os.chdir(original_cwd)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.geometry_registry = {}
        
        # Initialize parameter sampler directly
        fitted_file = AORTA_PATH / 'data/processed/fitted_distributions.json'
        if fitted_file.exists():
            print(f"✅ Found fitted distributions: {fitted_file}")
            # Change back to AORTA dir for sampler initialization
            os.chdir(str(AORTA_PATH))
            self.sampler = AAAParameterSampler(str(fitted_file))
            os.chdir(original_cwd)
        else:
            print(f"❌ Missing fitted distributions: {fitted_file}")
            raise FileNotFoundError(f"Missing: {fitted_file}")
        
        print(f"📁 Minimal AORTA model initialized, output: {self.output_dir}")
        
    def create_geometry_params(self, params_dict):
        """Create AAAGeometryParams from parameter dictionary."""
        geom_params = AAAGeometryParams()
        
        # Map parameters
        if 'neck_d1' in params_dict:
            geom_params.neck_diameter_1 = params_dict['neck_d1']
        if 'neck_d2' in params_dict:
            geom_params.neck_diameter_2 = params_dict['neck_d2']
        if 'sac_d1' in params_dict:
            geom_params.sac_diameter_1 = params_dict['sac_d1']
        if 'sac_d2' in params_dict:
            geom_params.sac_diameter_2 = params_dict['sac_d2']
        if 'sac_d3' in params_dict:
            geom_params.sac_diameter_3 = params_dict['sac_d3']
        if 'sac_l' in params_dict:
            geom_params.sac_length = params_dict['sac_l']
        if 'max_d' in params_dict:
            geom_params.max_diameter = params_dict['max_d']
        if 'distal_d' in params_dict:
            geom_params.distal_diameter = params_dict['distal_d']
        
        return geom_params
    
    def create_anatomical_points(self, geom_params):
        """Create anatomical points from geometry parameters."""
        # Use default template positions
        template = AnatomicalTemplate()
        
        points = [
            AnatomicalPoint(0, 0, "Inlet", geom_params.neck_diameter_1),
            AnatomicalPoint(0, 10, "Control point", geom_params.neck_diameter_1),
            AnatomicalPoint(0, 20, "Control point", geom_params.neck_diameter_1),
            AnatomicalPoint(0, 30, "Neck 1", geom_params.neck_diameter_1),
            AnatomicalPoint(10, 47, "Neck 2", geom_params.neck_diameter_2),
            AnatomicalPoint(20, 75.95, "Maximum Aneurysm", getattr(geom_params, 'max_diameter', 45.0)),
            AnatomicalPoint(10, 124.9, "Distal", getattr(geom_params, 'distal_diameter', 18.0))
        ]
        
        return points
    
    def evaluate(self, samples):
        """Generate geometries for parameter samples."""
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        
        n_samples, n_params = samples.shape
        print(f"🔄 Minimal evaluation: {n_samples} samples, {n_params} parameters")
        
        results = []
        
        for i, sample in enumerate(samples):
            case_id = f"minimal_{i:04d}"
            success = self._generate_single_geometry(case_id, sample, n_params)
            results.append(1.0 if success else 0.0)
        
        return np.array(results).reshape(-1, 1)
    
    def _generate_single_geometry(self, case_id, sample, n_params):
        """Generate single geometry with minimal approach."""
        try:
            print(f"\n🏗️  Generating {case_id}...")
            
            # Map sample to parameters
            if n_params >= 6:
                params = {
                    'neck_d1': float(sample[0]),
                    'neck_d2': float(sample[1]),
                    'sac_d1': float(sample[2]),
                    'sac_d2': float(sample[3]),
                    'sac_d3': float(sample[4]),
                    'sac_l': float(sample[5])
                }
            elif n_params == 4:
                params = {
                    'neck_d1': float(sample[0]),
                    'neck_d2': float(sample[1]),
                    'max_d': float(sample[2]),
                    'distal_d': float(sample[3])
                }
            else:
                raise ValueError(f"Unsupported parameter count: {n_params}")
            
            print(f"   📋 Parameters: {params}")
            
            # Create minimal configuration objects
            geom_params = self.create_geometry_params(params)
            anatomical_points = self.create_anatomical_points(geom_params)
            
            # Create minimal config-like object
            class MinimalConfig:
                def __init__(self):
                    self.geometry_params = geom_params
                    self.anatomical_points = anatomical_points
                    # Add vessel settings
                    class VesselSettings:
                        num_points = 100
                        num_circumference_vertices = 100
                        perturbation_range = 0.15
                        num_cycles = 4
                    self.vessel_settings = VesselSettings()
            
            config = MinimalConfig()
            
            # Change to AORTA directory for geometry generation
            os.chdir(str(AORTA_PATH))
            
            try:
                # Generate base geometry
                print(f"   🔧 Generating base geometry...")
                base_geometry = generate_base_geometry_without_perturbation(config)
                
                # Apply perturbation
                print(f"   🎲 Applying perturbation...")
                geometry = apply_perturbation(base_geometry, config)
                
            finally:
                # Always change back
                os.chdir(original_cwd)
            
            # Save geometry
            case_dir = self.output_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Save STL
            stl_file = case_dir / 'wall.stl'
            print(f"   💾 Saving STL: {stl_file}")
            
            if 'patches' in geometry:
                save_stl_from_patches(geometry['patches'], str(stl_file))
                print(f"   ✅ STL saved successfully")
            else:
                print(f"   ⚠️  No patches in geometry")
            
            # Save parameters
            param_file = case_dir / 'parameters.json'
            with open(param_file, 'w') as f:
                json.dump(params, f, indent=4)
            
            # Register
            self.geometry_registry[case_id] = {
                'parameters': params,
                'stl_file': str(stl_file),
                'case_dir': str(case_dir),
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_generated_geometries(self):
        """Return geometry registry."""
        return self.geometry_registry


def test_minimal_wrapper():
    """Test the minimal wrapper."""
    print("🧪 Testing Minimal AORTA Wrapper")
    print("=" * 50)
    
    # Create model
    model = MinimalAortaModel(output_dir='./test_minimal_output')
    
    # Test sample
    test_sample = np.array([[22.0, 18.0, 32.0, 28.0, 34.0, 45.0]])
    print(f"Test sample: {test_sample}")
    
    result = model.evaluate(test_sample)
    print(f"Result: {result}")
    
    # Show registry
    print(f"Generated: {model.get_generated_geometries()}")
    
    return result[0, 0] == 1.0


if __name__ == "__main__":
    success = test_minimal_wrapper()
    print(f"\n{'✅ Minimal test passed!' if success else '❌ Minimal test failed!'}")