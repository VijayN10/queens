# aorta_queens_model.py
import sys
from pathlib import Path
import numpy as np
import json

# Setup paths
QUEENS_PATH = Path('/home/a11evina/queens/src').resolve()
AORTA_PATH = Path('/home/a11evina/Aorta/ofCaseGen/Method_4').resolve()

sys.path.insert(0, str(QUEENS_PATH))
sys.path.insert(0, str(AORTA_PATH))

# Correct QUEENS imports
from queens.models.simulation import Simulation

# AORTA imports
from config import ConfigParams
from main import generate_base_geometry_without_perturbation, apply_perturbation
from src.vesselGen.save_stl_from_patches import save_stl_from_patches

class AortaGeometryModel(Simulation):
    """QUEENS-compatible wrapper for AORTA geometry generation."""
    
    def __init__(self, output_dir='./queens_geometries', enable_perturbation=True):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_perturbation = enable_perturbation
        self.geometry_registry = {}
        
    def evaluate(self, samples, **kwargs):
        """
        Generate geometries for given parameter samples.
        
        Args:
            samples: numpy array of shape (n_samples, 4)
                    [neck_d1, neck_d2, max_d, distal_d]
        Returns:
            results: numpy array of outputs (for now, just success flag)
        """
        results = []
        
        for i, sample in enumerate(samples):
            case_id = f"case_{i:04d}"
            print(f"\nGenerating geometry {case_id}...")
            
            try:
                # Create AORTA config with sample parameters
                config = ConfigParams()
                
                # Set parameters from QUEENS sample
                config.geometry_params.neck_diameter_1 = float(sample[0])
                config.geometry_params.neck_diameter_2 = float(sample[1])
                config.geometry_params.max_diameter = float(sample[2])
                config.geometry_params.distal_diameter = float(sample[3])
                
                # Update anatomical points with new parameters
                config.anatomical_points = config.anatomical_template.generate_points(
                    config.geometry_params
                )
                
                # Generate base geometry
                base_geometry = generate_base_geometry_without_perturbation(config)
                
                # Apply perturbation if enabled
                if self.enable_perturbation:
                    geometry = apply_perturbation(base_geometry, config)
                else:
                    geometry = base_geometry
                
                # Save geometry
                case_dir = self.output_dir / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                
                # Save STL
                stl_path = case_dir / 'wall.stl'
                save_stl_from_patches(geometry['patches'], str(stl_path))
                
                # Save parameters
                param_file = case_dir / 'parameters.json'
                params_dict = {
                    'neck_diameter_1': float(sample[0]),
                    'neck_diameter_2': float(sample[1]),
                    'max_diameter': float(sample[2]),
                    'distal_diameter': float(sample[3])
                }
                with open(param_file, 'w') as f:
                    json.dump(params_dict, f, indent=4)
                
                # Register geometry
                self.geometry_registry[case_id] = {
                    'parameters': params_dict,
                    'path': str(case_dir),
                    'stl_file': str(stl_path)
                }
                
                print(f"   ✅ Geometry saved to {case_dir}")
                
                # For now, return success flag (later can return actual metrics)
                results.append([1.0])  # Success
                
            except Exception as e:
                print(f"   ❌ Error generating geometry: {e}")
                results.append([0.0])  # Failure
        
        return np.array(results)