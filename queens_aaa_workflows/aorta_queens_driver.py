# aorta_queens_driver.py
import sys
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any

# Add your AORTA path
sys.path.append('/home/a11evina/Aorta/ofCaseGen/Method_4')

# Import AORTA components
from config import ConfigParams, Demographics, MorphingSettings
from src.vesselStats.parameter_sampler import AAAParameterSampler
from main import generate_base_geometry_without_perturbation, apply_perturbation

class AortaGeometryDriver:
    """QUEENS-compatible driver for AORTA geometry generation."""
    
    def __init__(self, base_config_path=None, output_dir='./queens_geometries'):
        """Initialize the AORTA geometry driver."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store base configuration
        self.base_config = ConfigParams() if not base_config_path else self.load_config(base_config_path)
        
        # Initialize parameter sampler
        self.sampler = AAAParameterSampler(
            str(self.base_config.paths.fitted_params_file)
        )
        
        # Track generated geometries
        self.geometry_registry = {}
        
    def generate_from_parameters(self, parameters: Dict[str, float], 
                                 case_id: str = None) -> Dict[str, Any]:
        """
        Generate geometry from specific parameters.
        
        Parameters:
            parameters: Dict with keys:
                - neck_diameter_1
                - neck_diameter_2
                - max_diameter
                - distal_diameter
            case_id: Unique identifier for this case
        """
        if case_id is None:
            case_id = f"geom_{len(self.geometry_registry)}"
        
        print(f"\nGenerating geometry {case_id}...")
        
        # Create new config with these specific parameters
        config = ConfigParams()
        
        # Override sampled parameters with provided ones
        config.geometry_params.neck_diameter_1 = parameters['neck_diameter_1']
        config.geometry_params.neck_diameter_2 = parameters['neck_diameter_2']
        config.geometry_params.max_diameter = parameters['max_diameter']
        config.geometry_params.distal_diameter = parameters['distal_diameter']
        
        # Update anatomical points with new parameters
        config.anatomical_points = config.anatomical_template.generate_points(
            config.geometry_params
        )
        
        # Generate base geometry
        base_geometry = generate_base_geometry_without_perturbation(config)
        
        # Apply perturbation if needed
        perturbed_geometry = apply_perturbation(base_geometry, config)
        
        # Save geometry
        geometry_path = self.output_dir / case_id
        geometry_path.mkdir(parents=True, exist_ok=True)
        
        # Save STL and metadata
        self._save_geometry(perturbed_geometry, geometry_path, parameters)
        
        # Register geometry
        self.geometry_registry[case_id] = {
            'parameters': parameters,
            'path': str(geometry_path),
            'config': config
        }
        
        return {
            'case_id': case_id,
            'geometry_path': str(geometry_path),
            'parameters': parameters,
            'success': True
        }
    
    def _save_geometry(self, geometry: Dict, output_path: Path, parameters: Dict):
        """Save geometry files and metadata."""
        from src.vesselGen.save_stl_from_patches import save_stl_from_patches
        
        # Save STL
        stl_path = output_path / 'wall.stl'
        save_stl_from_patches(geometry['patches'], str(stl_path))
        
        # Save parameters
        param_file = output_path / 'parameters.json'
        with open(param_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        # Save centerline
        centerline_file = output_path / 'centerline.json'
        centerline_data = {
            'points': geometry['centerline'].tolist() if isinstance(
                geometry['centerline'], np.ndarray
            ) else geometry['centerline']
        }
        with open(centerline_file, 'w') as f:
            json.dump(centerline_data, f, indent=4)