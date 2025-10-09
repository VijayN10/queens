#!/usr/bin/env python3
"""
QUEENS-compatible AORTA geometry model wrapper.

This module provides the AortaGeometryModel class that inherits from QUEENS Simulation
to enable uncertainty quantification of abdominal aortic aneurysm geometries.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Setup paths
sys.path.insert(0, '/home/a11evina/queens/src')
sys.path.insert(0, '/home/a11evina/Aorta/ofCaseGen/Method_4')

# QUEENS imports
from queens.models.simulation import Simulation
from queens.schedulers.pool import Pool

# AORTA imports
try:
    from config import ConfigParams
    from main import generate_base_geometry_without_perturbation, apply_perturbation
    from src.vesselGen.save_stl_from_patches import save_stl_from_patches
    print("✅ AORTA imports successful")
except ImportError as e:
    print(f"❌ AORTA import error: {e}")
    print("⚠️  Make sure AORTA is properly installed at /home/a11evina/Aorta/")
    # Don't raise here, let the class definition proceed


class AortaGeometryDriver:
    """Driver component for AORTA geometry generation."""
    
    def __init__(self, output_dir='./geometries', demographic=('F', '70-79')):
        """
        Initialize AORTA geometry driver.
        
        Args:
            output_dir: Directory to save generated geometries
            demographic: Tuple of (gender, age_group) for parameter fitting
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gender, self.age_group = demographic
        self.geometry_counter = 0
        self.files_to_copy = []  # Required by QUEENS Scheduler
        
    def evaluate(self, samples):
        """
        Generate geometries for parameter samples.
        
        Args:
            samples: numpy array of shape (n_samples, 4)
                    [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]
        
        Returns:
            numpy array of results (1.0 for success, 0.0 for failure)
        """
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            
        results = []
        
        for i, sample in enumerate(samples):
            try:
                success = self._generate_single_geometry(sample, sample_id=i)
                results.append(1.0 if success else 0.0)
            except Exception as e:
                print(f"❌ Error generating geometry for sample {i}: {e}")
                results.append(0.0)
                
        return np.array(results)
    
    def _generate_single_geometry(self, sample, sample_id):
        """Generate a single geometry from parameters."""
        try:
            # Map QUEENS parameters to AORTA config
            config = self._create_aorta_config(sample, sample_id)
            
            # Generate base geometry
            base_geometry = generate_base_geometry_without_perturbation(config)
            
            # Apply perturbations (morphological transformations)
            final_geometry = apply_perturbation(base_geometry, config)
            
            # Save as STL file
            output_file = self.output_dir / f'aorta_geometry_{sample_id:03d}.stl'
            save_stl_from_patches(final_geometry, str(output_file))
            
            print(f"✅ Generated geometry {sample_id}: {output_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate geometry {sample_id}: {e}")
            return False
    
    def _create_aorta_config(self, sample, sample_id):
        """Create AORTA configuration from QUEENS parameters."""
        # Unpack QUEENS parameters
        neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter = sample
        
        # Create AORTA config
        config = ConfigParams()
        
        # Map geometric parameters
        config.neck_diameter_1 = float(neck_diameter_1)
        config.neck_diameter_2 = float(neck_diameter_2)
        config.max_diameter = float(max_diameter)
        config.distal_diameter = float(distal_diameter)
        
        # Set demographic info
        config.gender = self.gender
        config.age_group = self.age_group
        
        # Set output configuration
        config.case_id = f'queens_case_{sample_id:03d}'
        config.output_dir = str(self.output_dir)
        
        return config


class AortaGeometryModel(Simulation):
    """QUEENS-compatible AORTA geometry model."""
    
    def __init__(self, output_dir='./geometries', demographic=('F', '70-79')):
        """
        Initialize the AORTA geometry model for QUEENS.
        
        Args:
            output_dir: Directory to save generated geometries
            demographic: Tuple of (gender, age_group)
        """
        # Create driver and scheduler components required by QUEENS
        driver = AortaGeometryDriver(output_dir, demographic)
        scheduler = Pool(experiment_name='aorta_geometry')
        
        # Initialize parent Simulation class
        super().__init__(scheduler=scheduler, driver=driver)
        
        # Store additional attributes
        self.output_dir = Path(output_dir)
        self.demographic = demographic
        
        print(f"✅ AortaGeometryModel initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Demographic: {demographic}")


# Test function for direct usage
def test_aorta_model():
    """Test the AORTA model directly."""
    print("🧪 Testing AORTA Geometry Model")
    print("=" * 50)
    
    # Create model
    model = AortaGeometryModel(
        output_dir='./test_geometries',
        demographic=('F', '70-79')
    )
    
    # Test samples: [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]
    test_samples = np.array([
        [22.0, 24.0, 52.0, 18.0],  # Typical AAA
        [25.0, 28.0, 60.0, 20.0],  # Larger AAA
    ])
    
    print(f"Test samples shape: {test_samples.shape}")
    print("Parameters: [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter]")
    
    for i, sample in enumerate(test_samples):
        print(f"Sample {i+1}: {sample}")
    
    # Generate geometries
    results = model.evaluate(test_samples)
    
    print(f"\nResults: {results}")
    success_count = np.sum(results == 1.0)
    print(f"✅ Successfully generated {success_count}/{len(results)} geometries")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = test_aorta_model()
    if success:
        print("\n🎉 AORTA model test passed!")
    else:
        print("\n❌ AORTA model test failed!")