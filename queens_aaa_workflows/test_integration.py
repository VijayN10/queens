# test_integration.py
import numpy as np
from aorta_queens_driver import AortaGeometryDriver

def test_single_geometry():
    """Test single geometry generation."""
    
    print("Testing AORTA-QUEENS integration...")
    
    # Initialize driver
    driver = AortaGeometryDriver(output_dir='./test_geometry')
    
    # Define test parameters
    test_params = {
        'neck_diameter_1': 22.5,
        'neck_diameter_2': 23.0,
        'max_diameter': 45.0,
        'distal_diameter': 18.5
    }
    
    # Generate geometry
    result = driver.generate_from_parameters(test_params, 'test_case_001')
    
    if result['success']:
        print("✅ Geometry generation successful!")
        print(f"   Output path: {result['geometry_path']}")
        print(f"   Parameters: {result['parameters']}")
    else:
        print("❌ Geometry generation failed!")
    
    return result

if __name__ == "__main__":
    test_single_geometry()