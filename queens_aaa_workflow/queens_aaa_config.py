#!/usr/bin/env python3
"""
QUEENS-AORTA Parameter Configuration Module

This module defines the parameter distributions for abdominal aortic aneurysm (AAA)
geometry generation, providing both fitted distributions from clinical data and
fixed fallback distributions.
"""

import sys
import numpy as np
from pathlib import Path

# Setup QUEENS imports
sys.path.insert(0, '/home/a11evina/queens/src')

from queens.parameters import Parameters
from queens.distributions import Normal

# AORTA data path (for fitted distributions)
AORTA_DATA_PATH = '/home/a11evina/Aorta/ofCaseGen/Method_4/data'

def load_fitted_distributions(gender='F', age_group='70-79'):
    """
    Load fitted parameter distributions from AORTA clinical data.
    
    Args:
        gender: 'M' or 'F'  
        age_group: Age range string (e.g., '70-79')
        
    Returns:
        Parameters object with fitted distributions
        
    Raises:
        Exception: If fitted data is not available
    """
    # Attempt to load fitted data from AORTA
    data_file = Path(AORTA_DATA_PATH) / 'fitted_distributions' / f'{gender}_{age_group}.json'
    
    if not data_file.exists():
        raise Exception(f"No data for {gender} age {age_group}")
        
    # This would load actual fitted distributions
    # For now, we'll raise an exception to fall back to fixed distributions
    raise Exception(f"Fitted data loading not yet implemented for {gender} age {age_group}")


def create_aaa_parameters():
    """
    Create AAA parameter distributions using fixed (non-fitted) values.
    
    These are based on clinical literature and provide reasonable bounds
    for abdominal aortic aneurysm geometries.
    
    Returns:
        Parameters object with Normal distributions for geometry parameters
    """
    
    # Neck diameter 1 (mm) - proximal neck diameter
    # Clinical range: 18-30mm, typical: 22-26mm
    neck_diameter_1 = Normal(
        mean=2.5,  # 25mm in cm
        covariance=0.04  # std ≈ 2mm in cm
    )
    
    # Neck diameter 2 (mm) - distal neck diameter  
    # Typically slightly larger than neck 1
    # Clinical range: 20-32mm, typical: 24-30mm
    neck_diameter_2 = Normal(
        mean=2.8,  # 28mm in cm
        covariance=0.04  # std ≈ 2mm in cm
    )
    
    # Maximum aneurysm diameter (mm) - key diagnostic parameter
    # AAA definition: ≥30mm (3cm), typical range: 30-80mm
    # Mean around 55mm for surgical candidates
    max_diameter = Normal(
        mean=5.5,  # 55mm in cm
        covariance=0.25  # std ≈ 5mm in cm
    )
    
    # Distal diameter (mm) - diameter at distal end
    # Typically smaller than max, larger than normal aorta
    # Clinical range: 15-25mm, typical: 18-24mm
    distal_diameter = Normal(
        mean=2.2,  # 22mm in cm
        covariance=0.04  # std ≈ 2mm in cm
    )
    
    # Create QUEENS Parameters object
    parameters = Parameters(
        neck_diameter_1=neck_diameter_1,
        neck_diameter_2=neck_diameter_2,
        max_diameter=max_diameter,
        distal_diameter=distal_diameter
    )
    
    return parameters


def get_parameter_bounds():
    """
    Get reasonable bounds for AAA parameters for validation.

    Returns:
        dict: Parameter bounds in mm
    """
    return {
        'neck_diameter_1': (15, 35),    # mm
        'neck_diameter_2': (18, 38),    # mm
        'max_diameter': (30, 80),       # mm
        'distal_diameter': (12, 30),    # mm
    }


def validate_parameters(sample):
    """
    Validate that parameter sample is within reasonable clinical bounds.

    Args:
        sample: Array of [neck_diameter_1, neck_diameter_2, max_diameter, distal_diameter] in mm

    Returns:
        bool: True if parameters are valid

    Raises:
        ValueError: If parameters are invalid with explanation
    """
    neck_d1, neck_d2, max_d, distal_d = sample
    bounds = get_parameter_bounds()

    # Check individual parameter bounds
    if not (bounds['neck_diameter_1'][0] <= neck_d1 <= bounds['neck_diameter_1'][1]):
        raise ValueError(f"neck_diameter_1 ({neck_d1:.2f} mm) outside bounds {bounds['neck_diameter_1']} mm")

    if not (bounds['neck_diameter_2'][0] <= neck_d2 <= bounds['neck_diameter_2'][1]):
        raise ValueError(f"neck_diameter_2 ({neck_d2:.2f} mm) outside bounds {bounds['neck_diameter_2']} mm")

    if not (bounds['max_diameter'][0] <= max_d <= bounds['max_diameter'][1]):
        raise ValueError(f"max_diameter ({max_d:.2f} mm) outside bounds {bounds['max_diameter']} mm")

    if not (bounds['distal_diameter'][0] <= distal_d <= bounds['distal_diameter'][1]):
        raise ValueError(f"distal_diameter ({distal_d:.2f} mm) outside bounds {bounds['distal_diameter']} mm")

    # Check logical constraints
    if max_d <= max(neck_d1, neck_d2):
        raise ValueError(f"max_diameter ({max_d:.2f} mm) must be larger than neck diameters")

    if abs(neck_d2 - neck_d1) > 10.0:  # Necks shouldn't differ by more than 10mm
        raise ValueError(f"Neck diameter difference too large: {abs(neck_d2 - neck_d1):.2f} mm")

    return True


def print_parameter_summary(parameters):
    """Print a summary of the parameter distributions."""
    print("🔍 AAA Parameter Distributions Summary")
    print("=" * 50)
    
    for name, distribution in parameters.dict.items():
        mean_mm = (distribution.mean * 10).item()  # Convert cm to mm
        std_mm = (np.sqrt(distribution.covariance) * 10).item()
        print(f"{name:15s}: {mean_mm:5.1f} ± {std_mm:4.1f} mm")
    
    print("\n📊 Clinical Context:")
    print("   - Normal aorta diameter: ~20mm")
    print("   - AAA definition: ≥30mm max diameter")
    print("   - Surgical threshold: ~55mm max diameter")
    print("   - Parameters in cm for AORTA compatibility")


# Test function
def test_parameter_creation():
    """Test parameter creation and validation."""
    print("🧪 Testing AAA Parameter Creation")
    print("=" * 50)
    
    # Create parameters
    parameters = create_aaa_parameters()
    print_parameter_summary(parameters)
    
    # Test parameter validation
    print("\n🔍 Testing Parameter Validation:")
    
    # Valid sample (in mm)
    valid_sample = [25, 28, 55, 22]
    try:
        validate_parameters(valid_sample)
        print(f"✅ Valid sample: {valid_sample} mm")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")

    # Invalid sample (max < neck)
    invalid_sample = [25, 28, 20, 22]  # max_diameter too small
    try:
        validate_parameters(invalid_sample)
        print(f"❌ Should have failed: {invalid_sample} mm")
    except ValueError as e:
        print(f"✅ Correctly rejected invalid sample: {e}")
    
    return parameters


if __name__ == "__main__":
    params = test_parameter_creation()
    print(f"\n🎉 Parameter configuration test complete!")