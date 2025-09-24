#!/usr/bin/env python3
"""
Simple QUEENS-AORTA Integration Test
Tests basic integration without GlobalSettings complexity
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, '/home/a11evina/queens/src')
sys.path.insert(0, '/home/a11evina/Aorta/ofCaseGen/Method_4')

# QUEENS imports
from queens.parameters import Parameters
from queens.distributions import Normal
from queens.iterators import LatinHypercubeSampling
from queens.global_settings import GlobalSettings
from queens.schedulers import Pool

# Local imports - use the simple wrapper instead
from aorta_simple_wrapper import AortaGeometryModel

def main():
    """Simple test of QUEENS-AORTA integration."""
    
    print("=" * 60)
    print("Simple QUEENS-AORTA Test")
    print("=" * 60)
    
    print("1. Creating parameters...")
    parameters = Parameters(
        neck_d1=Normal(mean=22, covariance=4),      
        neck_d2=Normal(mean=18, covariance=2.25),   
        sac_d1=Normal(mean=32, covariance=6.25),    
        sac_d2=Normal(mean=28, covariance=4),       
        sac_d3=Normal(mean=34, covariance=9),       
        sac_l=Normal(mean=45, covariance=16),       
    )
    print("✅ Parameters created")
    
    print("2. Setting up GlobalSettings...")
    global_settings = GlobalSettings(
        experiment_name="aorta_test",
        output_dir="test_output"
    )
    print("✅ GlobalSettings created")
    
    print("3. Creating model...")
    # Use simple wrapper - no scheduler/driver needed
    model = AortaGeometryModel(output_dir="test_geometries")
    print("✅ Model created")
    
    print("4. Testing direct evaluation...")
    # Test direct evaluation first
    test_sample = np.array([[22.0, 18.0, 32.0, 28.0, 34.0, 45.0]])  # 6 parameters
    print(f"Test sample shape: {test_sample.shape}")
    
    result = model.evaluate(test_sample)
    print(f"✅ Direct evaluation successful! Result: {result}")
    
    print("5. Generating samples with LHS...")
    # For LHS, we need to create a minimal driver and scheduler if using Simulation
    # But with simple wrapper, we can use it directly
    
    # Generate samples directly
    lhs_samples = parameters.draw(2)  # Generate 2 samples directly from parameters
    print(f"LHS samples shape: {lhs_samples.shape}")
    
    print("6. Evaluating LHS samples...")
    lhs_results = model.evaluate(lhs_samples)
    print(f"✅ LHS evaluation successful! Results: {lhs_results}")
    
    return {
        'samples': lhs_samples,
        'results': lhs_results,
        'parameters': parameters,
        'num_samples': len(lhs_samples)
    }

if __name__ == "__main__":
    import numpy as np
    
    # Create output directories
    os.makedirs("test_output", exist_ok=True)
    os.makedirs("test_geometries", exist_ok=True)
    
    results = main()
    print(f"\n🎉 Test completed successfully!")
    print(f"Generated {results['num_samples']} geometry samples")