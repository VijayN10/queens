#!/usr/bin/env python
"""Full QUEENS-AORTA integration test."""

import sys
from pathlib import Path
import numpy as np

# Setup paths
sys.path.insert(0, '/home/a11evina/queens/src')
sys.path.insert(0, '/home/a11evina/Aorta/ofCaseGen/Method_4')

# QUEENS imports
from queens.global_settings import GlobalSettings
from queens.parameters import Parameters
from queens.distributions import Normal
from queens.iterators import LatinHypercubeSampling

# Import our model
from aorta_queens_model import AortaGeometryModel

def main():
    """Run geometry generation through QUEENS."""
    
    print("="*60)
    print("QUEENS-AORTA Full Integration")
    print("="*60)
    
    # Set up QUEENS global settings
    global_settings = GlobalSettings(
        experiment_name="aaa_geometry_generation",
        output_dir="./queens_output"
    )
    
    with global_settings:
        # Define parameters with distributions
        print("\n📊 Setting up parameters...")
        parameters = Parameters(
            neck_d1=Normal(mean=22, std=2),
            neck_d2=Normal(mean=23, std=2),
            max_d=Normal(mean=45, std=5),
            distal_d=Normal(mean=18, std=2)
        )
        
        # Create geometry model
        print("\n🏗️ Initializing AORTA geometry model...")
        model = AortaGeometryModel(
            output_dir='./geometries',
            enable_perturbation=True
        )
        
        # Set up Latin Hypercube Sampling
        print("\n🎲 Setting up Latin Hypercube Sampling...")
        lhs = LatinHypercubeSampling(
            parameters=parameters,
            seed=42
        )
        
        # Generate samples
        n_samples = 3  # Start with just 3 for testing
        print(f"\n📐 Drawing {n_samples} parameter samples...")
        samples = lhs.draw(n_samples)
        
        print("\nParameter samples:")
        for i, sample in enumerate(samples):
            print(f"  Sample {i+1}: neck_d1={sample[0]:.2f}, neck_d2={sample[1]:.2f}, "
                  f"max_d={sample[2]:.2f}, distal_d={sample[3]:.2f}")
        
        # Generate geometries
        print(f"\n🔄 Generating {n_samples} geometries...")
        results = model.evaluate(samples)
        
        # Summary
        print("\n" + "="*60)
        print("✅ GEOMETRY GENERATION COMPLETE")
        print("="*60)
        
        success_count = np.sum(results == 1.0)
        print(f"Successfully generated: {success_count}/{n_samples} geometries")
        print(f"Output directory: ./geometries/")
        
        if model.geometry_registry:
            print("\nGenerated files:")
            for case_id, info in model.geometry_registry.items():
                print(f"  - {case_id}: {info['stl_file']}")
        
        return results

if __name__ == "__main__":
    results = main()