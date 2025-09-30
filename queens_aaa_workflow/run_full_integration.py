#!/usr/bin/env python3
"""
Fixed QUEENS-AORTA Integration Test Script

This script demonstrates the complete integration between QUEENS and AORTA frameworks
for uncertainty quantification of abdominal aortic aneurysm (AAA) geometries.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Ensure we can import QUEENS
sys.path.insert(0, '/home/a11evina/queens/src')

# QUEENS imports
from queens.parameters import Parameters
from queens.distributions import Normal
from queens.iterators import LatinHypercubeSampling
from queens.global_settings import GlobalSettings

# Import our integration modules
from aorta_simple_wrapper import AortaGeometryModel
from queens_aaa_config import create_aaa_parameters, load_fitted_distributions

def create_output_directories():
    """Create required output directories."""
    directories = ['queens_output', 'geometries']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def main():
    """Main integration workflow."""
    
    print("🎯 Starting QUEENS-AORTA Integration Test")
    print("=" * 50)
    
    # Create output directories
    create_output_directories()
    
    try:
        # 1. Initialize GlobalSettings context
        output_dir = "queens_output"
        experiment_name = "aorta_integration_test"
        
        with GlobalSettings(
            experiment_name=experiment_name,
            output_dir=output_dir
        ) as global_settings:
            
            print("✅ GlobalSettings context active")
            
            # 2. Try to load fitted distributions, fall back to fixed if needed
            try:
                parameters = load_fitted_distributions('F', '70-79')
                print("✅ Loaded fitted parameter distributions")
            except Exception as e:
                print(f"⚠️  Could not load fitted distributions: {e}")
                print("🔄 Using fixed parameter distributions instead")
                parameters = create_aaa_parameters()
            
            # Display parameter information
            for name, distribution in parameters.dict.items():
                print(distribution)
                print()
            
            print(parameters)
            print()
            
            # 3. Create the AORTA geometry model
            model = AortaGeometryModel(
                output_dir='./geometries',
                demographic=('F', '70-79')
            )
            print("✅ AORTA model initialized")
            
            # 4. Create Latin Hypercube Sampler with model and global_settings
            iterator = LatinHypercubeSampling(
                model=model,                    # Required: model parameter
                parameters=parameters,          # Required: parameters
                global_settings=global_settings, # Required: global_settings
                seed=42,                        # Reproducible results
                num_samples=5,                  # Number of geometries to generate
                num_iterations=10,              # LHS optimization iterations
                criterion='maximin'             # LHS criterion
            )
            
            print(iterator)
            print()
            print("✅ Latin Hypercube Sampler initialized")
            
            # 5. Run the sampling workflow
            print("🚀 Starting geometry generation workflow...")
            
            # Pre-run: Generate samples
            iterator.pre_run()
            print(f"✅ Generated {len(iterator.samples)} sample points")
            
            # Core run: Evaluate model at sample points
            iterator.core_run()
            print(f"✅ Evaluated model at all sample points")
            
            # Post-run: Process results
            iterator.post_run()
            print("✅ Post-processing completed")
            
            # 6. Report results
            print("\n" + "=" * 50)
            print("📊 RESULTS SUMMARY")
            print("=" * 50)
            
            print(f"✅ Successfully generated {len(iterator.samples)} geometries")
            print(f"📁 Geometries saved to: ./geometries/")
            print(f"📊 QUEENS output saved to: {output_dir}/")
            
            # Display sample parameters
            print("\n🎯 Generated Parameter Samples:")
            param_names = list(parameters.dict.keys())
            for i, sample in enumerate(iterator.samples):
                print(f"  Sample {i+1}: {dict(zip(param_names, sample))}")
            
            # Check output files
            print("\n📁 Generated Files:")
            geometry_dir = Path('./geometries')
            if geometry_dir.exists():
                stl_files = list(geometry_dir.glob('*.stl'))
                print(f"  - {len(stl_files)} STL geometry files")
                for stl_file in stl_files:
                    print(f"    • {stl_file.name}")
            
            output_path = Path(output_dir)
            if output_path.exists():
                output_files = list(output_path.iterdir())
                print(f"  - {len(output_files)} QUEENS output files")
            
            print("\n🎉 Integration test completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ QUEENS-AORTA integration is working!")
        print("🚀 Ready for full uncertainty quantification workflow!")
    else:
        print("\n❌ Integration test failed - check errors above")
        sys.exit(1)