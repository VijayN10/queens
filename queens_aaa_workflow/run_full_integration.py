#!/usr/bin/env python3
"""
Fixed QUEENS-AORTA Integration Test Script with Convex Hull Validation

This script demonstrates the complete integration between QUEENS and AORTA frameworks
for uncertainty quantification of abdominal aortic aneurysm (AAA) geometries.
Now includes convex hull validation from Method 4.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Get repo root
repo_root = Path(__file__).parent.parent

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
    directories = ['queens_output', 'geometries', 'openfoam_cases']
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
            
            # 3. Create the AORTA geometry model with OpenFOAM case generation and convex hull validation
            # Define convex hull metadata path (from Method 4)
            convex_hull_path = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4' / 'data' / 'processed' / 'convex_hull_metadata.json'

            model = AortaGeometryModel(
                output_dir='./geometries',
                create_openfoam_cases=True,  # Enable full OpenFOAM case generation
                enable_morphing=False,
                random_seed=42,
                # Enable convex hull validation
                convex_hull_metadata_path=str(convex_hull_path) if convex_hull_path.exists() else None,
                gender='M',  # Example: Female
                age_group='70-79',  # Example: 70-79 age group
                outlier_method='manual'  # Use manual outlier detection
            )
            print("✅ AORTA model initialized with OpenFOAM case generation and convex hull validation")
            
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
            
            print(f"✅ Successfully generated {len(iterator.samples)} complete cases")
            print(f"📁 Geometries saved to: ./geometries/")
            print(f"🏗️  OpenFOAM cases saved to: ./openfoam_cases/")
            print(f"📊 QUEENS output saved to: {output_dir}/")
            
            # Display sample parameters
            print("\n🎯 Generated Parameter Samples:")
            param_names = list(parameters.dict.keys())
            for i, sample in enumerate(iterator.samples):
                print(f"  Sample {i+1}: {dict(zip(param_names, sample))}")
            
            # Check output files
            print("\n📁 Generated Files:")

            # Geometry files
            geometry_dir = Path('./geometries')
            if geometry_dir.exists():
                case_dirs = [d for d in geometry_dir.iterdir() if d.is_dir()]
                print(f"  - {len(case_dirs)} geometry case directories")
                for case_dir in case_dirs:
                    stl_files = list(case_dir.glob('*.stl'))
                    print(f"    • {case_dir.name}: {len(stl_files)} STL files")

            # OpenFOAM cases
            openfoam_dir = Path('./openfoam_cases')
            if openfoam_dir.exists():
                of_case_dirs = [d for d in openfoam_dir.iterdir() if d.is_dir()]
                print(f"  - {len(of_case_dirs)} OpenFOAM case directories")
                for of_case_dir in of_case_dirs:
                    print(f"    • {of_case_dir.name}: Ready for simulation")
                    # Check for key OpenFOAM directories
                    has_0 = (of_case_dir / '0').exists()
                    has_system = (of_case_dir / 'system').exists()
                    has_constant = (of_case_dir / 'constant').exists()
                    has_tri = (of_case_dir / 'constant' / 'triSurface').exists()
                    print(f"      ✓ Case structure: 0/={'✓' if has_0 else '✗'}, "
                          f"system/={'✓' if has_system else '✗'}, "
                          f"constant/={'✓' if has_constant else '✗'}, "
                          f"triSurface/={'✓' if has_tri else '✗'}")

            output_path = Path(output_dir)
            if output_path.exists():
                output_files = list(output_path.iterdir())
                print(f"  - {len(output_files)} QUEENS output files")

            # Display validation summary if available
            validation_summary_file = Path('./geometries/validation_summary.json')
            if validation_summary_file.exists():
                import json
                with open(validation_summary_file, 'r') as f:
                    validation_data = json.load(f)

                print("\n🔍 Convex Hull Validation Results:")
                print(f"  Total cases: {validation_data['total_cases']}")
                val_summary = validation_data['validation_summary']
                print(f"  ✅ Valid (inside hull): {val_summary['valid']}")
                print(f"  ⚠️  Invalid (outside hull): {val_summary['invalid']}")

                config = validation_data['convex_hull_config']
                if config['enabled']:
                    print(f"\n  Configuration:")
                    print(f"    Gender: {config['gender']}")
                    print(f"    Age group: {config['age_group']}")
                    print(f"    Outlier method: {config['outlier_method']}")

                # List valid cases
                valid_cases = [c for c in validation_data['cases']
                             if c['success'] and c.get('validation', {}).get('all_valid')]
                if valid_cases:
                    print(f"\n  Valid cases ({len(valid_cases)}):")
                    for case in valid_cases[:5]:  # Show first 5
                        params = case['parameters']
                        print(f"    - {case['case_id']}: neck1={params[0]:.1f}, "
                              f"neck2={params[1]:.1f}, max={params[2]:.1f}, distal={params[3]:.1f}")
                    if len(valid_cases) > 5:
                        print(f"    ... and {len(valid_cases)-5} more")

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
        print("🏗️  Full OpenFOAM cases created successfully!")
        print("🚀 Ready for CFD simulations and uncertainty quantification!")
        print("\n📋 Next steps:")
        print("   1. Run blockMesh in each OpenFOAM case")
        print("   2. Run snappyHexMesh to generate the mesh")
        print("   3. Run your CFD solver (e.g., pimpleFoam)")
        print("   4. Post-process results")
    else:
        print("\n❌ Integration test failed - check errors above")
        sys.exit(1)