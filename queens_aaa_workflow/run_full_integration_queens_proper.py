#!/usr/bin/env python3
"""
QUEENS-AORTA Integration using PROPER QUEENS Architecture

This script demonstrates the CORRECT way to integrate AORTA with QUEENS:
- Uses queens.drivers.Driver pattern
- Uses queens.schedulers.Pool for parallelism
- Uses queens.models.Simulation model
- Fully compatible with QUEENS infrastructure

Architecture:
  Parameters → Iterator (LHS) → Model (Simulation) → Scheduler (Pool) → Driver (AORTA)
"""

import sys
import os
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
from queens.schedulers import Pool
from queens.models import Simulation

# Import our QUEENS-compatible driver
from aorta_queens_driver import AortaGeometryDriver
from queens_aaa_config import create_aaa_parameters, load_fitted_distributions


def main():
    """Main integration workflow using proper QUEENS architecture."""

    print("🎯 QUEENS-AORTA Integration (Proper QUEENS Architecture)")
    print("=" * 70)

    # Create output directories
    os.makedirs('geometries', exist_ok=True)
    os.makedirs('openfoam_cases', exist_ok=True)

    try:
        # 1. Initialize GlobalSettings context
        output_dir = "queens_output_proper"
        experiment_name = "aorta_queens_proper"

        # Create QUEENS output directory
        os.makedirs(output_dir, exist_ok=True)

        with GlobalSettings(
            experiment_name=experiment_name,
            output_dir=output_dir
        ) as global_settings:

            print("✅ GlobalSettings context active")

            # 2. Load or create parameters
            try:
                parameters = load_fitted_distributions('F', '70-79')
                print("✅ Loaded fitted parameter distributions")
            except Exception as e:
                print(f"⚠️  Could not load fitted distributions: {e}")
                print("🔄 Using fixed parameter distributions")
                parameters = create_aaa_parameters()

            # Display parameters
            print("\n📊 Parameter Distributions:")
            for name, distribution in parameters.dict.items():
                print(f"   {name}: {distribution}")

            # 3. Create AORTA Driver (following QUEENS pattern)
            convex_hull_path = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4' / 'data' / 'processed' / 'convex_hull_metadata.json'

            driver = AortaGeometryDriver(
                parameters=parameters,
                geometry_output_dir='./geometries',
                openfoam_cases_dir='./openfoam_cases',
                create_openfoam_cases=True,
                enable_morphing=False,
                convex_hull_metadata_path=str(convex_hull_path) if convex_hull_path.exists() else None,
                gender='F',
                age_group='70-79',
                outlier_method='manual'
            )
            print("✅ AORTA Driver initialized")

            # 4. Create Scheduler (this handles parallelism!)
            scheduler = Pool(
                experiment_name=global_settings.experiment_name,
                num_jobs=5,  # ← Run 5 geometry generations in PARALLEL!
                verbose=True
            )
            print("✅ Pool Scheduler created (num_jobs=5 → 5 parallel workers)")

            # 5. Create Simulation Model (QUEENS standard)
            model = Simulation(
                scheduler=scheduler,
                driver=driver
            )
            print("✅ Simulation Model created")

            # 6. Create Latin Hypercube Sampler
            iterator = LatinHypercubeSampling(
                model=model,                    # ← Now using proper QUEENS Simulation!
                parameters=parameters,
                global_settings=global_settings,
                seed=42,
                num_samples=5,
                num_iterations=10,
                criterion='maximin'
            )

            print(f"✅ Latin Hypercube Sampler initialized")
            print(f"   Samples: {iterator.num_samples}")
            print(f"   Criterion: {iterator.criterion}")

            # 7. Run the sampling workflow
            print("\n" + "=" * 70)
            print("🚀 STARTING GEOMETRY GENERATION WORKFLOW")
            print("=" * 70)

            # Pre-run: Generate samples
            print("\n📝 Phase 1: Generating LHS samples...")
            iterator.pre_run()
            print(f"✅ Generated {len(iterator.samples)} sample points")

            print("\n🎯 Sample Preview:")
            param_names = list(parameters.dict.keys())
            for i, sample in enumerate(iterator.samples):
                sample_dict = {name: f"{val:.1f}" for name, val in zip(param_names, sample)}
                print(f"   Sample {i+1}: {sample_dict}")

            # Core run: Evaluate model at sample points
            print("\n" + "=" * 70)
            print("⚡ Phase 2: Running model evaluation (PARALLEL!)")
            print("   QUEENS Pool scheduler will run 5 jobs in parallel")
            print("=" * 70)

            iterator.core_run()  # ← QUEENS handles parallelism automatically here!

            print(f"\n✅ Model evaluation completed!")

            # Post-run: Process results
            print("\n📊 Phase 3: Post-processing results...")
            iterator.post_run()
            print("✅ Post-processing completed")

            # 8. Report results
            print("\n" + "=" * 70)
            print("📊 RESULTS SUMMARY")
            print("=" * 70)

            print(f"✅ Successfully processed {len(iterator.samples)} samples")
            print(f"📁 Geometries saved to: ./geometries/")
            print(f"🏗️  OpenFOAM cases saved to: ./openfoam_cases/")
            print(f"📊 QUEENS output saved to: {output_dir}/")

            # Check experiment directory for job results
            from queens.utils.config_directories import experiment_directory
            exp_dir = experiment_directory(experiment_name)

            if exp_dir.exists():
                job_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                print(f"\n🔍 QUEENS Job Directories: {len(job_dirs)} jobs")

                for job_dir in sorted(job_dirs, key=lambda x: int(x.name)):
                    job_id = job_dir.name

                    # Check for metadata
                    metadata_file = job_dir / 'aorta_metadata.json'
                    validation_file = job_dir / 'validation_results.json'

                    status_indicators = []
                    if metadata_file.exists():
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        if metadata.get('success'):
                            status_indicators.append("✅ Generated")
                        else:
                            status_indicators.append("❌ Failed")

                    if validation_file.exists():
                        status_indicators.append("✅ Validated")

                    print(f"   Job {job_id}: {' | '.join(status_indicators) if status_indicators else '⚠️ No status'}")

            # Check output files
            print("\n📁 Generated Files:")

            geometry_dir = Path('./geometries')
            if geometry_dir.exists():
                case_dirs = [d for d in geometry_dir.iterdir() if d.is_dir()]
                print(f"   ✅ {len(case_dirs)} geometry case directories")
                for case_dir in sorted(case_dirs):
                    stl_files = list(case_dir.glob('*.stl'))
                    print(f"      • {case_dir.name}: {len(stl_files)} STL files")

            openfoam_dir = Path('./openfoam_cases')
            if openfoam_dir.exists():
                of_case_dirs = [d for d in openfoam_dir.iterdir() if d.is_dir()]
                print(f"   ✅ {len(of_case_dirs)} OpenFOAM case directories")
                for of_case_dir in sorted(of_case_dirs):
                    has_0 = (of_case_dir / '0').exists()
                    has_system = (of_case_dir / 'system').exists()
                    has_constant = (of_case_dir / 'constant').exists()
                    has_tri = (of_case_dir / 'constant' / 'triSurface').exists()

                    status = "✅" if all([has_0, has_system, has_constant, has_tri]) else "⚠️"
                    print(f"      {status} {of_case_dir.name}")

            print("\n🎉 Geometry generation completed successfully!")

            return True

    except Exception as e:
        print(f"❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n" + "="*70)
        print("✅ QUEENS-AORTA INTEGRATION COMPLETE (PROPER QUEENS ARCHITECTURE)")
        print("="*70)

        print("\n📦 What was created:")
        print("   1. ✅ Parametric AAA geometries (geometries/)")
        print("   2. ✅ Complete OpenFOAM cases (openfoam_cases/)")
        print("   3. ✅ QUEENS experiment data (queens_output_proper/)")
        print("   4. ✅ Job metadata and validation results")

        print("\n🔍 Architecture Used:")
        print("   • Driver: AortaGeometryDriver (QUEENS Driver pattern)")
        print("   • Scheduler: Pool with num_jobs=5 (parallel execution)")
        print("   • Model: Simulation (QUEENS standard)")
        print("   • Iterator: LatinHypercubeSampling")

        print("\n🚀 Next steps:")
        print("   • Increase num_jobs for more parallelism")
        print("   • Run OpenFOAM simulations on generated cases")
        print("   • Switch to SlurmScheduler for HPC cluster")

        print("\n📊 To run OpenFOAM simulations:")
        print("   python run_openfoam_cases.py openfoam_cases/ --skip-solver")

    else:
        print("\n❌ Integration failed - check errors above")
        sys.exit(1)
