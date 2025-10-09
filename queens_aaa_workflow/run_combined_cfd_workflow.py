#!/usr/bin/env python3
"""
QUEENS-AORTA Combined CFD Workflow (Single Driver Approach)

This script demonstrates the combined driver that executes the FULL pipeline:
  Parameters → Geometry → OpenFOAM Case → Meshing → Solver → Results

Each job runs the complete workflow for one parameter sample.
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

# Import our combined driver
from aorta_cfd_driver import AortaCFDDriver
from queens_aaa_config import create_aaa_parameters, load_fitted_distributions


def main():
    """Main workflow using combined CFD driver."""

    print("🎯 QUEENS-AORTA Combined CFD Workflow")
    print("=" * 70)
    print("Single driver executes: Geometry → Meshing → Solver → Results")
    print("=" * 70)

    # Create output directories
    os.makedirs('geometries', exist_ok=True)
    os.makedirs('openfoam_cases', exist_ok=True)

    try:
        # 1. Initialize GlobalSettings
        output_dir = "queens_cfd_output"
        experiment_name = "aorta_cfd_combined"

        os.makedirs(output_dir, exist_ok=True)

        with GlobalSettings(
            experiment_name=experiment_name,
            output_dir=output_dir
        ) as global_settings:

            print("✅ GlobalSettings context active")

            # 2. Load parameters
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

            # 3. Create Combined CFD Driver
            convex_hull_path = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4' / 'data' / 'processed' / 'convex_hull_metadata.json'

            driver = AortaCFDDriver(
                parameters=parameters,
                geometry_output_dir='./geometries',
                openfoam_cases_dir='./openfoam_cases',
                enable_morphing=False,
                convex_hull_metadata_path=str(convex_hull_path) if convex_hull_path.exists() else None,
                gender='F',
                age_group='70-79',
                outlier_method='manual',
                # OpenFOAM simulation settings
                openfoam_bashrc="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc",
                solver="pimpleFoam",
                parallel=False,
                num_procs_sim=1,
                run_blockmesh=True,
                run_snappyhexmesh=True,
                run_solver=False,  # Set to True to run full CFD solver (takes long time!)
                # Result extraction
                extract_results=True,
                result_fields=['U', 'p'],
                result_time='latestTime'
            )
            print("✅ Combined CFD Driver initialized")

            # 4. Create Scheduler
            scheduler = Pool(
                experiment_name=global_settings.experiment_name,
                num_jobs=2,  # Run 2 complete pipelines in parallel
                verbose=True
            )
            print("✅ Pool Scheduler created (num_jobs=2)")

            # 5. Create Simulation Model
            model = Simulation(
                scheduler=scheduler,
                driver=driver
            )
            print("✅ Simulation Model created")

            # 6. Create Latin Hypercube Sampler
            iterator = LatinHypercubeSampling(
                model=model,
                parameters=parameters,
                global_settings=global_settings,
                seed=42,
                num_samples=3,  # Small number for testing
                num_iterations=10,
                criterion='maximin'
            )

            print(f"✅ Latin Hypercube Sampler initialized")
            print(f"   Samples: {iterator.num_samples}")
            print(f"   Criterion: {iterator.criterion}")

            # 7. Run the complete workflow
            print("\n" + "=" * 70)
            print("🚀 STARTING COMBINED CFD WORKFLOW")
            print("=" * 70)

            # Pre-run: Generate samples
            print("\n📝 Phase 1: Generating LHS samples...")
            iterator.pre_run()
            print(f"✅ Generated {len(iterator.samples)} sample points")

            print("\n🎯 Sample Preview (before filtering):")
            param_names = list(parameters.dict.keys())
            for i, sample in enumerate(iterator.samples):
                sample_dict = {name: f"{val:.1f}" for name, val in zip(param_names, sample)}
                print(f"   Sample {i+1}: {sample_dict}")

            # Filter samples to keep only those inside convex hull
            if convex_hull_path.exists() and driver.convex_hull_data:
                print("\n🔍 Filtering samples: keeping only those inside convex hull...")
                original_count = len(iterator.samples)
                filtered_samples = []

                for i, sample in enumerate(iterator.samples):
                    validation_results = driver._validate_parameters(sample)
                    if validation_results['all_valid']:
                        filtered_samples.append(sample)
                        print(f"   ✅ Sample {i+1}: INSIDE hull - keeping")
                    else:
                        print(f"   ❌ Sample {i+1}: OUTSIDE hull - skipping")

                iterator.samples = filtered_samples
                print(f"\n📊 Filtering complete: {len(filtered_samples)}/{original_count} samples inside convex hull")

                if len(filtered_samples) == 0:
                    print("⚠️  WARNING: No samples inside convex hull! Exiting.")
                    return False
            else:
                print("\n⚠️  No convex hull filtering applied (convex hull data not available)")

            print("\n🎯 Final Sample Set:")
            for i, sample in enumerate(iterator.samples):
                sample_dict = {name: f"{val:.1f}" for name, val in zip(param_names, sample)}
                print(f"   Sample {i+1}: {sample_dict}")

            # Core run: Execute FULL pipeline for each sample
            print("\n" + "=" * 70)
            print("⚡ Phase 2: Running FULL CFD pipeline (Geometry → Mesh → Solver)")
            print("   QUEENS Pool scheduler will run jobs in parallel")
            print("=" * 70)

            iterator.core_run()

            print(f"\n✅ Complete CFD workflow executed!")

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

                    # Check for pipeline results
                    pipeline_file = job_dir / 'pipeline_results.json'

                    if pipeline_file.exists():
                        import json
                        with open(pipeline_file, 'r') as f:
                            pipeline_data = json.load(f)

                        success = "✅" if pipeline_data.get('success') else "❌"
                        stages = " → ".join(pipeline_data.get('stages_completed', []))
                        print(f"   Job {job_id}: {success} Stages: {stages}")

                        if not pipeline_data.get('success') and 'error' in pipeline_data:
                            print(f"      Error: {pipeline_data['error']}")

            # Check output files
            print("\n📁 Generated Files:")

            geometry_dir = Path('./geometries')
            if geometry_dir.exists():
                case_dirs = [d for d in geometry_dir.iterdir() if d.is_dir()]
                print(f"   ✅ {len(case_dirs)} geometry directories")

            openfoam_dir = Path('./openfoam_cases')
            if openfoam_dir.exists():
                of_case_dirs = [d for d in openfoam_dir.iterdir() if d.is_dir()]
                print(f"   ✅ {len(of_case_dirs)} OpenFOAM case directories")

                for of_case_dir in sorted(of_case_dirs):
                    # Check mesh
                    mesh_exists = (of_case_dir / 'constant' / 'polyMesh' / 'points').exists()
                    # Check solver results
                    time_dirs = [d for d in of_case_dir.iterdir()
                                if d.is_dir() and d.name.replace('.', '').replace('-', '').isdigit()]
                    solver_ran = len(time_dirs) > 1  # More than just '0' directory

                    mesh_status = "✅ mesh" if mesh_exists else "⚠️ no mesh"
                    solver_status = "✅ solver" if solver_ran else "⏭️ no solver"

                    print(f"      {of_case_dir.name}: {mesh_status}, {solver_status}")

            print("\n🎉 Combined CFD workflow completed!")

            return True

    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n" + "="*70)
        print("✅ QUEENS-AORTA COMBINED CFD WORKFLOW COMPLETE")
        print("="*70)

        print("\n📦 What was created:")
        print("   1. ✅ Parametric AAA geometries")
        print("   2. ✅ Complete OpenFOAM cases with meshes")
        print("   3. ✅ CFD simulation results (if solver was enabled)")
        print("   4. ✅ QUEENS experiment data with pipeline results")

        print("\n🔍 Architecture:")
        print("   • Single Driver: AortaCFDDriver (Complete pipeline)")
        print("   • Scheduler: Pool (parallel execution of complete workflows)")
        print("   • Model: Simulation (QUEENS standard)")
        print("   • Iterator: LatinHypercubeSampling")

        print("\n🚀 Next steps:")
        print("   • Set run_solver=True to run actual CFD solver")
        print("   • Increase num_samples and num_jobs for production runs")
        print("   • Customize _extract_results() in driver for your QoI")
        print("   • Switch to SlurmScheduler for HPC cluster execution")

    else:
        print("\n❌ Workflow failed - check errors above")
        sys.exit(1)
