#!/usr/bin/env python3
"""
FINAL FIXED Combined OpenFOAM + ParaView test script.
Runs OpenFOAM simulations and automatically processes results with ParaView.

KEY FIXES:
1. Uses custom OpenFoam driver that passes case_dir (not output_dir) to data_processor
2. Correct directory path handling for QUEENS  
3. Proper job directory naming (numeric, not job_*)
4. Fixed result extraction logic

The issue was: Jobscript base class calls data_processor.get_data_from_file(output_dir) 
where output_dir = job_dir/output/, but ParaView needs job_dir/ (the case directory).
"""

import sys
import os
import glob
import numpy as np
from pathlib import Path

# Adjust path to your QUEENS installation
sys.path.insert(0, '/home/a11evina/queens/src')

# Import the FIXED OpenFOAM driver
from queens.drivers.openfoam import OpenFoam 
from queens.distributions import Uniform
from queens.parameters import Parameters
from queens.schedulers import Pool
from queens.models import Simulation
from queens.iterators import MonteCarlo
from queens.global_settings import GlobalSettings
from queens.main import run_iterator
from queens.data_processors.paraview import DataProcessorParaview
from queens.utils.io import load_result


def create_combined_openfoam_paraview_workflow():
    """Create combined OpenFOAM + ParaView workflow with FIXED driver."""
    print("🔧 Phase 1: Setting up FIXED combined OpenFOAM + ParaView workflow...")
    
    # Global settings
    global_settings = GlobalSettings(
        experiment_name="openfoam_paraview_fixed",
        output_dir="./combined_output"
    )
    
    with global_settings:
        # Parameter definitions
        parameters = Parameters(
            lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
            initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
        )
        print("✅ Parameters defined")
        
        # ParaView data processor
        data_processor = DataProcessorParaview(
            field_name="U",
            file_name_identifier="foam.foam",
            file_options_dict={},
            probe_locations=[
                (0.05, 0.05, 0.005),  # center
                (0.01, 0.01, 0.005),  # bottom_left
                (0.09, 0.09, 0.005),  # top_right
            ],
            data_fields=['U', 'p'],
            time_step=-1
        )
        print("✅ ParaView data processor created")
        
        # FIXED OpenFOAM driver with integrated data processor
        driver = OpenFoam(  # <-- Using our FIXED driver
            parameters=parameters,
            case_template_dir="./cavity_template",
            solver="icoFoam",
            parallel=False,
            num_procs=1,
            container_command=None,
            openfoam_bashrc="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc",
            data_processor=data_processor  # This will now work correctly!
        )
        print("✅ FIXED OpenFOAM driver created with ParaView integration")
        
        # Scheduler
        scheduler = Pool(
            experiment_name=global_settings.experiment_name,
            num_jobs=2,
            verbose=True
        )
        print("✅ Scheduler created")
        
        # Model
        model = Simulation(scheduler=scheduler, driver=driver)
        print("✅ Model created")
        
        # Iterator (Monte Carlo)
        iterator = MonteCarlo(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            num_samples=5,
            seed=42,
            result_description={"write_results": True, "plot_results": False},
        )
        print(f"✅ Iterator created → ready to generate {iterator.num_samples} samples\n")
        
        return iterator, global_settings


def run_combined_workflow():
    """Run the complete workflow."""
    # Phase 1: Setup
    iterator, gs = create_combined_openfoam_paraview_workflow()
    if iterator is None:
        sys.exit(1)
    
    # Phase 2: Run simulations with integrated post-processing
    print("🚀 Phase 2: Running OpenFOAM simulations with FIXED ParaView post-processing...")
    run_iterator(iterator, global_settings=gs)
    
    # Phase 3: Analyze results and create surrogate data
    print("📊 Phase 3: Analyzing results and preparing surrogate modeling data...")
    X, Y = analyze_results(gs)
    
    return X, Y


def analyze_results(global_settings):
    """Analyze the combined results and create surrogate modeling pickle file."""
    try:
        # Load QUEENS results
        results = load_result(global_settings.result_file(".pickle"))
        
        print("\n=== SIMULATION RESULTS ===")
        print(f"Available keys in results: {list(results.keys())}")
        
        # Handle different possible result structures
        if 'input_data' in results:
            samples_key = 'input_data'
        elif 'samples' in results:
            samples_key = 'samples'
        elif 'input' in results:
            samples_key = 'input'
        elif 'raw_input_data' in results:
            samples_key = 'raw_input_data'
        else:
            print("❌ Cannot find input samples in results")
            print("Available keys:", list(results.keys()))
            return None, None
        
        X = results[samples_key]
        print(f"Number of samples: {len(X)}")
        print(f"Parameter samples shape: {X.shape}")
        
        # Debug: Let's see what's in raw_output_data
        if 'raw_output_data' in results:
            print(f"raw_output_data keys: {list(results['raw_output_data'].keys())}")
            print(f"raw_output_data structure: {type(results['raw_output_data'])}")
        
        # Extract X (parameters) and Y (probe outputs) for surrogate modeling
        X = results[samples_key]  # n × 2 matrix [lid_velocity, initial_pressure]
        
        if 'raw_output_data' in results and 'result' in results['raw_output_data']:
            Y = np.array(results['raw_output_data']['result'])
            
            print(f"✅ Successfully extracted data for surrogate modeling:")
            print(f"   X (parameters): {X.shape} - [lid_velocity, initial_pressure]")
            print(f"   Y (probe data): {Y.shape} - [probe0(Ux,Uy,Uz,p), probe1(...), probe2(...)]")
            
            # Check if we got valid data (not all zeros)
            if np.all(Y == 0):
                print("❌ ERROR: Still getting zero probe data! Debug needed...")
                debug_job_execution(global_settings)
                return X, Y  # Still return for analysis
            else:
                print("🎉 SUCCESS: Got non-zero probe data!")
            
            # Create surrogate modeling dataset
            surrogate_data = {
                'X_train': X,
                'Y_train': Y,
                'parameter_names': ['lid_velocity', 'initial_pressure'],
                'output_description': {
                    'probe_locations': [
                        (0.05, 0.05, 0.005),  # center
                        (0.01, 0.01, 0.005),  # bottom_left
                        (0.09, 0.09, 0.005),  # top_right
                    ],
                    'fields': ['Ux', 'Uy', 'Uz', 'p'],
                    'total_outputs': Y.shape[1]
                },
                'metadata': {
                    'solver': 'icoFoam',
                    'case_type': 'lid_driven_cavity',
                    'num_samples': len(X),
                    'parameter_ranges': {
                        'lid_velocity': [0.5, 2.0],
                        'initial_pressure': [-0.1, 0.1]
                    }
                }
            }
            
            # Save pickle file for surrogate modeling
            surrogate_file = Path(global_settings.output_dir) / f"{global_settings.experiment_name}_surrogate_data.pkl"
            import pickle
            with open(surrogate_file, 'wb') as f:
                pickle.dump(surrogate_data, f)
            
            print(f"\n🎯 SURROGATE MODELING DATA SAVED:")
            print(f"   File: {surrogate_file}")
            print(f"   Format: X_train{X.shape}, Y_train{Y.shape}")
            
            # Show sample data for verification
            print(f"\n📊 SAMPLE DATA PREVIEW:")
            for i in range(min(3, len(X))):
                print(f"   Sample {i}:")
                print(f"     Input:  lid_vel={X[i,0]:.3f}, init_p={X[i,1]:.3f}")
                if len(Y[i]) >= 12:
                    print(f"     Output: probe0=[{Y[i,0]:.4f},{Y[i,1]:.4f},{Y[i,2]:.4f},{Y[i,3]:.4f}], ...")
                else:
                    print(f"     Output: {Y[i]}")
            
            # Statistical summary
            print(f"\n📈 STATISTICAL SUMMARY:")
            print(f"   Parameter ranges:")
            print(f"     lid_velocity: [{X[:,0].min():.3f}, {X[:,0].max():.3f}] (mean: {X[:,0].mean():.3f})")
            print(f"     initial_pressure: [{X[:,1].min():.3f}, {X[:,1].max():.3f}] (mean: {X[:,1].mean():.3f})")
            print(f"   Output statistics:")
            print(f"     Mean: {np.mean(Y, axis=0)}")
            print(f"     Std:  {np.std(Y, axis=0)}")
            
        else:
            print("❌ No output data found - ParaView processing may have failed")
            Y = None
        
        # Check individual job directories for debugging
        debug_job_execution(global_settings)
        
        return X, Y if Y is not None else None
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def debug_job_execution(global_settings):
    """Debug job execution with correct directory paths."""
    print(f"\n=== JOB EXECUTION DEBUG ===")
    
    # Use correct QUEENS experiment directory structure
    from queens.utils.config_directories import experiment_directory
    experiment_dir = experiment_directory(global_settings.experiment_name)
    
    print(f"QUEENS experiment directory: {experiment_dir}")
    print(f"Directory exists: {experiment_dir.exists()}")
    
    if not experiment_dir.exists():
        print("❌ QUEENS experiment directory not found!")
        return
    
    # Find job directories - Look for numeric directories (0, 1, 2, etc.)
    job_dirs = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            job_dirs.append(item)
    
    job_dirs.sort(key=lambda x: int(x.name))  # Sort by numeric value
    
    print(f"Found {len(job_dirs)} job directories:")
    
    successful_jobs = 0
    for job_dir in job_dirs:
        job_id = job_dir.name
        success_indicators = []
        
        # Check OpenFOAM case structure
        if (job_dir / "0").exists():
            success_indicators.append("✅ OF_case")
        else:
            success_indicators.append("❌ OF_case")
        
        # Check solver execution
        log_files = list(job_dir.glob("log.*")) + list(job_dir.glob("output/*.log"))
        if log_files:
            success_indicators.append("✅ solver")
        else:
            success_indicators.append("❌ solver")
        
        # Check ParaView processing - NOW THIS SHOULD WORK!
        probe_files = list(job_dir.glob("probe_data.json"))
        if probe_files:
            success_indicators.append("✅ probes")
            successful_jobs += 1
            
            # Show actual probe data
            try:
                import json
                with open(probe_files[0], 'r') as f:
                    probe_data = json.load(f)
                
                # Extract a sample value to verify non-zero data
                first_probe = list(probe_data.keys())[0]
                sample_value = probe_data[first_probe]
                
                print(f"   Job {job_id}: {' | '.join(success_indicators)} | Sample: {sample_value}")
            except:
                print(f"   Job {job_id}: {' | '.join(success_indicators)} | Data: ERROR reading")
        else:
            success_indicators.append("❌ probes")
            print(f"   Job {job_id}: {' | '.join(success_indicators)}")
        
        # List actual contents for debugging
        contents = [item.name for item in job_dir.iterdir()][:10]  # First 10 items
        print(f"     Contents: {contents}")
    
    print(f"\n✅ Successfully processed: {successful_jobs}/{len(job_dirs)} jobs")


if __name__ == "__main__":
    print("🎯 QUEENS: FIXED Combined OpenFOAM + ParaView → Surrogate Modeling Pipeline")
    print("=" * 80)
    print("🔧 Using custom OpenFoam driver that passes case_dir to data_processor")
    print("=" * 80)
    
    # Run the full workflow with the FIXED driver
    X, Y = run_combined_workflow()
    
    # Show surrogate modeling example
    if X is not None and Y is not None:
        create_surrogate_modeling_example()
    
    print("\n✨ Workflow completed with FIXED OpenFOAM driver!")