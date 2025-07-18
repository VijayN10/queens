#!/usr/bin/env python3
"""
Test the OpenFOAM driver with cavity case
"""

import sys
sys.path.insert(0, '/Users/user/Desktop/Vijay-Nandurdikar-PhD/queens/src')  # Adjust path to your QUEENS installation

from queens.drivers.openfoam import OpenFoam, OpenFoamLogProcessor
from queens.distributions import Uniform
from queens.parameters import Parameters
from queens.schedulers import Pool
from queens.models import Simulation
from queens.iterators import MonteCarlo
from queens.global_settings import GlobalSettings

def test_openfoam_driver():
    """Test OpenFOAM driver initialization and setup."""
    
    print("Testing OpenFOAM Driver Phase 1...")
    
    # Set up global settings
    global_settings = GlobalSettings(
        experiment_name="openfoam_driver_test",
        output_dir="./test_output"
    )
    
    with global_settings:
        # Define parameters
        parameters = Parameters(
            lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
            initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
        )
        
        # # Create data processor (Not working)
        # data_processor = OpenFoamLogProcessor(
        #     extract_residuals=True,
        #     extract_forces=False
        # )
        
        # Test driver initialization
        try:
            driver = OpenFoam(
                parameters=parameters,
                case_template_dir="./cavity_template",
                solver="icoFoam",
                parallel=False,
                num_procs=1,
                container_command="openfoam9-macos -d $HOME/openfoam",
                # data_processor=data_processor
            )
            print("✅ Driver initialized successfully")
            
            # Test scheduler
            scheduler = Pool(
                experiment_name=global_settings.experiment_name,
                num_jobs=1,
                verbose=True
            )
            print("✅ Scheduler created successfully")
            
            # Test model
            model = Simulation(scheduler=scheduler, driver=driver)
            print("✅ Model created successfully")
            
            # Test iterator (don't run yet, just create)
            iterator = MonteCarlo(
                model=model,
                parameters=parameters,
                global_settings=global_settings,
                num_samples=2,
                seed=42,
                result_description={"write_results": True}
            )
            print("✅ Iterator created successfully")
            
            print("\n🎉 OpenFOAM Driver Phase 1 test passed!")
            print("Ready to run with: run_iterator(iterator, global_settings)")
            
            return iterator, global_settings
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None

if __name__ == "__main__":
    test_openfoam_driver()