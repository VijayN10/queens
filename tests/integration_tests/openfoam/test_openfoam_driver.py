#!/usr/bin/env python3
"""
Test the OpenFOAM driver with cavity case, including actual case generation.
"""

import sys
import os
import glob
sys.path.insert(0, '/Users/user/Desktop/Vijay-Nandurdikar-PhD/queens/src')  # Adjust path to your QUEENS installation

from queens.drivers.openfoam import OpenFoam
from queens.distributions import Uniform
from queens.parameters import Parameters
from queens.schedulers import Pool
from queens.models import Simulation
from queens.iterators import MonteCarlo
from queens.global_settings import GlobalSettings
from queens.main import run_iterator


def test_openfoam_driver():
    """Test OpenFOAM driver initialization and setup."""
    print("🔧 Phase 1: Initialization checks...")
    
    # 1) Global settings
    global_settings = GlobalSettings(
        experiment_name="openfoam_driver_test",
        output_dir="./test_output"
    )
    
    with global_settings:
        # 2) Parameter definitions
        parameters = Parameters(
            lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
            initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
        )
        
        # 3) Driver
        driver = OpenFoam(
            parameters=parameters,
            case_template_dir="./cavity_template",
            solver="icoFoam",
            parallel=False,
            num_procs=1,
            container_command=None,  # or your container cmd if needed
            openfoam_bashrc="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc"
        )
        print("✅ Driver initialized")
        
        # 4) Scheduler
        scheduler = Pool(
            experiment_name=global_settings.experiment_name,
            num_jobs=1,
            verbose=True
        )
        print("✅ Scheduler created")
        
        # 5) Model
        model = Simulation(scheduler=scheduler, driver=driver)
        print("✅ Model created")
        
        # 6) Iterator (Monte Carlo)
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


if __name__ == "__main__":
    iterator, gs = test_openfoam_driver()
    if iterator is None:
        sys.exit(1)
    
    # Phase 2: actually run and generate the case‐folders
    print("🚀 Phase 2: Generating cases and rendering templates...")
    run_iterator(iterator, global_settings=gs)
    
    # Phase 3: quick check of output directories
    out_base = os.path.join(gs.output_dir, gs.experiment_name)
    jobs = sorted(glob.glob(os.path.join(out_base, "job_*")))
    print(f"\nGenerated {len(jobs)} jobs under:\n  {out_base}\n")
    for j in jobs:
        print("  ", os.path.basename(j), "→ contains:", os.listdir(j))
