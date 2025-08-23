#!/usr/bin/env python3
"""
Simplified surrogate data generation using OpenFOAM + probe values.
Based on existing test_openfoam_driver.py structure.
"""

import sys
import numpy as np
from pathlib import Path
import pickle
# Add the queens src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from queens.drivers.openfoam import OpenFoam
from queens.distributions import Uniform
from queens.parameters import Parameters
from queens.schedulers import Pool
from queens.models import Simulation
from queens.iterators import MonteCarlo
from queens.global_settings import GlobalSettings
from queens.main import run_iterator
from queens.data_processors.paraview import DataProcessorParaviewSubprocess


def generate_training_data(num_samples=5):
    """Generate training data using OpenFOAM + DataProcessorParaviewSubprocess."""

    # Create output directory if it doesn't exist
    from pathlib import Path
    Path("./surrogate_data").mkdir(exist_ok=True)    

    global_settings = GlobalSettings(
        experiment_name="surrogate_training",
        output_dir="./surrogate_data"
    )
    
    with global_settings:
        # Simple parameter ranges
        parameters = Parameters(
            lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
            initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
        )
        
        # Data processor for probe extraction
        data_processor = DataProcessorParaviewSubprocess(
            probe_locations=[(0.05, 0.05, 0.005), (0.01, 0.01, 0.005), (0.09, 0.09, 0.005)],
            data_fields=['U', 'p']
        )
        
        # OpenFOAM driver with data processor
        openfoam_driver = OpenFoam(
            parameters=parameters,
            case_template_dir="./cavity_template",
            solver="icoFoam",
            parallel=False,
            num_procs=1,
            openfoam_bashrc="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc",
            data_processor=data_processor  # Pass data processor to driver
        )
        
        scheduler = Pool(
            experiment_name=global_settings.experiment_name,
            num_jobs=2,
            verbose=True
        )
        
        # Simulation model without data processor
        model = Simulation(
            scheduler=scheduler,
            driver=openfoam_driver
        )
        
        iterator = MonteCarlo(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            num_samples=num_samples,
            seed=42,
            result_description={"write_results": True, "plot_results": False},
        )
        
        print(f"🚀 Generating {num_samples} samples...")
        run_iterator(iterator, global_settings=global_settings)
        
        # Extract processed data
        X_train, y_train = extract_processed_data(global_settings)
        
        return X_train, y_train


def extract_processed_data(global_settings):
    """Extract processed probe data from QUEENS results."""
    
    output_dir = Path(global_settings.output_dir) / global_settings.experiment_name
    
    # Load results from QUEENS output
    results_file = output_dir / "results.json"  # or results.pkl
    if results_file.exists():
        if results_file.suffix == '.json':
            import json
            with open(results_file, 'r') as f:
                data = json.load(f)
        else:
            with open(results_file, 'rb') as f:
                data = pickle.load(f)
        
        # Extract parameters and results
        X_train = np.array([sample['parameters'] for sample in data])
        y_train = np.array([sample['results'] for sample in data])
    else:
        print(f"Warning: Results file not found at {results_file}")
        return np.array([]), np.array([])
    
    # Save dataset
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    
    print(f"✅ Saved {len(X_train)} samples")
    return X_train, y_train


def extract_final_probe_value(probe_file):
    """Helper function kept for compatibility."""
    return [0]  # Not needed with DataProcessorParaviewSubprocess


if __name__ == "__main__":
    # Generate simple training data
    X_train, y_train = generate_training_data(num_samples=5)
    
    print(f"Training data shape:")
    print(f"  X (lid_velocity, pressure): {X_train.shape}")
    print(f"  y (velocity_mag, pressure):  {y_train.shape}")