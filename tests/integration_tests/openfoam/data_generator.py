#!/usr/bin/env python3
"""
Simplified surrogate data generation using OpenFOAM + probe values.
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
from queens.data_processors.paraview import DataProcessorParaview


def generate_training_data(num_samples=5):
    """Generate training data using OpenFOAM + DataProcessorParaview."""
    
    # Create output directory
    Path("./surrogate_data").mkdir(exist_ok=True)    

    global_settings = GlobalSettings(
        experiment_name="surrogate_training",
        output_dir="./surrogate_data"
    )
    
    with global_settings:
        # Parameter ranges
        parameters = Parameters(
            lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
            initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
        )
        
        # Data processor for probe extraction
        data_processor = DataProcessorParaview(
            probe_locations=[(0.05, 0.05, 0.005), (0.02, 0.02, 0.005), (0.08, 0.08, 0.005)],
            data_fields=['U', 'p']
        )
        
        # OpenFOAM driver
        openfoam_driver = OpenFoam(
            parameters=parameters,
            case_template_dir="./cavity_template",
            solver="icoFoam",
            parallel=False,
            num_procs=1,
            openfoam_bashrc="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc",
            data_processor=data_processor
        )
        
        scheduler = Pool(
            experiment_name=global_settings.experiment_name,
            num_jobs=2,
            verbose=True
        )
        
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
    from queens.utils.io import load_result
    
    results_file = Path(global_settings.output_dir) / global_settings.experiment_name / "results.json"
    
    if not results_file.exists():
        print(f"Warning: Results file not found at {results_file}")
        return np.array([]), np.array([])
    
    try:
        results = load_result(results_file)
        
        X_data = []
        y_data = []
        
        if hasattr(results, 'output_dict') and results.output_dict:
            for sample_data in results.output_dict.values():
                if 'parameters' in sample_data and 'model_response' in sample_data:
                    # Parameters (lid_velocity, initial_pressure)
                    params = sample_data['parameters']
                    X_data.append([params.get('lid_velocity', 0), params.get('initial_pressure', 0)])
                    
                    # Probe data response
                    response = sample_data['model_response']
                    if isinstance(response, np.ndarray) and len(response) > 0:
                        y_data.append(response)
                    else:
                        y_data.append(np.zeros(12))  # 3 probes × (3 U + 1 p) = 12
        
        X_train = np.array(X_data)
        y_train = np.array(y_data)
        
        print(f"Training data shape:")
        print(f"  X (lid_velocity, pressure): {X_train.shape}")
        print(f"  y (velocity_mag, pressure):  {y_train.shape}")
        
        return X_train, y_train
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return np.array([]), np.array([])


if __name__ == "__main__":
    X, y = generate_training_data(5)
    
    # Save training data
    output_file = "surrogate_training_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    
    print(f"\n✅ Training data saved to {output_file}")
    if len(X) > 0:
        print(f"Sample X: {X[0]}")
        print(f"Sample y: {y[0]}")