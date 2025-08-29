#!/usr/bin/env python3
"""Demo: Surrogate modeling with OpenFOAM/ParaView data."""

# Example: Using the generated data for surrogate modeling in QUEENS



import pickle
import numpy as np
from queens.models.surrogate_models import GaussianProcess
from queens.distributions import Uniform
from queens.parameters import Parameters
from queens.global_settings import GlobalSettings
import sys

# Adjust path to your QUEENS installation
sys.path.insert(0, '/home/a11evina/queens/src')

# Load the generated training data
with open("combined_output/openfoam_paraview_combined_surrogate_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data['X_train']  # (n × 2) parameters
Y_train = data['Y_train']  # (n × 12) probe outputs

print(f"Training data: X{X_train.shape}, Y{Y_train.shape}")

# Set up QUEENS for surrogate modeling
global_settings = GlobalSettings(
    experiment_name="cavity_surrogate",
    output_dir="./surrogate_output"
)

with global_settings:
    # Define parameter space (same as training)
    parameters = Parameters(
        lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
        initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
    )
    
    # Create Gaussian Process surrogate model
    surrogate_model = GaussianProcess(
        parameters=parameters,
        training_input=X_train,
        training_output=Y_train,
        # You can select specific outputs if needed:
        # training_output=Y_train[:, 0:4]  # Only first probe
    )
    
    # Train the surrogate
    surrogate_model.train()
    
    # Make predictions
    X_test = np.array([[1.2, 0.05], [1.8, -0.02]])  # Test points
    Y_pred, Y_std = surrogate_model.predict(X_test)
    
    print(f"Predictions: {Y_pred}")
    print(f"Uncertainties: {Y_std}")
    
    # Use with QUEENS iterators for UQ, optimization, etc.
    from queens.iterators import MonteCarlo
    # ... continue with uncertainty quantification
