#!/usr/bin/env python3
"""Demo: Surrogate modeling with OpenFOAM/ParaView data."""

import pickle
import numpy as np
import sys
import os
from pathlib import Path

# Fix the Python path - add QUEENS source directory properly
queens_src_path = Path('/home/a11evina/queens/src').resolve()
if str(queens_src_path) not in sys.path:
    sys.path.insert(0, str(queens_src_path))

# Alternative method: Set PYTHONPATH environment variable
# os.environ['PYTHONPATH'] = str(queens_src_path) + ':' + os.environ.get('PYTHONPATH', '')

# Now import QUEENS modules
try:
    from queens.models.surrogates.gaussian_process import GaussianProcess
    from queens.distributions import Uniform
    from queens.parameters import Parameters
    from queens.global_settings import GlobalSettings
    print("✅ QUEENS modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available modules in sys.path:")
    for path in sys.path:
        if 'queens' in path:
            print(f"  {path}")
    sys.exit(1)

# Load the generated training data
try:
    with open("combined_output/openfoam_paraview_fixed_surrogate_data.pkl", "rb") as f:
        data = pickle.load(f)
    print("✅ Training data loaded successfully")
except FileNotFoundError:
    print("❌ Pickle file not found. Make sure data_generator.py has run successfully.")
    sys.exit(1)

X_train = data['X_train']  # (n × 2) parameters
Y_train = data['Y_train']  # (n × 12) probe outputs

print(f"Training data shapes: X{X_train.shape}, Y{Y_train.shape}")
print(f"Parameter ranges: {X_train.min(axis=0)} to {X_train.max(axis=0)}")
print(f"Output ranges: {Y_train.min(axis=0)} to {Y_train.max(axis=0)}")

# Set up QUEENS for surrogate modeling
try:
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
        
        print("✅ Parameters defined")
        
        # Create Gaussian Process surrogate model
        # Set dimension_lengthscales to number of input parameters
        surrogate_model = GaussianProcess(
            dimension_lengthscales=X_train.shape[1],  # 2 for lid_velocity and initial_pressure
        )
        
        print("✅ Gaussian Process model created")
        
        # Set up the surrogate with training data using the setup method
        surrogate_model.setup(X_train, Y_train)
        
        # Train the surrogate
        print("🔄 Training surrogate model...")
        surrogate_model.train()
        print("✅ Surrogate model trained")
        
        # Make predictions on test points
        X_test = np.array([
            [1.2, 0.05],   # Test point 1
            [1.8, -0.02],  # Test point 2
            [0.8, 0.08],   # Test point 3
            [1.5, -0.05]   # Test point 4
        ])
        
        print("🔄 Making predictions...")
        prediction_dict = surrogate_model.predict(X_test)
        Y_pred = prediction_dict["result"]
        Y_std = np.sqrt(prediction_dict["variance"])
        
        print("\n" + "="*60)
        print("SURROGATE MODEL PREDICTIONS")
        print("="*60)
        
        for i, (x_test, y_pred, y_std) in enumerate(zip(X_test, Y_pred, Y_std)):
            print(f"\nTest point {i+1}: [vel={x_test[0]:.2f}, p={x_test[1]:.3f}]")
            print(f"  Predictions (12 probes): {y_pred}")
            print(f"  Uncertainties (12 probes): {y_std}")
            print(f"  Mean prediction: {y_pred.mean():.4f} ± {y_std.mean():.4f}")
        
        print("\n✅ Surrogate modeling demonstration completed!")
        print("\nNext steps:")
        print("- Use surrogate for uncertainty quantification with MonteCarlo")
        print("- Perform sensitivity analysis")
        print("- Optimize parameters using surrogate")
        
except Exception as e:
    print(f"❌ Error during surrogate modeling: {e}")
    import traceback
    traceback.print_exc()