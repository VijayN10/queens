#!/usr/bin/env python3
"""Train and save surrogate model with OpenFOAM/ParaView data."""

import pickle
import numpy as np
import sys
import os
from pathlib import Path

# Add QUEENS source directory to Python path
queens_src_path = Path('/home/a11evina/queens/src').resolve()
if str(queens_src_path) not in sys.path:
    sys.path.insert(0, str(queens_src_path))

# Import QUEENS modules
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

def train_and_save_surrogate():
    """Train surrogate model and save it to disk."""
    
    # Load the generated training data
    try:
        with open("surrogate_data_output/cavity_flow_surrogate_surrogate_data.pkl", "rb") as f:
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
            experiment_name="cavity_surrogate_training",
            output_dir="./trained_models"
        )
        
        with global_settings:
            # Define parameter space (same as training data)
            parameters = Parameters(
                lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
                initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
            )
            
            print("✅ Parameters defined")
            
            # Create Gaussian Process surrogate model
            surrogate_model = GaussianProcess(
                dimension_lengthscales=X_train.shape[1],  # 2 for lid_velocity and initial_pressure
            )
            
            print("✅ Gaussian Process model created")
            
            # Set up the surrogate with training data
            surrogate_model.setup(X_train, Y_train)
            
            # Train the surrogate
            print("🔄 Training surrogate model...")
            surrogate_model.train()
            print("✅ Surrogate model trained")
            
            # Save the trained model
            model_save_path = "trained_models/cavity_surrogate_model.pkl"
            os.makedirs("trained_models", exist_ok=True)
            
            # Create a save dictionary with model and metadata
            save_data = {
                'surrogate_model': surrogate_model,
                'parameters': parameters,
                'training_data_shape': X_train.shape,
                'output_data_shape': Y_train.shape,
                'parameter_ranges': {
                    'min': X_train.min(axis=0).tolist(),
                    'max': X_train.max(axis=0).tolist()
                },
                'metadata': {
                    'experiment_name': 'cavity_surrogate_training',
                    'model_type': 'GaussianProcess',
                    'trained_on': str(Path.cwd()),
                }
            }
            
            with open(model_save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"✅ Trained model saved to: {model_save_path}")
            
            # Validate the saved model by loading it back
            print("🔄 Validating saved model...")
            with open(model_save_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            loaded_model = loaded_data['surrogate_model']
            
            # Test prediction to ensure it works
            X_test = np.array([[1.2, 0.05]])
            prediction_dict = loaded_model.predict(X_test)
            
            print("✅ Model validation successful!")
            print(f"   Test prediction shape: {prediction_dict['result'].shape}")
            
            return model_save_path
            
    except Exception as e:
        print(f"❌ Error during surrogate modeling: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎯 TRAINING SURROGATE MODEL")
    print("=" * 50)
    
    saved_model_path = train_and_save_surrogate()
    
    if saved_model_path:
        print(f"\n🎉 SUCCESS!")
        print(f"Trained model saved to: {saved_model_path}")
        print("\nNext step: Use prediction script to make predictions")
    else:
        print("\n❌ FAILED to train and save model")
        sys.exit(1)