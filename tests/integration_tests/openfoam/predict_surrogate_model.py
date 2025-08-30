#!/usr/bin/env python3
"""Load trained surrogate model and make predictions."""

import pickle
import numpy as np
import sys
from pathlib import Path

# Add QUEENS source directory to Python path
queens_src_path = Path('/home/a11evina/queens/src').resolve()
if str(queens_src_path) not in sys.path:
    sys.path.insert(0, str(queens_src_path))

def load_and_predict():
    """Load trained surrogate model and make predictions."""
    
    # Load the trained model
    model_path = "trained_models/cavity_surrogate_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        print("✅ Trained model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Make sure you've run the training script first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Extract model and metadata
    surrogate_model = saved_data['surrogate_model']
    parameters = saved_data['parameters']
    metadata = saved_data['metadata']
    param_ranges = saved_data['parameter_ranges']
    
    print(f"Model type: {metadata['model_type']}")
    print(f"Training data shape: {saved_data['training_data_shape']}")
    print(f"Output data shape: {saved_data['output_data_shape']}")
    print(f"Parameter ranges: {param_ranges}")
    
    # Define test points for prediction
    X_test = np.array([
        [1.2, 0.05],   # Test point 1
        [1.8, -0.02],  # Test point 2
        [0.8, 0.08],   # Test point 3
        [1.5, -0.05],  # Test point 4
        [1.0, 0.0],    # Test point 5 (center)
    ])
    
    print(f"\n🔄 Making predictions on {len(X_test)} test points...")
    
    try:
        # Make predictions
        prediction_dict = surrogate_model.predict(X_test)
        Y_pred = prediction_dict["result"]
        Y_std = np.sqrt(prediction_dict["variance"])
        
        print("\n" + "="*70)
        print("SURROGATE MODEL PREDICTIONS")
        print("="*70)
        
        for i, (x_test, y_pred, y_std) in enumerate(zip(X_test, Y_pred, Y_std)):
            print(f"\nTest point {i+1}: [vel={x_test[0]:.2f}, p={x_test[1]:.3f}]")
            print(f"  Predictions (12 probes): {y_pred}")
            print(f"  Uncertainties (12 probes): {y_std}")
            print(f"  Mean prediction: {y_pred.mean():.4f} ± {y_std.mean():.4f}")
            print(f"  Max prediction: {y_pred.max():.4f}")
            print(f"  Min prediction: {y_pred.min():.4f}")
        
        # Statistical summary
        print(f"\n📊 PREDICTION STATISTICS:")
        print(f"Overall mean prediction: {Y_pred.mean():.4f}")
        print(f"Overall std prediction: {Y_pred.std():.4f}")
        print(f"Mean uncertainty: {Y_std.mean():.4f}")
        
        return Y_pred, Y_std
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def interactive_prediction():
    """Allow user to input custom test points."""
    
    model_path = "trained_models/cavity_surrogate_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        surrogate_model = saved_data['surrogate_model']
        param_ranges = saved_data['parameter_ranges']
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n🎯 INTERACTIVE PREDICTION MODE")
    print("="*50)
    print(f"Parameter ranges:")
    print(f"  lid_velocity: {param_ranges['min'][0]:.2f} to {param_ranges['max'][0]:.2f}")
    print(f"  initial_pressure: {param_ranges['min'][1]:.3f} to {param_ranges['max'][1]:.3f}")
    
    while True:
        try:
            print(f"\nEnter parameters (or 'quit' to exit):")
            
            vel_input = input("lid_velocity: ")
            if vel_input.lower() == 'quit':
                break
            
            press_input = input("initial_pressure: ")
            if press_input.lower() == 'quit':
                break
            
            # Convert to float and create test point
            vel = float(vel_input)
            press = float(press_input)
            X_test = np.array([[vel, press]])
            
            # Make prediction
            prediction_dict = surrogate_model.predict(X_test)
            Y_pred = prediction_dict["result"]
            Y_std = np.sqrt(prediction_dict["variance"])
            
            print(f"\nPrediction for [vel={vel:.2f}, p={press:.3f}]:")
            print(f"  Mean prediction: {Y_pred.mean():.4f} ± {Y_std.mean():.4f}")
            print(f"  Probe predictions: {Y_pred[0]}")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Exiting interactive mode")

if __name__ == "__main__":
    print("🎯 SURROGATE MODEL PREDICTIONS")
    print("=" * 50)
    
    # Run standard predictions
    Y_pred, Y_std = load_and_predict()
    
    if Y_pred is not None:
        print(f"\n✅ Predictions completed successfully!")
        
        # Ask if user wants interactive mode
        try:
            response = input(f"\nWould you like to try interactive prediction mode? (y/n): ")
            if response.lower().startswith('y'):
                interactive_prediction()
        except KeyboardInterrupt:
            print(f"\n👋 Goodbye!")
    else:
        print(f"\n❌ Prediction failed!")
        sys.exit(1)