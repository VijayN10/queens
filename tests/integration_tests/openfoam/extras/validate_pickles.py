#!/usr/bin/env python3
"""
Comprehensive validation script for QUEENS pickle files from OpenFOAM + ParaView workflow.
Checks both standard QUEENS results and custom surrogate modeling data.
"""

import pickle
import numpy as np
from pathlib import Path
import json
import sys

# Add QUEENS to path if needed
sys.path.insert(0, '/home/a11evina/queens/src')

from queens.utils.io import load_pickle, print_pickled_data, load_result
from queens.global_settings import GlobalSettings


def validate_queens_standard_pickle(pickle_path):
    """
    Validate the standard QUEENS results pickle file.
    
    Args:
        pickle_path (str/Path): Path to the QUEENS results pickle file
        
    Returns:
        dict: Validation results with status and extracted data
    """
    print(f"\n🔍 VALIDATING QUEENS STANDARD PICKLE: {pickle_path}")
    print("=" * 60)
    
    validation_results = {
        'file_exists': False,
        'loadable': False,
        'has_input_data': False,
        'has_output_data': False,
        'data_shapes_valid': False,
        'non_zero_outputs': False,
        'extracted_data': None
    }
    
    try:
        # Check file existence
        file_path = Path(pickle_path)
        if not file_path.exists():
            print(f"❌ File does not exist: {pickle_path}")
            return validation_results
        
        validation_results['file_exists'] = True
        print(f"✅ File exists: {file_path.name} ({file_path.stat().st_size} bytes)")
        
        # Try loading the pickle file
        try:
            # Method 1: Use QUEENS load_result (preferred)
            results = load_result(pickle_path)
            validation_results['loadable'] = True
            print("✅ File loaded successfully with QUEENS load_result()")
        except Exception as e:
            print(f"❌ QUEENS load_result failed: {e}")
            # Method 2: Direct pickle load as fallback
            try:
                with open(pickle_path, 'rb') as f:
                    results = pickle.load(f)
                validation_results['loadable'] = True
                print("✅ File loaded successfully with direct pickle.load()")
            except Exception as e2:
                print(f"❌ Direct pickle.load also failed: {e2}")
                return validation_results
        
        # Inspect file structure
        print(f"\n📊 PICKLE FILE STRUCTURE:")
        print(f"Available keys: {list(results.keys())}")
        
        # Check for input data (parameter samples)
        input_keys = ['input_data', 'samples', 'raw_input_data']
        input_data = None
        for key in input_keys:
            if key in results:
                input_data = results[key]
                validation_results['has_input_data'] = True
                print(f"✅ Found input data under key: '{key}'")
                print(f"   Shape: {input_data.shape}")
                print(f"   Type: {type(input_data)}")
                break
        
        if input_data is None:
            print(f"❌ No input data found. Looked for keys: {input_keys}")
        
        # Check for output data
        output_data = None
        if 'raw_output_data' in results:
            if 'result' in results['raw_output_data']:
                output_data = results['raw_output_data']['result']
                validation_results['has_output_data'] = True
                print(f"✅ Found output data under 'raw_output_data']['result']'")
                print(f"   Shape: {output_data.shape}")
                print(f"   Type: {type(output_data)}")
            else:
                print("❌ 'raw_output_data' exists but no 'result' key found")
                print(f"   Available subkeys: {list(results['raw_output_data'].keys())}")
        else:
            print("❌ No 'raw_output_data' found")
        
        # Validate data shapes and relationships
        if input_data is not None and output_data is not None:
            if len(input_data) == len(output_data):
                validation_results['data_shapes_valid'] = True
                print(f"✅ Input and output shapes are consistent:")
                print(f"   Samples: {len(input_data)}")
                print(f"   Input parameters per sample: {input_data.shape[1] if len(input_data.shape) > 1 else 1}")
                print(f"   Output values per sample: {output_data.shape[1] if len(output_data.shape) > 1 else 1}")
            else:
                print(f"❌ Shape mismatch: {len(input_data)} inputs vs {len(output_data)} outputs")
        
        # Check for non-zero outputs (critical for OpenFOAM + ParaView)
        if output_data is not None:
            if np.any(output_data != 0):
                validation_results['non_zero_outputs'] = True
                print("✅ Output data contains non-zero values (ParaView processing successful)")
                print(f"   Output range: [{np.min(output_data):.6f}, {np.max(output_data):.6f}]")
                print(f"   Mean: {np.mean(output_data):.6f}")
                print(f"   Std: {np.std(output_data):.6f}")
            else:
                print("❌ WARNING: All output data is zero! ParaView processing may have failed")
        
        # Show sample data
        if input_data is not None and output_data is not None:
            print(f"\n📋 SAMPLE DATA (first 3 samples):")
            for i in range(min(3, len(input_data))):
                if len(input_data.shape) > 1:
                    input_str = f"[{', '.join([f'{x:.4f}' for x in input_data[i]])}]"
                else:
                    input_str = f"{input_data[i]:.4f}"
                
                if len(output_data.shape) > 1:
                    output_str = f"[{', '.join([f'{x:.4f}' for x in output_data[i][:6]])}{'...' if output_data.shape[1] > 6 else ''}]"
                else:
                    output_str = f"{output_data[i]:.4f}"
                
                print(f"   Sample {i}: Input={input_str} → Output={output_str}")
        
        # Store extracted data for further analysis
        validation_results['extracted_data'] = {
            'input': input_data,
            'output': output_data,
            'full_results': results
        }
        
        # Check for additional statistical outputs
        stat_keys = ['mean', 'var', 'std']
        print(f"\n📈 STATISTICAL OUTPUTS:")
        for key in stat_keys:
            if key in results:
                print(f"   ✅ {key}: {results[key]}")
            else:
                print(f"   ❌ {key}: Not found")
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    return validation_results


def validate_surrogate_pickle(pickle_path):
    """
    Validate the custom surrogate modeling pickle file.
    
    Args:
        pickle_path (str/Path): Path to the surrogate data pickle file
        
    Returns:
        dict: Validation results
    """
    print(f"\n🧠 VALIDATING SURROGATE MODELING PICKLE: {pickle_path}")
    print("=" * 60)
    
    validation_results = {
        'file_exists': False,
        'loadable': False,
        'has_required_keys': False,
        'data_shapes_valid': False,
        'metadata_complete': False,
        'ready_for_ml': False
    }
    
    try:
        # Check file existence
        file_path = Path(pickle_path)
        if not file_path.exists():
            print(f"❌ File does not exist: {pickle_path}")
            return validation_results
        
        validation_results['file_exists'] = True
        print(f"✅ File exists: {file_path.name} ({file_path.stat().st_size} bytes)")
        
        # Load pickle file
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            validation_results['loadable'] = True
            print("✅ File loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load pickle file: {e}")
            return validation_results
        
        # Check required keys for surrogate modeling
        required_keys = ['X_train', 'Y_train', 'parameter_names', 'output_description', 'metadata']
        missing_keys = []
        
        print(f"\n📋 CHECKING REQUIRED KEYS:")
        for key in required_keys:
            if key in data:
                print(f"   ✅ {key}")
            else:
                print(f"   ❌ {key} - MISSING!")
                missing_keys.append(key)
        
        if not missing_keys:
            validation_results['has_required_keys'] = True
            print("✅ All required keys present")
        else:
            print(f"❌ Missing keys: {missing_keys}")
            return validation_results
        
        # Validate data shapes
        X_train = data['X_train']
        Y_train = data['Y_train']
        
        print(f"\n📐 DATA SHAPES:")
        print(f"   X_train: {X_train.shape} (samples × parameters)")
        print(f"   Y_train: {Y_train.shape} (samples × outputs)")
        
        if X_train.shape[0] == Y_train.shape[0]:
            validation_results['data_shapes_valid'] = True
            print("✅ X and Y have matching sample counts")
        else:
            print(f"❌ Sample count mismatch: X={X_train.shape[0]}, Y={Y_train.shape[0]}")
        
        # Check expected dimensions for OpenFOAM cavity case
        expected_n_params = 2  # lid_velocity, initial_pressure
        expected_n_outputs = 12  # 3 probes × 4 fields
        
        if X_train.shape[1] == expected_n_params:
            print(f"✅ Expected parameter count: {expected_n_params}")
        else:
            print(f"❌ Unexpected parameter count: {X_train.shape[1]} (expected {expected_n_params})")
        
        if Y_train.shape[1] == expected_n_outputs:
            print(f"✅ Expected output count: {expected_n_outputs}")
        else:
            print(f"❌ Unexpected output count: {Y_train.shape[1]} (expected {expected_n_outputs})")
        
        # Check metadata completeness
        metadata = data['metadata']
        output_desc = data['output_description']
        
        print(f"\n📝 METADATA VALIDATION:")
        
        # Check parameter names
        param_names = data['parameter_names']
        expected_params = ['lid_velocity', 'initial_pressure']
        if param_names == expected_params:
            print(f"✅ Parameter names: {param_names}")
        else:
            print(f"❌ Parameter names mismatch: {param_names} (expected {expected_params})")
        
        # Check probe locations
        probe_locations = output_desc.get('probe_locations', [])
        expected_probes = 3
        if len(probe_locations) == expected_probes:
            print(f"✅ Probe locations ({len(probe_locations)} probes):")
            for i, loc in enumerate(probe_locations):
                print(f"      Probe {i}: {loc}")
        else:
            print(f"❌ Probe count mismatch: {len(probe_locations)} (expected {expected_probes})")
        
        # Check parameter ranges
        param_ranges = metadata.get('parameter_ranges', {})
        expected_ranges = {
            'lid_velocity': [0.5, 2.0],
            'initial_pressure': [-0.1, 0.1]
        }
        
        ranges_valid = True
        for param, expected_range in expected_ranges.items():
            if param in param_ranges:
                actual_range = param_ranges[param]
                if actual_range == expected_range:
                    print(f"✅ {param} range: {actual_range}")
                else:
                    print(f"❌ {param} range mismatch: {actual_range} (expected {expected_range})")
                    ranges_valid = False
            else:
                print(f"❌ Missing parameter range: {param}")
                ranges_valid = False
        
        if ranges_valid:
            validation_results['metadata_complete'] = True
        
        # Check if data is ready for ML
        if (validation_results['has_required_keys'] and 
            validation_results['data_shapes_valid'] and 
            validation_results['metadata_complete']):
            validation_results['ready_for_ml'] = True
            print("🎉 FILE IS READY FOR SURROGATE MODELING!")
        
        # Data quality checks
        print(f"\n🔬 DATA QUALITY CHECKS:")
        
        # Check for NaN or infinite values
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("❌ Input data contains NaN or infinite values")
        else:
            print("✅ Input data is clean (no NaN/inf)")
        
        if np.isnan(Y_train).any() or np.isinf(Y_train).any():
            print("❌ Output data contains NaN or infinite values")
        else:
            print("✅ Output data is clean (no NaN/inf)")
        
        # Check for zero variance (constant outputs)
        output_vars = np.var(Y_train, axis=0)
        zero_var_outputs = np.sum(output_vars == 0)
        if zero_var_outputs > 0:
            print(f"⚠️  WARNING: {zero_var_outputs} outputs have zero variance (constant values)")
        else:
            print("✅ All outputs have non-zero variance")
        
        # Check parameter ranges actually used
        print(f"\n📊 ACTUAL DATA RANGES:")
        for i, param_name in enumerate(data['parameter_names']):
            actual_min = np.min(X_train[:, i])
            actual_max = np.max(X_train[:, i])
            expected_range = param_ranges.get(param_name, [None, None])
            print(f"   {param_name}: [{actual_min:.4f}, {actual_max:.4f}] (expected {expected_range})")
        
        print(f"\n   Output statistics:")
        print(f"   Mean outputs: [{np.min(np.mean(Y_train, axis=0)):.6f}, {np.max(np.mean(Y_train, axis=0)):.6f}]")
        print(f"   Output std: [{np.min(np.std(Y_train, axis=0)):.6f}, {np.max(np.std(Y_train, axis=0)):.6f}]")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return validation_results


def quick_pickle_inspector(pickle_path):
    """
    Quick inspection of any pickle file structure.
    
    Args:
        pickle_path (str/Path): Path to pickle file
    """
    print(f"\n🔎 QUICK PICKLE INSPECTION: {pickle_path}")
    print("=" * 50)
    
    try:
        # Use QUEENS built-in inspector
        print_pickled_data(Path(pickle_path))
    except Exception as e:
        print(f"QUEENS inspector failed: {e}")
        
        # Fallback manual inspection
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            def inspect_item(key, item, indent=0):
                spaces = "  " * indent
                item_type = type(item).__name__
                
                if isinstance(item, dict):
                    print(f"{spaces}{key} ({item_type}) - {len(item)} keys:")
                    for subkey, subitem in list(item.items())[:5]:  # Show first 5 items
                        inspect_item(subkey, subitem, indent+1)
                    if len(item) > 5:
                        print(f"{spaces}  ... ({len(item)-5} more items)")
                
                elif isinstance(item, (list, tuple)):
                    print(f"{spaces}{key} ({item_type}) - Length: {len(item)}")
                    if len(item) > 0:
                        print(f"{spaces}  First item type: {type(item[0])}")
                
                elif isinstance(item, np.ndarray):
                    print(f"{spaces}{key} ({item_type}) - Shape: {item.shape}, dtype: {item.dtype}")
                    if item.size > 0:
                        print(f"{spaces}  Range: [{np.min(item):.6f}, {np.max(item):.6f}]")
                
                else:
                    item_str = str(item)
                    if len(item_str) > 50:
                        item_str = item_str[:50] + "..."
                    print(f"{spaces}{key} ({item_type}): {item_str}")
            
            print("Manual inspection:")
            for key, item in data.items():
                inspect_item(key, item)
                
        except Exception as e2:
            print(f"Manual inspection also failed: {e2}")


def validate_openfoam_workflow_results(base_dir="./combined_output"):
    """
    Complete validation of OpenFOAM + ParaView workflow results.
    
    Args:
        base_dir (str): Base directory containing the pickle files
    """
    print("🎯 COMPLETE OPENFOAM + PARAVIEW WORKFLOW VALIDATION")
    print("=" * 80)
    
    base_path = Path(base_dir)
    
    # Look for expected pickle files
    standard_pickle = base_path / "openfoam_paraview_fixed.pickle"
    surrogate_pickle = base_path / "openfoam_paraview_fixed_surrogate_data.pkl"
    
    results = {}
    
    # Validate standard QUEENS pickle
    if standard_pickle.exists():
        results['standard'] = validate_queens_standard_pickle(standard_pickle)
    else:
        print(f"❌ Standard pickle not found: {standard_pickle}")
        results['standard'] = {'file_exists': False}
    
    # Validate surrogate pickle
    if surrogate_pickle.exists():
        results['surrogate'] = validate_surrogate_pickle(surrogate_pickle)
    else:
        print(f"❌ Surrogate pickle not found: {surrogate_pickle}")
        results['surrogate'] = {'file_exists': False}
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print("=" * 40)
    
    standard_ok = results['standard'].get('ready_for_ml', False) if 'standard' in results else False
    surrogate_ok = results['surrogate'].get('ready_for_ml', False) if 'surrogate' in results else False
    
    if standard_ok and surrogate_ok:
        print("🎉 SUCCESS: Both pickle files are valid and ready for use!")
    elif standard_ok:
        print("⚠️  PARTIAL: Standard pickle is valid, but surrogate pickle has issues")
    elif surrogate_ok:
        print("⚠️  PARTIAL: Surrogate pickle is valid, but standard pickle has issues")
    else:
        print("❌ FAILURE: Both pickle files have validation issues")
    
    return results


def demonstrate_loading_workflow():
    """Demonstrate how to properly load and use the pickle files."""
    print(f"\n💡 DEMONSTRATION: How to load and use your pickle files")
    print("=" * 60)
    
    code_example = '''
# Method 1: Load QUEENS standard results
from queens.utils.io import load_result

results = load_result("combined_output/openfoam_paraview_fixed.pickle")
X = results['input_data']  # or 'samples' or 'raw_input_data'
Y = results['raw_output_data']['result']

print(f"Loaded {len(X)} samples with {X.shape[1]} parameters and {Y.shape[1]} outputs")

# Method 2: Load surrogate modeling data
import pickle

with open("combined_output/openfoam_paraview_fixed_surrogate_data.pkl", "rb") as f:
    surrogate_data = pickle.load(f)

X_train = surrogate_data['X_train']
Y_train = surrogate_data['Y_train']
probe_locations = surrogate_data['output_description']['probe_locations']

# Method 3: Use with QUEENS Gaussian Process
from queens.models.surrogate_models import GaussianProcess
from queens.parameters import Parameters
from queens.distributions import Uniform

parameters = Parameters(
    lid_velocity=Uniform(lower_bound=0.5, upper_bound=2.0),
    initial_pressure=Uniform(lower_bound=-0.1, upper_bound=0.1),
)

# Create and train surrogate model
gp_model = GaussianProcess(
    parameters=parameters,
    training_input=X_train,
    training_output=Y_train,
)
gp_model.train()

# Make predictions
X_test = np.array([[1.5, 0.0]])  # Test point
Y_pred, Y_std = gp_model.predict(X_test)
'''
    
    print(code_example)


if __name__ == "__main__":
    print("🔍 QUEENS PICKLE FILE VALIDATOR")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Validate specific file
        pickle_path = sys.argv[1]
        if pickle_path.endswith('_surrogate_data.pkl'):
            validate_surrogate_pickle(pickle_path)
        else:
            validate_queens_standard_pickle(pickle_path)
        quick_pickle_inspector(pickle_path)
    else:
        # Validate entire workflow
        validate_openfoam_workflow_results()
        demonstrate_loading_workflow()