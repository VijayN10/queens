#!/usr/bin/env python3
"""
Test script using your existing DataProcessorParaview driver.
Simple pattern similar to OpenFOAM driver tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, '/home/a11evina/queens/src')

# Import your working ParaView driver
# Assuming you save your implementation as 'paraview.py' 
from queens.data_processors.paraview import DataProcessorParaview

def main():
    """Test your ParaView driver - similar to OpenFOAM test pattern."""
    
    print("=== Creating DataProcessor Instance ===")
    # Create processor with your working implementation
    processor = DataProcessorParaview(
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
    
    print("\n=== Testing Single Case ===")
    case_path = Path("/home/a11evina/queens-experiments/openfoam_driver_test/1")
    try:
        results = processor.get_data_from_file(case_path)
        print(f"Results shape: {results.shape}")
        print(f"Data: {results}")
    except Exception as e:
        print(f"Error processing case 1: {e}")
    
    print("\n=== Testing Multiple Cases ===")
    base_path = Path("/home/a11evina/queens-experiments/openfoam_driver_test")
    for case_id in [0, 1, 2, 3, 4]:
        case_path = base_path / str(case_id)
        if case_path.exists():
            try:
                results = processor.get_data_from_file(case_path)
                print(f"Case {case_id}: Shape={results.shape}, Data={results}")
            except Exception as e:
                print(f"Case {case_id}: Error - {e}")
        else:
            print(f"Case {case_id}: Not found")

if __name__ == "__main__":
    main()