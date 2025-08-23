#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, '/home/a11evina/queens/src')

from queens.data_processors.paraview import DataProcessorParaviewSubprocess

# Create processor with cavity domain coordinates
processor = DataProcessorParaviewSubprocess(
    field_name="U",
    file_name_identifier="foam.foam",
    file_options_dict={},
    probe_locations=[
        (0.05, 0.05, 0.005),  # center
        (0.01, 0.01, 0.005),  # bottom_left
        (0.09, 0.09, 0.005),  # top_right
    ],
    data_fields=['U', 'p']
)

# Test single case
case_path = Path("/home/a11evina/queens-experiments/openfoam_driver_test/1")
results = processor.get_data_from_file(case_path)
print(f"Results shape: {results.shape}")
print(f"Data: {results}")

# Test all cases
base_path = Path("/home/a11evina/queens-experiments/openfoam_driver_test")
for case_id in range(5):  # Cases 0-4
    case_path = base_path / str(case_id)
    if case_path.exists():
        results = processor.get_data_from_file(case_path)
        print(f"Case {case_id}: {results}")
    else:
        print(f"Case {case_id}: Not found")