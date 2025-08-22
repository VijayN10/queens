#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

# Use the subprocess version instead
from queens.data_processors.paraview import DataProcessorParaviewSubprocess

processor = DataProcessorParaviewSubprocess(
    field_name="U",
    file_name_identifier="*",
    file_options_dict={},
    paraview_script_path="simple_cavity_probes.py"
)

case_path = "/home/a11evina/queens-experiments/openfoam_driver_test/1"
results = processor.get_data_from_file(case_path)
print(f"Results shape: {results.shape}")
print(f"Data:\n{results}")