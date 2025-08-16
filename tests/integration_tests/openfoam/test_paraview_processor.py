#!/usr/bin/env python3
"""
Test the ParaView data processor with OpenFOAM cavity case.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

from queens.data_processors.paraview import DataProcessorParaview

processor = DataProcessorParaview(
    field_name="U",
    file_name_identifier="*",
    file_options_dict={}
)

# Test with your cavity case
case_path = "/home/a11evina/queens-experiments/openfoam_driver_test/1"
results = processor.get_data_from_file(case_path)
print(f"Results shape: {results.shape}")
print(f"Data:\n{results}")