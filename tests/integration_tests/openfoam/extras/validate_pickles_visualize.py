# Use this instead of online parsers
import pickle
import numpy as np

with open('combined_output/openfoam_paraview_fixed.pickle', 'rb') as f:
    data = pickle.load(f)

# This will show actual floating-point values:
print("Input data:", data['input_data'])
print("Output data:", data['raw_output_data']['result'])