import sys
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/a11evina/queens/src')
sys.path.insert(0, '/home/a11evina/Aorta/ofCaseGen/Method_4')

# Correct QUEENS imports
from queens.models.simulation import Simulation
print("✅ Simulation class imported successfully")

from queens.models._model import Model  
print("✅ Model base class imported successfully")

# Test AORTA
from config import ConfigParams
print("✅ AORTA ConfigParams imported successfully")

print("\n✅ All imports working!")