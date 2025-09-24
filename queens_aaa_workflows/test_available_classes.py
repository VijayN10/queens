# test_available_classes.py
import sys
sys.path.insert(0, '/home/a11evina/queens/src')

# Check what's in simulation module
from queens.models import simulation
print("Classes in simulation:", dir(simulation))

# Check what's in _model module  
from queens.models import _model
print("\nClasses in _model:", dir(_model))

# Check what's directly available from models
import queens.models
print("\nDirectly from queens.models:", dir(queens.models))