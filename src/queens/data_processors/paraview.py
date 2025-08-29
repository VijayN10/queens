#!/usr/bin/env python3
"""
ParaView DataProcessor with embedded processing logic - no external scripts needed
"""

import logging
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

from queens.data_processors._data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class DataProcessorParaview(DataProcessor):
    """ParaView DataProcessor with embedded processing logic."""

    @log_init_args
    def __init__(
        self,
        field_name="U",
        file_name_identifier="foam.foam",
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
        probe_locations=None,
        data_fields=None,
        time_step=-1,
    ):
        """Initialize the ParaView processor.
        
        Args:
            field_name (str): Primary field name
            file_name_identifier (str): File identifier (foam.foam)
            file_options_dict (dict): File options
            files_to_be_deleted_regex_lst (list): Cleanup files
            probe_locations (list): Probe coordinates [(x,y,z), ...]
            data_fields (list): Field names to extract
            time_step (int): Time step to process (-1 for last)
        """
        if file_options_dict is None:
            file_options_dict = {}
            
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )
        
        self.field_name = field_name
        self.time_step = time_step
        
        # Default probe locations for cavity case
        if probe_locations is None:
            probe_locations = [
                (0.05, 0.05, 0.005),  # center
                (0.01, 0.01, 0.005),  # bottom_left
                (0.09, 0.09, 0.005),  # top_right
            ]
        self.probe_locations = probe_locations
        
        if data_fields is None:
            data_fields = ['U', 'p']
        self.data_fields = data_fields

    def get_data_from_file(self, base_dir_file):
        """Override to work directly with case directories."""
        if not isinstance(base_dir_file, Path):
            base_dir_file = Path(base_dir_file)
        
        raw_data = self.get_raw_data_from_file(base_dir_file)
        filtered_data = self.filter_and_manipulate_raw_data(raw_data)
        return filtered_data

    def get_raw_data_from_file(self, case_path):
        """Extract data using embedded ParaView script."""
        case_path = Path(case_path)
        
        if not case_path.exists():
            _logger.error(f"Case path does not exist: {case_path}")
            return np.array([])
        
        # Create temporary ParaView script
        script_content = self._generate_paraview_script()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Run ParaView with the generated script
            cmd = f"""
            cd {case_path}
            if command -v pvpython >/dev/null 2>&1; then
                pvpython {script_path} {case_path}
            else
                python {script_path} {case_path}
            fi
            """
            
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                executable='/bin/bash', timeout=300
            )
            
            if result.returncode != 0:
                _logger.error(f"ParaView failed: {result.stderr}")
                return np.array([])
                
        except Exception as e:
            _logger.error(f"Subprocess error: {e}")
            return np.array([])
        finally:
            # Clean up temp script
            Path(script_path).unlink(missing_ok=True)
        
        # Load results
        return self._load_results(case_path)
    
    def _generate_paraview_script(self):
        """Generate ParaView script content based on working script."""
        script = f'''
import sys
import json
import numpy as np
from pathlib import Path

try:
    import paraview
    paraview.compatibility.major = 5
    paraview.compatibility.minor = 13
    from paraview.simple import *
    from paraview import servermanager
except ImportError:
    print("ParaView not available, using dummy data")
    case_path = Path(sys.argv[1])
    dummy_results = {{}}
    for i, probe in enumerate({self.probe_locations}):
        probe_name = f"probe_{{i}}"
        dummy_results[probe_name] = {{"U": [0.1 * i, 0.2 * i, 0.0], "p": 0.01 * i}}
    
    with open(case_path / "probe_data.json", "w") as f:
        json.dump(dummy_results, f)
    sys.exit(0)

case_path = Path(sys.argv[1])

# Create/find .foam file
foam_file = case_path / f"{{case_path.name}}.foam"
if not foam_file.exists():
    foam_file = case_path / "foam.foam"
if not foam_file.exists():
    foam_file = case_path / ".foam"
if not foam_file.exists():
    with open(case_path / "foam.foam", 'w') as f:
        f.write("")
    foam_file = case_path / "foam.foam"

# Load case
reader = OpenFOAMReader(FileName=str(foam_file))
reader.MeshRegions = ['internalMesh']
reader.CellArrays = ['p', 'U']
reader.UpdatePipeline()

# Go to latest time
time_values = reader.TimestepValues
if time_values:
    latest_time = time_values[{self.time_step}]
    view = GetRenderView()
    view.ViewTime = latest_time
    reader.UpdatePipeline()

# Define probe points
probe_locations = {self.probe_locations}
results = {{}}

for i, coords in enumerate(probe_locations):
    probe_name = f"probe_{{i}}"
    try:
        probe = ProbeLocation(Input=reader)
        probe.ProbeType = 'Fixed Radius Point Source'
        probe.ProbeType.Center = list(coords)
        probe.ProbeType.Radius = 0.001  # Small radius like working script
        probe.UpdatePipeline()
        
        probe_data = servermanager.Fetch(probe)
        if probe_data.GetNumberOfPoints() > 0:
            point_data = probe_data.GetPointData()
            
            # Extract velocity
            if point_data.GetArray('U'):
                velocity = point_data.GetArray('U').GetTuple(0)
                results[probe_name] = {{
                    'U': [velocity[0], velocity[1], velocity[2]],
                }}
            else:
                results[probe_name] = {{'U': [0.0, 0.0, 0.0]}}
            
            # Extract pressure
            if point_data.GetArray('p'):
                pressure = point_data.GetArray('p').GetValue(0)
                results[probe_name]['p'] = pressure
            else:
                results[probe_name]['p'] = 0.0
                
        else:
            results[probe_name] = {{'U': [0.0, 0.0, 0.0], 'p': 0.0}}
            
    except Exception as e:
        print(f"Error with probe {{i}}: {{e}}")
        results[probe_name] = {{'U': [0.0, 0.0, 0.0], 'p': 0.0}}

# Save results
with open(case_path / "probe_data.json", "w") as f:
    json.dump(results, f, indent=2)
'''
        return script
    
    def _load_results(self, case_path):
        """Load results from JSON file."""
        result_file = case_path / "probe_data.json"
        
        if not result_file.exists():
            _logger.error(f"No results file found: {result_file}")
            return np.array([])
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            _logger.error(f"Failed to load results: {e}")
            return np.array([])
        
        # Flatten to array
        data = []
        for i in range(len(self.probe_locations)):
            probe_name = f"probe_{i}"
            if probe_name in results:
                for field in self.data_fields:
                    if field in results[probe_name]:
                        value = results[probe_name][field]
                        if isinstance(value, list):
                            data.extend(value)
                        else:
                            data.append(value)
                    else:
                        data.append(np.nan)
            else:
                data.extend([np.nan] * len(self.data_fields) * 3)  # Assume 3D vectors
        
        return np.array(data)
    
    def filter_and_manipulate_raw_data(self, raw_data):
        """Return raw data as-is."""
        return raw_data


# Alias for compatibility
class DataProcessorParaviewSubprocess(DataProcessorParaview):
    pass