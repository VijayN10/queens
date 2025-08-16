#!/usr/bin/env python3
"""
Simple Queens DataProcessor for OpenFOAM using ParaView
Just implements get_raw_data_from_file method with the macro code
"""

import logging
import json
from pathlib import Path
import numpy as np

# Import ParaView modules
try:
    import paraview
    paraview.compatibility.major = 5
    paraview.compatibility.minor = 13
    from paraview.simple import *
    from paraview import servermanager
except ImportError as e:
    logging.error(f"Error importing ParaView: {e}")
    raise ImportError("ParaView is required for this data processor") from e

from queens.data_processors.pvd_file import DataProcessorPvd
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class DataProcessorParaview(DataProcessorPvd):
    """Simple Probe Extraction DataProcessor for OpenFOAM cases using ParaView.
    
    Inherits from DataProcessorPvd and only implements get_raw_data_from_file.
    Uses the existing filter_and_manipulate_raw_data from parent class.
    Can be used for any OpenFOAM case with probe extraction.
    """

    @log_init_args
    def __init__(
        self,
        field_name="U",  # Required by parent class
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
        time_steps=None,
        block=0,
        point_data=True,
    ):
        """Initialize the ParaView processor.
        
        Args:
            field_name (str): Field name (required by parent)
            file_name_identifier (str): Identifier for case directory
            file_options_dict (dict): Options dictionary
            files_to_be_deleted_regex_lst (list): Files to delete
            time_steps (list): Time steps to process
            block (int): Block index for multiblock data
            point_data (bool): Whether to use point data
        """
        super().__init__(
            field_name=field_name,
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
            time_steps=time_steps,
            block=block,
            point_data=point_data,
        )

    def get_raw_data_from_file(self, file_path):
        """Get the raw data from OpenFOAM case using ParaView.
        
        This method contains the core logic from original ParaView macro.

        Args:
            file_path (str): Actual path to the OpenFOAM case directory.

        Returns:
            raw_data (np.array): Probe data in numpy array format.
        """
        case_path = Path(file_path)
        _logger.info(f"Processing OpenFOAM case: {case_path}")
        
        # Physical domain (0.1m × 0.1m × 0.01m) -  original values
        X_CENTER, Y_CENTER, Z_CENTER = 0.05, 0.05, 0.005
        
        # Create/find .foam file -  original logic
        foam_file = str(case_path / f"{case_path.name}.foam")
        if not Path(foam_file).exists():
            with open(foam_file, 'w') as f:
                f.write("")
        
        # Load case -  original logic
        reader = OpenFOAMReader(FileName=foam_file)
        reader.MeshRegions = ['internalMesh']
        reader.CellArrays = ['p', 'U']
        reader.UpdatePipeline()
        
        # Go to latest time -  original logic
        time_values = reader.TimestepValues
        if time_values:
            latest_time = time_values[-1]
            view = GetRenderView()
            view.ViewTime = latest_time
            reader.UpdatePipeline()
        
        # Define probe points -  original definitions
        probe_points = {
            'center': [0.05, 0.05, 0.005],
            'bottom_left': [0.01, 0.01, 0.005],
            'bottom_right': [0.09, 0.01, 0.005],
            'top_left': [0.01, 0.09, 0.005],
            'top_right': [0.09, 0.09, 0.005],
        }
        
        results = {}
        processed_data = []
        
        _logger.info("Extracting probe data...")
        for probe_name, coords in probe_points.items():
            try:
                #  original probe logic
                probe = ProbeLocation(Input=reader)
                probe.ProbeType = 'Fixed Radius Point Source'
                probe.ProbeType.Center = coords
                probe.ProbeType.Radius = 0.001
                probe.UpdatePipeline()
                
                probe_data = servermanager.Fetch(probe)
                if probe_data.GetNumberOfPoints() > 0:
                    point_data = probe_data.GetPointData()
                    velocity = point_data.GetArray('U').GetTuple(0)
                    pressure = point_data.GetArray('p').GetValue(0)
                    
                    results[probe_name] = {
                        'u_velocity': velocity[0],
                        'v_velocity': velocity[1],
                        'pressure': pressure
                    }
                    
                    # Add to processed_data array (row per probe)
                    processed_data.append([velocity[0], velocity[1], pressure])
                    
                    _logger.info(f"✅ {probe_name}: U={velocity[0]:.6f}, V={velocity[1]:.6f}, p={pressure:.6f}")
                else:
                    _logger.warning(f"❌ {probe_name}: No data")
                    results[probe_name] = None
                    processed_data.append([np.nan, np.nan, np.nan])
                    
            except Exception as e:
                _logger.error(f"❌ {probe_name}: Error - {e}")
                results[probe_name] = None
                processed_data.append([np.nan, np.nan, np.nan])
        
        # Save results to JSON (optional, from  original script)
        output_file = case_path / "cavity_probes.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        _logger.info(f"Results saved to: {output_file}")
        
        # Convert to numpy array and return
        raw_data = np.array(processed_data)
        _logger.info(f"Processed {len(probe_points)} probe points, data shape: {raw_data.shape}")
        
        return raw_data

    # filter_and_manipulate_raw_data is inherited from DataProcessorPvd!