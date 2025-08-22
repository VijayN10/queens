#!/usr/bin/env python3
"""
Subprocess-based ParaView DataProcessor for QUEENS
Uses 'module load paraview' instead of direct imports
"""

import logging
import json
import subprocess
import shutil
from pathlib import Path
import numpy as np

from queens.data_processors.pvd_file import DataProcessorPvd
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class DataProcessorParaview(DataProcessorPvd):
    """ParaView DataProcessor using subprocess calls with module load."""

    @log_init_args
    def __init__(
        self,
        field_name="U",
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
        time_steps=None,
        block=0,
        point_data=True,
        paraview_script_path="simple_cavity_probes.py",
    ):
        """Initialize the subprocess ParaView processor."""
        super().__init__(
            field_name=field_name,
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
            time_steps=time_steps,
            block=block,
            point_data=point_data,
        )
        
        self.paraview_script_path = Path(paraview_script_path)

    def get_raw_data_from_file(self, file_path):
        """Get raw data using subprocess ParaView call."""
        case_path = Path(file_path)
        _logger.info(f"Processing case: {case_path}")
        
        # Copy script to case directory
        local_script = case_path / "paraview_script.py"
        shutil.copy2(self.paraview_script_path, local_script)
        
        # Run ParaView via subprocess
        cmd = f"""
        module load paraview
        cd {case_path}
        pvpython {local_script} {case_path}
        """
        
        try:
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
        
        # Load results
        probe_file = case_path / "cavity_probes.json"
        if not probe_file.exists():
            _logger.error("No probe data found")
            return np.array([])
        
        with open(probe_file, 'r') as f:
            results = json.load(f)
        
        # Convert to array
        probe_order = ['center', 'bottom_left', 'bottom_right', 'top_left', 'top_right']
        processed_data = []
        
        for probe_name in probe_order:
            if probe_name in results and results[probe_name] is not None:
                probe = results[probe_name]
                processed_data.extend([
                    probe['u_velocity'],
                    probe['v_velocity'],
                    probe['pressure']
                ])
            else:
                processed_data.extend([np.nan, np.nan, np.nan])
        
        return np.array(processed_data)