#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""QUEENS main.

Main module of QUEENS containing the high-level control routine for
input file workflow.
"""

import logging
import time

_logger = logging.getLogger(__name__)


def run_iterator(iterator, global_settings):
    """Run the main queens iterator.

    Args:
        iterator (Iterator): Main queens iterator
        global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                  and the output directory
    """
    global_settings.print_git_information()

    start_time_calc = time.time()

    _logger.info(
        "%s for experiment: %s", iterator.__class__.__name__, global_settings.experiment_name
    )
    _logger.info("")
    _logger.info("Starting Analysis...")
    _logger.info("")

    try:
        iterator.run()
    except Exception as exception:
        _logger.exception(exception)
        global_settings.__exit__(None, None, None)
        # TODO: Write iterator in pickle file # pylint: disable=fixme
        raise exception

    end_time_calc = time.time()
    _logger.info("")
    _logger.info("Time for CALCULATION: %s s", end_time_calc - start_time_calc)
    _logger.info("")
