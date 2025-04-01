"""
Accelerometer signal class implementation.

This module defines the AccelerometerSignal class for motion data.
"""

import pandas as pd
import numpy as np
# Removed unused typing imports
from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType
# Removed unused signal imports

class AccelerometerSignal(TimeSeriesSignal):
    """
    Class for accelerometer signals.
    
    Accelerometer signals measure acceleration forces in three axes (X, Y, Z).
    """
    _is_abstract = False
    signal_type = SignalType.ACCELEROMETER
    required_columns = ['x', 'y', 'z']

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation
    # Operations like compute_magnitude and compute_angle are now registered
    # in their respective output signal modules (magnitude_signal.py, angle_signal.py)
    # and invoked via apply_operation("compute_magnitude") etc. on an
    # AccelerometerSignal instance.
