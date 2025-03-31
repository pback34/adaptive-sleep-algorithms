"""
Angle signal class implementation.

This module defines the AngleSignal class for accelerometer-derived orientation angles.
"""

from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType

class AngleSignal(TimeSeriesSignal):
    """
    Class for accelerometer-derived angle signals.
    
    Angle signals represent orientation in terms of pitch and roll derived
    from accelerometer data. Pitch represents the forward/backward tilt,
    while roll represents the left/right tilt.
    """
    _is_abstract = False
    signal_type = SignalType.ANGLE # Use the dedicated ANGLE type
    required_columns = ['pitch', 'roll']

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation
