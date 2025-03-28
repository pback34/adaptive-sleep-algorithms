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
    signal_type = SignalType.ACCELEROMETER
    required_columns = ['pitch', 'roll']
    
    def get_sampling_rate(self) -> float:
        """
        Get the sampling rate of the angle signal.
        
        Returns:
            The sampling rate in Hz.
        """
        return super().get_sampling_rate() or 50.0  # Default to 50Hz if calculation fails
