"""
Magnitude signal class implementation.

This module defines the MagnitudeSignal class for accelerometer magnitude data.
"""

from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType

class MagnitudeSignal(TimeSeriesSignal):
    """
    Class for accelerometer magnitude signals.
    
    Magnitude signals represent the scalar magnitude (sqrt(x^2 + y^2 + z^2))
    of the 3D accelerometer vector, providing a single value that represents
    overall movement intensity regardless of direction.
    """
    _is_abstract = False
    signal_type = SignalType.ACCELEROMETER
    required_columns = ['magnitude']
    
    def get_sampling_rate(self) -> float:
        """
        Get the sampling rate of the magnitude signal.
        
        Returns:
            The sampling rate in Hz.
        """
        return super().get_sampling_rate() or 50.0  # Default to 50Hz if calculation fails
