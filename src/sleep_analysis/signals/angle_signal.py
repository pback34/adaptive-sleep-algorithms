"""
Angle signal class implementation.

This module defines the AngleSignal class for accelerometer-derived orientation angles.
"""

# Added imports
# Added imports
import pandas as pd
import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING
from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType, Unit # Import Unit

# Forward reference for type hinting
if TYPE_CHECKING:
    from .accelerometer_signal import AccelerometerSignal


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
    _default_units = { # Define default units for this signal type
        'pitch': Unit.DEGREES,
        'roll': Unit.DEGREES
    }

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

    @classmethod
    def from_accelerometer(cls, acc_signal: 'AccelerometerSignal', **params) -> 'AngleSignal':
        """
        Create an AngleSignal from an AccelerometerSignal.

        Args:
            acc_signal: The source AccelerometerSignal instance.
            **params: Optional parameters passed to the compute_angle method.

        Returns:
            A new AngleSignal instance.
        """
        # Ensure the input is the correct type
        from .accelerometer_signal import AccelerometerSignal
        if not isinstance(acc_signal, AccelerometerSignal):
            raise TypeError("Input signal must be an instance of AccelerometerSignal")

        # Call the instance method directly (no longer using registry)
        result_signal = acc_signal.compute_angle(**params)

        # Ensure the result is of the expected type
        if not isinstance(result_signal, cls):
             raise TypeError(f"compute_angle did not return an AngleSignal instance, got {type(result_signal).__name__}")

        return result_signal
