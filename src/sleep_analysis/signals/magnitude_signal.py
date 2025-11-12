"""
Magnitude signal class implementation.

This module defines the MagnitudeSignal class for accelerometer magnitude data.
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


class MagnitudeSignal(TimeSeriesSignal):
    """
    Class for accelerometer magnitude signals.
    
    Magnitude signals represent the scalar magnitude (sqrt(x^2 + y^2 + z^2))
    of the 3D accelerometer vector, providing a single value that represents
    overall movement intensity regardless of direction.
    """
    _is_abstract = False
    signal_type = SignalType.MAGNITUDE
    required_columns = ['magnitude']
    # Default unit depends on the source, but we can set a common default.
    # The __init__ logic will try to infer from source if possible,
    # otherwise this default might be used if created standalone.
    # For derivation via apply_operation, the source unit logic was removed,
    # so this default will be used if the source had no units.
    _default_units = {
        'magnitude': Unit.MILLI_G # Defaulting to milli-g, assuming ACC source
    }


    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

    @classmethod
    def from_accelerometer(cls, acc_signal: 'AccelerometerSignal', **params) -> 'MagnitudeSignal':
        """
        Create a MagnitudeSignal from an AccelerometerSignal.

        Args:
            acc_signal: The source AccelerometerSignal instance.
            **params: Optional parameters passed to the compute_magnitude method.

        Returns:
            A new MagnitudeSignal instance.
        """
        # Ensure the input is the correct type to prevent errors
        from .accelerometer_signal import AccelerometerSignal
        if not isinstance(acc_signal, AccelerometerSignal):
            raise TypeError("Input signal must be an instance of AccelerometerSignal")

        # Call the instance method directly (no longer using registry)
        result_signal = acc_signal.compute_magnitude(**params)

        # Ensure the result is of the expected type
        if not isinstance(result_signal, cls):
             raise TypeError(f"compute_magnitude did not return a MagnitudeSignal instance, got {type(result_signal).__name__}")

        return result_signal
