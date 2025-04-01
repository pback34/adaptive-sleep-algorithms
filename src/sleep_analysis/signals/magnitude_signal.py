"""
Magnitude signal class implementation.

This module defines the MagnitudeSignal class for accelerometer magnitude data.
"""

# Added imports
import pandas as pd
import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING
from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType

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

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

    @classmethod
    def from_accelerometer(cls, acc_signal: 'AccelerometerSignal', **params) -> 'MagnitudeSignal':
        """
        Create a MagnitudeSignal from an AccelerometerSignal using the registered operation.

        Args:
            acc_signal: The source AccelerometerSignal instance.
            **params: Optional parameters passed to the 'compute_magnitude' operation.

        Returns:
            A new MagnitudeSignal instance.
        """
        # Ensure the input is the correct type to prevent errors
        # (Type checking helps but runtime check adds safety)
        from .accelerometer_signal import AccelerometerSignal # Local import for runtime check
        if not isinstance(acc_signal, AccelerometerSignal):
            raise TypeError("Input signal must be an instance of AccelerometerSignal")

        # Delegate creation to the apply_operation method of the source signal
        # which uses the registry system.
        result_signal = acc_signal.apply_operation("compute_magnitude", **params)

        # Ensure the result is of the expected type
        if not isinstance(result_signal, cls):
             raise TypeError(f"Operation 'compute_magnitude' did not return a MagnitudeSignal instance, got {type(result_signal).__name__}")

        return result_signal


# --- Registered Operation Function ---
# This function is registered with AccelerometerSignal but defined here
# to keep the creation logic close to the MagnitudeSignal class and avoid
# modifying accelerometer_signal.py when adding new derived signals.

# Import AccelerometerSignal here for the registration decorator
from .accelerometer_signal import AccelerometerSignal

@AccelerometerSignal.register("compute_magnitude", output_class=MagnitudeSignal)
def compute_magnitude(data_list: List[pd.DataFrame], parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Registered function to compute scalar magnitude from accelerometer data.

    Args:
        data_list: List containing the input signal's DataFrame (expected from AccelerometerSignal).
        parameters: Dictionary of parameters (unused in this operation).

    Returns:
        DataFrame containing the 'magnitude' column, suitable for MagnitudeSignal.

    Raises:
        ValueError: If the input DataFrame does not contain 'x', 'y', 'z' columns.
    """
    if not data_list:
        raise ValueError("Input data_list is empty")
    df = data_list[0]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("Input DataFrame must contain 'x', 'y', and 'z' columns")

    result = pd.DataFrame(index=df.index)
    # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
    result['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    return result
