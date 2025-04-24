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
        Create an AngleSignal from an AccelerometerSignal using the registered operation.

        Args:
            acc_signal: The source AccelerometerSignal instance.
            **params: Optional parameters passed to the 'compute_angle' operation.

        Returns:
            A new AngleSignal instance.
        """
        # Ensure the input is the correct type
        from .accelerometer_signal import AccelerometerSignal # Local import for runtime check
        if not isinstance(acc_signal, AccelerometerSignal):
            raise TypeError("Input signal must be an instance of AccelerometerSignal")

        # Delegate creation to the apply_operation method of the source signal
        result_signal = acc_signal.apply_operation("compute_angle", **params)

        # Ensure the result is of the expected type
        if not isinstance(result_signal, cls):
             raise TypeError(f"Operation 'compute_angle' did not return an AngleSignal instance, got {type(result_signal).__name__}")

        return result_signal


# --- Registered Operation Function ---
# This function is registered with AccelerometerSignal but defined here.

# Import AccelerometerSignal here for the registration decorator
from .accelerometer_signal import AccelerometerSignal

@AccelerometerSignal.register("compute_angle", output_class=AngleSignal)
def compute_angle(data_list: List[pd.DataFrame], parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Registered function to compute pitch and roll angles from accelerometer data.

    Args:
        data_list: List containing the input signal's DataFrame (expected from AccelerometerSignal).
        parameters: Dictionary of parameters (unused in this operation).

    Returns:
        DataFrame containing 'pitch' and 'roll' columns in degrees, suitable for AngleSignal.

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
    y_z_square = df['y']**2 + df['z']**2
    # Compute pitch: arctan(x / sqrt(y² + z²)) in degrees
    # Handle potential division by zero if y and z are both zero
    sqrt_y_z_square = np.sqrt(y_z_square)
    # Avoid division by zero; if sqrt is zero, pitch is +/- 90 degrees depending on x
    result['pitch'] = np.where(
        sqrt_y_z_square == 0,
        np.sign(df['x']) * 90,
        np.arctan2(df['x'], sqrt_y_z_square) * 180 / np.pi
    )
    # Compute roll: arctan(y / z) in degrees
    # Handle potential division by zero if z is zero
    result['roll'] = np.where(
        df['z'] == 0,
        np.sign(df['y']) * 90 if 'y' in df else 0, # Roll is +/- 90 if z=0 and y!=0
        np.arctan2(df['y'], df['z']) * 180 / np.pi
    )
    return result
