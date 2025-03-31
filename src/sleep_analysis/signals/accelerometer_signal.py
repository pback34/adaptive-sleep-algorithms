"""
Accelerometer signal class implementation.

This module defines the AccelerometerSignal class for motion data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any # Added import
from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType
from .magnitude_signal import MagnitudeSignal
from .angle_signal import AngleSignal

class AccelerometerSignal(TimeSeriesSignal):
    """
    Class for accelerometer signals.
    
    Accelerometer signals measure acceleration forces in three axes (X, Y, Z).
    """
    _is_abstract = False
    signal_type = SignalType.ACCELEROMETER
    required_columns = ['x', 'y', 'z']

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

# --- Registered Operation Functions ---

@AccelerometerSignal.register("compute_magnitude", output_class=MagnitudeSignal)
def compute_magnitude_registered(data_list: List[pd.DataFrame], parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Registered function to compute scalar magnitude from accelerometer data.

    Args:
        data_list: List containing the input signal's DataFrame.
        parameters: Dictionary of parameters (unused in this operation).

    Returns:
        DataFrame containing the 'magnitude' column.
    """
    df = data_list[0]
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("Input DataFrame must contain 'x', 'y', and 'z' columns")

    result = pd.DataFrame(index=df.index)
    result['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    return result

@AccelerometerSignal.register("compute_angle", output_class=AngleSignal)
def compute_angle_registered(data_list: List[pd.DataFrame], parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Registered function to compute pitch and roll angles from accelerometer data.

    Args:
        data_list: List containing the input signal's DataFrame.
        parameters: Dictionary of parameters (unused in this operation).

    Returns:
        DataFrame containing 'pitch' and 'roll' columns in degrees.
    """
    df = data_list[0]
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("Input DataFrame must contain 'x', 'y', and 'z' columns")

    result = pd.DataFrame(index=df.index)
    y_z_square = df['y']**2 + df['z']**2
    # Compute pitch: arctan(x / sqrt(y² + z²)) in degrees
    result['pitch'] = np.arctan2(df['x'], np.sqrt(y_z_square)) * 180 / np.pi
    # Compute roll: arctan(y / z) in degrees
    result['roll'] = np.arctan2(df['y'], df['z']) * 180 / np.pi
    return result


# --- Instance Methods (Implementation moved to registered functions) ---

    # Renamed original implementation method
    def _compute_magnitude_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic to compute scalar magnitude.
        
        The magnitude is calculated as sqrt(x² + y² + z²) for each sample.
        
        Args:
            **parameters: Optional parameters (unused in basic implementation)
            
        Args:
            inplace: If True, attempts to modify the signal in place (not recommended for this op).
            **parameters: Optional parameters passed to the registered operation.

        Args:
            df: Input DataFrame with 'x', 'y', 'z' columns.

        Returns:
            DataFrame containing the 'magnitude' column.
        """
        if not all(col in df.columns for col in ['x', 'y', 'z']):
             raise ValueError("Input DataFrame must contain 'x', 'y', and 'z' columns for _compute_magnitude_impl")
        result_df = pd.DataFrame(index=df.index)
        result_df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        return result_df

    # Renamed original implementation method
    def _compute_angle_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic to compute pitch and roll angles.
        
        Pitch represents the forward/backward tilt, calculated as arctan(x / sqrt(y² + z²)).
        Roll represents the left/right tilt, calculated as arctan(y / z).
        Both angles are returned in degrees.
        
        Args:
            **parameters: Optional parameters (unused in basic implementation)
            
        Args:
            inplace: If True, attempts to modify the signal in place (not recommended for this op).
            **parameters: Optional parameters passed to the registered operation.

        Args:
            df: Input DataFrame with 'x', 'y', 'z' columns.

        Returns:
            DataFrame containing 'pitch' and 'roll' columns in degrees.
        """
        if not all(col in df.columns for col in ['x', 'y', 'z']):
             raise ValueError("Input DataFrame must contain 'x', 'y', and 'z' columns for _compute_angle_impl")
        result_df = pd.DataFrame(index=df.index)
        y_z_square = df['y']**2 + df['z']**2
        result_df['pitch'] = np.arctan2(df['x'], np.sqrt(y_z_square)) * 180 / np.pi
        result_df['roll'] = np.arctan2(df['y'], df['z']) * 180 / np.pi
        return result_df

# --- Public Methods (Wrappers calling apply_operation) ---

    def compute_magnitude(self, inplace: bool = False, **parameters):
        """
        Compute the scalar magnitude by calling the registered operation.
        """
        # This now calls apply_operation, which will use the registry
        # because getattr won't find a method named 'compute_magnitude'
        return self.apply_operation("compute_magnitude", inplace=inplace, **parameters)

    def compute_angle(self, inplace: bool = False, **parameters):
        """
        Compute pitch and roll angles by calling the registered operation.
        """
        # This now calls apply_operation, which will use the registry
        # because getattr won't find a method named 'compute_angle'
        return self.apply_operation("compute_angle", inplace=inplace, **parameters)
