"""
Accelerometer signal class implementation.

This module defines the AccelerometerSignal class for motion data.
"""

import pandas as pd
import numpy as np
# Removed unused typing imports
from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType, Unit # Import Unit
# Removed unused signal imports

class AccelerometerSignal(TimeSeriesSignal):
    """
    Class for accelerometer signals.
    
    Accelerometer signals measure acceleration forces in three axes (X, Y, Z).
    """
    _is_abstract = False
    signal_type = SignalType.ACCELEROMETER
    required_columns = ['x', 'y', 'z']
    _default_units = { # Define default units for this signal type
        'x': Unit.MILLI_G, # Assuming milli-g as standard
        'y': Unit.MILLI_G,
        'z': Unit.MILLI_G
    }

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

    def compute_magnitude(self, **parameters) -> 'MagnitudeSignal':
        """
        Compute the scalar magnitude of the 3D accelerometer vector.

        Calculates magnitude as sqrt(x² + y² + z²), providing a single value
        that represents overall movement intensity regardless of direction.

        Args:
            **parameters: Additional parameters (unused currently, reserved for future use).

        Returns:
            MagnitudeSignal: A new MagnitudeSignal instance with the computed magnitude.

        Raises:
            ValueError: If the signal data does not contain 'x', 'y', 'z' columns.
        """
        # Avoid circular import by importing here
        from .magnitude_signal import MagnitudeSignal

        df = self.get_data()
        if df is None or df.empty:
            raise ValueError("Cannot compute magnitude from empty signal data")

        if not all(col in df.columns for col in ['x', 'y', 'z']):
            raise ValueError("AccelerometerSignal must contain 'x', 'y', and 'z' columns")

        # Calculate magnitude: sqrt(x² + y² + z²)
        result_data = pd.DataFrame(index=df.index)
        result_data['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

        # Create new signal with proper metadata
        # Operation index is the index of the last operation on the source signal
        operation_index = len(self.metadata.operations) - 1
        new_metadata = {
            'name': f"{self.metadata.name}_magnitude",
            'derived_from': [(self.metadata.signal_id, operation_index)],
            'operations': [],  # Will be updated by MagnitudeSignal init
        }

        # Create MagnitudeSignal instance
        magnitude_signal = MagnitudeSignal(
            data=result_data,
            metadata=new_metadata,
            handler=self.handler
        )

        # Record the operation in the new signal's metadata
        from ..core.metadata import OperationInfo
        op_info = OperationInfo(
            operation_name="compute_magnitude",
            parameters=parameters
        )
        magnitude_signal.metadata.operations.append(op_info)

        return magnitude_signal

    def compute_angle(self, **parameters) -> 'AngleSignal':
        """
        Compute pitch and roll angles from the accelerometer data.

        Pitch represents the forward/backward tilt, while roll represents
        the left/right tilt, both in degrees.

        Args:
            **parameters: Additional parameters (unused currently, reserved for future use).

        Returns:
            AngleSignal: A new AngleSignal instance with computed pitch and roll angles.

        Raises:
            ValueError: If the signal data does not contain 'x', 'y', 'z' columns.
        """
        # Avoid circular import by importing here
        from .angle_signal import AngleSignal

        df = self.get_data()
        if df is None or df.empty:
            raise ValueError("Cannot compute angles from empty signal data")

        if not all(col in df.columns for col in ['x', 'y', 'z']):
            raise ValueError("AccelerometerSignal must contain 'x', 'y', and 'z' columns")

        # Calculate angles
        result_data = pd.DataFrame(index=df.index)

        y_z_square = df['y']**2 + df['z']**2
        sqrt_y_z_square = np.sqrt(y_z_square)

        # Compute pitch: arctan(x / sqrt(y² + z²)) in degrees
        # Handle division by zero: if sqrt is zero, pitch is ±90 degrees depending on x
        result_data['pitch'] = np.where(
            sqrt_y_z_square == 0,
            np.sign(df['x']) * 90,
            np.arctan2(df['x'], sqrt_y_z_square) * 180 / np.pi
        )

        # Compute roll: arctan(y / z) in degrees
        # Handle division by zero: if z is zero, roll is ±90 degrees depending on y
        result_data['roll'] = np.where(
            df['z'] == 0,
            np.sign(df['y']) * 90,
            np.arctan2(df['y'], df['z']) * 180 / np.pi
        )

        # Create new signal with proper metadata
        # Operation index is the index of the last operation on the source signal
        operation_index = len(self.metadata.operations) - 1
        new_metadata = {
            'name': f"{self.metadata.name}_angle",
            'derived_from': [(self.metadata.signal_id, operation_index)],
            'operations': [],  # Will be updated by AngleSignal init
        }

        # Create AngleSignal instance
        angle_signal = AngleSignal(
            data=result_data,
            metadata=new_metadata,
            handler=self.handler
        )

        # Record the operation in the new signal's metadata
        from ..core.metadata import OperationInfo
        op_info = OperationInfo(
            operation_name="compute_angle",
            parameters=parameters
        )
        angle_signal.metadata.operations.append(op_info)

        return angle_signal
