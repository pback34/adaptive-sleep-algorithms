"""
Accelerometer signal class implementation.

This module defines the AccelerometerSignal class for motion data.
"""

import pandas as pd
import numpy as np
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
    
    def get_sampling_rate(self) -> float:
        """
        Get the sampling rate of the accelerometer signal.
        
        Returns:
            The sampling rate in Hz.
        """
        # First try to calculate from index if possible
        data = self.get_data()
        if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            # Calculate from the timestamps
            try:
                timedeltas = data.index.to_series().diff().dropna()
                if not timedeltas.empty:
                    # Calculate median time delta in seconds
                    median_delta = timedeltas.median().total_seconds()
                    if median_delta > 0:
                        return 1.0 / median_delta
            except Exception:
                pass  # Fall back to metadata
                
        # Fall back to metadata or default
        sample_rate_str = getattr(self.metadata, 'sample_rate', '50Hz')
        try:
            # Extract numeric part from string like "50Hz"
            return float(sample_rate_str.replace('Hz', ''))
        except (ValueError, AttributeError):
            return 50.0  # Default value
            
    @TimeSeriesSignal.output_class(MagnitudeSignal)
    def compute_magnitude(self, **parameters):
        """
        Compute the scalar magnitude of accelerometer data.
        
        The magnitude is calculated as sqrt(x² + y² + z²) for each sample.
        
        Args:
            **parameters: Optional parameters (unused in basic implementation)
            
        Returns:
            DataFrame containing the 'magnitude' column
        """
        df = self.get_data()
        
        # Compute magnitude: sqrt(x² + y² + z²)
        result = pd.DataFrame(index=df.index)
        result['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        
        return result
        
    @TimeSeriesSignal.output_class(AngleSignal)
    def compute_angle(self, **parameters):
        """
        Compute pitch and roll angles from accelerometer data.
        
        Pitch represents the forward/backward tilt, calculated as arctan(x / sqrt(y² + z²)).
        Roll represents the left/right tilt, calculated as arctan(y / z).
        Both angles are returned in degrees.
        
        Args:
            **parameters: Optional parameters (unused in basic implementation)
            
        Returns:
            DataFrame containing 'pitch' and 'roll' columns in degrees
        """
        df = self.get_data()
        
        # Create result DataFrame with same index
        result = pd.DataFrame(index=df.index)
        
        # Compute pitch: arctan(x / sqrt(y² + z²)) in degrees
        # This gives the tilt forward/backward
        y_z_square = df['y']**2 + df['z']**2
        result['pitch'] = np.arctan2(df['x'], np.sqrt(y_z_square)) * 180 / np.pi
        
        # Compute roll: arctan(y / z) in degrees
        # This gives the tilt left/right
        result['roll'] = np.arctan2(df['y'], df['z']) * 180 / np.pi
        
        return result
