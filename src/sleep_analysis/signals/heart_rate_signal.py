"""
Heart rate signal class implementation.

This module defines the HeartRateSignal class for heart rate data.
"""

from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType

class HeartRateSignal(TimeSeriesSignal):
    """
    Class for heart rate signals.
    
    Heart rate signals typically measure beats per minute (BPM) and may include 
    heart rate variability (HRV) data measured in milliseconds, which represents 
    the variation in time between consecutive heartbeats.
    """
    _is_abstract = False
    signal_type = SignalType.HEART_RATE
    required_columns = ['hr']
    optional_columns = ['hrv']  # HRV is optional as not all HR signals include it

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

    def get_hrv_stats(self):
        """
        Calculate basic statistics from the HRV data if available.
        
        Returns:
            Dictionary containing HRV statistics including mean, median, 
            standard deviation, and RMSSD (Root Mean Square of Successive Differences).
            Returns None if HRV data is not available.
        """
        import pandas as pd
        import numpy as np
        
        df = self.get_data()
        
        if 'hrv' not in df.columns:
            return None
            
        # Filter out missing values
        hrv_values = df['hrv'].dropna()
        
        if len(hrv_values) < 2:
            return {
                'mean': hrv_values.mean() if len(hrv_values) > 0 else None,
                'count': len(hrv_values),
                'available': False  # Not enough data for meaningful statistics
            }
            
        # Calculate successive differences for RMSSD
        successive_diffs = hrv_values.diff().dropna()
        rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else None
        
        return {
            'mean': hrv_values.mean(),
            'median': hrv_values.median(),
            'std': hrv_values.std(),
            'min': hrv_values.min(),
            'max': hrv_values.max(),
            'rmssd': rmssd,  # Root Mean Square of Successive Differences
            'count': len(hrv_values),
            'available': True
        }
