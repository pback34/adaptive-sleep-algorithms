"""
PPG signal class implementation.

This module defines the PPGSignal class for photoplethysmography data.
"""

import pandas as pd
from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType

class PPGSignal(TimeSeriesSignal):
    """
    Class for photoplethysmography (PPG) signals.
    
    PPG signals measure blood volume changes in the microvascular bed of tissue.
    """
    _is_abstract = False
    signal_type = SignalType.PPG
    required_columns = ['value']
    
    def get_sampling_rate(self) -> float:
        """
        Get the sampling rate of the PPG signal.
        
        Returns:
            The sampling rate in Hz.
        """
        # First try to calculate from index if possible
        data = self.get_data()
        if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            # Calculate from the first two timestamps
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
        sample_rate_str = getattr(self.metadata, 'sample_rate', '100Hz')
        try:
            # Extract numeric part from string like "100Hz"
            return float(sample_rate_str.replace('Hz', ''))
        except (ValueError, AttributeError):
            return 100.0  # Default value


@PPGSignal.register("normalize")
def mock_normalize(data_list, parameters):
    """
    Mock implementation of PPG normalization for testing.
    
    Args:
        data_list: List of data arrays to normalize
        parameters: Normalization parameters
        
    Returns:
        Normalized data (in this mock, just returns the input data)
    """
    return data_list[0]
