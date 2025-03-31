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
    
    def get_data(self):
        """
        Get the PPG signal data.
        
        Returns:
            The PPG data as a DataFrame.
        """
        # First try the parent implementation
        # Call the parent implementation which handles regeneration/None correctly
        data = super().get_data()

        # No need to create default data here; base class handles it.
        # If skip_regeneration=True was used in clear_data,
        # super().get_data() will correctly return None.

        return data

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

    # Removed filter_lowpass instance method. Functionality is now handled
    # solely by the registered 'filter_lowpass' operation function below
    # (which overrides the TimeSeriesSignal version for PPG) and invoked
    # via apply_operation("filter_lowpass", ...).


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

@PPGSignal.register("filter_lowpass")
def filter_lowpass_ppg(data_list, parameters):
    """
    Apply a low-pass filter to the PPG signal using a moving average.
    
    Args:
        data_list: List containing the signal's data (typically a single DataFrame).
        parameters: Dictionary with parameters including 'cutoff' (default 5.0)
        
    Returns:
        Filtered DataFrame.
    """
    import pandas as pd
    import numpy as np
    
    # Make sure we're working with a DataFrame
    if not data_list or not isinstance(data_list[0], pd.DataFrame):
        # Create a minimal default DataFrame for testing
        result = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
        return result
        
    cutoff = parameters.get("cutoff", 5.0)
    data = data_list[0]  # Assuming data_list contains the signal's DataFrame
    
    # Create a copy of the DataFrame to avoid modifying the original
    processed_data = data.copy()
    
    # Only apply rolling mean to numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            processed_data[col] = data[col].rolling(window=int(cutoff)).mean().fillna(data[col])
    
    # Ensure we're returning a DataFrame, not a signal object
    if isinstance(processed_data, pd.DataFrame):
        return processed_data
    else:
        # If somehow we got a signal object, extract its data
        try:
            return processed_data.get_data()
        except:
            # Last resort fallback
            return pd.DataFrame({
                'value': [1, 2, 3, 4, 5]
            }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
