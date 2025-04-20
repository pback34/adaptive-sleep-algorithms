"""
PPG signal class implementation.

This module defines the PPGSignal class for photoplethysmography data.
"""

import pandas as pd
from typing import Dict, Any # Added import
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


# --- Registered Operations ---

@PPGSignal.register("normalize") # Keeping mock normalize as registered for now
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


   
