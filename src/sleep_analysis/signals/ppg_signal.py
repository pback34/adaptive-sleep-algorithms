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


    # --- Instance Methods (Including Overrides) ---

    def filter_lowpass(self, cutoff: float = 5.0, **other_params) -> pd.DataFrame:
        """
        Apply a low-pass filter specifically for PPG signals (core logic).

        This method overrides the default TimeSeriesSignal implementation.
        It performs the calculation and returns the resulting DataFrame.
        Metadata updates and instance handling are managed by apply_operation.

        Args:
            cutoff: The window size for the moving average. Defaults to 5.0.
            **other_params: Additional parameters.

        Returns:
            A DataFrame containing the filtered PPG data.
        """
        import pandas as pd # Local imports
        import numpy as np
        import logging
        logger = logging.getLogger(__name__)

        window_size = int(cutoff)
        if window_size < 1:
             raise ValueError("Cutoff (window size) for moving average must be at least 1.")

        data = self.get_data() # Get current data
        if data is None:
             raise ValueError("Cannot apply PPG filter_lowpass: signal data is None.")

        logger.debug(f"Applying PPG-specific rolling mean with window size {window_size}")
        processed_data = data.copy()
        # PPG typically has 'value' column, but apply to all numeric just in case
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if not numeric_cols.empty:
            for col in numeric_cols:
                processed_data[col] = data[col].rolling(window=window_size, min_periods=1).mean()
            logger.debug(f"Applied rolling mean to PPG columns: {list(numeric_cols)}")
        else:
            logger.warning("No numeric columns found in PPG signal to apply low-pass filter.")

        return processed_data
