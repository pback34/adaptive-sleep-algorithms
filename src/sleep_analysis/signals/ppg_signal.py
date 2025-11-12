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

    def normalize(self, inplace: bool = False, **parameters) -> 'PPGSignal':
        """
        Mock implementation of PPG normalization for testing.

        This is a placeholder implementation that returns the signal unchanged.
        In a real implementation, this would normalize the PPG values.

        Args:
            inplace: If True, modify this signal in place. If False, return a new signal.
            **parameters: Additional normalization parameters (unused in mock).

        Returns:
            PPGSignal: The normalized signal (self if inplace=True, new instance otherwise).
        """
        # Mock implementation: just return the data as-is
        if inplace:
            # For inplace, we don't modify anything in this mock
            # But we should still record the operation in metadata
            from ..core.metadata import OperationInfo
            op_info = OperationInfo(
                operation_name="normalize",
                parameters=parameters
            )
            self.metadata.operations.append(op_info)
            return self
        else:
            # For non-inplace, create a new signal with the same data
            new_data = self.get_data().copy()
            # Operation index is the index of the last operation on the source signal
            # If no operations exist, this will be -1
            operation_index = len(self.metadata.operations) - 1
            new_metadata = {
                'name': f"{self.metadata.name}_normalized",
                'derived_from': [(self.metadata.signal_id, operation_index)],
                'operations': [],  # Will be updated below
            }

            # Create new PPGSignal instance
            new_signal = PPGSignal(
                data=new_data,
                metadata=new_metadata,
                handler=self.handler
            )

            # Record the operation in the new signal's metadata
            from ..core.metadata import OperationInfo
            op_info = OperationInfo(
                operation_name="normalize",
                parameters=parameters
            )
            new_signal.metadata.operations.append(op_info)

            return new_signal


   
