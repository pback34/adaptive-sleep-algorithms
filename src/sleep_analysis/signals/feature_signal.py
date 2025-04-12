"""
Class definition for Feature Signals derived from epoch-based analysis.
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from ..core.signal_data import SignalData
from ..signal_types import SignalType
from ..core.metadata_handler import MetadataHandler

class FeatureSignal(SignalData):
    """
    Class for signals containing features computed over epochs.

    These signals typically result from applying aggregation functions
    (e.g., mean, std, correlation) over defined time windows (epochs)
    of one or more source TimeSeriesSignals. The data is a DataFrame
    where the index represents epoch start times and columns represent
    the computed features.
    """
    signal_type = SignalType.FEATURES
    # FeatureSignals don't have inherent required columns in the same way
    # TimeSeriesSignals do; columns are defined by the features computed.
    required_columns = []

    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None, handler: Optional[MetadataHandler] = None):
        """
        Initialize a FeatureSignal instance.

        Args:
            data: pandas DataFrame containing feature data, indexed by epoch start time.
            metadata: Optional metadata dictionary. Must include feature-specific fields
                      like 'epoch_window_length', 'epoch_step_size', 'feature_names',
                      and 'source_signal_keys'.
            handler: Optional metadata handler.

        Raises:
            ValueError: If data is not a DataFrame or lacks a DatetimeIndex.
            ValueError: If required feature-specific metadata is missing.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("FeatureSignal data must be a pandas DataFrame.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("FeatureSignal data must have a DatetimeIndex (epoch start times).")

        # Initialize metadata first to check required fields
        temp_metadata = metadata or {}
        # Ensure signal_type is set correctly for validation during super().__init__
        temp_metadata['signal_type'] = self.signal_type

        # Check for required feature-specific metadata before calling super().__init__
        required_feature_meta = ['epoch_window_length', 'epoch_step_size', 'feature_names', 'source_signal_keys']
        missing_meta = [field for field in required_feature_meta if field not in temp_metadata or temp_metadata[field] is None]
        if missing_meta:
            raise ValueError(f"Missing required metadata fields for FeatureSignal: {missing_meta}")

        # Validate feature_names match DataFrame columns, *unless* columns are MultiIndex
        if not isinstance(data.columns, pd.MultiIndex):
            if 'feature_names' in temp_metadata and isinstance(temp_metadata['feature_names'], list):
                # Check if the set of feature names matches the set of column names
                if set(temp_metadata['feature_names']) != set(data.columns):
                    raise ValueError(f"Metadata 'feature_names' {temp_metadata['feature_names']} do not match DataFrame columns {list(data.columns)}")
            else:
                # If feature_names wasn't provided correctly or is not a list, derive it from columns
                temp_metadata['feature_names'] = list(data.columns)
        else:
            # If columns are MultiIndex, feature_names metadata is expected to be empty or not used for this validation
            # Ensure feature_names is an empty list in metadata if columns are MultiIndex
            if 'feature_names' not in temp_metadata or temp_metadata['feature_names']:
                 temp_metadata['feature_names'] = []


        # Call parent initializer AFTER validation
        super().__init__(data=data, metadata=temp_metadata, handler=handler)

        # Ensure feature_names metadata matches columns after potential superclass modifications
        if set(self.metadata.feature_names) != set(self._data.columns):
             # This case should ideally not be reached due to earlier checks
             self.metadata.feature_names = list(self._data.columns)


    def get_data(self) -> pd.DataFrame:
        """
        Get the feature data DataFrame.

        Returns:
            pandas DataFrame containing features indexed by epoch start time.
        """
        # FeatureSignal data regeneration is complex and likely requires
        # re-running the specific feature extraction operation via the
        # collection/workflow. We don't attempt it here.
        if self._data is None:
             raise RuntimeError("FeatureSignal data has been cleared and cannot be regenerated directly. Re-run the feature extraction workflow step.")
        return self._data

    def apply_operation(self, operation_name: str, inplace: bool = False, **parameters) -> 'SignalData':
        """
        Apply an operation. FeatureSignals typically don't have operations applied
        directly to them in the same way as TimeSeriesSignals. Raises NotImplementedError.
        """
        # Feature signals are usually endpoints or inputs to combination/export.
        # Defining operations *on* feature signals is not standard in this plan.
        raise NotImplementedError(f"Operations cannot be applied directly to FeatureSignal objects. "
                                  f"Operation '{operation_name}' is not supported.")

    # FeatureSignals don't have a sampling rate in the traditional sense.
    def get_sampling_rate(self) -> None:
        """Returns None, as FeatureSignals don't have a sampling rate."""
        return None

    def _update_sample_rate_metadata(self):
        """Does nothing, as FeatureSignals don't have a sampling rate."""
        pass
