"""
Class definition for Feature objects derived from epoch-based analysis.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import logging # Added import
import uuid # Added import

# Removed SignalData import
from ..core.metadata import FeatureMetadata # Import new metadata class
from ..core.metadata_handler import MetadataHandler # Keep if needed for handler logic

# Removed SignalType import

logger = logging.getLogger(__name__) # Initialize logger

class Feature: # Renamed from FeatureSignal, removed inheritance from SignalData
    """
    Class representing epoch-based features derived from TimeSeriesSignals.

    Holds feature data (DataFrame indexed by epoch start time) and associated
    FeatureMetadata detailing provenance and generation parameters.
    """
    # Removed signal_type and required_columns class attributes

    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None, handler: Optional[MetadataHandler] = None):
        """
        Initialize a Feature instance.

        Args:
            data: pandas DataFrame containing feature data, indexed by epoch start time.
            metadata: Dictionary of metadata values used to initialize FeatureMetadata.
                      Must include 'epoch_window_length', 'epoch_step_size',
                      'feature_names', 'source_signal_keys', and 'source_signal_ids'.
            handler: Optional metadata handler (less commonly used directly for Features).

        Raises:
            ValueError: If data is not a DataFrame or lacks a DatetimeIndex.
            ValueError: If required feature-specific metadata is missing or invalid.
            TypeError: If metadata argument is not a dictionary.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Feature data must be a pandas DataFrame.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Feature data must have a DatetimeIndex (epoch start times).")
        if metadata is None:
             raise ValueError("Metadata dictionary must be provided to initialize Feature.")
        if not isinstance(metadata, dict):
             raise TypeError("Metadata argument must be a dictionary.")

        # Initialize metadata using the provided dictionary
        metadata_dict = metadata.copy() # Work on a copy

        # Check for required feature-specific metadata before creating FeatureMetadata
        required_feature_meta = ['epoch_window_length', 'epoch_step_size', 'feature_names', 'source_signal_keys', 'source_signal_ids']
        missing_meta = [field for field in required_feature_meta if field not in metadata_dict or metadata_dict[field] is None]
        if missing_meta:
            raise ValueError(f"Missing required metadata fields for Feature: {missing_meta}")

        # Convert timedelta strings if necessary
        for field in ['epoch_window_length', 'epoch_step_size']:
            if field in metadata_dict and isinstance(metadata_dict[field], str):
                try:
                    metadata_dict[field] = pd.Timedelta(metadata_dict[field])
                except ValueError:
                     raise ValueError(f"Invalid format for {field}: '{metadata_dict[field]}'. Use pandas Timedelta string format.")

        # Validate feature_names match DataFrame columns (simple names)
        # This validation assumes feature_names contains the *simple* names
        # and the DataFrame columns might be simple or MultiIndex from the operation.
        if 'feature_names' in metadata_dict and isinstance(metadata_dict['feature_names'], list):
             feature_names_set = set(metadata_dict['feature_names'])
             # If data columns are MultiIndex, feature_names should list the last level values
             if isinstance(data.columns, pd.MultiIndex):
                  last_level_values = set(data.columns.get_level_values(-1))
                  if feature_names_set != last_level_values:
                       logger.warning(f"Metadata 'feature_names' {sorted(list(feature_names_set))} may not perfectly match the last level of MultiIndex columns {sorted(list(last_level_values))} for feature '{metadata_dict.get('name', 'unnamed')}'")
                       # Attempt to fix metadata if mismatch detected
                       metadata_dict['feature_names'] = sorted(list(last_level_values))
             # If data columns are simple, they should match feature_names
             elif feature_names_set != set(data.columns):
                  raise ValueError(f"Metadata 'feature_names' {sorted(list(feature_names_set))} do not match DataFrame columns {list(data.columns)} for feature '{metadata_dict.get('name', 'unnamed')}'")
        elif not data.empty: # Only attempt derivation if data is not empty
             # If feature_names wasn't provided correctly, derive from columns if possible
             logger.warning(f"Metadata 'feature_names' missing or invalid for feature '{metadata_dict.get('name', 'unnamed')}'. Deriving from data columns.")
             if isinstance(data.columns, pd.MultiIndex):
                  metadata_dict['feature_names'] = sorted(list(set(data.columns.get_level_values(-1))))
             else:
                  metadata_dict['feature_names'] = list(data.columns)
        else: # Data is empty, ensure feature_names is empty list
             metadata_dict['feature_names'] = []


        # Create the FeatureMetadata instance
        # Filter metadata_dict to only include valid FeatureMetadata fields
        # Corrected: Iterate over the keys directly
        valid_fields = set(FeatureMetadata.__dataclass_fields__.keys())
        filtered_metadata_dict = {k: v for k, v in metadata_dict.items() if k in valid_fields}
        try:
            # Ensure feature_id is generated if not provided
            if 'feature_id' not in filtered_metadata_dict:
                 filtered_metadata_dict['feature_id'] = str(uuid.uuid4())
            self.metadata = FeatureMetadata(**filtered_metadata_dict)
        except TypeError as e:
            logger.error(f"Error creating FeatureMetadata: {e}. Provided dict: {filtered_metadata_dict}")
            raise TypeError(f"Failed to initialize FeatureMetadata: {e}") from e

        # Store data directly
        self._data = data
        self.handler = handler # Store handler if provided

        # Ensure name is set (using key/name from metadata_dict or fallback)
        if not self.metadata.name:
             self.metadata.name = f"feature_{self.metadata.feature_id[:8]}"


    def get_data(self) -> pd.DataFrame:
        """
        Get the feature data DataFrame.

        Returns:
            pandas DataFrame containing features indexed by epoch start time.
        """
        # Feature data regeneration is complex and likely requires
        # re-running the specific feature extraction operation via the
        # collection/workflow. We don't attempt it here.
        if self._data is None:
             raise RuntimeError("Feature data has been cleared and cannot be regenerated directly. Re-run the feature extraction workflow step.")
        return self._data

    # Removed apply_operation, get_sampling_rate, _update_sample_rate_metadata
    # Add __repr__ for better debugging
    def __repr__(self) -> str:
        shape = self._data.shape if self._data is not None else "Cleared"
        return f"<Feature name='{self.metadata.name}' id='{self.metadata.feature_id[:8]}' shape={shape}>"
