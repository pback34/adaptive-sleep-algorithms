"""
Class definition for Feature objects derived from epoch-based analysis.
"""

from typing import Dict, Any, Optional, List, Callable
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

    Supports lazy evaluation: Features can be created with a computation function
    that is only executed when get_data() is first called, saving memory and
    computation time for unused features.
    """
    # Removed signal_type and required_columns class attributes

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        handler: Optional[MetadataHandler] = None,
        lazy: bool = False,
        computation_function: Optional[Callable] = None,
        computation_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Feature instance.

        Supports two modes:
        1. Eager evaluation: Provide data directly (traditional mode)
        2. Lazy evaluation: Provide computation_function and defer computation

        Args:
            data: pandas DataFrame containing feature data, indexed by epoch start time.
                  Optional if using lazy evaluation.
            metadata: Dictionary of metadata values used to initialize FeatureMetadata.
                      Must include 'epoch_window_length', 'epoch_step_size',
                      'feature_names', 'source_signal_keys', and 'source_signal_ids'.
            handler: Optional metadata handler (less commonly used directly for Features).
            lazy: If True, enables lazy evaluation mode. Requires computation_function.
            computation_function: Function to call to compute the data (lazy mode only).
                                 Should return a pandas DataFrame.
            computation_args: Arguments to pass to computation_function (lazy mode only).

        Raises:
            ValueError: If data/computation setup is invalid or metadata is missing/invalid.
            TypeError: If metadata argument is not a dictionary.

        Examples:
            Eager evaluation:
                >>> feature = Feature(data=df, metadata=meta_dict)

            Lazy evaluation:
                >>> feature = Feature(
                ...     metadata=meta_dict,
                ...     lazy=True,
                ...     computation_function=compute_stats,
                ...     computation_args={'signals': signals, 'params': params}
                ... )
        """
        # Validate lazy evaluation setup
        if lazy:
            if computation_function is None:
                raise ValueError("lazy=True requires computation_function to be provided")
            if not callable(computation_function):
                raise TypeError("computation_function must be callable")
            if data is not None:
                logger.warning("Lazy mode enabled but data provided. Data will be ignored.")
                data = None
        else:
            if data is None:
                raise ValueError("Non-lazy mode requires data to be provided")
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Feature data must be a pandas DataFrame.")
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Feature data must have a DatetimeIndex (epoch start times).")

        # Validate metadata
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

        # Validate feature_names match DataFrame columns (simple names) - only for eager mode
        if not lazy and data is not None:
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

        # Store data or lazy computation setup
        self._data = data
        self._lazy = lazy
        self._computation_function = computation_function
        self._computation_args = computation_args or {}
        self._computed = not lazy  # Track if data has been computed
        self.handler = handler # Store handler if provided

        # Ensure name is set (using key/name from metadata_dict or fallback)
        if not self.metadata.name:
             self.metadata.name = f"feature_{self.metadata.feature_id[:8]}"

    def _compute_data(self) -> pd.DataFrame:
        """
        Internal method to compute the feature data using the stored function.

        Returns:
            Computed feature DataFrame.

        Raises:
            RuntimeError: If computation fails or lazy mode not properly set up.
        """
        if not self._lazy:
            raise RuntimeError("_compute_data called on non-lazy Feature")

        if self._computation_function is None:
            raise RuntimeError("No computation function available for lazy Feature")

        logger.debug(f"Computing lazy feature data for '{self.metadata.name}'")

        try:
            # Call the computation function with stored arguments
            data = self._computation_function(**self._computation_args)

            # Validate the computed data
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Computation function must return DataFrame, got {type(data).__name__}")
            if not isinstance(data.index, pd.DatetimeIndex):
                raise TypeError("Computed data must have DatetimeIndex")

            return data

        except Exception as e:
            logger.error(f"Failed to compute lazy feature '{self.metadata.name}': {e}")
            raise RuntimeError(f"Lazy feature computation failed: {e}") from e

    def get_data(self) -> pd.DataFrame:
        """
        Get the feature data DataFrame.

        For lazy features, this triggers computation on first call and caches the result.
        For eager features, this returns the stored data directly.

        Returns:
            pandas DataFrame containing features indexed by epoch start time.

        Raises:
            RuntimeError: If data is cleared or computation fails.
        """
        # Check if data needs to be computed (lazy mode)
        if self._lazy and not self._computed:
            self._data = self._compute_data()
            self._computed = True
            logger.info(f"Lazy feature '{self.metadata.name}' computed and cached")

        # Return stored data
        if self._data is None:
             raise RuntimeError(
                 f"Feature data for '{self.metadata.name}' has been cleared and cannot be regenerated. "
                 "Re-run the feature extraction workflow step."
             )
        return self._data

    def clear_data(self):
        """
        Clear the stored data to free memory.

        For lazy features, data can be recomputed on next get_data() call.
        For eager features, data cannot be recovered.
        """
        if self._lazy:
            logger.debug(f"Clearing lazy feature data for '{self.metadata.name}' (can be recomputed)")
            self._computed = False
        else:
            logger.warning(f"Clearing eager feature data for '{self.metadata.name}' (cannot be recovered)")

        self._data = None

    def is_lazy(self) -> bool:
        """Check if this Feature uses lazy evaluation."""
        return self._lazy

    def is_computed(self) -> bool:
        """Check if lazy feature data has been computed."""
        return self._computed

    # Removed apply_operation, get_sampling_rate, _update_sample_rate_metadata
    # Add __repr__ for better debugging
    def __repr__(self) -> str:
        if self._lazy and not self._computed:
            status = "Lazy(not computed)"
        elif self._data is not None:
            status = f"shape={self._data.shape}"
        else:
            status = "Cleared"
        return f"<Feature name='{self.metadata.name}' id='{self.metadata.feature_id[:8]}' {status}>"
