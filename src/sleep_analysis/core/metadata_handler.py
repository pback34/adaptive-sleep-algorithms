"""
Metadata handler for standardized metadata management.

This module provides a centralized handler for managing signal metadata
in a consistent way across standalone signals and collections.
"""

from typing import Dict, Any, Optional, List, Union # Added Union
import uuid
import pandas as pd # Added import
# Updated import to use TimeSeriesMetadata and FeatureMetadata
from ..core.metadata import TimeSeriesMetadata, FeatureMetadata, OperationInfo

class MetadataHandler:
    """
    Centralized handler for managing metadata for all signals.
    
    This class provides methods to initialize, update, and modify metadata
    in a consistent way, ensuring that all signals follow the same patterns
    regardless of whether they are standalone or part of a collection.
    """
    
    def __init__(self, default_values: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata handler with optional default values.
        
        Args:
            default_values: Dictionary of default values for metadata fields
        """
        self.default_values = default_values or {}
        self.required_fields = ["signal_id"]  # Signal ID is always required for TimeSeriesMetadata

    # Updated to specifically handle TimeSeriesMetadata initialization
    def initialize_metadata(self, **kwargs) -> TimeSeriesMetadata:
        """
        Create TimeSeriesMetadata with defaults and specified overrides.

        This method creates a new TimeSeriesMetadata instance with default values,
        then applies any provided overrides from kwargs. It is specific to TimeSeries signals.
        
        Args:
            **kwargs: Metadata field values to override defaults

        Returns:
            Initialized TimeSeriesMetadata instance

        Raises:
            ValueError: If a required field is missing or invalid for TimeSeriesMetadata
        """
        # Start with default values
        metadata_dict = dict(self.default_values)
        
        # Generate a signal_id if not provided
        if "signal_id" not in metadata_dict and "signal_id" not in kwargs:
            metadata_dict["signal_id"] = str(uuid.uuid4())
            
        # Apply overrides from kwargs
        metadata_dict.update(kwargs)
        
        # Filter out any keys that are not valid for TimeSeriesMetadata
        # Need to import pandas here if not already imported globally
        import pandas as pd
        from dataclasses import fields
        valid_fields = {f.name for f in fields(TimeSeriesMetadata)} # Use TimeSeriesMetadata
        filtered_dict = {}
        for k, v in metadata_dict.items():
            if k in valid_fields:
                # Special handling for Timedelta conversion if needed (e.g., from string)
                if k in ['epoch_window_length', 'epoch_step_size'] and isinstance(v, str):
                    try:
                        filtered_dict[k] = pd.Timedelta(v)
                    except ValueError:
                        raise ValueError(f"Invalid format for {k}: '{v}'. Use pandas Timedelta string format (e.g., '30s', '5m').")
                else:
                    filtered_dict[k] = v

        # Create the TimeSeriesMetadata instance
        metadata = TimeSeriesMetadata(**filtered_dict) # Use TimeSeriesMetadata

        # Ensure required fields are set (for TimeSeriesMetadata)
        for field in self.required_fields: # self.required_fields currently only contains 'signal_id'
            if not getattr(metadata, field, None):
                raise ValueError(f"Required metadata field '{field}' is missing")
                
        return metadata

    # Updated type hint to handle both metadata types
    def update_metadata(self, metadata: Union[TimeSeriesMetadata, FeatureMetadata], **kwargs) -> None:
        """
        Update existing metadata (TimeSeries or Feature) with new values.

        Args:
            metadata: The TimeSeriesMetadata or FeatureMetadata instance to update.
            **kwargs: New field values to apply.
        """
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

    # Updated type hint to handle both metadata types
    def set_name(self, metadata: Union[TimeSeriesMetadata, FeatureMetadata], name: Optional[str] = None, key: Optional[str] = None) -> None:
        """
        Set the 'name' field using a fallback strategy. Prioritizes key if provided.

        Works for both TimeSeriesMetadata and FeatureMetadata.

        Args:
            metadata: The TimeSeriesMetadata or FeatureMetadata instance to update.
            name: Explicit name to set (used only if key is not provided).
            key: Collection key to use (highest priority).
        """
        # --- MODIFIED LOGIC ---
        if key:
            # If key is provided, it represents the signal's identity in the collection. Use it.
            metadata.name = key
        elif name:
            # If no key, but explicit name is given, use that.
            metadata.name = name
        elif not metadata.name:
            # If no key and no explicit name, and name isn't already set, fallback to signal_id.
            metadata.name = f"signal_{metadata.signal_id[:8]}"
        # If key is None, name is None, and metadata.name already exists, we do nothing (preserve existing name).
        # --- END MODIFIED LOGIC ---

    # Updated type hint to handle both metadata types
    def record_operation(self, metadata: Union[TimeSeriesMetadata, FeatureMetadata], operation_name: str, parameters: Dict[str, Any]) -> None:
        """
        Record an operation in the metadata's operation history.

        Works for both TimeSeriesMetadata and FeatureMetadata.

        Args:
            metadata: The TimeSeriesMetadata or FeatureMetadata instance to update.
            operation_name: Name of the operation performed.
            parameters: Parameters used for the operation.
        """
        if metadata.operations is None: # Should not happen with default_factory=list
            metadata.operations = []

        # Sanitize parameters before storing
        sanitized_params = self._sanitize_parameters(parameters)

        operation_info = OperationInfo(operation_name, sanitized_params)
        metadata.operations.append(operation_info)

    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize operation parameters for safe storage in metadata.

        Converts complex objects like DataFrames or Index objects into
        string representations.

        Args:
            params: The original parameters dictionary.

        Returns:
            A new dictionary with sanitized parameters.
        """
        sanitized = {}
        if not isinstance(params, dict):
             # Return input directly if not a dict (shouldn't happen with **kwargs)
             return params

        for key, value in params.items():
            if isinstance(value, pd.DataFrame):
                sanitized[key] = f"<DataFrame shape={value.shape}>"
            elif isinstance(value, pd.Series):
                sanitized[key] = f"<Series size={len(value)}>"
            elif isinstance(value, pd.Index):
                 # More specific for DatetimeIndex
                 if isinstance(value, pd.DatetimeIndex):
                      sanitized[key] = f"<DatetimeIndex size={len(value)} freq={value.freq}>"
                 else:
                      sanitized[key] = f"<Index size={len(value)}>"
            # Add checks for other large or non-serializable types if needed
            # elif isinstance(value, np.ndarray):
            #     sanitized[key] = f"<ndarray shape={value.shape}>"
            else:
                # Assume other types are serializable (or handle specific cases)
                sanitized[key] = value
        return sanitized
