"""
Signal collection for managing multiple signals.

This module defines the SignalCollection class, which serves as a container
for all signals in the framework and provides methods for adding, retrieving,
and managing signals.
"""

# Standard library imports
import math
import warnings
import uuid
import logging
import time
import os
import glob
from dataclasses import fields, dataclass # Added dataclass
from datetime import datetime # Added datetime
from typing import Dict, List, Any, Optional, Type, Tuple, Callable, Union

# Third-party imports
import pandas as pd
import numpy as np
from enum import Enum # Added import

# Local application imports
# Updated metadata imports
from .metadata import TimeSeriesMetadata, FeatureMetadata, CollectionMetadata, OperationInfo, FeatureType
from .signal_data import SignalData
# Updated signal_types import
from ..signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from .metadata_handler import MetadataHandler
from ..signals.time_series_signal import TimeSeriesSignal
# Import Feature from its new location
from ..features.feature import Feature
# Removed FeatureSignal import as Feature is now used
# from ..signals.feature_signal import FeatureSignal
from ..utils import str_to_enum

import functools # Added for decorator
import inspect # Added for __init_subclass__

import inspect # Added for __init_subclass__

# Initialize logger for the module
logger = logging.getLogger(__name__)


# --- Decorator Definition (Simple Function) ---
def register_collection_operation(operation_name: str):
    """
    Decorator to mark a SignalCollection method as a registered collection operation.
    Stores the operation name in the '_collection_op_name' attribute of the function.
    """
    def decorator(func: Callable):
        setattr(func, '_collection_op_name', operation_name)
        # Optional: Use functools.wraps if needed, but primarily for marking here
        # @functools.wraps(func)
        # def wrapper(*args, **kwargs):
        #     return func(*args, **kwargs)
        # setattr(wrapper, '_collection_op_name', operation_name)
        # return wrapper
        return func # Return the original function marked with the attribute
    return decorator


# Define standard rates: factors of 1000 Hz plus rates corresponding to multi-second periods
# Periods >= 1s: 1s (1Hz), 2s (0.5Hz), 4s (0.25Hz), 5s (0.2Hz), 10s (0.1Hz)
STANDARD_RATES = sorted(list(set([0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000])))

class SignalCollection:
    """
    Container for managing multiple signals with automatic indexing.
    
    SignalCollection serves as the central hub and single source of truth
    for all signals in the system, including imported signals, intermediate
    signals, and derived signals with base name indexing (e.g., 'ppg_0').
    """
    # Registry for multi-signal operations (used for operations creating new signals)
    multi_signal_registry: Dict[str, Tuple[Callable, Type[SignalData]]] = {}
    # Registry for collection-level operations (modifying collection state or signals within)
    # Populated AFTER class definition below
    collection_operation_registry: Dict[str, Callable] = {}

    # REMOVED __init_subclass__ method

    def __init__(self, metadata: Optional[Dict[str, Any]] = None, metadata_handler: Optional[MetadataHandler] = None):
        """
        Initialize a SignalCollection instance.
        
        Args:
            metadata: Optional dictionary with collection-level metadata
            metadata_handler: Optional metadata handler, will create one if not provided
        """
        # Initialize separate dictionaries for time series signals and features
        self.time_series_signals: Dict[str, TimeSeriesSignal] = {}
        self.features: Dict[str, Feature] = {}

        # Initialize the metadata handler
        self.metadata_handler = metadata_handler or MetadataHandler()
        
        # Initialize collection metadata
        metadata = metadata or {}
        self.metadata = CollectionMetadata(
            collection_id=metadata.get("collection_id", f"collection_{str(uuid.uuid4())[:8]}"),
            subject_id=metadata.get("subject_id", "subject_unknown"),
            session_id=metadata.get("session_id"),
            start_datetime=metadata.get("start_datetime"),
            end_datetime=metadata.get("end_datetime"),
            timezone=metadata.get("timezone", "UTC"),
            study_info=metadata.get("study_info", {}),
            device_info=metadata.get("device_info", {}),
            notes=metadata.get("notes", ""),
            protocol_id=metadata.get("protocol_id"),
            data_acquisition_notes=metadata.get("data_acquisition_notes")
        )
        # Attributes to store the generated aligned dataframe and its parameters
        self._aligned_dataframe: Optional[pd.DataFrame] = None
        self._aligned_dataframe_params: Optional[Dict[str, Any]] = None
        self._merge_tolerance: Optional[pd.Timedelta] = None # Store tolerance used in merge_asof

        # Alignment parameters calculated by generate_alignment_grid
        self.target_rate: Optional[float] = None
        self.ref_time: Optional[pd.Timestamp] = None
        self.grid_index: Optional[pd.DatetimeIndex] = None
        self._alignment_params_calculated: bool = False # Flag indicating if grid params are set

        # Epoch grid parameters calculated by generate_epoch_grid
        self.epoch_grid_index: Optional[pd.DatetimeIndex] = None
        self.global_epoch_window_length: Optional[pd.Timedelta] = None
        self.global_epoch_step_size: Optional[pd.Timedelta] = None
        self._epoch_grid_calculated: bool = False # Flag indicating if epoch grid params are set

        # Attributes for storing the summary dataframe
        self._summary_dataframe: Optional[pd.DataFrame] = None
        self._summary_dataframe_params: Optional[Dict[str, Any]] = None

        # Attribute for storing the combined feature matrix
        self._combined_feature_matrix: Optional[pd.DataFrame] = None

    def add_time_series_signal(self, key: str, signal: TimeSeriesSignal) -> None:
        """
        Add a TimeSeriesSignal to the collection.

        Args:
            key: Unique identifier for the signal in this collection.
            signal: The TimeSeriesSignal instance to add.

        Raises:
            ValueError: If a signal with the given key already exists in time_series_signals
                        or if the input is not a TimeSeriesSignal.
            TypeError: If signal is not a TimeSeriesSignal instance.
        """
        if not isinstance(signal, TimeSeriesSignal):
            raise TypeError(f"Signal provided for key '{key}' is not a TimeSeriesSignal (type: {type(signal).__name__}).")
        if key in self.time_series_signals:
            raise ValueError(f"TimeSeriesSignal with key '{key}' already exists in the collection.")

        # Check for signal_id uniqueness across *all* signals (time series and features)
        existing_ids = {s.metadata.signal_id for s in self.time_series_signals.values()} | \
                       {f.metadata.feature_id for f in self.features.values()} # Check feature IDs too
        if signal.metadata.signal_id in existing_ids:
            new_id = str(uuid.uuid4())
            logger.warning(f"TimeSeriesSignal ID '{signal.metadata.signal_id}' conflicts with an existing signal/feature ID. Assigning new ID: {new_id}")
            signal.metadata.signal_id = new_id

        # Validate timestamp index and timezone
        self._validate_timestamp_index(signal) # Checks for DatetimeIndex
        try:
            # Optional: Validate timezone consistency
                signal_tz = signal.get_data().index.tz
                collection_tz_str = self.metadata.timezone # The string name from collection metadata

                # Convert collection timezone string to tzinfo object for robust comparison if possible
                collection_tz = None
                if collection_tz_str:
                    try:
                        # Use pandas to interpret the timezone string robustly
                        collection_tz = pd.Timestamp('now', tz=collection_tz_str).tz
                    except Exception as tz_parse_err:
                        logger.warning(f"Could not parse collection timezone string '{collection_tz_str}' for validation: {tz_parse_err}")
                        # Proceed without robust comparison if collection TZ parsing fails
                        collection_tz = None # Ensure it's None if parsing failed

                # Perform comparison using string representations for robustness
                signal_tz_str = str(signal_tz) if signal_tz is not None else "None"

                if signal_tz is None and collection_tz_str != "None": # Check against string "None" if collection_tz_str could be None
                     logger.warning(f"Signal '{key}' has a naive timestamp index (timezone: {signal_tz_str}), while collection timezone is '{collection_tz_str}'. Potential inconsistency.")
                     # Optionally raise ValueError("Signal timezone mismatch: Signal is naive, collection is aware.")
                elif signal_tz is not None and collection_tz_str == "None":
                     logger.warning(f"Signal '{key}' has timezone '{signal_tz_str}', while collection timezone is not set ('{collection_tz_str}'). Potential inconsistency.")
                elif signal_tz is not None and collection_tz_str != "None" and signal_tz_str != collection_tz_str:
                     # Compare string representations
                     logger.warning(f"Signal '{key}' timezone string ('{signal_tz_str}') does not match collection timezone string ('{collection_tz_str}'). Potential inconsistency.")
                     # Optionally raise ValueError("Signal timezone mismatch")
                # No warning if both are None or if string representations match

        except AttributeError:
            logger.warning(f"Could not access index or timezone for signal '{key}' during validation.")
        except Exception as val_err:
            logger.error(f"Error during timezone validation for signal '{key}': {val_err}", exc_info=True)

        # Set the signal's name to the key if not already set
        if signal.handler:
            signal.handler.set_name(signal.metadata, key=key)
        else:
            signal.handler = self.metadata_handler
            signal.handler.set_name(signal.metadata, key=key)

        self.time_series_signals[key] = signal

    def add_feature(self, key: str, feature: Feature) -> None:
        """
        Add a Feature object to the collection.

        Args:
            key: Unique identifier for the feature set in this collection.
            feature: The Feature instance to add.

        Raises:
            ValueError: If a feature with the given key already exists.
            TypeError: If feature is not a Feature instance.
        """
        if not isinstance(feature, Feature):
            raise TypeError(f"Object provided for key '{key}' is not a Feature (type: {type(feature).__name__}).")
        if key in self.features:
            raise ValueError(f"Feature with key '{key}' already exists in the collection.")

        # Check for feature_id uniqueness across *all* signals/features
        existing_ids = {s.metadata.signal_id for s in self.time_series_signals.values()} | \
                       {f.metadata.feature_id for f in self.features.values()}
        if feature.metadata.feature_id in existing_ids:
            new_id = str(uuid.uuid4())
            logger.warning(f"Feature ID '{feature.metadata.feature_id}' conflicts with an existing signal/feature ID. Assigning new ID: {new_id}")
            feature.metadata.feature_id = new_id

        # Set the feature's name to the key if not already set
        if feature.handler:
            feature.handler.set_name(feature.metadata, key=key) # Assuming handler can handle FeatureMetadata
        else:
            feature.handler = self.metadata_handler
            feature.handler.set_name(feature.metadata, key=key) # Assuming handler can handle FeatureMetadata

        self.features[key] = feature

    def add_signal_with_base_name(self, base_name: str, signal: Union[TimeSeriesSignal, Feature]) -> str:
        """
        Add a TimeSeriesSignal or Feature with a base name, appending an index if needed.

        Args:
            base_name: Base name for the signal/feature (e.g., "ppg", "hr_stats").
            signal: The TimeSeriesSignal or Feature instance to add.

        Returns:
            The key assigned to the signal/feature (e.g., "ppg_0", "hr_stats_1").

        Raises:
            ValueError: If the base name is empty.
            TypeError: If signal is not a TimeSeriesSignal or Feature.
        """
        if not base_name:
            raise ValueError("Base name cannot be empty")

        target_dict = None
        if isinstance(signal, TimeSeriesSignal):
            target_dict = self.time_series_signals
        elif isinstance(signal, Feature):
            target_dict = self.features
        else:
            raise TypeError(f"Input must be a TimeSeriesSignal or Feature, got {type(signal).__name__}")

        index = 0
        while True:
            key = f"{base_name}_{index}"
            if key not in target_dict:
                # Use the appropriate add method
                if isinstance(signal, TimeSeriesSignal):
                    self.add_time_series_signal(key, signal)
                else: # Must be Feature
                    self.add_feature(key, signal)
                return key
            index += 1

    def get_time_series_signal(self, key: str) -> TimeSeriesSignal:
        """Retrieve a TimeSeriesSignal by its key."""
        if key not in self.time_series_signals:
            raise KeyError(f"No TimeSeriesSignal with key '{key}' found in the collection.")
        return self.time_series_signals[key]

    def get_feature(self, key: str) -> Feature:
        """Retrieve a Feature object by its key."""
        if key not in self.features:
            raise KeyError(f"No Feature with key '{key}' found in the collection.")
        return self.features[key]

    def get_signal(self, key: str) -> Union[TimeSeriesSignal, Feature]:
        """
        Retrieve a TimeSeriesSignal or Feature by its key.

        Checks both time_series_signals and features dictionaries.

        Args:
            key: The key used when adding the signal or feature.

        Returns:
            The requested TimeSeriesSignal or Feature instance.

        Raises:
            KeyError: If no signal or feature exists with the specified key.
        """
        if key in self.time_series_signals:
            return self.time_series_signals[key]
        elif key in self.features:
            return self.features[key]
        else:
            raise KeyError(f"No TimeSeriesSignal or Feature with key '{key}' found in the collection.")

    def get_signals(self, input_spec: Union[str, Dict[str, Any], List[str], None] = None,
                   signal_type: Union[SignalType, str, None] = None,
                   feature_type: Union[FeatureType, str, None] = None, # Added feature_type filter
                   criteria: Dict[str, Any] = None,
                   base_name: str = None) -> List[Union[TimeSeriesSignal, Feature]]:
        """
        Retrieve TimeSeriesSignals and/or Features based on flexible criteria.

        Searches both `time_series_signals` and `features` containers.

        Args:
            input_spec: Can be:
                        - String ID or base name ("ppg", "ppg_0", "hr_stats_0")
                        - Dictionary with criteria/base_name
                        - List of string IDs or base names
            signal_type: A SignalType enum/string to filter TimeSeriesSignals.
            feature_type: A FeatureType enum/string to filter Features.
            criteria: Dictionary of metadata field/value pairs to match (searches
                      both TimeSeriesMetadata and FeatureMetadata).
            base_name: Base name to filter signals/features (e.g., "ppg", "hr_stats").

        Returns:
            List of matching TimeSeriesSignal and/or Feature instances.
        """
        results = []
        search_space = {**self.time_series_signals, **self.features} # Combine both dicts for searching

        # --- Prepare Criteria ---
        processed_criteria = criteria.copy() if criteria else {}

        # Add signal_type to criteria if provided
        if signal_type is not None:
            st = str_to_enum(signal_type, SignalType) if isinstance(signal_type, str) else signal_type
            processed_criteria["signal_type"] = st

        # Add feature_type to criteria if provided
        if feature_type is not None:
            ft = str_to_enum(feature_type, FeatureType) if isinstance(feature_type, str) else feature_type
            processed_criteria["feature_type"] = ft

        # --- Process input_spec ---
        if input_spec is not None:
            if isinstance(input_spec, dict):
                # Dictionary spec: extract base_name and merge criteria
                if "base_name" in input_spec:
                    base_name = input_spec["base_name"]
                if "criteria" in input_spec:
                    spec_criteria = self._process_enum_criteria(input_spec["criteria"])
                    processed_criteria.update(spec_criteria)

            elif isinstance(input_spec, list):
                # List spec: recursively call get_signals for each item
                for spec_item in input_spec:
                    # Pass down existing filters
                    results.extend(self.get_signals(input_spec=spec_item,
                                                   signal_type=signal_type,
                                                   feature_type=feature_type,
                                                   criteria=criteria, # Pass original criteria dict
                                                   base_name=base_name)) # Pass original base_name
                # Deduplicate results based on object ID
                return list({id(s): s for s in results}.values())

            else: # String spec
                spec_str = str(input_spec)
                # Check if it's an exact key
                if spec_str in search_space:
                    signal = search_space[spec_str]
                    if self._matches_criteria(signal, processed_criteria):
                        return [signal]
                    else:
                        return [] # Key found but doesn't match criteria
                # If not an exact key, treat it as a base name
                else:
                    base_name = spec_str

        # --- Apply Filtering (Base Name and Criteria) ---
        if base_name:
            # Filter by base name first
            filtered_signals = []
            for key, signal in search_space.items():
                 # Check if key matches the base name pattern (e.g., "basename_0")
                 if key.startswith(f"{base_name}_") and key[len(base_name)+1:].isdigit():
                      filtered_signals.append(signal)
        else:
            # If no base name, start with all signals/features
            filtered_signals = list(search_space.values())

        # Apply criteria filtering
        final_results = [s for s in filtered_signals if self._matches_criteria(s, processed_criteria)]

        # Deduplicate final results
        return list({id(s): s for s in final_results}.values())

    def _process_enum_criteria(self, criteria_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to convert string enum values in criteria to Enum objects."""
        processed = {}
        enum_map = {
            "signal_type": SignalType,
            "sensor_type": SensorType,
            "sensor_model": SensorModel,
            "body_position": BodyPosition,
            "feature_type": FeatureType,
            # Add other enum fields here if needed
        }
        for key, value in criteria_dict.items():
            if key in enum_map and isinstance(value, str):
                try:
                    processed[key] = str_to_enum(value, enum_map[key])
                except ValueError:
                    logger.warning(f"Invalid enum value '{value}' for criteria key '{key}'. Keeping as string.")
                    processed[key] = value # Keep original string if conversion fails
            else:
                processed[key] = value
        return processed


    def _matches_criteria(self, signal: Union[TimeSeriesSignal, Feature], criteria: Dict[str, Any]) -> bool:
        """Check if a TimeSeriesSignal or Feature matches all criteria."""
        if not criteria:
            return True

        metadata_obj = signal.metadata # Get the correct metadata object

        for key, value in criteria.items():
            # Handle nested fields (e.g., "sensor_info.device_id") - applies only to TimeSeriesMetadata
            if "." in key and isinstance(metadata_obj, TimeSeriesMetadata):
                parts = key.split(".", 1)
                container_name, field_name = parts
                if hasattr(metadata_obj, container_name):
                    container = getattr(metadata_obj, container_name)
                    if isinstance(container, dict) and field_name in container:
                        if container[field_name] != value:
                            return False
                    else: return False # Container not dict or field missing
                else: return False # Container attribute missing
            # Handle standard fields
            elif hasattr(metadata_obj, key):
                if getattr(metadata_obj, key) != value:
                    return False
            else: # Field doesn't exist on this metadata type
                return False
        return True

    def update_time_series_metadata(self, signal: TimeSeriesSignal, metadata_spec: Dict[str, Any]) -> None:
        """Update a TimeSeriesSignal's metadata."""
        if not isinstance(signal, TimeSeriesSignal):
             raise TypeError(f"Expected TimeSeriesSignal, got {type(signal).__name__}")

        # Process enum fields specifically for TimeSeriesMetadata
        processed_metadata = {}
        enum_map = {
            "signal_type": SignalType, "sensor_type": SensorType,
            "sensor_model": SensorModel, "body_position": BodyPosition, "units": Unit
        }
        for field, enum_cls in enum_map.items():
            if field in metadata_spec and isinstance(metadata_spec[field], str):
                try:
                    processed_metadata[field] = str_to_enum(metadata_spec[field], enum_cls)
                except ValueError:
                     logger.warning(f"Invalid enum value '{metadata_spec[field]}' for field '{field}'. Skipping update.")

        # Handle sensor_info separately
        if "sensor_info" in metadata_spec and isinstance(metadata_spec["sensor_info"], dict):
            if signal.metadata.sensor_info is None:
                signal.metadata.sensor_info = {}
            signal.metadata.sensor_info.update(metadata_spec["sensor_info"])

        # Add other valid TimeSeriesMetadata fields
        valid_fields = {f.name for f in fields(TimeSeriesMetadata)}
        for field in valid_fields:
            if field in metadata_spec and field not in processed_metadata and field != "sensor_info":
                processed_metadata[field] = metadata_spec[field]

        # Use the metadata handler to update
        handler = signal.handler or self.metadata_handler
        handler.update_metadata(signal.metadata, **processed_metadata)

    def update_feature_metadata(self, feature: Feature, metadata_spec: Dict[str, Any]) -> None:
        """Update a Feature's metadata."""
        if not isinstance(feature, Feature):
             raise TypeError(f"Expected Feature, got {type(feature).__name__}")

        processed_metadata = {}
        # Process FeatureType enum
        if "feature_type" in metadata_spec and isinstance(metadata_spec["feature_type"], str):
             try:
                  processed_metadata["feature_type"] = str_to_enum(metadata_spec["feature_type"], FeatureType)
             except ValueError:
                  logger.warning(f"Invalid enum value '{metadata_spec['feature_type']}' for field 'feature_type'. Skipping update.")

        # Add other valid FeatureMetadata fields
        valid_fields = {f.name for f in fields(FeatureMetadata)}
        for field in valid_fields:
            if field in metadata_spec and field not in processed_metadata:
                 # Special handling for timedeltas if provided as strings
                 if field in ['epoch_window_length', 'epoch_step_size'] and isinstance(metadata_spec[field], str):
                      try:
                           processed_metadata[field] = pd.Timedelta(metadata_spec[field])
                      except ValueError:
                           logger.warning(f"Invalid timedelta format '{metadata_spec[field]}' for field '{field}'. Skipping update.")
                 else:
                      processed_metadata[field] = metadata_spec[field]

        # Use the metadata handler to update
        handler = feature.handler or self.metadata_handler
        handler.update_metadata(feature.metadata, **processed_metadata) # Assumes handler works with FeatureMetadata


    def set_index_config(self, index_fields: List[str]) -> None:
        """Configure the multi-index fields for combined *time-series* exports."""
        valid_fields = {f.name for f in fields(TimeSeriesMetadata)}
        invalid = [f for f in index_fields if f not in valid_fields]
        if invalid:
            raise ValueError(f"Invalid index_config fields (must be TimeSeriesMetadata attributes): {invalid}")
        self.metadata.index_config = index_fields
        logger.info(f"Set time-series index_config to: {index_fields}")

    def set_feature_index_config(self, index_fields: List[str]) -> None:
        """Configure the multi-index fields for combined *feature* matrix exports."""
        # Fields can come from FeatureMetadata or propagated TimeSeriesMetadata fields
        valid_feature_fields = {f.name for f in fields(FeatureMetadata)}
        # Allow fields potentially propagated from TimeSeriesMetadata as well
        # This check is less strict here; the combine_features step validates access
        # valid_ts_fields = {f.name for f in fields(TimeSeriesMetadata)}
        # all_possible_fields = valid_feature_fields.union(valid_ts_fields)
        # invalid = [f for f in index_fields if f not in all_possible_fields]
        # if invalid:
        #     raise ValueError(f"Invalid feature_index_config fields: {invalid}")
        self.metadata.feature_index_config = index_fields
        logger.info(f"Set feature_index_config to: {index_fields}")


    def _validate_timestamp_index(self, signal: TimeSeriesSignal) -> None:
        """Validate that a TimeSeriesSignal has a proper DatetimeIndex."""
        # logger = logging.getLogger(__name__) # Logger already defined at module level
        if not isinstance(signal, TimeSeriesSignal):
             # This check might be redundant if called from add_time_series_signal, but safe
             logger.warning(f"Attempted to validate timestamp index on non-TimeSeriesSignal: {signal.metadata.name}")
             return
        try:
            data = signal.get_data()
            if data is None:
                 logger.warning(f"Signal {signal.metadata.name} has None data. Cannot validate index.")
                 return # Allow None data
            if isinstance(data, pd.DataFrame):
                if not isinstance(data.index, pd.DatetimeIndex):
                    logger.error(f"Signal {signal.metadata.name} (ID: {signal.metadata.signal_id}) doesn't have a DatetimeIndex.")
                    raise ValueError(f"TimeSeriesSignal '{signal.metadata.name}' must have a DatetimeIndex.")
            # else: Allow non-DataFrame data? Current signals are DataFrame based.
        except Exception as e:
             logger.error(f"Error validating timestamp index for signal {signal.metadata.name}: {e}", exc_info=True)
             raise ValueError(f"Failed to validate timestamp index for signal '{signal.metadata.name}'.") from e

    def get_target_sample_rate(self, user_specified=None):
        """Determine the target sample rate for time-series alignment."""
        if user_specified is not None:
            return float(user_specified) # Ensure float

        # Calculate max rate only from TimeSeriesSignals
        valid_rates = [
            s.get_sampling_rate() for s in self.time_series_signals.values() # Use time_series_signals
            if s.get_sampling_rate() is not None and s.get_sampling_rate() > 0
        ]

        if not valid_rates:
             logger.warning("No valid positive sampling rates found in TimeSeriesSignals. Defaulting target rate to 100 Hz.")
             return 100.0

        max_rate = max(valid_rates)

        if max_rate <= 0: # Should not happen with filter > 0
             logger.warning("Max sampling rate is not positive. Defaulting target rate to 100 Hz.")
             return 100.0

        # Find the largest standard rate <= the maximum rate found
        valid_standard_rates = [r for r in STANDARD_RATES if r <= max_rate]
        chosen_rate = max(valid_standard_rates) if valid_standard_rates else min(STANDARD_RATES)
        logger.info(f"Determined target sample rate: {chosen_rate} Hz (based on max TimeSeriesSignal rate {max_rate:.4f} Hz)")
        return chosen_rate


    def get_nearest_standard_rate(self, rate):
        """Find the nearest standard rate to a given sample rate."""
        # logger = logging.getLogger(__name__) # Logger already defined
        if rate is None or rate <= 0:
            logger.warning(f"Invalid rate ({rate}) provided to get_nearest_standard_rate. Returning default rate 1 Hz.")
            return 1.0

        nearest_rate = min(STANDARD_RATES, key=lambda r: abs(r - rate))
        logger.debug(f"Nearest standard rate to {rate:.4f} Hz is {nearest_rate} Hz.")
        return nearest_rate

    def get_reference_time(self, target_period: pd.Timedelta) -> pd.Timestamp:
        """Compute the reference timestamp for grid alignment based on TimeSeriesSignals."""
        # logger = logging.getLogger(__name__) # Logger already defined

        min_times = []
        # Iterate only over time_series_signals
        for signal in self.time_series_signals.values():
             try:
                 data = signal.get_data()
                 if data is not None and isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                     min_times.append(data.index.min())
             except Exception as e:
                  logger.warning(f"Could not get start time for signal {signal.metadata.name}: {e}")

        if not min_times:
            logger.warning("No valid timestamps found in TimeSeriesSignals. Using default reference time 1970-01-01 UTC.")
            return pd.Timestamp("1970-01-01", tz='UTC')

        min_time = min(min_times)
        logger.debug(f"Earliest timestamp found across TimeSeriesSignals: {min_time}")

        # Ensure reference time is timezone-aware if min_time is
        epoch = pd.Timestamp("1970-01-01", tz=min_time.tz)

        delta_ns = (min_time - epoch).total_seconds() * 1e9
        target_period_ns = target_period.total_seconds() * 1e9

        if target_period_ns == 0:
             logger.error("Target period is zero, cannot calculate reference time.")
             raise ValueError("Target period cannot be zero for reference time calculation.")

        num_periods = np.floor(delta_ns / target_period_ns)
        ref_time = epoch + pd.Timedelta(nanoseconds=num_periods * target_period_ns)
        logger.debug(f"Calculated reference time: {ref_time} based on target period {target_period}")
        return ref_time


    def _calculate_grid_index(self, target_rate: float, ref_time: pd.Timestamp) -> Optional[pd.DatetimeIndex]:
        """Calculates the final DatetimeIndex grid based on TimeSeriesSignals."""
        # logger = logging.getLogger(__name__) # Logger already defined

        if target_rate <= 0:
            logger.error(f"Invalid target_rate ({target_rate}) for grid calculation.")
            return None

        target_period = pd.Timedelta(seconds=1 / target_rate)

        min_times = []
        max_times = []
        # Iterate only over time_series_signals
        for signal in self.time_series_signals.values():
            try:
                data = signal.get_data()
                if data is not None and isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                    # Ensure timezone consistency with ref_time before comparison
                    data_index_tz = data.index.tz_convert(ref_time.tz) if data.index.tz is not None else data.index.tz_localize(ref_time.tz)
                    min_times.append(data_index_tz.min())
                    max_times.append(data_index_tz.max())
            except Exception as e:
                 logger.warning(f"Could not get time range for signal {signal.metadata.name}: {e}")


        if not min_times or not max_times:
            logger.warning("No valid timestamps found in TimeSeriesSignals. Cannot create grid index.")
            return None

        earliest_start = min(min_times)
        latest_end = max(max_times)
        logger.info(f"Overall time range for grid (from TimeSeriesSignals): {earliest_start} to {latest_end}")

        target_period_ns = target_period.total_seconds() * 1e9
        if target_period_ns == 0:
             logger.error("Target period is zero, cannot calculate grid index.")
             return None

        start_offset_ns = (earliest_start - ref_time).total_seconds() * 1e9
        end_offset_ns = (latest_end - ref_time).total_seconds() * 1e9

        periods_to_start = np.floor(start_offset_ns / target_period_ns)
        periods_to_end = np.ceil(end_offset_ns / target_period_ns)

        grid_start = ref_time + pd.Timedelta(nanoseconds=periods_to_start * target_period_ns)
        grid_end = ref_time + pd.Timedelta(nanoseconds=periods_to_end * target_period_ns)

        if grid_start > grid_end:
            logger.warning(f"Calculated grid_start ({grid_start}) is after grid_end ({grid_end}). Attempting swap.")
            if abs((grid_start - grid_end).total_seconds()) < (target_period.total_seconds() * 0.5):
                 grid_start, grid_end = grid_end, grid_start
            else:
                 logger.error("Invalid grid range calculated.")
                 return None

        try:
            grid_index = pd.date_range(
                start=grid_start, end=grid_end, freq=target_period, name='timestamp'
            )
            grid_index = grid_index.tz_convert(ref_time.tz) if grid_index.tz is not None else grid_index.tz_localize(ref_time.tz)
            logger.info(f"Calculated grid_index with {len(grid_index)} points from {grid_index.min()} to {grid_index.max()}")
            return grid_index
        except Exception as e:
            logger.error(f"Error creating date_range for grid index: {e}", exc_info=True)
            return None

    # Decorator now just marks the method
    @register_collection_operation("generate_alignment_grid")
    def generate_alignment_grid(self, target_sample_rate: Optional[float] = None) -> 'SignalCollection':
        """Calculates and stores the alignment grid parameters based on TimeSeriesSignals."""
        logger.info(f"Starting alignment grid parameter calculation with target_sample_rate={target_sample_rate}")
        start_time = time.time()
        self._alignment_params_calculated = False # Reset flag

        # --- Filter for TimeSeriesSignals ---
        # Check the dedicated dictionary
        if not self.time_series_signals:
            logger.error("No time-series signals found in the collection. Cannot calculate alignment grid.")
            raise RuntimeError("No time-series signals found in the collection to calculate alignment grid.")

        # --- Determine Target Rate (uses time_series_signals internally now) ---
        try:
            self.target_rate = self.get_target_sample_rate(target_sample_rate)
            if self.target_rate is None or self.target_rate <= 0:
                 raise ValueError(f"Calculated invalid target rate: {self.target_rate}")
            target_period = pd.Timedelta(seconds=1 / self.target_rate)
            logger.info(f"Using target rate: {self.target_rate} Hz (Period: {target_period})")
        except Exception as e:
            logger.error(f"Failed to determine target sample rate: {e}", exc_info=True)
            raise RuntimeError(f"Failed to determine target sample rate: {e}") from e

        # --- Determine Reference Time (uses time_series_signals internally now) ---
        try:
            self.ref_time = self.get_reference_time(target_period)
            logger.info(f"Using reference time: {self.ref_time}")
        except Exception as e:
            logger.error(f"Failed to determine reference time: {e}", exc_info=True)
            raise RuntimeError(f"Failed to determine reference time: {e}") from e

        # --- Calculate Grid Index (uses time_series_signals internally now) ---
        try:
            self.grid_index = self._calculate_grid_index(self.target_rate, self.ref_time)
            if self.grid_index is None or self.grid_index.empty:
                raise ValueError("Calculated grid index is None or empty.")
        except Exception as e:
            logger.error(f"Failed to calculate grid index: {e}", exc_info=True)
            self.grid_index = None
            raise RuntimeError(f"Failed to calculate a valid grid index for alignment: {e}") from e

        self._alignment_params_calculated = True
        logger.info(f"Alignment grid parameters calculated in {time.time() - start_time:.2f} seconds.")
        return self

    # --- New Epoch Grid Generation ---
    @register_collection_operation("generate_epoch_grid")
    def generate_epoch_grid(self, start_time: Optional[Union[str, pd.Timestamp]] = None, end_time: Optional[Union[str, pd.Timestamp]] = None) -> 'SignalCollection':
        """
        Calculates and stores the global epoch grid based on collection settings.

        Uses `epoch_grid_config` from `CollectionMetadata` and the time range
        of `time_series_signals` to create a common `epoch_grid_index`.

        Args:
            start_time: Optional override for the grid start time.
            end_time: Optional override for the grid end time.

        Returns:
            The SignalCollection instance (self) with epoch grid parameters set.

        Raises:
            RuntimeError: If `epoch_grid_config` is missing or invalid, or if
                          no time-series signals are found to determine the range.
            ValueError: If start/end time overrides are invalid.
        """
        logger.info("Starting global epoch grid calculation...")
        op_start_time = time.time()
        self._epoch_grid_calculated = False # Reset flag

        # --- Get Config ---
        config = self.metadata.epoch_grid_config
        if not config or "window_length" not in config or "step_size" not in config:
            raise RuntimeError("Missing or incomplete 'epoch_grid_config' in collection metadata. Cannot generate epoch grid.")

        try:
            window_length = pd.Timedelta(config["window_length"])
            step_size = pd.Timedelta(config["step_size"])
            if window_length <= pd.Timedelta(0) or step_size <= pd.Timedelta(0):
                raise ValueError("window_length and step_size must be positive.")
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid epoch_grid_config parameters: {e}") from e

        self.global_epoch_window_length = window_length
        self.global_epoch_step_size = step_size
        logger.info(f"Using global epoch parameters: window={window_length}, step={step_size}")

        # --- Determine Time Range ---
        min_times = []
        max_times = []
        collection_tz = pd.Timestamp('now', tz=self.metadata.timezone).tz # Get collection tz object

        for signal in self.time_series_signals.values():
            try:
                data = signal.get_data()
                if data is not None and isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                    # Ensure timezone consistency before comparison
                    data_index_tz = data.index.tz_convert(collection_tz) if data.index.tz is not None else data.index.tz_localize(collection_tz)
                    min_times.append(data_index_tz.min())
                    max_times.append(data_index_tz.max())
            except Exception as e:
                 logger.warning(f"Could not get time range for signal {signal.metadata.name} for epoch grid: {e}")

        if not min_times or not max_times:
            raise RuntimeError("No valid time ranges found in TimeSeriesSignals. Cannot determine epoch grid range.")

        # --- Apply Overrides ---
        try:
            grid_start = pd.Timestamp(start_time, tz=collection_tz) if start_time else min(min_times)
            grid_end = pd.Timestamp(end_time, tz=collection_tz) if end_time else max(max_times)
            # Ensure overrides are timezone-aware consistent with collection
            if grid_start.tz is None: grid_start = grid_start.tz_localize(collection_tz)
            else: grid_start = grid_start.tz_convert(collection_tz)
            if grid_end.tz is None: grid_end = grid_end.tz_localize(collection_tz)
            else: grid_end = grid_end.tz_convert(collection_tz)

        except Exception as e:
            raise ValueError(f"Invalid start_time or end_time override for epoch grid: {e}") from e

        if grid_start >= grid_end:
             raise ValueError(f"Epoch grid start time ({grid_start}) must be before end time ({grid_end}).")

        logger.info(f"Epoch grid time range: {grid_start} to {grid_end}")

        # --- Generate Epoch Index ---
        try:
            # Generate epoch start times using the step_size as frequency
            self.epoch_grid_index = pd.date_range(
                start=grid_start,
                end=grid_end, # date_range includes end if it falls on frequency step
                freq=step_size,
                name='epoch_start_time',
                inclusive='left' # Only include start times <= grid_end
            )
            # Filter out any start times where the window would begin after the grid ends
            # This check might be slightly redundant with inclusive='left' but safer
            self.epoch_grid_index = self.epoch_grid_index[self.epoch_grid_index <= grid_end]

            if self.epoch_grid_index.empty:
                 logger.warning("Generated epoch grid index is empty.")
                 # Keep empty index, subsequent steps should handle this

            # Ensure timezone matches collection
            self.epoch_grid_index = self.epoch_grid_index.tz_convert(collection_tz) if self.epoch_grid_index.tz is not None else self.epoch_grid_index.tz_localize(collection_tz)

            logger.info(f"Calculated epoch_grid_index with {len(self.epoch_grid_index)} points.")

        except Exception as e:
            logger.error(f"Error creating date_range for epoch grid index: {e}", exc_info=True)
            self.epoch_grid_index = None # Mark as failed
            raise RuntimeError(f"Failed to calculate epoch grid index: {e}") from e

        self._epoch_grid_calculated = True
        logger.info(f"Epoch grid calculated in {time.time() - op_start_time:.2f} seconds.")
        return self

    def apply_multi_signal_operation(self, operation_name: str, input_signal_keys: List[str], parameters: Dict[str, Any]) -> Union[TimeSeriesSignal, Feature]:
        """
        Applies an operation that takes multiple signals as input and produces a single output.

        Handles operations registered in `multi_signal_registry`.

        Args:
            operation_name: Name of the operation (e.g., "feature_statistics").
            input_signal_keys: List of keys for the input TimeSeriesSignals.
            parameters: Dictionary of parameters for the operation.

        Returns:
            The resulting Feature or TimeSeriesSignal object.

        Raises:
            ValueError: If operation is not found, inputs are invalid, or prerequisites are not met.
            RuntimeError: If the operation execution fails.
        """
        logger.info(f"Applying multi-signal operation '{operation_name}' to inputs: {input_signal_keys}")

        if operation_name not in self.multi_signal_registry:
            raise ValueError(f"Multi-signal operation '{operation_name}' not found in registry.")

        operation_func, output_class = self.multi_signal_registry[operation_name]

        # --- Input Resolution and Validation ---
        input_signals: List[TimeSeriesSignal] = []
        for key in input_signal_keys:
            try:
                signal = self.get_time_series_signal(key) # Ensure inputs are TimeSeriesSignals
                input_signals.append(signal)
            except KeyError:
                raise ValueError(f"Input TimeSeriesSignal key '{key}' not found for operation '{operation_name}'.")

        if not input_signals:
            raise ValueError(f"No valid input TimeSeriesSignals resolved for operation '{operation_name}'.")

        # --- Prerequisite Checks (Specific to Feature Extraction) ---
        # Import Feature here to avoid circular dependency at module level if needed
        from ..features.feature import Feature
        is_feature_op = issubclass(output_class, Feature) # Check if output is a Feature
        if is_feature_op:
            if not self._epoch_grid_calculated or self.epoch_grid_index is None or self.epoch_grid_index.empty:
                raise RuntimeError(f"Cannot execute feature operation '{operation_name}': generate_epoch_grid must be run successfully first.")
            # REMOVE the lines that add global params to the 'parameters' dict
            # parameters['epoch_grid_index'] = self.epoch_grid_index # Keep this if needed, but it's passed separately below
            # parameters['global_epoch_window_length'] = self.global_epoch_window_length # REMOVE
            # parameters['global_epoch_step_size'] = self.global_epoch_step_size         # REMOVE

        # --- Function Execution ---
        try:
            logger.debug(f"Executing operation function '{operation_func.__name__}'...")
            if is_feature_op:
                # Make a copy to avoid modifying the original dict
                params_copy = parameters.copy()
                # Remove epoch_grid_index from parameters dict as it's passed separately
                # Also remove window_length/step_size if they were accidentally left in params
                params_copy.pop('epoch_grid_index', None)
                params_copy.pop('global_epoch_window_length', None)
                params_copy.pop('global_epoch_step_size', None)

                # Call feature function with explicit global args
                result_object = operation_func(
                    signals=input_signals,
                    epoch_grid_index=self.epoch_grid_index, # Pass grid index separately
                    parameters=params_copy,                 # Pass remaining specific params
                    global_window_length=self.global_epoch_window_length, # Pass global window explicitly
                    global_step_size=self.global_epoch_step_size          # Pass global step explicitly
                )
            else:
                 # For non-feature ops, pass parameters as before
                 result_object = operation_func(signals=input_signals, **parameters)

            logger.debug(f"Operation function '{operation_func.__name__}' completed.")
        except Exception as e:
            logger.error(f"Error executing multi-signal operation function '{operation_func.__name__}': {e}", exc_info=True)
            # Add context about which operation failed
            raise RuntimeError(f"Execution of operation '{operation_name}' failed.") from e # Keep original error context

        # --- Result Validation ---
        if not isinstance(result_object, output_class):
            raise TypeError(f"Operation '{operation_name}' returned unexpected type {type(result_object).__name__}. Expected {output_class.__name__}.")

        # --- Metadata Propagation (for Feature outputs) ---
        if isinstance(result_object, Feature):
            logger.debug(f"Propagating metadata for Feature result of '{operation_name}'...")
            feature_meta = result_object.metadata
            fields_to_propagate = self.metadata.feature_index_config # Fields defined in collection config

            if fields_to_propagate:
                if len(input_signals) == 1:
                    # Single input: Copy directly
                    source_meta = input_signals[0].metadata
                    for field in fields_to_propagate:
                        if hasattr(source_meta, field) and hasattr(feature_meta, field):
                            value = getattr(source_meta, field)
                            setattr(feature_meta, field, value)
                            logger.debug(f"  Propagated '{field}' = {value} (from single source)")
                        elif hasattr(feature_meta, field):
                             # Field exists in FeatureMetadata but not source TimeSeriesMetadata
                             logger.debug(f"  Field '{field}' exists in FeatureMetadata but not in source TimeSeriesMetadata. Skipping propagation.")
                        # else: Field doesn't exist in FeatureMetadata, ignore.

                elif len(input_signals) > 1:
                    # Multiple inputs: Check for common values
                    for field in fields_to_propagate:
                        if hasattr(feature_meta, field): # Only propagate if field exists in FeatureMetadata
                            values = set()
                            all_sources_have_field = True
                            for source_signal in input_signals:
                                if hasattr(source_signal.metadata, field):
                                    values.add(getattr(source_signal.metadata, field))
                                else:
                                     all_sources_have_field = False
                                     logger.debug(f"  Source signal '{source_signal.metadata.name}' missing field '{field}' for propagation.")
                                     break # If one source doesn't have it, we can't determine commonality

                            if not all_sources_have_field:
                                 logger.debug(f"  Field '{field}' not present in all source TimeSeriesSignals. Setting to None.")
                                 setattr(feature_meta, field, None) # Or handle as needed
                            elif len(values) == 1:
                                common_value = values.pop()
                                setattr(feature_meta, field, common_value)
                                logger.debug(f"  Propagated '{field}' = {common_value} (common value)")
                            else:
                                # Different values found
                                setattr(feature_meta, field, "mixed") # Use "mixed" string indicator
                                logger.debug(f"  Propagated '{field}' = 'mixed' (values differ: {values})")
                        # else: Field doesn't exist in FeatureMetadata, ignore.
            else:
                 logger.debug("No feature_index_config set. Skipping metadata propagation.")

            # Ensure source signal IDs and keys are set (should be done by feature function, but double-check)
            if not feature_meta.source_signal_ids:
                 feature_meta.source_signal_ids = [s.metadata.signal_id for s in input_signals]
            if not feature_meta.source_signal_keys:
                 feature_meta.source_signal_keys = [s.metadata.name for s in input_signals] # Use name as key

        logger.info(f"Successfully applied multi-signal operation '{operation_name}'. Result type: {type(result_object).__name__}")
        return result_object


    # --- Collection Operation Dispatch ---

    def apply_operation(self, operation_name: str, **parameters: Any) -> Any: # type: ignore
        """
        Applies a registered collection-level operation by name.

        Looks up the operation in the `collection_operation_registry` and executes
        the corresponding method on this instance, passing the provided parameters.

        Args:
            operation_name: The name of the operation to execute (must be registered).
            **parameters: Keyword arguments to pass to the registered operation method.

        Returns:
            The result returned by the executed operation method (often `self` or `None`).

        Raises:
            ValueError: If the operation_name is not found in the registry.
            Exception: If the underlying operation method raises an exception.
        """
        logger.info(f"Applying collection operation '{operation_name}' with parameters: {parameters}")
        if operation_name not in self.collection_operation_registry:
            logger.error(f"Collection operation '{operation_name}' not found in registry.")
            raise ValueError(f"Collection operation '{operation_name}' not found.")

        operation_method = self.collection_operation_registry[operation_name]

        try:
            # Call the registered method, passing 'self' as the first argument.
            result = operation_method(self, **parameters)
            logger.info(f"Successfully applied collection operation '{operation_name}'.")
            return result
        except Exception as e:
            logger.error(f"Error executing collection operation '{operation_name}': {e}", exc_info=True)
            # Re-raise the exception to be handled by the caller (e.g., WorkflowExecutor)
            raise

    # --- End Collection Operation Dispatch ---


    def get_signals_from_input_spec(self, input_spec: Union[str, Dict[str, Any], List[str], None] = None) -> List[Union[TimeSeriesSignal, Feature]]:
        """
        Get signals/features based on an input specification. Alias for get_signals.
        """
        # Just call the main get_signals method
        return self.get_signals(input_spec=input_spec)

    @register_collection_operation("apply_grid_alignment")
    def apply_grid_alignment(self, method: str = 'nearest', signals_to_align: Optional[List[str]] = None):
        """Applies grid alignment to specified TimeSeriesSignals in place."""
        if not self._alignment_params_calculated or self.grid_index is None or self.grid_index.empty:
            logger.error("Cannot apply grid alignment: generate_alignment_grid must be run successfully first.")
            raise RuntimeError("generate_alignment_grid must be run successfully before applying grid alignment.")

        allowed_methods = ['nearest', 'pad', 'ffill', 'backfill', 'bfill']
        if method not in allowed_methods:
             logger.warning(f"Alignment method '{method}' not in allowed list {allowed_methods}. Using 'nearest'.")
             method = 'nearest'

        logger.info(f"Applying grid alignment in-place to TimeSeriesSignals using method '{method}'...")
        start_time = time.time()
        # Determine target keys: specified list or all time_series_signals
        target_keys = signals_to_align if signals_to_align is not None else list(self.time_series_signals.keys())

        processed_count = 0
        skipped_count = 0
        error_signals = []

        for key in target_keys:
            try:
                # Use get_time_series_signal to ensure correct type and existence
                signal = self.get_time_series_signal(key)

                current_data = signal.get_data()
                if current_data is None or current_data.empty:
                    logger.warning(f"Skipping alignment for TimeSeriesSignal '{key}': data is None or empty.")
                    skipped_count += 1
                    continue

                logger.debug(f"Calling apply_operation('reindex_to_grid') for TimeSeriesSignal '{key}'...")
                signal.apply_operation(
                    'reindex_to_grid',
                    inplace=True,
                    grid_index=self.grid_index,
                    method=method
                )
                logger.debug(f"Successfully applied 'reindex_to_grid' operation to TimeSeriesSignal '{key}'.")
                processed_count += 1
            except KeyError:
                 logger.warning(f"TimeSeriesSignal key '{key}' specified for alignment not found.")
                 skipped_count += 1
            except Exception as e:
                logger.error(f"Failed to apply 'reindex_to_grid' operation to TimeSeriesSignal '{key}': {e}", exc_info=True)
                warnings.warn(f"Failed to apply grid alignment to TimeSeriesSignal '{key}': {e}")
                error_signals.append(key)

        logger.info(f"Grid alignment application finished in {time.time() - start_time:.2f} seconds. "
                    f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {len(error_signals)}")

        if error_signals:
            raise RuntimeError(f"Failed to apply grid alignment to the following TimeSeriesSignals: {', '.join(error_signals)}")

    @register_collection_operation("align_and_combine_signals")
    def align_and_combine_signals(self) -> None:
        """Aligns TimeSeriesSignals using merge_asof and combines them."""
        if not self._alignment_params_calculated or self.grid_index is None or self.grid_index.empty:
            logger.error("Cannot align and combine signals: generate_alignment_grid must be run successfully first.")
            raise RuntimeError("generate_alignment_grid must be run successfully before aligning and combining signals.")

        logger.info("Aligning and combining TimeSeriesSignals using merge_asof...")
        start_time = time.time()

        target_period = pd.Timedelta(seconds=1 / self.target_rate) if self.target_rate else None
        if target_period is None or target_period.total_seconds() <= 0:
            logger.warning("Grid index frequency is missing or invalid. Using default merge tolerance (1ms).")
            tolerance = pd.Timedelta(milliseconds=1)
        else:
            tolerance_ns = target_period.total_seconds() * 1e9 / 2
            tolerance = pd.Timedelta(nanoseconds=tolerance_ns + 1)
        self._merge_tolerance = tolerance
        logger.debug(f"Using merge_asof tolerance: {self._merge_tolerance}")

        target_df = pd.DataFrame({'timestamp': self.grid_index})
        aligned_signal_dfs = {}
        error_signals = []

        # Iterate only over time_series_signals
        for key, signal in self.time_series_signals.items():
            if signal.metadata.temporary:
                logger.debug(f"Skipping temporary TimeSeriesSignal '{key}' for combined export.")
                continue

            try:
                signal_df = signal.get_data()
                if signal_df is None or not isinstance(signal_df, pd.DataFrame) or signal_df.empty:
                    logger.warning(f"TimeSeriesSignal '{key}' has no valid data, skipping merge_asof.")
                    continue
                if not isinstance(signal_df.index, pd.DatetimeIndex):
                     logger.warning(f"TimeSeriesSignal '{key}' data does not have a DatetimeIndex, skipping merge_asof.")
                     continue

                source_df = signal_df.reset_index()
                if source_df.columns[0] != 'timestamp':
                     source_df = source_df.rename(columns={source_df.columns[0]: 'timestamp'})

                if source_df['timestamp'].dt.tz is None:
                     source_df['timestamp'] = source_df['timestamp'].dt.tz_localize(self.grid_index.tz)
                else:
                     source_df['timestamp'] = source_df['timestamp'].dt.tz_convert(self.grid_index.tz)

                source_df = source_df.sort_values('timestamp')

                logger.debug(f"Aligning TimeSeriesSignal '{key}' using merge_asof...")
                aligned_df = pd.merge_asof(
                    target_df, source_df, on='timestamp', direction='nearest', tolerance=tolerance
                )

                aligned_df = aligned_df.set_index('timestamp')
                original_cols = [col for col in signal_df.columns if col in aligned_df.columns]

                if aligned_df[original_cols].isnull().all().all():
                    logger.warning(f"TimeSeriesSignal '{key}' resulted in all NaN values after merge_asof alignment. Skipping.")
                else:
                    aligned_signal_dfs[key] = aligned_df[original_cols]
                    logger.debug(f"TimeSeriesSignal '{key}' aligned via merge_asof. Shape: {aligned_signal_dfs[key].shape}.")

            except Exception as e:
                logger.error(f"Error processing TimeSeriesSignal '{key}' with merge_asof: {e}", exc_info=True)
                error_signals.append(key)

        if error_signals:
            warnings.warn(f"Failed merge_asof alignment for TimeSeriesSignals: {', '.join(error_signals)}. Combining successful ones.")

        if not aligned_signal_dfs:
            logger.warning("No TimeSeriesSignals were successfully aligned using merge_asof. Storing empty DataFrame.")
            self._aligned_dataframe = pd.DataFrame(index=self.grid_index)
            self._aligned_dataframe_params = self._get_current_alignment_params("merge_asof")
            return

        combined_df = self._perform_concatenation(aligned_signal_dfs, self.grid_index, is_feature=False) # Specify not feature

        self._aligned_dataframe = combined_df
        self._aligned_dataframe_params = self._get_current_alignment_params("merge_asof")

        logger.info(f"Successfully aligned and combined {len(aligned_signal_dfs)} TimeSeriesSignals using merge_asof "
                    f"in {time.time() - start_time:.2f} seconds. Stored shape: {combined_df.shape}")


    def _perform_concatenation(self, aligned_dfs: Dict[str, pd.DataFrame], grid_index: pd.DatetimeIndex, is_feature: bool) -> pd.DataFrame:
        """Internal helper to concatenate aligned dataframes, handling MultiIndex based on type."""
        if not aligned_dfs:
            return pd.DataFrame(index=grid_index)

        # Determine which index config and metadata source to use
        index_config = self.metadata.feature_index_config if is_feature else self.metadata.index_config
        source_dict = self.features if is_feature else self.time_series_signals
        metadata_attr = 'metadata' # Both Feature and TimeSeriesSignal have .metadata

        combined_df: pd.DataFrame

        if is_feature:
            # --- Feature Concatenation Logic ---
            logger.info("Using simplified concatenation for features (pd.concat with keys).")
            # aligned_dfs keys are feature set names ("hr_features", "accel_mag_features")
            # Values are DataFrames from feat.get_data(), which already have MultiIndex columns (signal_key, feature)
            try:
                # Concatenate using the feature set keys as the top level
                combined_df = pd.concat(aligned_dfs, axis=1)
                # Ensure the index matches the grid index exactly
                combined_df = combined_df.reindex(grid_index)
                # Name the levels appropriately (top level is the feature set key)
                if isinstance(combined_df.columns, pd.MultiIndex):
                     # Check if existing levels have names, preserve if possible, otherwise assign default
                     current_names = list(combined_df.columns.names)
                     # Expected structure: [None, 'signal_key', 'feature'] if source had 2 levels
                     # Or just [None, 'feature'] if source had 1 level (e.g., sleep_stage_mode)
                     # We want: ['feature_set', 'signal_key', 'feature'] or ['feature_set', 'feature']

                     # Determine expected number of levels from the first non-empty df
                     expected_levels = 0
                     for df_val in aligned_dfs.values():
                          if not df_val.empty and isinstance(df_val.columns, pd.MultiIndex):
                               expected_levels = df_val.columns.nlevels
                               break
                          elif not df_val.empty: # Simple index
                               expected_levels = 1
                               break

                     if expected_levels == 2: # e.g., from feature_statistics
                          combined_df.columns.names = ['feature_set'] + ['signal_key', 'feature']
                     elif expected_levels == 1: # e.g., from sleep_stage_mode
                          combined_df.columns.names = ['feature_set'] + ['feature']
                     else: # Fallback or empty case
                          # Assign names based on actual number of levels created by concat
                          num_levels = combined_df.columns.nlevels
                          if num_levels > 0:
                               new_names = ['feature_set'] + [f'level_{i+1}' for i in range(num_levels - 1)]
                               combined_df.columns.names = new_names[:num_levels] # Ensure correct length

                     logger.debug(f"Feature concatenation resulted in MultiIndex columns with names: {combined_df.columns.names}")

            except Exception as e:
                logger.error(f"Error during feature concatenation using pd.concat: {e}", exc_info=True)
                # Fallback to empty dataframe with grid index
                combined_df = pd.DataFrame(index=grid_index)

        else:
            # --- Time-Series Concatenation Logic (Existing) ---
            index_config = self.metadata.index_config
            source_dict = self.time_series_signals
            metadata_attr = 'metadata'

            if index_config:
                logger.info("Using MultiIndex for combined time-series columns.")
                multi_index_tuples = []
                final_columns_data = {}

                for key, signal_aligned_df in aligned_dfs.items():
                    signal_obj = source_dict.get(key)
                    if not signal_obj:
                         logger.warning(f"Could not find original object for key '{key}' during concatenation. Skipping.")
                         continue
                    metadata_obj = getattr(signal_obj, metadata_attr)

                    for col_name in signal_aligned_df.columns: # col_name is simple string here
                        metadata_values = []
                        for field in index_config:
                            value = getattr(metadata_obj, field, None)
                            value = key if value is None and field == 'name' else value
                            value = "N/A" if value is None else value
                            value = value.name if isinstance(value, Enum) else str(value)
                            metadata_values.append(value)
                        metadata_values.append(col_name)
                        tuple_key = tuple(metadata_values)
                        multi_index_tuples.append(tuple_key)
                        final_columns_data[tuple_key] = signal_aligned_df[col_name]

                if final_columns_data:
                    level_names = index_config + ['column']
                    multi_idx = pd.MultiIndex.from_tuples(multi_index_tuples, names=level_names)
                    combined_df = pd.DataFrame(final_columns_data, index=grid_index)
                    if not combined_df.empty:
                         combined_df.columns = multi_idx
                    else:
                         combined_df = pd.DataFrame(index=grid_index, columns=multi_idx)
                    logger.debug(f"Applied MultiIndex. Final level names: {combined_df.columns.names}")
                else:
                    logger.warning("No data available to create MultiIndex columns.")
                    level_names = index_config + ['column']
                    empty_multi_idx = pd.MultiIndex.from_tuples([], names=level_names)
                    combined_df = pd.DataFrame(index=grid_index, columns=empty_multi_idx)
            else:
                logger.info("Using simple column names (key_colname) for combined time-series dataframe.")
                simple_concat_list = []
                for key, signal_aligned_df in aligned_dfs.items():
                     prefix = key
                     if len(signal_aligned_df.columns) == 1:
                          renamed_df = signal_aligned_df.rename(columns={signal_aligned_df.columns[0]: prefix})
                          simple_concat_list.append(renamed_df)
                     else:
                          prefixed_df = signal_aligned_df.add_prefix(f"{prefix}_")
                          simple_concat_list.append(prefixed_df)

                if not simple_concat_list:
                     logger.warning("No data available for simple column concatenation.")
                     combined_df = pd.DataFrame(index=grid_index)
                else:
                     combined_df = pd.concat(simple_concat_list, axis=1)
                     combined_df = combined_df.reindex(grid_index)

        # Final Cleanup (applies to both feature and time-series)
        if not combined_df.empty:
            initial_rows = len(combined_df)
            combined_df = combined_df.dropna(axis=0, how='all')
            final_rows = len(combined_df)
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} rows with all NaN values during concatenation.")
        else:
            logger.info("Combined dataframe was empty before NaN removal during concatenation.")

        return combined_df

    def _get_current_alignment_params(self, method_used: str) -> Dict[str, Any]:
        """Helper to gather current alignment parameters for storage."""
        return {
            "method_used": method_used,
            "target_rate": self.target_rate,
            "ref_time": self.ref_time,
            "grid_index_size": len(self.grid_index) if self.grid_index is not None else 0,
            "grid_start": self.grid_index.min() if self.grid_index is not None and not self.grid_index.empty else None,
            "grid_end": self.grid_index.max() if self.grid_index is not None and not self.grid_index.empty else None,
            "tolerance": self._merge_tolerance # Include tolerance used (relevant for merge_asof)
        }

    # --- Stored Combined Dataframe Access ---

    def get_stored_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Returns the internally stored combined *time-series* dataframe."""
        if self._aligned_dataframe is None:
             logger.debug("Stored combined time-series dataframe has not been generated yet.")
        return self._aligned_dataframe

    def get_stored_combined_feature_matrix(self) -> Optional[pd.DataFrame]:
         """Returns the internally stored combined *feature* matrix."""
         if self._combined_feature_matrix is None:
              logger.debug("Stored combined feature matrix has not been generated yet.")
         return self._combined_feature_matrix


    def get_stored_combination_params(self) -> Optional[Dict[str, Any]]:
         """Returns the parameters used to generate the stored combined time-series dataframe."""
         if self._aligned_dataframe_params is None:
              logger.debug("Stored combined time-series dataframe parameters are not available.")
         return self._aligned_dataframe_params

    @register_collection_operation("combine_aligned_signals")
    def combine_aligned_signals(self) -> None:
        """Combines TimeSeriesSignals modified in-place by apply_grid_alignment."""
        if not self._alignment_params_calculated or self.grid_index is None or self.grid_index.empty:
            logger.error("Cannot combine snapped signals: generate_alignment_grid must be run successfully first.")
            raise RuntimeError("generate_alignment_grid must be run successfully before combining snapped signals.")
        if not self.time_series_signals: # Check specific dict
             logger.warning("No TimeSeriesSignals in collection to combine.")
             self._aligned_dataframe = pd.DataFrame(index=self.grid_index)
             self._aligned_dataframe_params = self._get_current_alignment_params("outer_join_reindex")
             return

        logger.info("Combining in-place snapped TimeSeriesSignals using outer join and reindexing...")
        start_time = time.time()

        snapped_signal_dfs = {}
        error_signals = []
        # Iterate only over time_series_signals
        for key, signal in self.time_series_signals.items():
            if signal.metadata.temporary:
                logger.debug(f"Skipping temporary TimeSeriesSignal '{key}' for combined export.")
                continue

            try:
                signal_df = signal.get_data() # Get the potentially modified data
                if signal_df is None or signal_df.empty:
                    logger.warning(f"TimeSeriesSignal '{key}' has no data after snapping, skipping combination.")
                    continue
                if not isinstance(signal_df.index, pd.DatetimeIndex):
                    logger.error(f"TimeSeriesSignal '{key}' index is not DatetimeIndex after snapping attempt.")
                    error_signals.append(key)
                    continue

                # Use _perform_concatenation's logic implicitly by preparing dict
                snapped_signal_dfs[key] = signal_df
                logger.debug(f"Collected snapped data for TimeSeriesSignal '{key}'. Shape: {signal_df.shape}")

            except Exception as e:
                logger.error(f"Error accessing data for TimeSeriesSignal '{key}': {e}", exc_info=True)
                error_signals.append(key)

        if error_signals:
            raise RuntimeError(f"Failed to combine signals. Errors occurred while accessing data for: {', '.join(error_signals)}")

        if not snapped_signal_dfs:
            logger.warning("No valid snapped TimeSeriesSignals found to combine. Storing empty DataFrame.")
            self._aligned_dataframe = pd.DataFrame(index=self.grid_index)
            self._aligned_dataframe_params = self._get_current_alignment_params("outer_join_reindex")
            return

        # --- Perform Outer Join & Reindex using helper ---
        # Note: _perform_concatenation handles the join logic implicitly when is_feature=False
        # It builds the structure needed for concatenation, which effectively does the join+reindex
        # Let's rename the helper or adjust logic slightly for clarity if needed.
        # For now, assume _perform_concatenation handles this correctly.
        combined_df_final = self._perform_concatenation(snapped_signal_dfs, self.grid_index, is_feature=False)

        # --- Store Result and Parameters ---
        self._aligned_dataframe = combined_df_final
        self._aligned_dataframe_params = self._get_current_alignment_params("outer_join_reindex")

        logger.info(f"Successfully combined {len(snapped_signal_dfs)} snapped TimeSeriesSignals using outer join and reindex "
                    f"in {time.time() - start_time:.2f} seconds. Stored shape: {combined_df_final.shape}")

    @register_collection_operation("combine_features")
    def combine_features(self, inputs: List[str], feature_index_config: Optional[List[str]] = None) -> None: # Made config optional
        """
        Combines multiple Feature objects into a single combined feature matrix.

        Retrieves specified Feature objects, validates their indices against the
        collection's `epoch_grid_index`, and concatenates their data column-wise.
        Constructs a MultiIndex for the columns of the resulting DataFrame based
        on the provided `feature_index_config` (read from collection metadata if
        not provided here) and the metadata of the source Feature objects.
        Stores the result in `self._combined_feature_matrix`.

        Args:
            inputs: List of keys identifying the input Feature objects in `self.features`.
                    Can contain base names, which will be resolved.
            feature_index_config: Optional list of metadata field names to override
                                  the collection's default `feature_index_config`.

        Raises:
            RuntimeError: If the epoch grid hasn't been calculated.
            ValueError: If inputs are missing, invalid, not Feature objects,
                        have indices mismatched with the epoch grid, or if
                        `feature_index_config` is invalid.
            TypeError: If input dataframes cannot be concatenated.
        """
        if not self._epoch_grid_calculated or self.epoch_grid_index is None or self.epoch_grid_index.empty:
            raise RuntimeError("Cannot combine features: generate_epoch_grid must be run successfully first.")
        if not inputs:
            raise ValueError("No input signals specified for combine_features.")

        # Use provided config or fallback to collection's config
        config_to_use = feature_index_config if feature_index_config is not None else self.metadata.feature_index_config
        if not config_to_use:
             logger.warning("No feature_index_config provided or set on collection. Combined feature columns will not have a MultiIndex.")

        logger.info(f"Combining features from inputs: {inputs} using index config: {config_to_use}")
        start_time = time.time()

        # --- Resolve input keys (handle base names) ---
        resolved_keys = []
        for key_spec in inputs:
            if key_spec in self.features: # Check features dict
                resolved_keys.append(key_spec)
            else:
                found_match = False
                for existing_key in self.features.keys(): # Check features dict
                    if existing_key.startswith(f"{key_spec}_") and existing_key[len(key_spec)+1:].isdigit():
                        resolved_keys.append(existing_key)
                        found_match = True
                if not found_match:
                    raise ValueError(f"Input specification '{key_spec}' for combine_features does not match any existing feature key or base name.")

        if not resolved_keys:
             raise ValueError(f"Input specification {inputs} for combine_features resolved to an empty list.")
        logger.debug(f"Resolved combine_features input {inputs} to keys: {resolved_keys}")

        # --- Retrieve and Validate Input Features ---
        input_features: List[Feature] = []
        for key in resolved_keys:
            feature = self.get_feature(key) # Raises KeyError if not found

            feature_data = feature.get_data()
            if not isinstance(feature_data.index, pd.DatetimeIndex):
                raise TypeError(f"Input Feature '{key}' does not have a DatetimeIndex.")

            # --- Strict Index Validation against epoch_grid_index ---
            if not feature_data.index.equals(self.epoch_grid_index):
                logger.error(f"Index mismatch for Feature '{key}'. Expected index matching epoch_grid_index (size {len(self.epoch_grid_index)}), "
                             f"but got index size {len(feature_data.index)}.")
                if len(feature_data.index) == len(self.epoch_grid_index):
                     diff = self.epoch_grid_index.difference(feature_data.index)
                     logger.error(f"Index values differ. Example differences (grid vs feature): {diff[:5]}...")
                raise ValueError(f"Input Feature '{key}' index does not match the collection's epoch_grid_index. Ensure feature generation used the global grid.")

            input_features.append(feature)

        if not input_features:
            logger.warning("No valid Feature objects found to combine.")
            self._combined_feature_matrix = pd.DataFrame(index=self.epoch_grid_index)
            return

        # --- Prepare DataFrames for Concatenation using helper ---
        # Pass is_feature=True to use feature_index_config
        combined_df = self._perform_concatenation(
            {feat.metadata.name: feat.get_data() for feat in input_features},
            self.epoch_grid_index,
            is_feature=True
        )

        # --- Store Result ---
        self._combined_feature_matrix = combined_df
        # Optionally store the config used
        # self._combined_feature_matrix_config = config_to_use

        logger.info(f"Successfully combined {len(input_features)} feature objects "
                    f"in {time.time() - start_time:.2f} seconds. Stored matrix shape: {combined_df.shape}")


    def apply_and_store_operation(self, signal_key: str, operation_name: str,
                                 parameters: Dict[str, Any], output_key: str) -> Union[TimeSeriesSignal, Feature]:
        """
        Apply an operation to a TimeSeriesSignal and store the result.

        Note: This currently only supports operations on TimeSeriesSignals.
        Operations on Features are not standard.

        Args:
            signal_key: Key of the TimeSeriesSignal to operate on.
            operation_name: Name of the operation to apply (must be registered
                            in the TimeSeriesSignal's registry).
            parameters: Parameters for the operation.
            output_key: Key to use when storing the result (must be unique).

        Returns:
            The resulting TimeSeriesSignal that was stored.

        Raises:
            KeyError: If the signal key doesn't exist or is not a TimeSeriesSignal.
            ValueError: If the operation fails or output_key exists.
        """
        signal = self.get_time_series_signal(signal_key) # Ensures it's a TimeSeriesSignal
        result = signal.apply_operation(operation_name, **parameters) # inplace=False is default

        # Check result type and add to appropriate dictionary
        if isinstance(result, TimeSeriesSignal):
             self.add_time_series_signal(output_key, result)
        # elif isinstance(result, Feature): # Should not happen from TimeSeriesSignal.apply_operation
        #      self.add_feature(output_key, result)
        else:
             raise TypeError(f"Operation '{operation_name}' on signal '{signal_key}' returned unexpected type {type(result).__name__}")

        return result

    def apply_operation_to_signals(self, signal_keys: List[str], operation_name: str,
                                  parameters: Dict[str, Any], inplace: bool = False,
                                  output_keys: Optional[List[str]] = None) -> List[Union[TimeSeriesSignal, Feature]]:
        """
        Apply an operation to multiple TimeSeriesSignals.

        Note: This currently only supports operations on TimeSeriesSignals.

        Args:
            signal_keys: List of keys for TimeSeriesSignals to operate on.
            operation_name: Name of the operation to apply.
            parameters: Parameters for the operation.
            inplace: Whether to apply the operation in place.
            output_keys: Keys for storing results (required if inplace=False).

        Returns:
            List of TimeSeriesSignals that were created or modified.

        Raises:
            ValueError: If inplace=False and output_keys mismatch, or if a key
                        is not a valid TimeSeriesSignal key.
        """
        if not inplace and (not output_keys or len(output_keys) != len(signal_keys)):
            raise ValueError("Must provide matching output_keys when inplace=False")

        results = []
        for i, key in enumerate(signal_keys):
            signal = self.get_time_series_signal(key) # Ensures TimeSeriesSignal

            if inplace:
                signal.apply_operation(operation_name, inplace=True, **parameters)
                results.append(signal)
            else:
                result = signal.apply_operation(operation_name, **parameters)
                output_key = output_keys[i] # type: ignore
                # Check result type and add
                if isinstance(result, TimeSeriesSignal):
                     self.add_time_series_signal(output_key, result)
                # elif isinstance(result, Feature): # Should not happen
                #      self.add_feature(output_key, result)
                else:
                     raise TypeError(f"Operation '{operation_name}' on signal '{key}' returned unexpected type {type(result).__name__}")
                results.append(result)

        return results # type: ignore

    def import_signals_from_source(self, importer_instance, source: str,
                                  spec: Dict[str, Any]) -> List[TimeSeriesSignal]:
        """
        Import TimeSeriesSignals from a source using the specified importer.

        Args:
            importer_instance: The importer instance to use.
            source: Source path or identifier.
            spec: Import specification containing configuration.

        Returns:
            List of imported TimeSeriesSignals.

        Raises:
            ValueError: If the source doesn't exist or no signals can be imported.
            TypeError: If the imported object is not a TimeSeriesSignal.
        """
        signal_type_str = spec["signal_type"]
        strict_validation = spec.get("strict_validation", True)

        # --- Determine expected output type based on signal_type_str ---
        # This is a basic check; importers might return subclasses.
        # We primarily expect TimeSeriesSignal results from importers.
        expected_type = TimeSeriesSignal
        # Add logic here if certain signal_type strings imply Feature outputs, though unlikely for importers.
        # if signal_type_str == "some_feature_type":
        #     expected_type = Feature

        imported_objects: List[Any] = [] # Use Any initially

        # --- File Pattern Handling ---
        if "file_pattern" in spec:
            if not os.path.isdir(source):
                if strict_validation: raise ValueError(f"Source directory not found: {source}")
                else: warnings.warn(f"Source directory not found: {source}, skipping"); return []

            # Delegate pattern handling to importer if supported
            if hasattr(importer_instance, 'import_signals'):
                try:
                    # Assume import_signals returns a list of the expected type
                    imported_objects = importer_instance.import_signals(source, signal_type_str)
                except FileNotFoundError as e:
                     if strict_validation: raise e
                     else: warnings.warn(f"No files found matching pattern in {source} for importer: {e}"); return []
                except Exception as e:
                    if strict_validation: raise
                    else: warnings.warn(f"Error importing from {source} with pattern: {e}, skipping"); return []
            else:
                # Manual globbing if importer doesn't handle patterns
                file_pattern = os.path.join(source, spec["file_pattern"])
                matching_files = glob.glob(file_pattern)
                if not matching_files:
                    if strict_validation: raise ValueError(f"No files found matching pattern: {file_pattern}")
                    else: warnings.warn(f"No files found matching pattern: {file_pattern}, skipping"); return []

                for file_path in matching_files:
                    try:
                        # Assume import_signal returns a single object
                        signal_obj = importer_instance.import_signal(file_path, signal_type_str)
                        imported_objects.append(signal_obj)
                    except Exception as e:
                        if strict_validation: raise
                        else: warnings.warn(f"Error importing {file_path}: {e}, skipping")
        else:
            # --- Regular File Import ---
            if not os.path.exists(source):
                if strict_validation: raise ValueError(f"Source file not found: {source}")
                else: warnings.warn(f"Source file not found: {source}, skipping"); return []
            try:
                 # Assume import_signal for single file source
                 signal_obj = importer_instance.import_signal(source, signal_type_str)
                 imported_objects.append(signal_obj)
            except Exception as e:
                 if strict_validation: raise
                 else: warnings.warn(f"Error importing {source}: {e}, skipping"); return []


        # --- Validate and Filter Results ---
        validated_signals: List[TimeSeriesSignal] = []
        for obj in imported_objects:
            if isinstance(obj, expected_type):
                 # Further check if it's specifically TimeSeriesSignal for this method
                 if isinstance(obj, TimeSeriesSignal):
                      validated_signals.append(obj)
                 else:
                      # This case should be rare if expected_type is TimeSeriesSignal
                      logger.warning(f"Importer returned object of type {type(obj).__name__} which is not TimeSeriesSignal. Skipping.")
            else:
                logger.warning(f"Importer returned unexpected type {type(obj).__name__} (expected {expected_type.__name__}). Skipping.")

        return validated_signals


    def add_imported_signals(self, signals: List[TimeSeriesSignal], base_name: str,
                           start_index: int = 0) -> List[str]:
        """Add imported TimeSeriesSignals to the collection with sequential indexing."""
        keys = []
        current_index = start_index

        for signal in signals:
            if not isinstance(signal, TimeSeriesSignal):
                 logger.warning(f"Skipping object of type {type(signal).__name__} during add_imported_signals (expected TimeSeriesSignal).")
                 continue
            key = f"{base_name}_{current_index}"
            try:
                self.add_time_series_signal(key, signal) # Use specific add method
                keys.append(key)
                current_index += 1
            except ValueError as e:
                 # Handle case where key might already exist unexpectedly
                 logger.error(f"Failed to add imported signal with key '{key}': {e}. Trying next index.")
                 # Try incrementing index again to find next available slot
                 current_index += 1
                 key = f"{base_name}_{current_index}"
                 try:
                      self.add_time_series_signal(key, signal)
                      keys.append(key)
                      current_index += 1
                 except ValueError as e2:
                      logger.error(f"Failed again to add imported signal with key '{key}': {e2}. Skipping this signal.")


        return keys

    def _format_summary_cell(self, x, col_name):
        """Helper function to format a single cell for the summary DataFrame printout."""
        # --- Start Debugging ---
        logger.debug(f"_format_summary_cell received value of type: {type(x)} for column '{col_name}'")
        if isinstance(x, (pd.Series, pd.DataFrame)):
            # Log detailed error if a Series/DataFrame is received
            logger.error(f"!!! Unexpected Series/DataFrame received in _format_summary_cell for column '{col_name}'. Value:\n{x}")
            # Optionally return a distinct error string for the output table
            # return "<ERROR: Unexpected Series/DataFrame>"
        # --- End Debugging ---

        # --- Start Corrected Logic ---
        # Handle lists, tuples, dicts FIRST
        if isinstance(x, (list, tuple, dict)):
             # Show type and length to avoid large strings
             try:
                  # Check if empty before accessing length
                  if not x:
                       return f"<{type(x).__name__} len=0>"
                  else:
                       return f"<{type(x).__name__} len={len(x)}>"
             except TypeError: # Handle unsized objects (should be rare for list/tuple/dict)
                  return f"<{type(x).__name__}>"
        # Handle Enums next
        elif isinstance(x, Enum):
            return x.name
        # Handle Timestamps/Datetimes
        elif isinstance(x, (pd.Timestamp, datetime)):
            # pd.Timestamp handles NaT check implicitly in strftime
            try:
                # Check for NaT explicitly before formatting
                if pd.isna(x):
                     return 'NaT'
                # Attempt to format with timezone, fallback if naive or fails
                return x.strftime('%Y-%m-%d %H:%M:%S %Z')
            except ValueError: # Handle cases where %Z might fail or other formatting issues
                try:
                     return x.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError: # Fallback if all formatting fails
                     return str(x) # Use default string representation
        # Handle Timedeltas
        elif isinstance(x, pd.Timedelta):
             if pd.isna(x):
                  return 'NaT'
             return str(x) # Default Timedelta string representation
        # Handle specific column formatting (like data_shape tuple)
        elif col_name == 'data_shape' and isinstance(x, tuple):
             return str(x) # Format shape tuple as string

        # NOW it's safe to use pd.isna() for remaining scalar types
        elif pd.isna(x): # Handles None, np.nan, etc. for non-list/enum/time types
            return 'N/A'

        # Fallback for other scalar types (numbers, strings, bools, etc.)
        else:
            return x # Keep other types as they are
        # --- End Corrected Logic ---

    @register_collection_operation("summarize_signals")
    def summarize_signals(self, fields_to_include: Optional[List[str]] = None, print_summary: bool = True) -> Optional[pd.DataFrame]:
        """Generates a summary table of TimeSeriesSignals and Features."""
        # Combine both signal types for summary generation
        all_items = {**self.time_series_signals, **self.features}

        if not all_items:
            logger.info("Signal collection is empty. No summary to generate.")
            self._summary_dataframe = pd.DataFrame() # Store empty DataFrame
            self._summary_dataframe_params = None
            if print_summary:
                print("Signal collection is empty.")
            return self._summary_dataframe

        # --- Determine Fields ---
        # Default fields combine common attributes and type-specific ones
        default_ts_fields = [f.name for f in fields(TimeSeriesMetadata)]
        default_feat_fields = [f.name for f in fields(FeatureMetadata)]
        # Combine and deduplicate, prioritize common names
        common_fields = set(default_ts_fields) & set(default_feat_fields)
        ts_only_fields = set(default_ts_fields) - common_fields
        feat_only_fields = set(default_feat_fields) - common_fields
        # Add calculated fields
        # REMOVE 'item_type' from this list
        calculated_fields = ['source_files_count', 'operations_count', 'feature_names_count', 'data_shape'] # Removed 'item_type'
        default_fields_ordered = ['key', 'item_type'] + sorted(list(common_fields)) + \
                                 sorted(list(ts_only_fields)) + sorted(list(feat_only_fields)) + \
                                 sorted(calculated_fields) # Now item_type only appears once

        fields_for_summary = fields_to_include if fields_to_include is not None else default_fields_ordered
        logger.info(f"Generating summary. Fields requested: {'Default' if fields_to_include is None else fields_to_include}")
        logger.debug(f"Actual fields being processed for summary: {fields_for_summary}")

        summary_data = []
        valid_ts_meta_fields = {f.name for f in fields(TimeSeriesMetadata)}
        valid_feat_meta_fields = {f.name for f in fields(FeatureMetadata)}

        for key, item in all_items.items():
            row_data = {'key': key}
            metadata_obj = item.metadata
            is_feature = isinstance(item, Feature)
            row_data['item_type'] = 'Feature' if is_feature else 'TimeSeries'
            valid_meta_fields = valid_feat_meta_fields if is_feature else valid_ts_meta_fields

            for field in fields_for_summary:
                if field in ['key', 'item_type']: continue # Already handled

                value = None
                try:
                    # --- Calculated Fields ---
                    if field == 'source_files_count':
                         value = len(metadata_obj.source_files) if hasattr(metadata_obj, 'source_files') and metadata_obj.source_files else 0
                    elif field == 'operations_count':
                         value = len(metadata_obj.operations) if hasattr(metadata_obj, 'operations') and metadata_obj.operations else 0
                    elif field == 'feature_names_count':
                         value = len(metadata_obj.feature_names) if hasattr(metadata_obj, 'feature_names') and metadata_obj.feature_names else (0 if is_feature else None) # Count only for features
                    elif field == 'data_shape':
                         try:
                              data = item.get_data()
                              value = data.shape if hasattr(data, 'shape') else None
                         except Exception: value = None
                    # --- Metadata Fields ---
                    elif field in valid_meta_fields:
                        value = getattr(metadata_obj, field, None)
                    else: # Field not applicable to this item type
                         value = None

                    row_data[field] = value # Store raw value

                except Exception as e:
                    logger.warning(f"Error accessing raw field '{field}' for item '{key}': {e}")
                    row_data[field] = None

            summary_data.append(row_data)

        # --- Create Raw DataFrame ---
        raw_summary_df = pd.DataFrame(summary_data)
        # Ensure all requested columns exist, filling with NaN if needed
        for col in fields_for_summary:
             if col not in raw_summary_df.columns:
                  raw_summary_df[col] = None
        # Reorder columns
        raw_summary_df = raw_summary_df[['key'] + [f for f in fields_for_summary if f != 'key']]
        raw_summary_df = raw_summary_df.set_index('key').sort_index()

        # --- Start Debugging ---
        logger.debug("Raw summary DataFrame info before formatting:")
        # Log dtypes to see if any column unexpectedly holds objects that might be Series
        logger.debug(f"dtypes:\n{raw_summary_df.dtypes}")
        # --- End Debugging ---

        # --- Store Raw DataFrame ---
        self._summary_dataframe = raw_summary_df.copy()
        self._summary_dataframe_params = {'fields_to_include': fields_for_summary, 'print_summary': print_summary}
        logger.info(f"Stored signal/feature summary DataFrame with shape {self._summary_dataframe.shape}")

        # --- Handle Printing ---
        if print_summary:
            formatted_summary_df = raw_summary_df.copy()
            # Apply formatting using the new helper function
            for col in formatted_summary_df.columns:
                # --- Start Debugging ---
                logger.debug(f"Attempting to format summary column: '{col}'")
                try:
                    # Pass the column name to the helper function for context-aware formatting
                    formatted_summary_df[col] = formatted_summary_df[col].apply(
                        lambda cell_value: self._format_summary_cell(cell_value, col)
                    )
                except ValueError as e:
                    # Catch the specific error to log context
                    if "The truth value of a Series is ambiguous" in str(e):
                        logger.error(f"ValueError caught while formatting column '{col}'. This likely means a cell contained a Series.")
                        # Optionally log the first few problematic values
                        try:
                            problematic_values = formatted_summary_df[col][formatted_summary_df[col].apply(lambda v: isinstance(v, pd.Series))]
                            logger.error(f"Problematic Series values found in column '{col}':\n{problematic_values.head()}")
                        except Exception as log_err:
                            logger.error(f"Could not log specific problematic values in column '{col}': {log_err}")
                    # Re-raise the original error after logging
                    raise e
                # --- End Debugging ---

            print("\n--- Signal Collection Summary ---")
            # Use pandas display options for better formatting if needed
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                 print(formatted_summary_df.to_string())
            print("-------------------------------\n")

        return self._summary_dataframe


# --- Populate the Collection Operation Registry ---
# Iterate through the methods of the class *after* it's defined
# and populate the registry based on the decorator attribute.
SignalCollection.collection_operation_registry = {} # Reset registry before population
for _method_name, _method_obj in inspect.getmembers(SignalCollection, predicate=inspect.isfunction):
    if hasattr(_method_obj, '_collection_op_name'):
        _op_name = getattr(_method_obj, '_collection_op_name')
        if _op_name in SignalCollection.collection_operation_registry:
             warnings.warn(f"Overwriting collection operation '{_op_name}' during registry population.")
        SignalCollection.collection_operation_registry[_op_name] = _method_obj
        logger.debug(f"Registered collection operation '{_op_name}' to method SignalCollection.{_method_name}")

# --- Register Multi-Signal Operations ---
SignalCollection.multi_signal_registry = {} # Reset registry before population
try:
    # Import feature extraction functions
    from ..operations.feature_extraction import (
        compute_feature_statistics,
        compute_sleep_stage_mode,
        compute_hrv_features,
        compute_movement_features,
        compute_correlation_features
    )
    # Import the Feature class (output type for these operations)
    from ..features.feature import Feature

    SignalCollection.multi_signal_registry.update({
        "feature_statistics": (compute_feature_statistics, Feature),
        "compute_sleep_stage_mode": (compute_sleep_stage_mode, Feature),
        "compute_hrv_features": (compute_hrv_features, Feature),
        "compute_movement_features": (compute_movement_features, Feature),
        "compute_correlation_features": (compute_correlation_features, Feature),
    })
    logger.debug("Registered multi-signal feature operations (statistics, sleep_stage_mode, hrv, movement, correlation).")
except ImportError as e:
    logger.warning(f"Could not import or register feature operations: {e}")


# Clean up temporary variables from the global scope of the module
# Check if variables exist before deleting
if '_method_name' in locals() or '_method_name' in globals(): del _method_name
if '_method_obj' in locals() or '_method_obj' in globals(): del _method_obj
if '_op_name' in locals() or '_op_name' in globals(): del _op_name
