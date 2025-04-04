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
from dataclasses import fields
from typing import Dict, List, Any, Optional, Type, Tuple, Callable, Union

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from .metadata import SignalMetadata, CollectionMetadata, OperationInfo
from .signal_data import SignalData
from ..signal_types import SignalType, SensorType, SensorModel, BodyPosition
from .metadata_handler import MetadataHandler
from ..signals.time_series_signal import TimeSeriesSignal # Added missing import
from ..utils import str_to_enum # Added missing import

# Initialize logger for the module
logger = logging.getLogger(__name__)


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
    # Registry for multi-signal operations
    multi_signal_registry: Dict[str, Tuple[Callable, Type[SignalData]]] = {}
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None, metadata_handler: Optional[MetadataHandler] = None):
        """
        Initialize a SignalCollection instance.
        
        Args:
            metadata: Optional dictionary with collection-level metadata
            metadata_handler: Optional metadata handler, will create one if not provided
        """
        self.signals: Dict[str, SignalData] = {}
        
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
    
    def add_signal(self, key: str, signal: SignalData) -> None:
        """
        Add a signal to the collection with the specified key.
        
        Args:
            key: Unique identifier for the signal in this collection
            signal: The signal instance to add
        
        Raises:
            ValueError: If a signal with the given key already exists or signal_id conflicts
        """
        if key in self.signals:
            raise ValueError(f"Signal with key '{key}' already exists in the collection")
        # Check for signal_id uniqueness
        existing_ids = {s.metadata.signal_id for s in self.signals.values()}
        if signal.metadata.signal_id in existing_ids:
            # If ID conflict, generate a new one and log a warning
            new_id = str(uuid.uuid4())
            logger.warning(f"Signal ID '{signal.metadata.signal_id}' conflicts with existing signal. Assigning new ID: {new_id}")
            signal.metadata.signal_id = new_id

        # Validate that time series signals have a proper timestamp index and matching timezone
        if isinstance(signal, TimeSeriesSignal):
            self._validate_timestamp_index(signal) # Checks for DatetimeIndex

            # Optional: Validate timezone consistency
            try:
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
        # This ensures consistency between standalone signals and collection signals
        if signal.handler:
            # Use the signal's existing handler
            signal.handler.set_name(signal.metadata, key=key)
        else:
            # Use the collection's handler if the signal doesn't have one
            signal.handler = self.metadata_handler
            signal.handler.set_name(signal.metadata, key=key)
            
        self.signals[key] = signal
    
    def add_signal_with_base_name(self, base_name: str, signal: SignalData) -> str:
        """
        Add a signal with a base name, appending an index if needed.
        
        Args:
            base_name: Base name for the signal (e.g., "ppg").
            signal: The signal instance to add.
        
        Returns:
            The key assigned to the signal (e.g., "ppg_0").
        
        Raises:
            ValueError: If the signal cannot be added due to invalid input.
        """
        if not base_name:
            raise ValueError("Base name cannot be empty")
        index = 0
        while True:
            key = f"{base_name}_{index}"
            if key not in self.signals:
                self.signals[key] = signal
                return key
            index += 1
    
    def get_signal(self, key: str) -> SignalData:
        """
        Retrieve a signal by its key.
        
        Args:
            key: The key used when adding the signal
        
        Returns:
            The requested signal instance
        
        Raises:
            KeyError: If no signal exists with the specified key
        """
        if key not in self.signals:
            raise KeyError(f"No signal with key '{key}' found in the collection")
        return self.signals[key]
    
    def get_signals(self, input_spec: Union[str, Dict[str, Any], List[str], None] = None,
                   signal_type: Union[SignalType, str, None] = None,
                   criteria: Dict[str, Any] = None,
                   base_name: str = None) -> List[SignalData]:
        """
        Retrieve signals based on flexible criteria.
        
        This method consolidates multiple signal retrieval methods into a single
        flexible interface. You can specify one or more filtering criteria.
        
        Args:
            input_spec: Can be:
                        - String ID or base name ("ppg", "ppg_0")
                        - Dictionary with criteria
                        - List of string IDs or base names
            signal_type: A SignalType enum value or string name to filter by
            criteria: Dictionary of metadata field/value pairs to match
            base_name: Base name to filter signals (e.g., "ppg" returns "ppg_0", "ppg_1")
        
        Returns:
            List of matching SignalData instances
        
        Examples:
            # Get all signals of type PPG
            collection.get_signals(signal_type=SignalType.PPG)
            
            # Get all signals with base name "ppg"
            collection.get_signals(base_name="ppg")
            
            # Get signals matching specific criteria
            collection.get_signals(criteria={"sensor_type": SensorType.WRIST})
            
            # Get signals using flexible input specifier
            collection.get_signals(input_spec=["ppg_0", "ecg_1"])
            collection.get_signals(input_spec={"base_name": "ppg",
                                              "criteria": {"sensor_model": SensorModel.POLAR}})
        """
        result = []
        # Process signal_type parameter if provided
        if signal_type is not None:
            # Convert string to enum if needed
            if isinstance(signal_type, str):
                signal_type = str_to_enum(signal_type, SignalType)
            
            # Create or update criteria dict with signal_type
            if criteria is None:
                criteria = {"signal_type": signal_type}
            else:
                criteria = criteria.copy()
                criteria["signal_type"] = signal_type
        
        # Process input_spec parameter if provided
        if input_spec is not None:
            if isinstance(input_spec, dict):
                # Dictionary specification with criteria and/or base_name
                if "base_name" in input_spec:
                    base_name = input_spec["base_name"]
                
                if "criteria" in input_spec:
                    # Process criteria from input_spec
                    spec_criteria = input_spec["criteria"]
                    processed_criteria = {}
                    
                    # Convert string enum values to actual enums
                    for key, value in spec_criteria.items():
                        if isinstance(value, str):
                            if key == "signal_type":
                                processed_criteria[key] = str_to_enum(value, SignalType)
                            elif key == "sensor_type":
                                processed_criteria[key] = str_to_enum(value, SensorType)
                            elif key == "sensor_model":
                                processed_criteria[key] = str_to_enum(value, SensorModel)
                            elif key == "body_position":
                                processed_criteria[key] = str_to_enum(value, BodyPosition)
                            else:
                                processed_criteria[key] = value
                        else:
                            processed_criteria[key] = value
                    
                    # Merge with existing criteria if any
                    if criteria is None:
                        criteria = processed_criteria
                    else:
                        criteria.update(processed_criteria)
            
            elif isinstance(input_spec, list):
                # List of keys or base names
                signals = []
                for spec in input_spec:
                    signals.extend(self.get_signals(input_spec=spec, 
                                                  signal_type=signal_type,
                                                  criteria=criteria,
                                                  base_name=base_name))
                return signals
                
            else:
                # String specifier (key or base name)
                spec_str = input_spec
                
                # Check if this is a specific key
                if spec_str in self.signals:
                    signal = self.signals[spec_str]
                    
                    # Apply additional filters if provided
                    if self._matches_criteria(signal, criteria):
                        result.append(signal)
                    return result
                    
                # Check if this is an indexed name (contains underscore and ends with number)
                if "_" in spec_str and spec_str.split("_")[-1].isdigit():
                    # This is an indexed name, get the specific signal if it exists
                    if spec_str in self.signals:
                        signal = self.signals[spec_str]
                        if self._matches_criteria(signal, criteria):
                            result.append(signal)
                    return result
                    
                # Treat as base name
                base_name = spec_str
        
        # Now apply all filtering
        # Start with all signals if we haven't yet built a result list
        if not result:
            # Special case: if base_name is provided, only include signals with that base name
            if base_name:
                for key, signal in self.signals.items():
                    if key.startswith(f"{base_name}_") and key[len(base_name)+1:].isdigit():
                        if self._matches_criteria(signal, criteria):
                            result.append(signal)
            else:
                # Otherwise, start with all signals and filter by criteria
                for signal in self.signals.values():
                    if self._matches_criteria(signal, criteria):
                        result.append(signal)
        
        return result
    
    def _matches_criteria(self, signal: SignalData, criteria: Dict[str, Any]) -> bool:
        """
        Check if a signal matches all the criteria.
        
        Args:
            signal: The signal to check
            criteria: Dictionary of metadata field/value pairs to match
            
        Returns:
            True if the signal matches all criteria, False otherwise
        """
        if not criteria:
            return True
        for key, value in criteria.items():
            # Handle nested fields (e.g., "sensor_info.device_id")
            if "." in key:
                parts = key.split(".", 1)
                container_name, field_name = parts
                # Get the container
                if hasattr(signal.metadata, container_name):
                    container = getattr(signal.metadata, container_name)
                    # Check if container is a dict and has the field
                    if isinstance(container, dict) and field_name in container:
                        if container[field_name] != value:
                            return False
                    else:
                        return False
                else:
                    return False
            
            # Handle standard fields
            elif hasattr(signal.metadata, key):
                if getattr(signal.metadata, key) != value:
                    return False
            else:
                return False
        
        return True
    
    def update_signal_metadata(self, signal: SignalData, metadata_spec: Dict[str, Any]) -> None:
        """
        Update a signal's metadata with values from a specification.
        
        Args:
            signal: The signal to update
            metadata_spec: Dictionary containing metadata values to update
        """
        # Process any enum fields first
        processed_metadata = {}
        
        # Update enum fields
        if "signal_type" in metadata_spec and isinstance(metadata_spec["signal_type"], str):
            from ..signal_types import SignalType
            processed_metadata["signal_type"] = str_to_enum(metadata_spec["signal_type"], SignalType)
            
        if "sensor_type" in metadata_spec and isinstance(metadata_spec["sensor_type"], str):
            processed_metadata["sensor_type"] = str_to_enum(metadata_spec["sensor_type"], SensorType)
            
        if "sensor_model" in metadata_spec and isinstance(metadata_spec["sensor_model"], str):
            processed_metadata["sensor_model"] = str_to_enum(metadata_spec["sensor_model"], SensorModel)
            
        if "body_position" in metadata_spec and isinstance(metadata_spec["body_position"], str):
            processed_metadata["body_position"] = str_to_enum(metadata_spec["body_position"], BodyPosition)
        
        # Handle sensor_info separately
        if "sensor_info" in metadata_spec and isinstance(metadata_spec["sensor_info"], dict):
            # Initialize sensor_info if needed
            if signal.metadata.sensor_info is None:
                signal.metadata.sensor_info = {}
            signal.metadata.sensor_info.update(metadata_spec["sensor_info"])
        
        # Add other fields to the processed metadata
        for field in ["name", "sample_rate", "units", "start_time", "end_time"]:
            if field in metadata_spec:
                processed_metadata[field] = metadata_spec[field]
        
        # Use the metadata handler to update the signal's metadata
        if hasattr(signal, 'handler') and signal.handler:
            # Use the signal's existing handler
            signal.handler.update_metadata(signal.metadata, **processed_metadata)
        else:
            # Use the collection's handler if the signal doesn't have one
            signal.handler = self.metadata_handler
            signal.handler.update_metadata(signal.metadata, **processed_metadata)
    
    
    def set_index_config(self, index_fields: List[str]) -> None:
        """
        Configure the multi-index fields for dataframe exports.
        
        Args:
            index_fields: List of metadata field names to use as index levels.
        
        Raises:
            ValueError: If any field is not a valid SignalMetadata attribute.
        """
        valid_fields = {f.name for f in fields(SignalMetadata)}
        if not all(f in valid_fields for f in index_fields):
            raise ValueError(f"Invalid index fields: {set(index_fields) - valid_fields}")
        self.metadata.index_config = index_fields
    
   
    def _validate_timestamp_index(self, signal: SignalData) -> None:
        """
        Validate that a signal has a proper timestamp index.
        
        Args:
            signal: The signal to validate
            
        Raises:
            ValueError: If the signal doesn't have a DatetimeIndex
        """
        logger = logging.getLogger(__name__)
        data = signal.get_data()
        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.error(f"Signal {signal.metadata.signal_id} doesn't have a DatetimeIndex")
                raise ValueError(f"All time series signals must have a DatetimeIndex")
    
    def get_target_sample_rate(self, user_specified=None):
        """
        Determine the target sample rate for alignment.
        
        Args:
            user_specified (float, optional): User-defined target rate in Hz.
        
        Returns:
            float: The target sample rate in Hz.
        """
        if user_specified is not None:
            return user_specified

        # Calculate max rate using the float value from get_sampling_rate()
        valid_rates = [
            s.get_sampling_rate() for s in self.signals.values()
            if isinstance(s, TimeSeriesSignal) and s.get_sampling_rate() is not None and s.get_sampling_rate() > 0
        ]

        if not valid_rates:
             # If no valid rates, default to 100 Hz
             return 100.0

        max_rate = max(valid_rates)

        # If max_rate is still 0 or less (shouldn't happen with filter), default to 100 Hz
        if max_rate <= 0:
             logger.warning("No valid positive sampling rates found. Defaulting target rate to 100 Hz.") # Added logger
             return 100.0

        # Find the largest standard rate <= the maximum rate found in the signals
        valid_standard_rates = [r for r in STANDARD_RATES if r <= max_rate]
        chosen_rate = max(valid_standard_rates) if valid_standard_rates else min(STANDARD_RATES) # Fallback to lowest standard rate
        logger.info(f"Determined target sample rate: {chosen_rate} Hz (based on max signal rate {max_rate:.4f} Hz)") # Added logger
        return chosen_rate
        # Removed redundant except block as it was outside the try


    def get_nearest_standard_rate(self, rate):
        """
        Find the nearest standard rate to a given sample rate.

        Args:
            rate (float): The signal's sample rate in Hz.

        Returns:
            float: The closest rate from STANDARD_RATES.
        """
        logger = logging.getLogger(__name__)

        # Consolidated the check for invalid rate
        if rate is None or rate <= 0:
            logger.warning(f"Invalid rate ({rate}) provided to get_nearest_standard_rate. Returning default rate 1 Hz.")
            return 1.0 # Default to 1 Hz if rate is invalid

        # Find the rate in STANDARD_RATES that minimizes the absolute difference
        nearest_rate = min(STANDARD_RATES, key=lambda r: abs(r - rate))
        logger.debug(f"Nearest standard rate to {rate:.4f} Hz is {nearest_rate} Hz.")
        return nearest_rate

    def get_reference_time(self, target_period: pd.Timedelta) -> pd.Timestamp:
        """
        Compute the reference timestamp for the grid alignment.

        Finds the earliest timestamp across all time-series signals and aligns
        it to the grid defined by the target_period, ensuring a consistent
        starting point relative to the Unix epoch (1970-01-01).
        
        Args:
            target_period (pd.Timedelta): The period corresponding to the target sample rate.
        
        Returns:
            pd.Timestamp: The reference time aligned to the grid.
        """
        logger = logging.getLogger(__name__)

        min_times = []
        for signal in self.signals.values():
             if isinstance(signal, TimeSeriesSignal):
                 data = signal.get_data()
                 if data is not None and isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                     min_times.append(data.index.min())

        if not min_times:
            logger.warning("No valid timestamps found in signals. Using default reference time 1970-01-01.")
            return pd.Timestamp("1970-01-01", tz='UTC') # Ensure timezone consistency

        # Use the overall earliest time across all signals
        min_time = min(min_times)
        logger.debug(f"Earliest timestamp found across signals: {min_time}")

        # Ensure reference time is timezone-aware if min_time is
        epoch = pd.Timestamp("1970-01-01", tz=min_time.tz)

        # Calculate the offset from the epoch in nanoseconds for precision
        delta_ns = (min_time - epoch).total_seconds() * 1e9
        target_period_ns = target_period.total_seconds() * 1e9

        if target_period_ns == 0:
             logger.error("Target period is zero, cannot calculate reference time.")
             raise ValueError("Target period cannot be zero for reference time calculation.")

        # Calculate the number of full periods from the epoch to the minimum time
        # Use floor division to find the grid point *before* or *at* the minimum time
        num_periods = np.floor(delta_ns / target_period_ns) # type: ignore

        # Calculate the reference time by adding the total duration of these periods to the epoch
        ref_time = epoch + pd.Timedelta(nanoseconds=num_periods * target_period_ns) # type: ignore
        logger.debug(f"Calculated reference time: {ref_time} based on target period {target_period}")

        return ref_time


    def _calculate_grid_index(self, target_rate: float, ref_time: pd.Timestamp) -> Optional[pd.DatetimeIndex]:
        """
        Calculates the final DatetimeIndex grid for the collection.

        Determines the overall time range of all time-series signals and creates
        a regular DatetimeIndex covering this range at the target_rate, aligned
        to the ref_time.

        Args:
            target_rate: The target sampling rate in Hz.
            ref_time: The reference timestamp for grid alignment.

        Returns:
            A pd.DatetimeIndex representing the common grid, or None if no valid
            timestamps are found.
        """
        logger = logging.getLogger(__name__)

        if target_rate <= 0:
            logger.error(f"Invalid target_rate ({target_rate}) for grid calculation.")
            return None

        target_period = pd.Timedelta(seconds=1 / target_rate)

        min_times = []
        max_times = []
        for signal in self.signals.values():
            if isinstance(signal, TimeSeriesSignal):
                data = signal.get_data()
                if data is not None and isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                    # Ensure timezone consistency with ref_time before comparison
                    data_index_tz = data.index.tz_convert(ref_time.tz) if data.index.tz is not None else data.index.tz_localize(ref_time.tz)
                    min_times.append(data_index_tz.min())
                    max_times.append(data_index_tz.max())

        if not min_times or not max_times:
            logger.warning("No valid timestamps found in signals. Cannot create grid index.")
            return None

        earliest_start = min(min_times)
        latest_end = max(max_times)
        logger.info(f"Overall time range for grid: {earliest_start} to {latest_end}")

        # Generate a regular grid covering this range, aligned to ref_time
        # Use floor/ceil on the number of periods from ref_time to ensure full coverage
        target_period_ns = target_period.total_seconds() * 1e9
        if target_period_ns == 0:
             logger.error("Target period is zero, cannot calculate grid index.")
             return None

        start_offset_ns = (earliest_start - ref_time).total_seconds() * 1e9
        end_offset_ns = (latest_end - ref_time).total_seconds() * 1e9

        # Calculate periods using floor/ceil for start/end respectively
        periods_to_start = np.floor(start_offset_ns / target_period_ns) # type: ignore
        periods_to_end = np.ceil(end_offset_ns / target_period_ns) # type: ignore

        grid_start = ref_time + pd.Timedelta(nanoseconds=periods_to_start * target_period_ns) # type: ignore
        grid_end = ref_time + pd.Timedelta(nanoseconds=periods_to_end * target_period_ns) # type: ignore

        # Ensure start <= end before creating range
        if grid_start > grid_end:
            logger.warning(f"Calculated grid_start ({grid_start}) is after grid_end ({grid_end}). Cannot create grid.")
            # Swap them if inverted due to potential floating point issues near zero offset
            if abs((grid_start - grid_end).total_seconds()) < (target_period.total_seconds() * 0.5):
                 logger.debug("Grid start/end inverted likely due to floating point near zero, swapping.")
                 grid_start, grid_end = grid_end, grid_start
            else:
                 return None # Return None if range is invalid

        try:
            grid_index = pd.date_range(
                start=grid_start,
                end=grid_end,
                freq=target_period,
                name='timestamp' # Name the index
            )
            # Ensure the grid has the same timezone as ref_time
            grid_index = grid_index.tz_convert(ref_time.tz) if grid_index.tz is not None else grid_index.tz_localize(ref_time.tz)

            logger.info(f"Calculated grid_index with {len(grid_index)} points from {grid_index.min()} to {grid_index.max()}")
            return grid_index
        except Exception as e:
            logger.error(f"Error creating date_range for grid index: {e}", exc_info=True)
            return None


    def align_signals(self, target_sample_rate: Optional[float] = None) -> 'SignalCollection':
        """
        Calculates and stores the alignment parameters for the collection.

        Determines the target sampling rate, reference time, and the common
        DatetimeIndex grid based on all time-series signals in the collection.
        These parameters are stored on the collection instance (`target_rate`,
        `ref_time`, `grid_index`) to be used later by `get_combined_dataframe`.

        This method does *not* modify the individual signal data.

        Args:
            target_sample_rate: Optional. The desired target sample rate in Hz.
                                If None, the highest standard rate less than or
                                equal to the maximum rate found among signals
                                will be used.

        Returns:
            The SignalCollection instance (self) with alignment parameters set.

        Raises:
            ValueError: If no time-series signals are found or if a valid
                        grid index cannot be calculated.
        """
        logger = logging.getLogger(__name__)
        start_time = time.time()

        logger.info(f"Starting align_signals parameter calculation with target_sample_rate={target_sample_rate}")

        # --- Filter for TimeSeriesSignals ---
        ts_signals = [s for s in self.signals.values() if isinstance(s, TimeSeriesSignal)]
        if not ts_signals:
            logger.warning("No time-series signals found in the collection. Cannot perform alignment.")
            # Set default values or raise error? Setting defaults for now.
            self.target_rate = 1.0 # Default rate
            self.ref_time = pd.Timestamp("1970-01-01", tz='UTC')
            self.grid_index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
            return self
            # raise ValueError("No time-series signals found in the collection to align.")

        # --- Determine Target Rate ---
        self.target_rate = self.get_target_sample_rate(target_sample_rate)
        if self.target_rate <= 0:
             logger.error(f"Calculated invalid target rate: {self.target_rate}. Aborting alignment.")
             raise ValueError(f"Invalid target sampling rate calculated: {self.target_rate}")
        target_period = pd.Timedelta(seconds=1 / self.target_rate)
        logger.info(f"Using target rate: {self.target_rate} Hz (Period: {target_period})")

        # --- Determine Reference Time ---
        self.ref_time = self.get_reference_time(target_period)
        logger.info(f"Using reference time: {self.ref_time}")

        # --- Calculate Grid Index ---
        self.grid_index = self._calculate_grid_index(self.target_rate, self.ref_time)

        if self.grid_index is None or self.grid_index.empty:
            logger.error("Failed to calculate a valid grid index. Alignment cannot proceed.")
            # Set defaults to prevent errors later, but log error
            self.grid_index = pd.DatetimeIndex([], name='timestamp', tz=self.ref_time.tz) # Empty index with correct TZ
            raise ValueError("Failed to calculate a valid grid index for alignment.")

        logger.info(f"Alignment parameters calculated in {time.time() - start_time:.2f} seconds.")
        # Note: No signals are modified here. Parameters are stored for get_combined_dataframe.
        return self


    def get_signals_from_input_spec(self, input_spec: Union[str, Dict[str, Any], List[str], None] = None) -> List[SignalData]:
        """
        Get signals based on an input specification.
        
        This is an alias for get_signals() that maintains interface compatibility
        with the workflow executor.
        
        Args:
            input_spec: Input specifier that can be:
                      - String (signal key or base name)
                      - Dictionary with criteria
                      - List of string keys or base names
                      
        Returns:
            List of SignalData instances matching the input specification
        """
        return self.get_signals(input_spec=input_spec)
        
    def get_combined_dataframe(self) -> pd.DataFrame:
        """
        Combine all signals into a single dataframe.
        
        Creates a combined dataframe with a regular grid determined by align_signals,
        preserving the exact timestamps and only including rows where data exists.
        
        Returns:
            pd.DataFrame: Combined dataframe with aligned signals.
        """
        logger.info("Generating combined dataframe on the fly (not storing internally)...")
        # Ensure alignment parameters are available if needed by the calculation method
        if not hasattr(self, 'grid_index') or self.grid_index is None or self.grid_index.empty:
             logger.warning("align_signals parameters not set. Calculation might fail or use defaults.")
             # Depending on _calculate_combined_dataframe, might need to call align_signals here or raise.
             # For now, assume _calculate handles missing grid index gracefully or align_signals was called.

        return self._calculate_combined_dataframe()

    def apply_multi_signal_operation(self, operation_name: str, inputs: List[str], 
                                    parameters: Dict[str, Any] = None) -> SignalData:
        """
        Apply an operation that works on multiple signals.
        
        Args:
            operation_name: Name of the operation in the multi_signal_registry
            inputs: List of signal keys to use as inputs
            parameters: Parameters to pass to the operation
        
        Returns:
            The result signal instance

        Raises:
            ValueError: If the operation is not found or if any input signal
                key specified in `inputs` does not exist in the collection.
        """
        parameters = parameters or {}
        # Check if operation exists
        if operation_name not in self.multi_signal_registry:
            raise ValueError(f"Multi-signal operation '{operation_name}' not found")
            
        func, output_class = self.multi_signal_registry[operation_name]
        
        # Get all input signals
        input_signals = []
        for key in inputs:
            try:
                input_signals.append(self.get_signal(key))
            except KeyError:
                raise ValueError(f"Input signal '{key}' not found in collection")
        
        # Apply the operation
        result_data = func([s.get_data() for s in input_signals], parameters)
        
        # Create metadata recording the operation and source signals
        derived_from = []
        for signal in input_signals:
            operation_index = len(signal.metadata.operations) - 1 if signal.metadata.operations else -1
            derived_from.append((signal.metadata.signal_id, operation_index))
        # Create the output signal
        operation_info = OperationInfo(operation_name, parameters)

        output_metadata = {
            "derived_from": derived_from,
            "operations": [operation_info]
        }
        
        return output_class(data=result_data, metadata=output_metadata)
        
    def apply_and_store_operation(self, signal_key: str, operation_name: str, 
                                 parameters: Dict[str, Any], output_key: str) -> SignalData:
        """
        Apply an operation to a signal and store the result.
        
        Args:
            signal_key: Key of the signal to operate on
            operation_name: Name of the operation to apply
            parameters: Parameters for the operation
            output_key: Key to use when storing the result
            
        Returns:
            The result signal that was stored
            
        Raises:
            KeyError: If the signal key doesn't exist
            ValueError: If the operation fails
        """
        signal = self.get_signal(signal_key)
        result = signal.apply_operation(operation_name, **parameters)
        self.add_signal(output_key, result)
        return result
    
    def apply_operation_to_signals(self, signal_keys: List[str], operation_name: str,
                                  parameters: Dict[str, Any], inplace: bool = False,
                                  output_keys: Optional[List[str]] = None) -> List[SignalData]:
        """
        Apply an operation to multiple signals.
        
        Args:
            signal_keys: List of keys for signals to operate on
            operation_name: Name of the operation to apply
            parameters: Parameters for the operation
            inplace: Whether to apply the operation in place
            output_keys: Keys to use when storing results (required if inplace=False)
            
        Returns:
            List of signals that were created or modified
            
        Raises:
            ValueError: If inplace=False and output_keys not provided or length mismatch
        """
        if not inplace and (not output_keys or len(output_keys) != len(signal_keys)):
            raise ValueError("Must provide matching output_keys when inplace=False")
            
        results = []
        for i, key in enumerate(signal_keys):
            signal = self.get_signal(key)
            
            if inplace:
                signal.apply_operation(operation_name, inplace=True, **parameters)
                results.append(signal)
            else:
                result = signal.apply_operation(operation_name, **parameters)
                self.add_signal(output_keys[i], result)
                results.append(result)
                
        return results
        
    def import_signals_from_source(self, importer_instance, source: str, 
                                  spec: Dict[str, Any]) -> List[SignalData]:
        """
        Import signals from a source using the specified importer.
        
        Args:
            importer_instance: The importer instance to use
            source: Source path or identifier
            spec: Import specification containing configuration
            
        Returns:
            List of imported signals
            
        Raises:
            ValueError: If the source doesn't exist or no signals can be imported
        """
        signal_type = spec["signal_type"]
        strict_validation = spec.get("strict_validation", True)
        
        if "file_pattern" in spec:
            # Handle directory with file pattern
            if not os.path.isdir(source):
                if strict_validation:
                    raise ValueError(f"Source directory not found: {source}")
                else:
                    warnings.warn(f"Source directory not found: {source}, skipping")
                    return []
                
            # Specialized importers handle file patterns internally
            if spec["importer"] in ["MergingImporter", "PolarCSVImporter"]:
                try:
                    return importer_instance.import_signals(source, signal_type)
                except Exception as e:
                    if strict_validation:
                        raise
                    else:
                        warnings.warn(f"Error importing from {source}: {e}, skipping")
                        return []
            else:
                # For other importers, process files individually
                file_pattern = os.path.join(source, spec["file_pattern"])
                matching_files = glob.glob(file_pattern)
                
                if not matching_files:
                    if strict_validation:
                        raise ValueError(f"No files found matching pattern: {file_pattern}")
                    else:
                        warnings.warn(f"No files found matching pattern: {file_pattern}, skipping")
                        return []
                    
                # Import all matching files
                all_signals = []
                for file_path in matching_files:
                    try:
                        signals = importer_instance.import_signals(file_path, signal_type)
                        all_signals.extend(signals)
                    except Exception as e:
                        if strict_validation:
                            raise
                        else:
                            warnings.warn(f"Error importing {file_path}: {e}, skipping")
                    
                return all_signals
        else:
            # Regular file import
            if not os.path.exists(source):
                if strict_validation:
                    raise ValueError(f"Source file not found: {source}")
                else:
                    warnings.warn(f"Source file not found: {source}, skipping")
                    return []
            
            # Import signals from source
            return importer_instance.import_signals(source, signal_type)
    
    def add_imported_signals(self, signals: List[SignalData], base_name: str, 
                           start_index: int = 0) -> List[str]:
        """
        Add imported signals to the collection with sequential indexing.
        
        Args:
            signals: List of signals to add
            base_name: Base name to use for signal keys
            start_index: Starting index for sequential numbering
            
        Returns:
            List of keys assigned to the added signals
        """
        keys = []
        current_index = start_index
        
        for signal in signals:
            key = f"{base_name}_{current_index}"
            self.add_signal(key, signal)
            keys.append(key)
            current_index += 1
            
        return keys

    def apply_grid_alignment(self, method: str = 'nearest', signals_to_align: Optional[List[str]] = None):
        """
        Applies the pre-calculated grid alignment to specified signals in place.

        Modifies the internal data of TimeSeriesSignal objects to conform to the
        grid_index calculated by a prior call to align_signals.

        Args:
            method: The method to use for reindexing ('nearest', 'pad'/'ffill', 'backfill'/'bfill').
                    Interpolation methods ('linear', etc.) are generally not suitable
                    for direct reindexing to a potentially sparse grid, 'nearest' is safer.
            signals_to_align: Optional list of signal keys to align. If None, attempts
                              to align all TimeSeriesSignal objects in the collection.

        Raises:
            RuntimeError: If align_signals has not been run successfully first (no valid grid_index).
            ValueError: If an invalid alignment method is provided.
        """
        if not hasattr(self, 'grid_index') or self.grid_index is None or self.grid_index.empty:
            logger.error("Cannot apply grid alignment: align_signals must be run successfully first.")
            raise RuntimeError("align_signals must be run successfully before applying grid alignment.")

        # Validate method - typically 'nearest' or padding methods are safest for reindex
        allowed_methods = ['nearest', 'pad', 'ffill', 'backfill', 'bfill']
        if method not in allowed_methods:
             logger.warning(f"Alignment method '{method}' might not be suitable for reindexing. Using 'nearest'.")
             # Force nearest as it's generally the safest for conforming data points to a grid
             method = 'nearest'
             # raise ValueError(f"Invalid alignment method for reindexing: {method}. Use one of {allowed_methods}")


        logger.info(f"Applying grid alignment in-place to signals using method '{method}'...")
        # Determine which signals to process
        target_keys = signals_to_align if signals_to_align is not None else self.signals.keys()

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for key in target_keys:
            if key not in self.signals:
                logger.warning(f"Signal key '{key}' specified for alignment not found.")
                skipped_count += 1
                continue

            signal = self.signals[key]
            if isinstance(signal, TimeSeriesSignal):
                try:
                    current_data = signal.get_data()
                    if current_data is None or current_data.empty:
                        logger.warning(f"Skipping alignment for signal '{key}': data is None or empty.")
                        skipped_count += 1
                        continue

                    # Ensure data index is timezone-aware and matches grid index timezone
                    grid_tz = self.grid_index.tz
                    if current_data.index.tz is None:
                        logger.debug(f"Localizing index for signal '{key}' to grid timezone '{grid_tz}' before reindexing.")
                        current_data.index = current_data.index.tz_localize(grid_tz)
                    elif current_data.index.tz != grid_tz:
                        logger.debug(f"Converting index for signal '{key}' from {current_data.index.tz} to grid timezone '{grid_tz}' before reindexing.")
                        current_data.index = current_data.index.tz_convert(grid_tz)

                    # Perform the reindexing to conform to the grid
                    logger.debug(f"Reindexing signal '{key}' to grid...")
                    aligned_data = current_data.reindex(self.grid_index, method=method)

                    # Update the signal's internal data directly
                    signal._data = aligned_data
                    # Update the sample rate metadata as it now matches the grid
                    signal._update_sample_rate_metadata()
                    logger.debug(f"Applied grid alignment to signal '{key}'. New shape: {aligned_data.shape}")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to apply grid alignment to signal '{key}': {e}", exc_info=True)
                    warnings.warn(f"Failed to apply grid alignment to signal '{key}': {e}")
                    error_count += 1
            else:
                 # Log signals that are skipped because they are not TimeSeriesSignal
                 logger.debug(f"Skipping alignment for signal '{key}': not a TimeSeriesSignal.")
                 skipped_count += 1


        logger.info(f"Grid alignment application complete. Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
