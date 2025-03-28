"""
Signal collection for managing multiple signals.

This module defines the SignalCollection class, which serves as a container
for all signals in the framework and provides methods for adding, retrieving,
and managing signals.
"""

from typing import Dict, List, Any, Optional, Type, Tuple, Callable, Union
from .metadata import SignalMetadata
import warnings
import pandas as pd
import uuid

from .signal_data import SignalData
from .metadata import CollectionMetadata
from ..signal_types import SignalType

# Define standard factors of 1000 Hz
FACTORS_OF_1000 = [1, 2, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000]

# Define standard factors of 1000 Hz
FACTORS_OF_1000 = [1, 2, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000]

class SignalCollection:
    """
    Container for managing multiple signals with automatic indexing.
    
    SignalCollection serves as the central hub and single source of truth
    for all signals in the system, including imported signals, intermediate
    signals, and derived signals with base name indexing (e.g., 'ppg_0').
    """
    # Registry for multi-signal operations
    multi_signal_registry: Dict[str, Tuple[Callable, Type[SignalData]]] = {}
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a SignalCollection instance.
        
        Args:
            metadata: Optional dictionary with collection-level metadata
        """
        self.signals: Dict[str, SignalData] = {}
        
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
            import uuid
            signal.metadata.signal_id = str(uuid.uuid4())
            
        # Validate that time series signals have a proper timestamp index
        from ..signals.time_series_signal import TimeSeriesSignal
        if isinstance(signal, TimeSeriesSignal):
            self._validate_timestamp_index(signal)
            
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
        from ..signal_types import SignalType, SensorType, SensorModel, BodyPosition
        from ..utils import str_to_enum
        
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
        from ..signal_types import SensorType, SensorModel, BodyPosition
        from ..utils import str_to_enum
        
        # Update enum fields
        if "signal_type" in metadata_spec and isinstance(metadata_spec["signal_type"], str):
            from ..signal_types import SignalType
            signal.metadata.signal_type = str_to_enum(metadata_spec["signal_type"], SignalType)
            
        if "sensor_type" in metadata_spec and isinstance(metadata_spec["sensor_type"], str):
            signal.metadata.sensor_type = str_to_enum(metadata_spec["sensor_type"], SensorType)
            
        if "sensor_model" in metadata_spec and isinstance(metadata_spec["sensor_model"], str):
            signal.metadata.sensor_model = str_to_enum(metadata_spec["sensor_model"], SensorModel)
            
        if "body_position" in metadata_spec and isinstance(metadata_spec["body_position"], str):
            signal.metadata.body_position = str_to_enum(metadata_spec["body_position"], BodyPosition)
        
        # Initialize sensor_info if needed
        if signal.metadata.sensor_info is None:
            signal.metadata.sensor_info = {}
        
        # Update sensor_info
        if "sensor_info" in metadata_spec and isinstance(metadata_spec["sensor_info"], dict):
            signal.metadata.sensor_info.update(metadata_spec["sensor_info"])
        
        # Update other metadata fields
        metadata_fields = ["name", "sample_rate", "units", "start_time", "end_time"]
        for field in metadata_fields:
            if field in metadata_spec:
                setattr(signal.metadata, field, metadata_spec[field])
    
    
    def set_index_config(self, index_fields: List[str]) -> None:
        """
        Configure the multi-index fields for dataframe exports.
        
        Args:
            index_fields: List of metadata field names to use as index levels.
        
        Raises:
            ValueError: If any field is not a valid SignalMetadata attribute.
        """
        from dataclasses import fields
        valid_fields = {f.name for f in fields(SignalMetadata)}
        if not all(f in valid_fields for f in index_fields):
            raise ValueError(f"Invalid index fields: {set(index_fields) - valid_fields}")
        self.metadata.index_config = index_fields
    
    # Method removed since it's no longer needed - multi-index creation is now
    # integrated directly into the get_combined_dataframe method
    
    def _validate_timestamp_index(self, signal: SignalData) -> None:
        """
        Validate that a signal has a proper timestamp index.
        
        Args:
            signal: The signal to validate
            
        Raises:
            ValueError: If the signal doesn't have a DatetimeIndex
        """
        import pandas as pd
        import logging
        
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
            
        max_rate = max((s.get_sampling_rate() or 0) for s in self.signals.values())
        
        # If no valid rates, default to 100 Hz
        if max_rate <= 0:
            return 100.0
            
        # Return the largest factor of 1000 Hz ≤ the maximum rate
        return max(f for f in FACTORS_OF_1000 if f <= max_rate)
    
    def get_nearest_factor(self, rate):
        """
        Find the nearest factor of 1000 Hz to a given sample rate.
        
        Args:
            rate (float): The signal's sample rate in Hz.
        
        Returns:
            int: The closest factor from FACTORS_OF_1000.
        """
        if rate is None or rate <= 0:
            return 100.0  # Default value
            
        return min(FACTORS_OF_1000, key=lambda f: abs(f - rate))
    
    def get_reference_time(self, target_period):
        """
        Compute the reference timestamp for the grid.
        
        Args:
            target_period (pd.Timedelta): The period corresponding to the target sample rate.
        
        Returns:
            pd.Timestamp: The reference time aligned to the grid.
        """
        import pandas as pd
        
        min_times = [s.get_data().index.min() for s in self.signals.values() 
                    if s.get_data() is not None and len(s.get_data()) > 0]
        
        if not min_times:
            return pd.Timestamp("1970-01-01")
            
        min_time = min(min_times)
        delta_ms = (min_time - pd.Timestamp("1970-01-01")).total_seconds() * 1000
        floored_delta = (delta_ms // (target_period.total_seconds() * 1000)) * target_period
        return pd.Timestamp("1970-01-01") + floored_delta
        
    def compute_optimal_index(self, target_sample_rate: Optional[float] = None) -> pd.DatetimeIndex:
        """
        Use the index from the signal with the most data points as the target index.
        
        This avoids creating a synthetic dense grid that might be much larger than
        what's needed, which can slow down alignment operations.

        Args:
            target_sample_rate: Not used in this implementation but kept for compatibility.

        Returns:
            A pd.DatetimeIndex from the signal with the most data points.
        """
        import pandas as pd
        import logging
        from ..signals.time_series_signal import TimeSeriesSignal
        
        logger = logging.getLogger(__name__)

        # Filter for time-series signals
        time_series_signals = [s for s in self.signals.values() if isinstance(s, TimeSeriesSignal)]
        if not time_series_signals:
            raise ValueError("No time-series signals found in the collection")

        # Find the signal with the most data points (which is likely the highest frequency)
        target_signal = max(time_series_signals, key=lambda s: len(s.get_data()))
        target_index = target_signal.get_data().index
        
        signal_id = target_signal.metadata.signal_id if hasattr(target_signal.metadata, 'signal_id') else 'unknown'
        logger.info(f"Using index from signal {signal_id} with {len(target_index)} points as target index")
        
        return target_index

    def align_signals(self, target_sample_rate: Optional[float] = None, method: str = "nearest", 
                     preserve_original: bool = True, inplace: bool = True,
                     resample_strategy: str = 'nearest') -> 'SignalCollection':
        """
        Align time-series signals to a common timeline.
        
        For signals with a sampling rate higher than the target, applies downsampling.
        For signals with a sampling rate lower than the target, keeps the original sampling 
        rate but aligns to the common timeline (no upsampling).

        Args:
            target_sample_rate: If None, use the highest sample rate; otherwise, resample to this rate.
            method: Resampling method for signals that need downsampling.
            preserve_original: If True, when upsampling, only keeps values at original timestamps.
            inplace: If True, modify signals in place; if False, add resampled signals with new keys.
            resample_strategy: Strategy for odd rates: 'downsample', 'upsample', 'nearest', or None.
            
        Returns:
            The SignalCollection (self if inplace=True, a new copy if inplace=False)
        """
        import logging
        import time
        import pandas as pd
        
        logger = logging.getLogger(__name__)
        start_time = time.time()
        
        logger.info(f"Starting align_signals operation with target_sample_rate={target_sample_rate}, method={method}")
        
        # Validate resample_strategy
        valid_strategies = ['downsample', 'upsample', 'nearest', None]
        if resample_strategy not in valid_strategies:
            raise ValueError(f"Invalid resample_strategy: {resample_strategy}. Must be 'downsample', 'upsample', 'nearest', or None.")
        
        # Determine the target sample rate and period
        target_rate = self.get_target_sample_rate(target_sample_rate)
        target_period = pd.Timedelta(milliseconds=1000 / target_rate)
        
        # Get reference time for the grid
        ref_time = self.get_reference_time(target_period)
        
        # Identify the signal with the most data points to use as our target index
        target_index = self.compute_optimal_index(target_sample_rate)
        if len(target_index) == 0:
            logger.warning("No valid target index found, nothing to align")
            return self
            
        logger.info(f"Using target index with {len(target_index)} points, from {target_index[0]} to {target_index[-1]}")
        
        # Process each signal
        signal_count = 0
        ts_signal_count = 0
        
        from ..signals.time_series_signal import TimeSeriesSignal
        for key, signal in self.signals.items():
            signal_count += 1
            
            if isinstance(signal, TimeSeriesSignal):
                ts_signal_count += 1
                signal_size = len(signal.get_data())
                signal_type = signal.signal_type.name if hasattr(signal.signal_type, 'name') else signal.signal_type
                
                logger.info(f"Processing signal {ts_signal_count}/{signal_count}: {key} ({signal_type}) with {signal_size} rows")
                signal_start_time = time.time()
                
                signal_rate = signal.get_sampling_rate()
                
                # Handle signals with valid rates
                if signal_rate and signal_rate > 0:
                    if signal_rate in FACTORS_OF_1000:
                        # Rate is a factor of 1000 Hz; snap to grid directly
                        logger.info(f"Signal {key} has standard rate {signal_rate} Hz (factor of 1000). Snapping to grid.")
                        data = signal.snap_to_grid(target_period, ref_time)
                        
                        if inplace:
                            logger.debug(f"Snapping {key} to grid in-place")
                            signal._data = data
                            # Record the operation in metadata
                            from ..core.metadata import OperationInfo
                            signal.metadata.operations.append(OperationInfo("snap_to_grid", {"target_period": str(target_period)}))
                        else:
                            logger.debug(f"Snapping {key} to grid and storing as {key}_aligned")
                            new_key = f"{key}_aligned"
                            # Copy the signal with the snapped data
                            new_signal = signal.copy_with_data(data)
                            # Record the operation in metadata
                            from ..core.metadata import OperationInfo
                            new_signal.metadata.operations.append(OperationInfo("snap_to_grid", {"target_period": str(target_period)}))
                            self.add_signal(new_key, new_signal)
                        
                        # Log some statistics about data density after snapping
                        has_data = data[data.notna().any(axis=1)]
                        logger.debug(f"After snapping {key}: {len(has_data)}/{len(data)} points have data ({len(has_data)/len(data)*100:.1f}% density)")
                    else:
                        # Odd rate; determine appropriate factor of 1000 Hz
                        if resample_strategy == 'downsample':
                            new_rate = max(f for f in FACTORS_OF_1000 if f <= signal_rate)
                            logger.info(f"Downsampling signal with rate {signal_rate:.2f} Hz to {new_rate} Hz")
                        elif resample_strategy == 'upsample':
                            new_rate = min(f for f in FACTORS_OF_1000 if f >= signal_rate)
                            logger.info(f"Upsampling signal with rate {signal_rate:.2f} Hz to {new_rate} Hz")
                        elif resample_strategy == 'nearest':
                            new_rate = self.get_nearest_factor(signal_rate)
                            logger.info(f"Resampling signal with rate {signal_rate:.2f} Hz to nearest factor {new_rate} Hz")
                        else:
                            raise ValueError(f"Signal '{key}' has rate {signal_rate} Hz which is not a factor of 1000 Hz. "
                                           f"Set resample_strategy to handle odd rates.")
                        
                        # Resample and align
                        data = signal.resample_to_rate(new_rate, target_period, ref_time)
                        
                        if inplace:
                            logger.debug(f"Applying resampling in-place for {key}")
                            signal._data = data
                            # Record the operation in metadata
                            from ..core.metadata import OperationInfo
                            signal.metadata.operations.append(OperationInfo("resample_to_rate", {"new_rate": new_rate}))
                        else:
                            logger.debug(f"Resampling {key} and storing as {key}_aligned")
                            new_key = f"{key}_aligned"
                            # Copy the signal with the resampled data
                            new_signal = signal.copy_with_data(data)
                            # Record the operation in metadata
                            from ..core.metadata import OperationInfo
                            new_signal.metadata.operations.append(OperationInfo("resample_to_rate", {"new_rate": new_rate}))
                            self.add_signal(new_key, new_signal)
                else:
                    # For signals with unknown rates, apply the standard resampling
                    # Use merge_asof for efficient alignment with proper tolerance
                    parameters = {
                        "target_index": target_index, 
                        "method": method,
                        "preserve_original": preserve_original
                    }
                    
                    if inplace:
                        logger.debug(f"Applying alignment in-place for {key}")
                        signal.apply_operation("resample", inplace=True, **parameters)
                        # Record the operation in metadata
                        from ..core.metadata import OperationInfo
                        signal.metadata.operations.append(OperationInfo("align", {"target_signal": "common_index"}))
                    else:
                        logger.debug(f"Applying alignment for {key} and storing as {key}_aligned")
                        aligned_signal = signal.apply_operation("resample", **parameters)
                        # Record the operation in metadata
                        from ..core.metadata import OperationInfo
                        aligned_signal.metadata.operations.append(OperationInfo("align", {"target_signal": "common_index"}))
                        new_key = f"{key}_aligned"
                        self.add_signal(new_key, aligned_signal)
                
                logger.info(f"Completed processing {key} in {time.time() - signal_start_time:.2f} seconds")
        
        # Store parameters for combined dataframe
        self.target_rate = target_rate
        self.ref_time = ref_time
        
        logger.info(f"align_signals operation complete: processed {ts_signal_count} signals in {time.time() - start_time:.2f} seconds")
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
            pd.DataFrame: Combined dataframe with aligned signals and consistent grid.
        """
        import pandas as pd
        import numpy as np
        import logging
        
        logger = logging.getLogger(__name__)
        
        if not self.signals:
            logger.warning("No signals to combine")
            return pd.DataFrame()
        
        logger.info(f"Creating combined dataframe from {len(self.signals)} signals")
        
        # Create a proper grid index based on aligned signals
        # If we have target_rate and ref_time from align_signals, use those
        if hasattr(self, 'target_rate') and hasattr(self, 'ref_time'):
            logger.info(f"Using alignment grid with rate {self.target_rate} Hz")
            target_period = pd.Timedelta(milliseconds=1000 / self.target_rate)
            
            # Find time range across all signals
            min_times = []
            max_times = []
            
            for signal in self.signals.values():
                data = signal.get_data()
                if data is not None and len(data) > 0:
                    # For each signal, find the first and last non-NaN values
                    non_null_rows = data.dropna(how='all')
                    if len(non_null_rows) > 0:
                        min_times.append(non_null_rows.index.min())
                        max_times.append(non_null_rows.index.max())
            
            if not min_times or not max_times:
                logger.warning("No valid timestamps found in signals")
                return pd.DataFrame()
            
            # Use earliest start and latest end time to preserve all data points
            earliest_start = min(min_times)
            latest_end = max(max_times)
            
            logger.info(f"Full time range with data: {earliest_start} to {latest_end}")
            
            # Generate a regular grid from ref_time through the full range
            # Make sure start is exactly on grid by calculating periods from ref_time
            periods_to_start = np.floor((earliest_start - self.ref_time) / target_period)
            periods_to_end = np.ceil((latest_end - self.ref_time) / target_period)
            
            grid_start = self.ref_time + (periods_to_start * target_period)
            grid_end = self.ref_time + (periods_to_end * target_period)
            
            grid_index = pd.date_range(
                start=grid_start, 
                end=grid_end, 
                freq=target_period
            )
            
            logger.info(f"Created regular grid with {len(grid_index)} points from {grid_start} to {grid_end}")
            
            # Create empty DataFrame with the regular grid
            combined_df = pd.DataFrame(index=grid_index)
        else:
            # If align_signals wasn't called, find common points in the data
            logger.info("align_signals not called; using existing signal timestamps")
            
            # Collect all unique timestamps from all signals
            all_timestamps = set()
            for signal in self.signals.values():
                data = signal.get_data()
                if data is not None and len(data) > 0:
                    all_timestamps.update(data.index)
            
            if not all_timestamps:
                logger.warning("No timestamps found in signals")
                return pd.DataFrame()
                
            # Sort timestamps for consistent order
            grid_index = pd.DatetimeIndex(sorted(all_timestamps))
            combined_df = pd.DataFrame(index=grid_index)
            
            logger.info(f"Created index with {len(grid_index)} unique timestamps")
            
        # Now add each signal's data to the combined dataframe
        if hasattr(self.metadata, 'index_config') and self.metadata.index_config:
            # Use multi-index for hierarchical columns
            columns_data = {}
            multi_index_tuples = []
            
            for key, signal in self.signals.items():
                try:
                    # Get the signal data
                    signal_df = signal.get_data()
                    if signal_df is None or len(signal_df) == 0:
                        continue
                        
                    # For each column in the signal, prepare the metadata tuple
                    for col_name in signal_df.columns:
                        # Create metadata values for multi-index
                        metadata_values = []
                        for field in self.metadata.index_config:
                            value = getattr(signal.metadata, field, None)
                            if value is None:
                                value = key if field == 'name' else "N/A"
                            # Convert enum to string
                            if hasattr(value, 'name'):
                                value = value.name
                            metadata_values.append(value)
                        
                        # Add column name to metadata and create tuple
                        metadata_values.append(col_name)
                        tuple_key = tuple(metadata_values)
                        multi_index_tuples.append(tuple_key)
                        
                        # Reindex to align with our grid, without filling NaN values
                        aligned_series = signal_df[col_name].reindex(combined_df.index)
                        columns_data[tuple_key] = aligned_series
                        
                except Exception as e:
                    logger.error(f"Error processing signal {key}: {e}")
                    import warnings
                    warnings.warn(f"Error processing signal {key}: {str(e)}")
            
            # Create multi-index columns
            if columns_data:
                multi_idx = pd.MultiIndex.from_tuples(
                    multi_index_tuples,
                    names=self.metadata.index_config + ['column']
                )
                
                # Create the combined DataFrame with multi-index columns
                combined_df = pd.DataFrame(
                    {col: data for col, data in columns_data.items()},
                    index=combined_df.index
                )
                combined_df.columns = multi_idx
                
                # Remove rows where all values are NaN
                non_empty_rows = combined_df.notna().any(axis=1)
                combined_df = combined_df.loc[non_empty_rows]
                
                # Log statistics
                logger.info(f"Combined dataframe has {len(combined_df)} rows and {len(combined_df.columns)} columns")
                non_null_counts = combined_df.count()
                if len(combined_df) > 0:
                    logger.info(f"Column data density: min={non_null_counts.min()/len(combined_df)*100:.1f}%, " +
                               f"max={non_null_counts.max()/len(combined_df)*100:.1f}%, " +
                               f"mean={non_null_counts.mean()/len(combined_df)*100:.1f}%")
        else:
            # Simple case without multi-index
            for key, signal in self.signals.items():
                try:
                    signal_df = signal.get_data()
                    if signal_df is None or len(signal_df) == 0:
                        continue
                    
                    # Add each column with appropriate prefix
                    if len(signal_df.columns) == 1:
                        col = signal_df.columns[0]
                        combined_df[key] = signal_df[col].reindex(combined_df.index)
                    else:
                        for col in signal_df.columns:
                            combined_df[f"{key}_{col}"] = signal_df[col].reindex(combined_df.index)
                            
                except Exception as e:
                    logger.error(f"Error processing signal {key}: {e}")
                    import warnings
                    warnings.warn(f"Error processing signal {key}: {str(e)}")
            
            # Remove rows where all values are NaN
            non_empty_rows = combined_df.notna().any(axis=1)
            combined_df = combined_df.loc[non_empty_rows]
            
            logger.info(f"Combined dataframe has {len(combined_df)} rows and {len(combined_df.columns)} columns")
        
        if len(combined_df) == 0:
            logger.warning("No data rows in combined dataframe")
            
        return combined_df

    
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
            ValueError: If the operation is not found or input signals don't exist
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
        from ..core.metadata import OperationInfo
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
        import os
        import glob
        import warnings
        
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
