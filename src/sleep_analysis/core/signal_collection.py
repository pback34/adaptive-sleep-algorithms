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
# Note: FeatureSignal was removed, Feature is now used instead
from ..utils import str_to_enum

# Import service classes for refactored architecture
from .repositories import SignalRepository
from .services import (
    SignalQueryService,
    MetadataManager,
    AlignmentGridService,
    EpochGridService,
    AlignmentExecutor,
    SignalCombinationService,
    OperationExecutor,
    DataImportService,
    SignalSummaryReporter
)
from .models import AlignmentGridState, EpochGridState, CombinationResult

import functools # Added for decorator
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

        # Initialize repository (will create dictionaries)
        self._repository = SignalRepository(
            metadata_handler=self.metadata_handler,
            collection_timezone=self.metadata.timezone
        )

        # Use repository's dictionaries as collection's dictionaries (single source of truth)
        self.time_series_signals = self._repository.time_series_signals
        self.features = self._repository.features

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

        # Initialize service classes (Phase 5 refactoring)
        self._query_service = SignalQueryService(repository=self._repository)
        self._metadata_manager = MetadataManager(metadata_handler=self.metadata_handler)
        self._alignment_grid_service = AlignmentGridService(repository=self._repository)
        self._epoch_grid_service = EpochGridService(repository=self._repository, collection_metadata=self.metadata)
        self._alignment_executor = AlignmentExecutor(repository=self._repository, alignment_grid_service=self._alignment_grid_service)
        # Combination service needs metadata and state objects (states initialized during operations)
        self._combination_service = None  # Will be created when needed with current state
        # Operation executor needs callbacks and registries
        self._operation_executor = OperationExecutor(
            collection_op_registry=self.collection_operation_registry,
            multi_signal_registry=self.multi_signal_registry,
            get_time_series_signal=self._repository.get_time_series_signal,
            get_feature=self._repository.get_feature,
            add_time_series_signal=self._repository.add_time_series_signal,
            add_feature=self._repository.add_feature,
            epoch_state=None,  # Will be updated when epoch grid is calculated
            feature_index_config=self.metadata.feature_index_config,
            global_epoch_window_length=self.global_epoch_window_length,
            global_epoch_step_size=self.global_epoch_step_size
        )
        self._import_service = DataImportService(
            add_time_series_signal=self._repository.add_time_series_signal
        )
        self._summary_reporter = SignalSummaryReporter()

    def _get_combination_service(self) -> SignalCombinationService:
        """
        Helper to create/update SignalCombinationService with current state.

        Returns:
            SignalCombinationService configured with current alignment and epoch states
        """
        # Create alignment state if calculated
        alignment_state = None
        if self._alignment_params_calculated:
            alignment_state = AlignmentGridState(
                target_rate=self.target_rate,
                reference_time=self.ref_time,
                grid_index=self.grid_index,
                merge_tolerance=self._merge_tolerance,
                is_calculated=self._alignment_params_calculated
            )

        # Create epoch state if calculated
        epoch_state = None
        if self._epoch_grid_calculated:
            epoch_state = EpochGridState(
                epoch_grid_index=self.epoch_grid_index,
                window_length=self.global_epoch_window_length,
                step_size=self.global_epoch_step_size,
                is_calculated=self._epoch_grid_calculated
            )

        return SignalCombinationService(
            metadata=self.metadata,
            alignment_state=alignment_state,
            epoch_state=epoch_state
        )

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
        # Delegate to repository (Phase 5 refactoring)
        self._repository.add_time_series_signal(key, signal)

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
        # Delegate to repository (Phase 5 refactoring)
        self._repository.add_feature(key, feature)

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
        # Delegate to repository (Phase 5 refactoring)
        return self._repository.add_signal_with_base_name(base_name, signal)

    def get_time_series_signal(self, key: str) -> TimeSeriesSignal:
        """Retrieve a TimeSeriesSignal by its key."""
        # Delegate to repository (Phase 5 refactoring)
        return self._repository.get_time_series_signal(key)

    def get_feature(self, key: str) -> Feature:
        """Retrieve a Feature object by its key."""
        # Delegate to repository (Phase 5 refactoring)
        return self._repository.get_feature(key)

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
        # Delegate to repository (Phase 5 refactoring)
        return self._repository.get_by_key(key)

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
        # Delegate to query service (Phase 5 refactoring)
        return self._query_service.get_signals(
            input_spec=input_spec,
            signal_type=signal_type,
            feature_type=feature_type,
            criteria=criteria,
            base_name=base_name
        )

    def _process_enum_criteria(self, criteria_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to convert string enum values in criteria to Enum objects."""
        # Delegate to query service (Phase 5 refactoring)
        return self._query_service._process_enum_criteria(criteria_dict)

    def _matches_criteria(self, signal: Union[TimeSeriesSignal, Feature], criteria: Dict[str, Any]) -> bool:
        """Check if a TimeSeriesSignal or Feature matches all criteria."""
        # Delegate to query service (Phase 5 refactoring)
        return self._query_service._matches_criteria(signal, criteria)

    def update_time_series_metadata(self, signal: TimeSeriesSignal, metadata_spec: Dict[str, Any]) -> None:
        """Update a TimeSeriesSignal's metadata."""
        # Delegate to metadata manager (Phase 5 refactoring)
        self._metadata_manager.update_time_series_metadata(signal, metadata_spec)

    def update_feature_metadata(self, feature: Feature, metadata_spec: Dict[str, Any]) -> None:
        """Update a Feature's metadata."""
        # Delegate to metadata manager (Phase 5 refactoring)
        self._metadata_manager.update_feature_metadata(feature, metadata_spec)

    def set_index_config(self, index_fields: List[str]) -> None:
        """Configure the multi-index fields for combined *time-series* exports."""
        # Delegate to metadata manager (Phase 5 refactoring)
        self._metadata_manager.validate_time_series_metadata_spec(index_fields)
        self.metadata.index_config = index_fields
        logger.info(f"Set time-series index_config to: {index_fields}")

    def set_feature_index_config(self, index_fields: List[str]) -> None:
        """Configure the multi-index fields for combined *feature* matrix exports."""
        # Delegate to metadata manager (Phase 5 refactoring)
        # Note: MetadataManager validates feature fields, but we're more lenient here
        # to allow propagated TimeSeriesMetadata fields
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
        # Delegate to alignment grid service (Phase 5 refactoring)
        logger.info(f"Starting alignment grid parameter calculation with target_sample_rate={target_sample_rate}")
        start_time = time.time()
        self._alignment_params_calculated = False # Reset flag

        alignment_state = self._alignment_grid_service.generate_alignment_grid(
            target_sample_rate=target_sample_rate
        )

        # Unpack state into collection attributes
        self.target_rate = alignment_state.target_rate
        self.ref_time = alignment_state.reference_time
        self.grid_index = alignment_state.grid_index
        self._merge_tolerance = alignment_state.merge_tolerance
        self._alignment_params_calculated = alignment_state.is_calculated

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
        # Delegate to epoch grid service (Phase 5 refactoring)
        logger.info("Starting global epoch grid calculation...")
        op_start_time = time.time()
        self._epoch_grid_calculated = False # Reset flag

        epoch_state = self._epoch_grid_service.generate_epoch_grid(
            start_time=start_time,
            end_time=end_time
        )

        # Unpack state into collection attributes
        self.epoch_grid_index = epoch_state.epoch_grid_index
        self.global_epoch_window_length = epoch_state.window_length
        self.global_epoch_step_size = epoch_state.step_size
        self._epoch_grid_calculated = epoch_state.is_calculated

        # Update operation executor with epoch state (Phase 5 refactoring)
        self._operation_executor.epoch_state = epoch_state
        self._operation_executor.global_epoch_window_length = self.global_epoch_window_length
        self._operation_executor.global_epoch_step_size = self.global_epoch_step_size

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
        # Delegate to OperationExecutor (Phase 5 refactoring)
        return self._operation_executor.apply_multi_signal_operation(
            operation_name=operation_name,
            input_signal_keys=input_signal_keys,
            parameters=parameters
        )


    # --- Collection Operation Dispatch ---

    def apply_operation(self, operation_name: str, **parameters: Any) -> Any:  # type: ignore
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
        # Delegate to OperationExecutor (Phase 5 refactoring)
        return self._operation_executor.apply_collection_operation(
            operation_name=operation_name,
            collection_instance=self,
            **parameters
        )

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
        # Delegate to alignment executor (Phase 5 refactoring)
        if not self._alignment_params_calculated or self.grid_index is None or self.grid_index.empty:
            logger.error("Cannot apply grid alignment: generate_alignment_grid must be run successfully first.")
            raise RuntimeError("generate_alignment_grid must be run successfully before applying grid alignment.")

        alignment_state = AlignmentGridState(
            target_rate=self.target_rate,
            reference_time=self.ref_time,
            grid_index=self.grid_index,
            merge_tolerance=self._merge_tolerance,
            is_calculated=self._alignment_params_calculated
        )

        logger.info(f"Applying grid alignment in-place to TimeSeriesSignals using method '{method}'...")
        start_time = time.time()

        processed_count = self._alignment_executor.apply_grid_alignment(
            method=method,
            signals_to_align=signals_to_align
        )

        logger.info(f"Grid alignment application finished in {time.time() - start_time:.2f} seconds. Processed: {processed_count}")

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
        # Delegate to SignalCombinationService (Phase 5 refactoring)
        service = self._get_combination_service()
        return service._perform_concatenation(
            aligned_dfs=aligned_dfs,
            grid_index=grid_index,
            is_feature=is_feature,
            time_series_signals=self.time_series_signals if not is_feature else None,
            features=self.features if is_feature else None
        )

    def _get_current_alignment_params(self, method_used: str) -> Dict[str, Any]:
        """Helper to gather current alignment parameters for storage."""
        # Delegate to SignalCombinationService (Phase 5 refactoring)
        service = self._get_combination_service()
        return service._get_alignment_params(method_used)

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
        # Delegate to SignalCombinationService (Phase 5 refactoring)
        service = self._get_combination_service()
        result = service.combine_aligned_signals(time_series_signals=self.time_series_signals)

        # Unpack result into collection attributes
        self._aligned_dataframe = result.dataframe
        self._aligned_dataframe_params = result.params

    @register_collection_operation("combine_features")
    def combine_features(self, inputs: List[str], feature_index_config: Optional[List[str]] = None) -> None:
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
        # Delegate to SignalCombinationService (Phase 5 refactoring)
        service = self._get_combination_service()
        result = service.combine_features(
            features=self.features,
            inputs=inputs,
            feature_index_config=feature_index_config
        )

        # Unpack result into collection attributes
        self._combined_feature_matrix = result.dataframe


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
        # Delegate to OperationExecutor (Phase 5 refactoring)
        return self._operation_executor.apply_and_store_operation(
            signal_key=signal_key,
            operation_name=operation_name,
            parameters=parameters,
            output_key=output_key
        )

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
        # Delegate to OperationExecutor (Phase 5 refactoring)
        return self._operation_executor.apply_operation_to_signals(
            signal_keys=signal_keys,
            operation_name=operation_name,
            parameters=parameters,
            inplace=inplace,
            output_keys=output_keys
        )

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
        # Delegate to DataImportService (Phase 5 refactoring)
        return self._import_service.import_signals_from_source(
            importer_instance=importer_instance,
            source=source,
            spec=spec
        )


    def add_imported_signals(self, signals: List[TimeSeriesSignal], base_name: str,
                           start_index: int = 0) -> List[str]:
        """Add imported TimeSeriesSignals to the collection with sequential indexing."""
        # Delegate to DataImportService (Phase 5 refactoring)
        return self._import_service.add_imported_signals(
            signals=signals,
            base_name=base_name,
            start_index=start_index
        )

    def _format_summary_cell(self, x, col_name):
        """Helper function to format a single cell for the summary DataFrame printout."""
        # Delegate to SignalSummaryReporter (Phase 5 refactoring)
        return self._summary_reporter._format_summary_cell(x, col_name)

    @register_collection_operation("summarize_signals")
    def summarize_signals(self, fields_to_include: Optional[List[str]] = None, print_summary: bool = True) -> Optional[pd.DataFrame]:
        """Generates a summary table of TimeSeriesSignals and Features."""
        # Delegate to SignalSummaryReporter (Phase 5 refactoring)
        summary_df = self._summary_reporter.summarize_signals(
            time_series_signals=self.time_series_signals,
            features=self.features,
            fields_to_include=fields_to_include,
            print_summary=print_summary
        )

        # Store the result in collection attributes for backward compatibility
        self._summary_dataframe = self._summary_reporter.get_summary_dataframe()
        self._summary_dataframe_params = self._summary_reporter.get_summary_params()

        return summary_df


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
    # Import algorithm operations
    from ..operations.algorithm_ops import (
        random_forest_sleep_staging,
        evaluate_sleep_staging
    )
    # Import the Feature class (output type for these operations)
    from ..features.feature import Feature

    SignalCollection.multi_signal_registry.update({
        "feature_statistics": (compute_feature_statistics, Feature),
        "compute_sleep_stage_mode": (compute_sleep_stage_mode, Feature),
        "compute_hrv_features": (compute_hrv_features, Feature),
        "compute_movement_features": (compute_movement_features, Feature),
        "compute_correlation_features": (compute_correlation_features, Feature),
        "random_forest_sleep_staging": (random_forest_sleep_staging, Feature),
        "evaluate_sleep_staging": (evaluate_sleep_staging, Feature),
    })
    logger.debug("Registered multi-signal operations (features, algorithms).")
except ImportError as e:
    logger.warning(f"Could not import or register operations: {e}")


# Clean up temporary variables from the global scope of the module
# Check if variables exist before deleting
if '_method_name' in locals() or '_method_name' in globals(): del _method_name
if '_method_obj' in locals() or '_method_obj' in globals(): del _method_obj
if '_op_name' in locals() or '_op_name' in globals(): del _op_name
