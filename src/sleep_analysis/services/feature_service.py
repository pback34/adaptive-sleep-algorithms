"""
Feature service for handling feature extraction operations.

This service encapsulates all logic related to generating epoch grids,
applying multi-signal operations (especially feature extraction),
and combining features.
"""

import time
import logging
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
import pandas as pd

from ..signals.time_series_signal import TimeSeriesSignal
from ..features.feature import Feature

if TYPE_CHECKING:
    from ..core.signal_collection import SignalCollection

logger = logging.getLogger(__name__)


class FeatureService:
    """
    Service for feature extraction and epoch-based operations.

    This service handles:
    - Generating epoch grids for feature extraction
    - Applying multi-signal operations (e.g., feature extraction)
    - Combining features from multiple sources
    - Metadata propagation for features
    """

    def generate_epoch_grid(
        self,
        collection: 'SignalCollection',
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None
    ) -> None:
        """
        Calculate and store the global epoch grid on the collection.

        Uses `epoch_grid_config` from `CollectionMetadata` and the time range
        of `time_series_signals` to create a common `epoch_grid_index`.

        Args:
            collection: The SignalCollection to generate epoch grid for
            start_time: Optional override for the grid start time
            end_time: Optional override for the grid end time

        Raises:
            RuntimeError: If `epoch_grid_config` is missing/invalid or no time-series signals found
            ValueError: If start/end time overrides are invalid
        """
        logger.info("Starting global epoch grid calculation...")
        op_start_time = time.time()
        collection._epoch_grid_calculated = False

        # Get config
        config = collection.metadata.epoch_grid_config
        if not config or "window_length" not in config or "step_size" not in config:
            raise RuntimeError("Missing or incomplete 'epoch_grid_config' in collection metadata. Cannot generate epoch grid.")

        try:
            window_length = pd.Timedelta(config["window_length"])
            step_size = pd.Timedelta(config["step_size"])
            if window_length <= pd.Timedelta(0) or step_size <= pd.Timedelta(0):
                raise ValueError("window_length and step_size must be positive.")
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid epoch_grid_config parameters: {e}") from e

        collection.global_epoch_window_length = window_length
        collection.global_epoch_step_size = step_size
        logger.info(f"Using global epoch parameters: window={window_length}, step={step_size}")

        # Determine time range
        min_times = []
        max_times = []
        collection_tz = pd.Timestamp('now', tz=collection.metadata.timezone).tz

        for signal in collection.time_series_signals.values():
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

        # Apply overrides
        try:
            grid_start = pd.Timestamp(start_time, tz=collection_tz) if start_time else min(min_times)
            grid_end = pd.Timestamp(end_time, tz=collection_tz) if end_time else max(max_times)

            # Ensure overrides are timezone-aware consistent with collection
            if grid_start.tz is None:
                grid_start = grid_start.tz_localize(collection_tz)
            else:
                grid_start = grid_start.tz_convert(collection_tz)

            if grid_end.tz is None:
                grid_end = grid_end.tz_localize(collection_tz)
            else:
                grid_end = grid_end.tz_convert(collection_tz)

        except Exception as e:
            raise ValueError(f"Invalid start_time or end_time override for epoch grid: {e}") from e

        if grid_start >= grid_end:
            raise ValueError(f"Epoch grid start time ({grid_start}) must be before end time ({grid_end}).")

        logger.info(f"Epoch grid time range: {grid_start} to {grid_end}")

        # Generate epoch index
        try:
            collection.epoch_grid_index = pd.date_range(
                start=grid_start,
                end=grid_end,
                freq=step_size,
                name='epoch_start_time',
                inclusive='left'
            )

            # Filter out any start times where the window would begin after the grid ends
            collection.epoch_grid_index = collection.epoch_grid_index[collection.epoch_grid_index <= grid_end]

            if collection.epoch_grid_index.empty:
                logger.warning("Generated epoch grid index is empty.")

            # Ensure timezone matches collection
            collection.epoch_grid_index = collection.epoch_grid_index.tz_convert(collection_tz) if collection.epoch_grid_index.tz is not None else collection.epoch_grid_index.tz_localize(collection_tz)

            logger.info(f"Calculated epoch_grid_index with {len(collection.epoch_grid_index)} points.")

        except Exception as e:
            logger.error(f"Error creating date_range for epoch grid index: {e}", exc_info=True)
            collection.epoch_grid_index = None
            raise RuntimeError(f"Failed to generate epoch grid index: {e}") from e

        collection._epoch_grid_calculated = True
        logger.info(f"Epoch grid calculation completed in {time.time() - op_start_time:.2f} seconds.")

    def apply_multi_signal_operation(
        self,
        collection: 'SignalCollection',
        operation_name: str,
        input_signal_keys: List[str],
        parameters: Dict[str, Any]
    ) -> Union[TimeSeriesSignal, Feature]:
        """
        Apply an operation that takes multiple signals as input and produces a single output.

        Args:
            collection: The SignalCollection containing the signals
            operation_name: Name of the operation (e.g., "feature_statistics")
            input_signal_keys: List of keys for the input TimeSeriesSignals
            parameters: Dictionary of parameters for the operation

        Returns:
            The resulting Feature or TimeSeriesSignal object

        Raises:
            ValueError: If operation not found, inputs invalid, or prerequisites not met
            RuntimeError: If the operation execution fails
        """
        logger.info(f"Applying multi-signal operation '{operation_name}' to inputs: {input_signal_keys}")

        if operation_name not in collection.multi_signal_registry:
            raise ValueError(f"Multi-signal operation '{operation_name}' not found in registry.")

        operation_func, output_class = collection.multi_signal_registry[operation_name]

        # Input resolution and validation
        input_signals: List[TimeSeriesSignal] = []
        for key in input_signal_keys:
            try:
                signal = collection.get_time_series_signal(key)
                input_signals.append(signal)
            except KeyError:
                raise ValueError(f"Input TimeSeriesSignal key '{key}' not found for operation '{operation_name}'.")

        if not input_signals:
            raise ValueError(f"No valid input TimeSeriesSignals resolved for operation '{operation_name}'.")

        # Prerequisite checks (specific to feature extraction)
        is_feature_op = issubclass(output_class, Feature)
        if is_feature_op:
            if not collection._epoch_grid_calculated or collection.epoch_grid_index is None or collection.epoch_grid_index.empty:
                raise RuntimeError(f"Cannot execute feature operation '{operation_name}': generate_epoch_grid must be run successfully first.")

        # Function execution
        try:
            logger.debug(f"Executing operation function '{operation_func.__name__}'...")
            if is_feature_op:
                # Make a copy to avoid modifying the original dict
                params_copy = parameters.copy()
                # Remove grid/global params if accidentally in parameters
                params_copy.pop('epoch_grid_index', None)
                params_copy.pop('global_epoch_window_length', None)
                params_copy.pop('global_epoch_step_size', None)

                # Call feature function with explicit global args
                result_object = operation_func(
                    signals=input_signals,
                    epoch_grid_index=collection.epoch_grid_index,
                    parameters=params_copy,
                    global_window_length=collection.global_epoch_window_length,
                    global_step_size=collection.global_epoch_step_size
                )
            else:
                # For non-feature ops, pass parameters as before
                result_object = operation_func(signals=input_signals, **parameters)

            logger.debug(f"Operation function '{operation_func.__name__}' completed.")
        except Exception as e:
            logger.error(f"Error executing multi-signal operation function '{operation_func.__name__}': {e}", exc_info=True)
            raise RuntimeError(f"Execution of operation '{operation_name}' failed.") from e

        # Result validation
        if not isinstance(result_object, output_class):
            raise TypeError(f"Operation '{operation_name}' returned unexpected type {type(result_object).__name__}. Expected {output_class.__name__}.")

        # Metadata propagation (for Feature outputs)
        if isinstance(result_object, Feature):
            self._propagate_metadata_to_feature(collection, result_object, input_signals, operation_name)

        return result_object

    def _propagate_metadata_to_feature(
        self,
        collection: 'SignalCollection',
        feature: Feature,
        input_signals: List[TimeSeriesSignal],
        operation_name: str
    ) -> None:
        """
        Propagate metadata from input signals to feature based on collection config.

        Args:
            collection: The SignalCollection
            feature: The Feature to propagate metadata to
            input_signals: The input TimeSeriesSignals
            operation_name: Name of the operation (for logging)
        """
        logger.debug(f"Propagating metadata for Feature result of '{operation_name}'...")
        feature_meta = feature.metadata
        fields_to_propagate = collection.metadata.feature_index_config

        if not fields_to_propagate:
            return

        if len(input_signals) == 1:
            # Single input: Copy directly
            source_meta = input_signals[0].metadata
            for field in fields_to_propagate:
                if hasattr(source_meta, field) and hasattr(feature_meta, field):
                    value = getattr(source_meta, field)
                    setattr(feature_meta, field, value)
                    logger.debug(f"  Propagated '{field}' = {value} (from single source)")
                elif hasattr(feature_meta, field):
                    logger.debug(f"  Field '{field}' exists in FeatureMetadata but not in source TimeSeriesMetadata. Skipping propagation.")

        elif len(input_signals) > 1:
            # Multiple inputs: Check for common values
            for field in fields_to_propagate:
                if hasattr(feature_meta, field):
                    values = set()
                    all_sources_have_field = True
                    for source_signal in input_signals:
                        if hasattr(source_signal.metadata, field):
                            values.add(getattr(source_signal.metadata, field))
                        else:
                            all_sources_have_field = False
                            logger.debug(f"  Source signal '{source_signal.metadata.name}' missing field '{field}' for propagation.")
                            break

                    if not all_sources_have_field:
                        logger.debug(f"  Field '{field}' not present in all source TimeSeriesSignals. Setting to None.")
                        setattr(feature_meta, field, None)
                    elif len(values) == 1:
                        # All sources have the same value
                        common_value = values.pop()
                        setattr(feature_meta, field, common_value)
                        logger.debug(f"  Propagated common '{field}' = {common_value} (from {len(input_signals)} sources)")
                    else:
                        # Sources have different values
                        logger.debug(f"  Field '{field}' has different values across sources: {values}. Setting to None.")
                        setattr(feature_meta, field, None)

    def combine_features(
        self,
        collection: 'SignalCollection',
        inputs: List[str]
    ) -> pd.DataFrame:
        """
        Combine multiple Feature objects into a single DataFrame.

        Args:
            collection: The SignalCollection containing the features
            inputs: List of feature keys to combine

        Returns:
            Combined DataFrame with all features

        Raises:
            ValueError: If inputs are invalid or features have incompatible indices
        """
        logger.info(f"Combining features: {inputs}")
        start_time = time.time()

        if not inputs:
            raise ValueError("No inputs specified for combine_features.")

        # Collect features
        features_to_combine = []
        for key in inputs:
            try:
                feature = collection.get_feature(key)
                features_to_combine.append(feature)
            except KeyError:
                logger.warning(f"Feature key '{key}' not found, skipping.")

        if not features_to_combine:
            raise ValueError("No valid features found to combine.")

        # Get dataframes
        dfs = [f.get_data() for f in features_to_combine]

        # Combine using outer join on index
        combined_df = pd.concat(dfs, axis=1, join='outer')

        logger.info(f"Combined {len(dfs)} features into dataframe with shape {combined_df.shape} in {time.time() - start_time:.2f} seconds.")

        return combined_df
