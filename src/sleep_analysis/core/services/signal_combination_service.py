"""
Signal combination service for combining time-series signals and features.

This module provides the SignalCombinationService class, which handles:
- Combining aligned time-series signals into a single DataFrame
- Combining features into a feature matrix
- Managing MultiIndex column structures
- Concatenating dataframes with proper alignment
"""

# Standard library imports
import logging
import time
from typing import Dict, List, Optional, Any
from enum import Enum

# Third-party imports
import pandas as pd

# Local application imports
from ..metadata import CollectionMetadata
from ..models import AlignmentGridState, EpochGridState, CombinationResult
from ...signals.time_series_signal import TimeSeriesSignal
from ...features.feature import Feature

# Initialize logger for the module
logger = logging.getLogger(__name__)


class SignalCombinationService:
    """
    Service for combining time-series signals and features into dataframes.

    This service handles:
    - Combining aligned time-series signals into a single DataFrame
    - Combining features into a feature matrix with MultiIndex columns
    - Managing index configuration and metadata
    - Validating alignment and epoch grids before combination

    Example:
        >>> service = SignalCombinationService(
        ...     metadata=collection_metadata,
        ...     alignment_state=alignment_state,
        ...     epoch_state=epoch_state
        ... )
        >>> result = service.combine_aligned_signals(time_series_signals)
        >>> df = result.dataframe
    """

    def __init__(
        self,
        metadata: CollectionMetadata,
        alignment_state: Optional[AlignmentGridState] = None,
        epoch_state: Optional[EpochGridState] = None
    ):
        """
        Initialize the SignalCombinationService.

        Args:
            metadata: Collection metadata containing index configurations
            alignment_state: Optional alignment grid state for time-series combination
            epoch_state: Optional epoch grid state for feature combination
        """
        self.metadata = metadata
        self.alignment_state = alignment_state
        self.epoch_state = epoch_state

    def combine_aligned_signals(
        self,
        time_series_signals: Dict[str, TimeSeriesSignal]
    ) -> CombinationResult:
        """
        Combines aligned TimeSeriesSignals into a single DataFrame.

        Retrieves data from all non-temporary time-series signals, validates alignment,
        and concatenates them using outer join and reindexing. The result is stored
        with combination parameters for reference.

        Args:
            time_series_signals: Dictionary of TimeSeriesSignal objects to combine

        Returns:
            CombinationResult containing the combined DataFrame and parameters

        Raises:
            RuntimeError: If alignment grid hasn't been calculated or is invalid
            RuntimeError: If errors occur while accessing signal data

        Example:
            >>> result = service.combine_aligned_signals(signals_dict)
            >>> combined_df = result.dataframe
            >>> print(combined_df.shape)
        """
        # Validate alignment state
        if not self.alignment_state or not self.alignment_state.is_valid():
            logger.error("Cannot combine signals: alignment grid must be calculated first.")
            raise RuntimeError("Alignment grid must be calculated before combining signals.")

        if self.alignment_state.grid_index is None or self.alignment_state.grid_index.empty:
            logger.error("Cannot combine signals: grid_index is None or empty.")
            raise RuntimeError("Alignment grid_index is invalid.")

        if not time_series_signals:
            logger.warning("No TimeSeriesSignals in collection to combine.")
            empty_df = pd.DataFrame(index=self.alignment_state.grid_index)
            params = self._get_alignment_params("outer_join_reindex")
            return CombinationResult(dataframe=empty_df, params=params, is_feature_matrix=False)

        logger.info("Combining aligned TimeSeriesSignals using outer join and reindexing...")
        start_time = time.time()

        # Collect signal dataframes
        signal_dfs = {}
        error_signals = []

        for key, signal in time_series_signals.items():
            if signal.metadata.temporary:
                logger.debug(f"Skipping temporary TimeSeriesSignal '{key}' for combined export.")
                continue

            try:
                signal_df = signal.get_data()
                if signal_df is None or signal_df.empty:
                    logger.warning(f"TimeSeriesSignal '{key}' has no data after alignment, skipping.")
                    continue
                if not isinstance(signal_df.index, pd.DatetimeIndex):
                    logger.error(f"TimeSeriesSignal '{key}' index is not DatetimeIndex.")
                    error_signals.append(key)
                    continue

                signal_dfs[key] = signal_df
                logger.debug(f"Collected data for TimeSeriesSignal '{key}'. Shape: {signal_df.shape}")

            except Exception as e:
                logger.error(f"Error accessing data for TimeSeriesSignal '{key}': {e}", exc_info=True)
                error_signals.append(key)

        if error_signals:
            raise RuntimeError(
                f"Failed to combine signals. Errors occurred for: {', '.join(error_signals)}"
            )

        if not signal_dfs:
            logger.warning("No valid signals found to combine. Returning empty DataFrame.")
            empty_df = pd.DataFrame(index=self.alignment_state.grid_index)
            params = self._get_alignment_params("outer_join_reindex")
            return CombinationResult(dataframe=empty_df, params=params, is_feature_matrix=False)

        # Perform concatenation
        combined_df = self._perform_concatenation(
            aligned_dfs=signal_dfs,
            grid_index=self.alignment_state.grid_index,
            is_feature=False,
            time_series_signals=time_series_signals
        )

        # Store result and parameters
        params = self._get_alignment_params("outer_join_reindex")

        logger.info(
            f"Successfully combined {len(signal_dfs)} signals using outer join and reindex "
            f"in {time.time() - start_time:.2f} seconds. Shape: {combined_df.shape}"
        )

        return CombinationResult(dataframe=combined_df, params=params, is_feature_matrix=False)

    def combine_features(
        self,
        features: Dict[str, Feature],
        inputs: List[str],
        feature_index_config: Optional[List[str]] = None
    ) -> CombinationResult:
        """
        Combines multiple Feature objects into a single feature matrix.

        Retrieves specified Feature objects, validates their indices against the
        epoch_grid_index, and concatenates their data column-wise. Constructs a
        MultiIndex for columns based on the provided feature_index_config and
        feature metadata.

        Args:
            features: Dictionary of all Feature objects
            inputs: List of keys identifying the Feature objects to combine.
                   Can contain base names, which will be resolved.
            feature_index_config: Optional list of metadata field names to override
                                 the collection's default feature_index_config

        Returns:
            CombinationResult containing the combined feature matrix

        Raises:
            RuntimeError: If epoch grid hasn't been calculated
            ValueError: If inputs are missing, invalid, not Feature objects,
                       have indices mismatched with the epoch grid
            TypeError: If input dataframes cannot be concatenated

        Example:
            >>> result = service.combine_features(
            ...     features_dict,
            ...     inputs=['hr_features', 'accel_features']
            ... )
            >>> feature_matrix = result.dataframe
        """
        # Validate epoch state
        if not self.epoch_state or not self.epoch_state.is_valid():
            raise RuntimeError("Cannot combine features: epoch grid must be calculated first.")

        if self.epoch_state.epoch_grid_index is None or self.epoch_state.epoch_grid_index.empty:
            raise RuntimeError("Epoch grid_index is None or empty.")

        if not inputs:
            raise ValueError("No input signals specified for combine_features.")

        # Use provided config or fallback to collection's config
        config_to_use = feature_index_config if feature_index_config is not None else self.metadata.feature_index_config
        if not config_to_use:
            logger.warning("No feature_index_config provided. Combined columns will not have MultiIndex.")

        logger.info(f"Combining features from inputs: {inputs} using config: {config_to_use}")
        start_time = time.time()

        # Resolve input keys (handle base names)
        resolved_keys = []
        for key_spec in inputs:
            if key_spec in features:
                resolved_keys.append(key_spec)
            else:
                found_match = False
                for existing_key in features.keys():
                    if existing_key.startswith(f"{key_spec}_") and existing_key[len(key_spec)+1:].isdigit():
                        resolved_keys.append(existing_key)
                        found_match = True
                if not found_match:
                    raise ValueError(
                        f"Input specification '{key_spec}' does not match any existing feature key or base name."
                    )

        if not resolved_keys:
            raise ValueError(f"Input specification {inputs} resolved to an empty list.")

        logger.debug(f"Resolved combine_features input {inputs} to keys: {resolved_keys}")

        # Retrieve and validate input features
        input_features: List[Feature] = []
        for key in resolved_keys:
            if key not in features:
                raise KeyError(f"Feature '{key}' not found in features dictionary.")

            feature = features[key]
            feature_data = feature.get_data()

            if not isinstance(feature_data.index, pd.DatetimeIndex):
                raise TypeError(f"Input Feature '{key}' does not have a DatetimeIndex.")

            # Strict index validation against epoch_grid_index
            if not feature_data.index.equals(self.epoch_state.epoch_grid_index):
                logger.error(
                    f"Index mismatch for Feature '{key}'. Expected index matching epoch_grid_index "
                    f"(size {len(self.epoch_state.epoch_grid_index)}), "
                    f"but got index size {len(feature_data.index)}."
                )
                if len(feature_data.index) == len(self.epoch_state.epoch_grid_index):
                    diff = self.epoch_state.epoch_grid_index.difference(feature_data.index)
                    logger.error(f"Index values differ. Example differences: {diff[:5]}...")
                raise ValueError(
                    f"Input Feature '{key}' index does not match epoch_grid_index. "
                    f"Ensure feature generation used the global grid."
                )

            input_features.append(feature)

        if not input_features:
            logger.warning("No valid Feature objects found to combine.")
            empty_df = pd.DataFrame(index=self.epoch_state.epoch_grid_index)
            return CombinationResult(dataframe=empty_df, params={}, is_feature_matrix=True)

        # Prepare DataFrames for concatenation
        feature_dfs = {feat.metadata.name: feat.get_data() for feat in input_features}

        combined_df = self._perform_concatenation(
            aligned_dfs=feature_dfs,
            grid_index=self.epoch_state.epoch_grid_index,
            is_feature=True,
            features=features
        )

        logger.info(
            f"Successfully combined {len(input_features)} features "
            f"in {time.time() - start_time:.2f} seconds. Matrix shape: {combined_df.shape}"
        )

        return CombinationResult(dataframe=combined_df, params={}, is_feature_matrix=True)

    def _perform_concatenation(
        self,
        aligned_dfs: Dict[str, pd.DataFrame],
        grid_index: pd.DatetimeIndex,
        is_feature: bool,
        time_series_signals: Optional[Dict[str, TimeSeriesSignal]] = None,
        features: Optional[Dict[str, Feature]] = None
    ) -> pd.DataFrame:
        """
        Internal helper to concatenate aligned dataframes with MultiIndex handling.

        Args:
            aligned_dfs: Dictionary of aligned DataFrames to concatenate
            grid_index: DatetimeIndex to use for the combined DataFrame
            is_feature: True for feature combination, False for time-series
            time_series_signals: Optional dict of time-series signals (for metadata)
            features: Optional dict of features (for metadata)

        Returns:
            Combined DataFrame with appropriate index structure
        """
        if not aligned_dfs:
            return pd.DataFrame(index=grid_index)

        # Determine which index config and metadata source to use
        index_config = self.metadata.feature_index_config if is_feature else self.metadata.index_config
        source_dict = features if is_feature else time_series_signals

        combined_df: pd.DataFrame

        if is_feature:
            # Feature concatenation logic
            logger.info("Using simplified concatenation for features (pd.concat with keys).")
            try:
                # Concatenate using feature set keys as top level
                combined_df = pd.concat(aligned_dfs, axis=1)
                combined_df = combined_df.reindex(grid_index)

                # Name the levels appropriately
                if isinstance(combined_df.columns, pd.MultiIndex):
                    # Determine expected number of levels from first non-empty df
                    expected_levels = 0
                    for df_val in aligned_dfs.values():
                        if not df_val.empty and isinstance(df_val.columns, pd.MultiIndex):
                            expected_levels = df_val.columns.nlevels
                            break
                        elif not df_val.empty:
                            expected_levels = 1
                            break

                    if expected_levels == 2:
                        combined_df.columns.names = ['feature_set', 'signal_key', 'feature']
                    elif expected_levels == 1:
                        combined_df.columns.names = ['feature_set', 'feature']
                    else:
                        # Fallback
                        num_levels = combined_df.columns.nlevels
                        if num_levels > 0:
                            new_names = ['feature_set'] + [f'level_{i+1}' for i in range(num_levels - 1)]
                            combined_df.columns.names = new_names[:num_levels]

                    logger.debug(f"Feature concatenation MultiIndex names: {combined_df.columns.names}")

            except Exception as e:
                logger.error(f"Error during feature concatenation: {e}", exc_info=True)
                combined_df = pd.DataFrame(index=grid_index)

        else:
            # Time-series concatenation logic
            if index_config:
                logger.info("Using MultiIndex for combined time-series columns.")
                multi_index_tuples = []
                final_columns_data = {}

                for key, signal_df in aligned_dfs.items():
                    signal_obj = source_dict.get(key) if source_dict else None
                    if not signal_obj:
                        logger.warning(f"Could not find object for key '{key}'. Skipping.")
                        continue
                    metadata_obj = signal_obj.metadata

                    for col_name in signal_df.columns:
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
                        final_columns_data[tuple_key] = signal_df[col_name]

                if final_columns_data:
                    level_names = index_config + ['column']
                    multi_idx = pd.MultiIndex.from_tuples(multi_index_tuples, names=level_names)
                    combined_df = pd.DataFrame(final_columns_data, index=grid_index)
                    if not combined_df.empty:
                        combined_df.columns = multi_idx
                    else:
                        combined_df = pd.DataFrame(index=grid_index, columns=multi_idx)
                    logger.debug(f"Applied MultiIndex. Level names: {combined_df.columns.names}")
                else:
                    logger.warning("No data available for MultiIndex columns.")
                    level_names = index_config + ['column']
                    empty_multi_idx = pd.MultiIndex.from_tuples([], names=level_names)
                    combined_df = pd.DataFrame(index=grid_index, columns=empty_multi_idx)
            else:
                logger.info("Using simple column names (key_colname) for combined dataframe.")
                simple_concat_list = []
                for key, signal_df in aligned_dfs.items():
                    if len(signal_df.columns) == 1:
                        renamed_df = signal_df.rename(columns={signal_df.columns[0]: key})
                        simple_concat_list.append(renamed_df)
                    else:
                        prefixed_df = signal_df.add_prefix(f"{key}_")
                        simple_concat_list.append(prefixed_df)

                if not simple_concat_list:
                    logger.warning("No data for simple concatenation.")
                    combined_df = pd.DataFrame(index=grid_index)
                else:
                    combined_df = pd.concat(simple_concat_list, axis=1)
                    combined_df = combined_df.reindex(grid_index)

        # Final cleanup: remove rows with all NaN
        if not combined_df.empty:
            initial_rows = len(combined_df)
            combined_df = combined_df.dropna(axis=0, how='all')
            final_rows = len(combined_df)
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} all-NaN rows.")
        else:
            logger.info("Combined dataframe was empty before NaN removal.")

        return combined_df

    def _get_alignment_params(self, method_used: str) -> Dict[str, Any]:
        """
        Helper to gather current alignment parameters for storage.

        Args:
            method_used: The alignment method that was used

        Returns:
            Dictionary of alignment parameters
        """
        if not self.alignment_state:
            return {"method_used": method_used}

        return {
            "method_used": method_used,
            "target_rate": self.alignment_state.target_rate,
            "ref_time": self.alignment_state.reference_time,
            "merge_tolerance": self.alignment_state.merge_tolerance,
            "grid_shape": self.alignment_state.grid_index.shape if self.alignment_state.grid_index is not None else None
        }
