"""
Alignment service for handling signal alignment operations.

This service encapsulates all logic related to generating alignment grids,
applying alignment to signals, and combining aligned signals.
"""

import time
import warnings
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ..core.signal_collection import SignalCollection

logger = logging.getLogger(__name__)

# Define standard rates: factors of 1000 Hz plus rates corresponding to multi-second periods
STANDARD_RATES = sorted(list(set([0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000])))


class AlignmentService:
    """
    Service for aligning time series signals to a common grid.

    This service handles:
    - Generating alignment grids based on signal sampling rates
    - Computing reference times and grid indices
    - Applying alignment to signals in place
    - Combining aligned signals using merge_asof
    """

    def generate_alignment_grid(
        self,
        collection: 'SignalCollection',
        target_sample_rate: Optional[float] = None
    ) -> None:
        """
        Calculate and store alignment grid parameters on the collection.

        Args:
            collection: The SignalCollection to generate alignment grid for
            target_sample_rate: Optional target sample rate in Hz

        Raises:
            RuntimeError: If no time series signals found or grid calculation fails
        """
        logger.info(f"Starting alignment grid parameter calculation with target_sample_rate={target_sample_rate}")
        start_time = time.time()
        collection._alignment_params_calculated = False

        # Check for time series signals
        if not collection.time_series_signals:
            logger.error("No time-series signals found in the collection. Cannot calculate alignment grid.")
            raise RuntimeError("No time-series signals found in the collection to calculate alignment grid.")

        # Determine target rate
        try:
            collection.target_rate = self.get_target_sample_rate(collection, target_sample_rate)
            if collection.target_rate is None or collection.target_rate <= 0:
                raise ValueError(f"Calculated invalid target rate: {collection.target_rate}")
            target_period = pd.Timedelta(seconds=1 / collection.target_rate)
            logger.info(f"Using target rate: {collection.target_rate} Hz (Period: {target_period})")
        except Exception as e:
            logger.error(f"Failed to determine target sample rate: {e}", exc_info=True)
            raise RuntimeError(f"Failed to determine target sample rate: {e}") from e

        # Determine reference time
        try:
            collection.ref_time = self.get_reference_time(collection, target_period)
            logger.info(f"Using reference time: {collection.ref_time}")
        except Exception as e:
            logger.error(f"Failed to determine reference time: {e}", exc_info=True)
            raise RuntimeError(f"Failed to determine reference time: {e}") from e

        # Calculate grid index
        try:
            collection.grid_index = self._calculate_grid_index(collection, collection.target_rate, collection.ref_time)
            if collection.grid_index is None or collection.grid_index.empty:
                raise ValueError("Calculated grid index is None or empty.")
        except Exception as e:
            logger.error(f"Failed to calculate grid index: {e}", exc_info=True)
            collection.grid_index = None
            raise RuntimeError(f"Failed to calculate a valid grid index for alignment: {e}") from e

        collection._alignment_params_calculated = True
        logger.info(f"Alignment grid parameters calculated in {time.time() - start_time:.2f} seconds.")

    def get_target_sample_rate(
        self,
        collection: 'SignalCollection',
        user_specified: Optional[float] = None
    ) -> float:
        """
        Determine the target sample rate for time-series alignment.

        Args:
            collection: The SignalCollection containing signals
            user_specified: Optional user-specified target rate

        Returns:
            Target sample rate in Hz
        """
        if user_specified is not None:
            return float(user_specified)

        # Calculate max rate only from TimeSeriesSignals
        valid_rates = [
            s.get_sampling_rate()
            for s in collection.time_series_signals.values()
            if s.get_sampling_rate() is not None and s.get_sampling_rate() > 0
        ]

        if not valid_rates:
            logger.warning("No valid positive sampling rates found in TimeSeriesSignals. Defaulting target rate to 100 Hz.")
            return 100.0

        max_rate = max(valid_rates)

        if max_rate <= 0:
            logger.warning("Max sampling rate is not positive. Defaulting target rate to 100 Hz.")
            return 100.0

        # Find the largest standard rate <= the maximum rate found
        valid_standard_rates = [r for r in STANDARD_RATES if r <= max_rate]
        chosen_rate = max(valid_standard_rates) if valid_standard_rates else min(STANDARD_RATES)
        logger.info(f"Determined target sample rate: {chosen_rate} Hz (based on max TimeSeriesSignal rate {max_rate:.4f} Hz)")
        return chosen_rate

    def get_reference_time(
        self,
        collection: 'SignalCollection',
        target_period: pd.Timedelta
    ) -> pd.Timestamp:
        """
        Compute the reference timestamp for grid alignment.

        Args:
            collection: The SignalCollection containing signals
            target_period: The target period for alignment

        Returns:
            Reference timestamp for grid alignment

        Raises:
            ValueError: If target period is zero
        """
        min_times = []
        for signal in collection.time_series_signals.values():
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

    def _calculate_grid_index(
        self,
        collection: 'SignalCollection',
        target_rate: float,
        ref_time: pd.Timestamp
    ) -> Optional[pd.DatetimeIndex]:
        """
        Calculate the final DatetimeIndex grid.

        Args:
            collection: The SignalCollection containing signals
            target_rate: Target sample rate in Hz
            ref_time: Reference time for grid alignment

        Returns:
            DatetimeIndex for the alignment grid or None if calculation fails
        """
        if target_rate <= 0:
            logger.error(f"Invalid target_rate ({target_rate}) for grid calculation.")
            return None

        target_period = pd.Timedelta(seconds=1 / target_rate)

        min_times = []
        max_times = []
        for signal in collection.time_series_signals.values():
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
            logger.error("No valid time ranges found in TimeSeriesSignals for grid calculation.")
            return None

        grid_start = min(min_times)
        grid_end = max(max_times)

        logger.debug(f"Grid index range: {grid_start} to {grid_end}")

        # Calculate number of periods from ref_time to grid_start
        periods_to_start = np.ceil((grid_start - ref_time) / target_period)
        periods_to_end = np.floor((grid_end - ref_time) / target_period)

        grid_start_aligned = ref_time + (periods_to_start * target_period)
        grid_end_aligned = ref_time + (periods_to_end * target_period)

        if grid_start_aligned > grid_end_aligned:
            logger.warning("Aligned grid start is after aligned grid end. Returning empty grid.")
            return pd.DatetimeIndex([], name='timestamp', tz=ref_time.tz)

        grid_index = pd.date_range(
            start=grid_start_aligned,
            end=grid_end_aligned,
            freq=target_period,
            name='timestamp'
        )

        logger.info(f"Created grid index with {len(grid_index)} points")
        return grid_index

    def apply_grid_alignment(
        self,
        collection: 'SignalCollection',
        method: str = 'nearest',
        signals_to_align: Optional[List[str]] = None
    ) -> None:
        """
        Apply grid alignment to specified TimeSeriesSignals in place.

        Args:
            collection: The SignalCollection containing signals to align
            method: Alignment method ('nearest', 'pad', 'ffill', 'backfill', 'bfill')
            signals_to_align: Optional list of signal keys to align. If None, aligns all.

        Raises:
            RuntimeError: If alignment grid not generated or alignment fails
        """
        if not collection._alignment_params_calculated or collection.grid_index is None or collection.grid_index.empty:
            logger.error("Cannot apply grid alignment: generate_alignment_grid must be run successfully first.")
            raise RuntimeError("generate_alignment_grid must be run successfully before applying grid alignment.")

        allowed_methods = ['nearest', 'pad', 'ffill', 'backfill', 'bfill']
        if method not in allowed_methods:
            logger.warning(f"Alignment method '{method}' not in allowed list {allowed_methods}. Using 'nearest'.")
            method = 'nearest'

        logger.info(f"Applying grid alignment in-place to TimeSeriesSignals using method '{method}'...")
        start_time = time.time()

        target_keys = signals_to_align if signals_to_align is not None else list(collection.time_series_signals.keys())

        processed_count = 0
        skipped_count = 0
        error_signals = []

        for key in target_keys:
            try:
                signal = collection.get_time_series_signal(key)

                current_data = signal.get_data()
                if current_data is None or current_data.empty:
                    logger.warning(f"Skipping alignment for TimeSeriesSignal '{key}': data is None or empty.")
                    skipped_count += 1
                    continue

                logger.debug(f"Calling apply_operation('reindex_to_grid') for TimeSeriesSignal '{key}'...")
                signal.apply_operation(
                    'reindex_to_grid',
                    inplace=True,
                    grid_index=collection.grid_index,
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

    def align_and_combine_signals(
        self,
        collection: 'SignalCollection'
    ) -> pd.DataFrame:
        """
        Align TimeSeriesSignals using merge_asof and combine them.

        Args:
            collection: The SignalCollection containing signals to align and combine

        Returns:
            Combined DataFrame with aligned signals

        Raises:
            RuntimeError: If alignment grid not generated
        """
        if not collection._alignment_params_calculated or collection.grid_index is None or collection.grid_index.empty:
            logger.error("Cannot align and combine signals: generate_alignment_grid must be run successfully first.")
            raise RuntimeError("generate_alignment_grid must be run successfully before aligning and combining signals.")

        logger.info("Aligning and combining TimeSeriesSignals using merge_asof...")
        start_time = time.time()

        target_period = pd.Timedelta(seconds=1 / collection.target_rate) if collection.target_rate else None
        if target_period is None or target_period.total_seconds() <= 0:
            logger.warning("Grid index frequency is missing or invalid. Using default merge tolerance (1ms).")
            tolerance = pd.Timedelta(milliseconds=1)
        else:
            tolerance_ns = target_period.total_seconds() * 1e9 / 2
            tolerance = pd.Timedelta(nanoseconds=tolerance_ns + 1)

        collection._merge_tolerance = tolerance
        logger.debug(f"Using merge_asof tolerance: {tolerance}")

        target_df = pd.DataFrame({'timestamp': collection.grid_index})
        aligned_signal_dfs = {}
        error_signals = []

        # Iterate only over time_series_signals
        for key, signal in collection.time_series_signals.items():
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
                    source_df['timestamp'] = source_df['timestamp'].dt.tz_localize(collection.grid_index.tz)
                else:
                    source_df['timestamp'] = source_df['timestamp'].dt.tz_convert(collection.grid_index.tz)

                # Rename columns to avoid collisions
                rename_map = {col: f"{key}_{col}" for col in source_df.columns if col != 'timestamp'}
                source_df = source_df.rename(columns=rename_map)
                aligned_signal_dfs[key] = source_df

            except Exception as e:
                logger.error(f"Error preparing TimeSeriesSignal '{key}' for merge_asof: {e}", exc_info=True)
                error_signals.append(key)

        if error_signals:
            logger.warning(f"Errors encountered preparing signals for merge: {', '.join(error_signals)}")

        # Perform merge_asof
        for key, source_df in aligned_signal_dfs.items():
            try:
                target_df = pd.merge_asof(
                    target_df,
                    source_df,
                    on='timestamp',
                    direction='nearest',
                    tolerance=tolerance
                )
                logger.debug(f"Merged TimeSeriesSignal '{key}' into combined dataframe")
            except Exception as e:
                logger.error(f"Error during merge_asof for TimeSeriesSignal '{key}': {e}", exc_info=True)

        # Set timestamp as index
        combined_df = target_df.set_index('timestamp')
        logger.info(f"Alignment and combination completed in {time.time() - start_time:.2f} seconds. "
                    f"Combined dataframe shape: {combined_df.shape}")

        return combined_df
