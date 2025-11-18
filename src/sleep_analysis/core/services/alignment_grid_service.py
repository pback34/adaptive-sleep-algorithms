"""
AlignmentGridService - Computes and manages alignment grid for time-series signals.

This service extracts alignment grid calculation logic from SignalCollection,
focusing on determining optimal grid parameters for signal alignment.
"""

import logging
import time
from typing import Optional, List
import pandas as pd
import numpy as np

from ..repositories.signal_repository import SignalRepository
from ..models.alignment_state import AlignmentGridState

logger = logging.getLogger(__name__)

# Standard sample rates (Hz) for grid alignment
STANDARD_RATES = sorted(list(set([0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000])))


class AlignmentGridService:
    """
    Computes and manages alignment grid for time-series signals.

    This service is responsible for calculating optimal grid parameters
    (target rate, reference time, grid index) for aligning time-series signals.

    The alignment grid provides a common temporal basis for combining signals
    with different sampling rates and time ranges.

    Attributes:
        repository: SignalRepository for accessing time-series signals
        _state: Current alignment grid state (AlignmentGridState)

    Example:
        >>> from src.sleep_analysis.core.repositories import SignalRepository
        >>> from src.sleep_analysis.core.services import AlignmentGridService
        >>>
        >>> repository = SignalRepository(metadata_handler, "UTC")
        >>> grid_service = AlignmentGridService(repository)
        >>>
        >>> # Generate alignment grid
        >>> state = grid_service.generate_alignment_grid(target_sample_rate=100.0)
        >>>
        >>> # Check if grid is valid
        >>> if state.is_valid():
        ...     print(f"Grid rate: {state.target_rate} Hz")
        ...     print(f"Grid size: {len(state.grid_index)}")
    """

    def __init__(self, repository: SignalRepository):
        """
        Initialize AlignmentGridService.

        Args:
            repository: SignalRepository instance for accessing time-series signals
        """
        self.repository = repository
        self._state: AlignmentGridState = AlignmentGridState()

    @property
    def state(self) -> AlignmentGridState:
        """
        Get current alignment grid state.

        Returns:
            Current AlignmentGridState instance
        """
        return self._state

    def generate_alignment_grid(self, target_sample_rate: Optional[float] = None) -> AlignmentGridState:
        """
        Generate alignment grid and return state.

        Calculates optimal grid parameters based on time-series signals in the repository:
        1. Determines target sample rate (user-specified or auto-calculated)
        2. Computes reference time aligned to epoch and target period
        3. Generates DatetimeIndex grid spanning all signals

        Args:
            target_sample_rate: Optional user-specified target rate in Hz.
                If None, uses the largest standard rate <= max signal rate.

        Returns:
            AlignmentGridState with calculated grid parameters

        Raises:
            RuntimeError: If no time-series signals found in repository
            ValueError: If calculated target rate is invalid (None or <= 0)

        Example:
            >>> state = grid_service.generate_alignment_grid(target_sample_rate=100.0)
            >>> print(f"Target rate: {state.target_rate} Hz")
            >>> print(f"Reference time: {state.reference_time}")
            >>> print(f"Grid points: {len(state.grid_index)}")
        """
        logger.info(f"Starting alignment grid parameter calculation with target_sample_rate={target_sample_rate}")
        start_time = time.time()

        # Check for time-series signals
        time_series_signals = self.repository.get_all_time_series()
        if not time_series_signals:
            logger.error("No time-series signals found in the repository. Cannot calculate alignment grid.")
            raise RuntimeError("No time-series signals found in the repository to calculate alignment grid.")

        # Determine target rate
        try:
            target_rate = self._get_target_sample_rate(target_sample_rate)
            if target_rate is None or target_rate <= 0:
                raise ValueError(f"Calculated invalid target rate: {target_rate}")
            target_period = pd.Timedelta(seconds=1 / target_rate)
            logger.info(f"Using target rate: {target_rate} Hz (Period: {target_period})")
        except Exception as e:
            logger.error(f"Failed to determine target sample rate: {e}", exc_info=True)
            raise RuntimeError(f"Failed to determine target sample rate: {e}") from e

        # Determine reference time
        try:
            ref_time = self._get_reference_time(target_period)
            logger.info(f"Using reference time: {ref_time}")
        except Exception as e:
            logger.error(f"Failed to determine reference time: {e}", exc_info=True)
            raise RuntimeError(f"Failed to determine reference time: {e}") from e

        # Calculate grid index
        try:
            grid_index = self._calculate_grid_index(target_rate, ref_time)
            if grid_index is None or grid_index.empty:
                raise RuntimeError("Failed to calculate valid grid index")
            logger.info(f"Calculated grid index with {len(grid_index)} points")
        except Exception as e:
            logger.error(f"Failed to calculate grid index: {e}", exc_info=True)
            raise RuntimeError(f"Failed to calculate grid index: {e}") from e

        # Create and store new state
        self._state = AlignmentGridState(
            target_rate=target_rate,
            reference_time=ref_time,
            grid_index=grid_index,
            merge_tolerance=None,  # Can be set later if needed
            is_calculated=True
        )

        elapsed = time.time() - start_time
        logger.info(f"Alignment grid calculation completed in {elapsed:.3f}s")

        return self._state

    def _get_target_sample_rate(self, user_specified: Optional[float] = None) -> float:
        """
        Determine the target sample rate for time-series alignment.

        If user specifies a rate, uses that value.
        Otherwise, finds the maximum rate across all time-series signals
        and selects the largest standard rate <= max rate.

        Args:
            user_specified: Optional user-specified rate in Hz

        Returns:
            Target sample rate in Hz

        Example:
            >>> # Auto-calculate from signals
            >>> rate = grid_service._get_target_sample_rate()
            >>>
            >>> # Use specified rate
            >>> rate = grid_service._get_target_sample_rate(100.0)
        """
        if user_specified is not None:
            return float(user_specified)

        # Calculate max rate from time-series signals
        time_series_signals = self.repository.get_all_time_series()
        valid_rates = [
            s.get_sampling_rate() for s in time_series_signals.values()
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

    def get_nearest_standard_rate(self, rate: float) -> float:
        """
        Find the nearest standard rate to a given sample rate.

        Standard rates are predefined common sampling rates that align well
        with typical signal processing requirements.

        Args:
            rate: Sample rate in Hz

        Returns:
            Nearest standard rate in Hz

        Example:
            >>> nearest = grid_service.get_nearest_standard_rate(99.5)
            >>> print(nearest)  # 100.0
        """
        if rate is None or rate <= 0:
            logger.warning(f"Invalid rate ({rate}) provided to get_nearest_standard_rate. Returning default rate 1 Hz.")
            return 1.0

        nearest_rate = min(STANDARD_RATES, key=lambda r: abs(r - rate))
        logger.debug(f"Nearest standard rate to {rate:.4f} Hz is {nearest_rate} Hz.")
        return nearest_rate

    def _get_reference_time(self, target_period: pd.Timedelta) -> pd.Timestamp:
        """
        Compute the reference timestamp for grid alignment.

        The reference time is aligned to the Unix epoch (1970-01-01) and the target period,
        ensuring that the grid starts at a time that is an integer multiple of the period
        from the epoch and <= the earliest signal timestamp.

        Args:
            target_period: Target sampling period (1/rate)

        Returns:
            Reference timestamp for grid alignment

        Raises:
            ValueError: If target period is zero

        Example:
            >>> target_period = pd.Timedelta(seconds=0.01)  # 100 Hz
            >>> ref_time = grid_service._get_reference_time(target_period)
        """
        time_series_signals = self.repository.get_all_time_series()

        min_times = []
        for signal in time_series_signals.values():
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
        """
        Calculate the final DatetimeIndex grid based on TimeSeriesSignals.

        Creates a uniform grid spanning from the earliest to latest signal timestamps,
        with points spaced at the target period (1/target_rate).

        Args:
            target_rate: Target sample rate in Hz
            ref_time: Reference timestamp for grid alignment

        Returns:
            DatetimeIndex grid, or None if calculation fails

        Example:
            >>> grid = grid_service._calculate_grid_index(100.0, ref_time)
            >>> print(f"Grid has {len(grid)} points")
        """
        if target_rate <= 0:
            logger.error(f"Invalid target_rate ({target_rate}) for grid calculation.")
            return None

        target_period = pd.Timedelta(seconds=1 / target_rate)

        min_times = []
        max_times = []

        time_series_signals = self.repository.get_all_time_series()
        for signal in time_series_signals.values():
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

        # Calculate total duration and number of periods
        total_duration = latest_end - earliest_start
        total_duration_ns = total_duration.total_seconds() * 1e9
        target_period_ns = target_period.total_seconds() * 1e9

        if target_period_ns <= 0:
            logger.error("Target period must be positive for grid index calculation.")
            return None

        num_periods = int(np.ceil(total_duration_ns / target_period_ns))

        # Generate grid from ref_time
        grid_start = ref_time
        grid_end = ref_time + pd.Timedelta(nanoseconds=num_periods * target_period_ns)

        try:
            grid_index = pd.date_range(
                start=grid_start,
                end=grid_end,
                freq=target_period,
                tz=ref_time.tz
            )
            logger.debug(f"Generated grid index from {grid_start} to {grid_end} with {len(grid_index)} points")
            return grid_index
        except Exception as e:
            logger.error(f"Failed to create date_range for grid: {e}")
            return None
