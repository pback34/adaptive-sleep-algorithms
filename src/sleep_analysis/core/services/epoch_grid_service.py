"""
EpochGridService - Computes and manages epoch grid for feature extraction.

This service extracts epoch grid calculation logic from SignalCollection,
focusing on generating time windows for feature extraction operations.
"""

import logging
import time
from typing import Optional, Union
import pandas as pd

from ..repositories.signal_repository import SignalRepository
from ..models.epoch_state import EpochGridState
from ..metadata import CollectionMetadata

logger = logging.getLogger(__name__)


class EpochGridService:
    """
    Computes and manages epoch grid for feature extraction.

    This service is responsible for calculating epoch-based time windows
    that are used for extracting features from time-series signals.

    The epoch grid defines a series of time windows (epochs) with specified
    window length and step size, covering the time range of signals in the collection.

    Attributes:
        repository: SignalRepository for accessing time-series signals
        collection_metadata: CollectionMetadata containing epoch_grid_config
        _state: Current epoch grid state (EpochGridState)

    Example:
        >>> from src.sleep_analysis.core.repositories import SignalRepository
        >>> from src.sleep_analysis.core.services import EpochGridService
        >>> from src.sleep_analysis.core.metadata import CollectionMetadata
        >>>
        >>> repository = SignalRepository(metadata_handler, "UTC")
        >>> collection_metadata = CollectionMetadata(
        ...     timezone="UTC",
        ...     epoch_grid_config={"window_length": "30s", "step_size": "30s"}
        ... )
        >>> epoch_service = EpochGridService(repository, collection_metadata)
        >>>
        >>> # Generate epoch grid
        >>> state = epoch_service.generate_epoch_grid()
        >>>
        >>> # Check if grid is valid
        >>> if state.is_valid():
        ...     print(f"Window length: {state.window_length}")
        ...     print(f"Number of epochs: {len(state.epoch_grid_index)}")
    """

    def __init__(self, repository: SignalRepository, collection_metadata: CollectionMetadata):
        """
        Initialize EpochGridService.

        Args:
            repository: SignalRepository instance for accessing time-series signals
            collection_metadata: CollectionMetadata instance with epoch_grid_config
        """
        self.repository = repository
        self.collection_metadata = collection_metadata
        self._state: EpochGridState = EpochGridState()

    @property
    def state(self) -> EpochGridState:
        """
        Get current epoch grid state.

        Returns:
            Current EpochGridState instance
        """
        return self._state

    def generate_epoch_grid(
        self,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None
    ) -> EpochGridState:
        """
        Generate epoch grid and return state.

        Uses `epoch_grid_config` from `CollectionMetadata` and the time range
        of `time_series_signals` to create a common `epoch_grid_index`.

        Args:
            start_time: Optional override for the grid start time.
                Can be a string (e.g., "2024-01-01 00:00:00") or pd.Timestamp.
            end_time: Optional override for the grid end time.
                Can be a string (e.g., "2024-01-01 23:59:59") or pd.Timestamp.

        Returns:
            EpochGridState with calculated epoch grid parameters

        Raises:
            RuntimeError: If `epoch_grid_config` is missing or invalid, or if
                          no time-series signals are found to determine the range.
            ValueError: If start/end time overrides are invalid or if
                       window_length/step_size are not positive.

        Example:
            >>> # Generate epoch grid using signal time ranges
            >>> state = epoch_service.generate_epoch_grid()
            >>>
            >>> # Generate epoch grid with custom time range
            >>> state = epoch_service.generate_epoch_grid(
            ...     start_time="2024-01-01 00:00:00",
            ...     end_time="2024-01-01 12:00:00"
            ... )
        """
        logger.info("Starting global epoch grid calculation...")
        op_start_time = time.time()

        # Get configuration from collection metadata
        config = self.collection_metadata.epoch_grid_config
        if not config or "window_length" not in config or "step_size" not in config:
            raise RuntimeError("Missing or incomplete 'epoch_grid_config' in collection metadata. Cannot generate epoch grid.")

        try:
            window_length = pd.Timedelta(config["window_length"])
            step_size = pd.Timedelta(config["step_size"])
            if window_length <= pd.Timedelta(0) or step_size <= pd.Timedelta(0):
                raise ValueError("window_length and step_size must be positive.")
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid epoch_grid_config parameters: {e}") from e

        logger.info(f"Using global epoch parameters: window={window_length}, step={step_size}")

        # Determine time range from signals
        time_series_signals = self.repository.get_all_time_series()
        if not time_series_signals:
            raise RuntimeError("No time-series signals found in repository. Cannot determine epoch grid range.")

        min_times = []
        max_times = []
        collection_tz = pd.Timestamp('now', tz=self.collection_metadata.timezone).tz

        for signal in time_series_signals.values():
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

        # Apply time range overrides if provided
        try:
            if start_time:
                # Handle both string and Timestamp inputs
                if isinstance(start_time, pd.Timestamp):
                    grid_start = start_time
                else:
                    grid_start = pd.Timestamp(start_time)
            else:
                grid_start = min(min_times)

            if end_time:
                # Handle both string and Timestamp inputs
                if isinstance(end_time, pd.Timestamp):
                    grid_end = end_time
                else:
                    grid_end = pd.Timestamp(end_time)
            else:
                grid_end = max(max_times)

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
            # Generate epoch start times using the step_size as frequency
            epoch_index = pd.date_range(
                start=grid_start,
                end=grid_end,
                freq=step_size,
                name='epoch_start_time',
                inclusive='left'  # Only include start times <= grid_end
            )

            # Filter out any start times where the window would begin after the grid ends
            epoch_index = epoch_index[epoch_index <= grid_end]

            if epoch_index.empty:
                logger.warning("Generated epoch grid index is empty.")

            # Ensure timezone matches collection
            epoch_index = epoch_index.tz_convert(collection_tz) if epoch_index.tz is not None else epoch_index.tz_localize(collection_tz)

            logger.info(f"Calculated epoch_grid_index with {len(epoch_index)} points.")

        except Exception as e:
            logger.error(f"Error creating date_range for epoch grid index: {e}", exc_info=True)
            raise RuntimeError(f"Failed to calculate epoch grid index: {e}") from e

        # Create and store new state
        self._state = EpochGridState(
            epoch_grid_index=epoch_index,
            window_length=window_length,
            step_size=step_size,
            is_calculated=True
        )

        elapsed = time.time() - op_start_time
        logger.info(f"Epoch grid calculated in {elapsed:.2f} seconds.")

        return self._state
