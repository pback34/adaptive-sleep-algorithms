"""Epoch grid state data class."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class EpochGridState:
    """Immutable state for epoch-based time windows.

    This data class encapsulates all parameters needed for epoch-based feature extraction,
    replacing scattered instance attributes in the original SignalCollection.

    Attributes:
        epoch_grid_index: DatetimeIndex representing epoch boundaries
        window_length: Duration of each epoch window (e.g., pd.Timedelta("30s"))
        step_size: Step size between epochs (e.g., pd.Timedelta("30s") for non-overlapping)
        is_calculated: Flag indicating if the epoch grid has been computed

    Example:
        >>> state = EpochGridState(
        ...     epoch_grid_index=pd.date_range("2025-01-01", periods=100, freq="30s", tz="UTC"),
        ...     window_length=pd.Timedelta("30s"),
        ...     step_size=pd.Timedelta("30s"),
        ...     is_calculated=True
        ... )
        >>> if state.is_valid():
        ...     print(f"Epoch grid ready with {len(state.epoch_grid_index)} epochs")
    """

    epoch_grid_index: Optional[pd.DatetimeIndex] = None
    window_length: Optional[pd.Timedelta] = None
    step_size: Optional[pd.Timedelta] = None
    is_calculated: bool = False

    def is_valid(self) -> bool:
        """Check if state is valid for feature extraction operations.

        Returns:
            True if the epoch grid has been calculated and contains valid data

        Example:
            >>> state = EpochGridState()
            >>> state.is_valid()
            False
            >>> state = EpochGridState(
            ...     epoch_grid_index=pd.date_range("2025-01-01", periods=10, freq="30s"),
            ...     window_length=pd.Timedelta("30s"),
            ...     is_calculated=True
            ... )
            >>> state.is_valid()
            True
        """
        return (
            self.is_calculated
            and self.epoch_grid_index is not None
            and not self.epoch_grid_index.empty
        )

    def __repr__(self) -> str:
        """Return detailed string representation."""
        if not self.is_calculated:
            return "EpochGridState(not calculated)"

        grid_size = len(self.epoch_grid_index) if self.epoch_grid_index is not None else 0
        return (
            f"EpochGridState("
            f"window={self.window_length}, "
            f"step={self.step_size}, "
            f"epochs={grid_size}"
            f")"
        )
