"""Alignment grid state data class."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class AlignmentGridState:
    """Immutable state for time-series alignment grid.

    This data class encapsulates all parameters needed for signal alignment,
    replacing scattered instance attributes in the original SignalCollection.

    Attributes:
        target_rate: Target sampling rate in Hz (e.g., 100.0 for 100 Hz)
        reference_time: Reference timestamp for grid alignment
        grid_index: DatetimeIndex representing the alignment grid
        merge_tolerance: Time tolerance for merging/alignment operations
        is_calculated: Flag indicating if the grid has been computed

    Example:
        >>> state = AlignmentGridState(
        ...     target_rate=100.0,
        ...     reference_time=pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        ...     grid_index=pd.date_range("2025-01-01", periods=1000, freq="10ms", tz="UTC"),
        ...     is_calculated=True
        ... )
        >>> if state.is_valid():
        ...     print(f"Grid ready with {len(state.grid_index)} points")
    """

    target_rate: Optional[float] = None
    reference_time: Optional[pd.Timestamp] = None
    grid_index: Optional[pd.DatetimeIndex] = None
    merge_tolerance: Optional[pd.Timedelta] = None
    is_calculated: bool = False

    def is_valid(self) -> bool:
        """Check if state is valid for alignment operations.

        Returns:
            True if the grid has been calculated and contains valid data

        Example:
            >>> state = AlignmentGridState()
            >>> state.is_valid()
            False
            >>> state = AlignmentGridState(
            ...     target_rate=100.0,
            ...     grid_index=pd.date_range("2025-01-01", periods=10, freq="10ms"),
            ...     is_calculated=True
            ... )
            >>> state.is_valid()
            True
        """
        return (
            self.is_calculated
            and self.grid_index is not None
            and not self.grid_index.empty
        )

    def __repr__(self) -> str:
        """Return detailed string representation."""
        if not self.is_calculated:
            return "AlignmentGridState(not calculated)"

        grid_size = len(self.grid_index) if self.grid_index is not None else 0
        return (
            f"AlignmentGridState("
            f"rate={self.target_rate}Hz, "
            f"ref_time={self.reference_time}, "
            f"grid_points={grid_size}"
            f")"
        )
