"""Combination result data class."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd


@dataclass
class CombinationResult:
    """Result of signal/feature combination operations.

    This data class encapsulates the result of combining multiple signals
    or features into a unified DataFrame, along with the parameters used.

    Attributes:
        dataframe: The combined DataFrame result
        params: Dictionary of parameters used for combination
        is_feature_matrix: Flag indicating if this is a feature matrix (vs time-series)

    Example:
        >>> result = CombinationResult(
        ...     dataframe=pd.DataFrame({"signal1": [1, 2, 3], "signal2": [4, 5, 6]}),
        ...     params={"method": "nearest", "target_rate": 100.0},
        ...     is_feature_matrix=False
        ... )
        >>> print(f"Combined {len(result.dataframe.columns)} signals")
    """

    dataframe: Optional[pd.DataFrame] = None
    params: Optional[Dict[str, Any]] = None
    is_feature_matrix: bool = False

    def is_valid(self) -> bool:
        """Check if result contains valid data.

        Returns:
            True if dataframe exists and is not empty

        Example:
            >>> result = CombinationResult()
            >>> result.is_valid()
            False
            >>> result = CombinationResult(dataframe=pd.DataFrame({"a": [1, 2, 3]}))
            >>> result.is_valid()
            True
        """
        return self.dataframe is not None and not self.dataframe.empty

    def __repr__(self) -> str:
        """Return detailed string representation."""
        if self.dataframe is None:
            return "CombinationResult(empty)"

        rows, cols = self.dataframe.shape
        result_type = "feature matrix" if self.is_feature_matrix else "time-series"
        return f"CombinationResult({result_type}, {rows} rows × {cols} cols)"
