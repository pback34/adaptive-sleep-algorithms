"""
Validation utilities for the sleep analysis framework.

This module provides reusable validation functions to ensure consistency
across the codebase and reduce duplication.
"""

from typing import List, Any, Type, Optional
import pandas as pd


def validate_not_empty(value: Any, name: str, error_type: Type[Exception] = ValueError):
    """
    Validate that a value is not None or empty.

    Args:
        value: The value to validate.
        name: Name of the parameter (for error message).
        error_type: Type of exception to raise (default: ValueError).

    Raises:
        ValueError (or specified error_type): If value is None or empty.

    Examples:
        >>> validate_not_empty([], "signals")
        Traceback (most recent call last):
        ...
        ValueError: No signals provided.

        >>> validate_not_empty(None, "epoch_grid")
        Traceback (most recent call last):
        ...
        ValueError: No epoch_grid provided.
    """
    if value is None:
        raise error_type(f"No {name} provided.")

    # Check for empty collections
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        raise error_type(f"No {name} provided.")

    # Check for empty pandas objects
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)) and value.empty:
        raise error_type(f"{name.capitalize()} cannot be empty.")


def validate_type(value: Any, expected_type: Type, name: str):
    """
    Validate that a value is of the expected type.

    Args:
        value: The value to validate.
        expected_type: Expected type or tuple of types.
        name: Name of the parameter (for error message).

    Raises:
        TypeError: If value is not of the expected type.

    Examples:
        >>> validate_type("test", str, "username")
        >>> validate_type(123, str, "username")
        Traceback (most recent call last):
        ...
        TypeError: username must be str, got int
    """
    if not isinstance(value, expected_type):
        expected_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
        actual_name = type(value).__name__
        raise TypeError(f"{name} must be {expected_name}, got {actual_name}")


def validate_all_types(items: List[Any], expected_type: Type, name: str):
    """
    Validate that all items in a list are of the expected type.

    Args:
        items: List of items to validate.
        expected_type: Expected type for all items.
        name: Name of the parameter (for error message).

    Raises:
        TypeError: If any item is not of the expected type.

    Examples:
        >>> from sleep_analysis.signals.time_series_signal import TimeSeriesSignal
        >>> validate_all_types([1, 2, 3], int, "numbers")
        >>> validate_all_types([1, "2", 3], int, "numbers")
        Traceback (most recent call last):
        ...
        TypeError: All numbers must be int instances.
    """
    if not all(isinstance(item, expected_type) for item in items):
        expected_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
        raise TypeError(f"All {name} must be {expected_name} instances.")


def validate_positive(value: Any, name: str, allow_zero: bool = False):
    """
    Validate that a numeric value is positive.

    Args:
        value: The value to validate.
        name: Name of the parameter (for error message).
        allow_zero: Whether to allow zero (default: False).

    Raises:
        ValueError: If value is not positive (or not >= 0 if allow_zero=True).

    Examples:
        >>> validate_positive(5, "window_length")
        >>> validate_positive(0, "window_length")
        Traceback (most recent call last):
        ...
        ValueError: window_length must be positive.

        >>> validate_positive(0, "count", allow_zero=True)
        >>> validate_positive(-1, "count", allow_zero=True)
        Traceback (most recent call last):
        ...
        ValueError: count must be non-negative.
    """
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


def validate_in_range(value: Any, name: str, min_val: Optional[Any] = None, max_val: Optional[Any] = None):
    """
    Validate that a value is within a specified range.

    Args:
        value: The value to validate.
        name: Name of the parameter (for error message).
        min_val: Minimum allowed value (inclusive, optional).
        max_val: Maximum allowed value (inclusive, optional).

    Raises:
        ValueError: If value is outside the specified range.

    Examples:
        >>> validate_in_range(5, "percentage", 0, 100)
        >>> validate_in_range(150, "percentage", 0, 100)
        Traceback (most recent call last):
        ...
        ValueError: percentage must be between 0 and 100.
    """
    if min_val is not None and max_val is not None:
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}.")
    elif min_val is not None:
        if value < min_val:
            raise ValueError(f"{name} must be at least {min_val}.")
    elif max_val is not None:
        if value > max_val:
            raise ValueError(f"{name} must be at most {max_val}.")


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str], df_name: str = "DataFrame"):
    """
    Validate that a DataFrame has all required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        df_name: Name of the DataFrame (for error message).

    Raises:
        ValueError: If any required columns are missing.

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> validate_dataframe_columns(df, ['a', 'b'])
        >>> validate_dataframe_columns(df, ['a', 'c'])
        Traceback (most recent call last):
        ...
        ValueError: DataFrame missing required columns: ['c']
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def validate_datetime_index(df: pd.DataFrame, df_name: str = "DataFrame"):
    """
    Validate that a DataFrame has a DatetimeIndex.

    Args:
        df: DataFrame to validate.
        df_name: Name of the DataFrame (for error message).

    Raises:
        ValueError: If DataFrame does not have a DatetimeIndex.

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2]}, index=pd.DatetimeIndex(['2020-01-01', '2020-01-02']))
        >>> validate_datetime_index(df)
        >>> df2 = pd.DataFrame({'a': [1, 2]})
        >>> validate_datetime_index(df2)
        Traceback (most recent call last):
        ...
        ValueError: DataFrame must have DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{df_name} must have DatetimeIndex")


def validate_parameters(parameters: dict, required: List[str], optional: Optional[List[str]] = None):
    """
    Validate that a parameters dictionary contains all required keys.

    Args:
        parameters: Dictionary of parameters to validate.
        required: List of required parameter names.
        optional: List of optional parameter names (for documentation purposes).

    Raises:
        ValueError: If any required parameters are missing.

    Examples:
        >>> params = {'window_length': '30s', 'step_size': '15s'}
        >>> validate_parameters(params, ['window_length'])
        >>> validate_parameters(params, ['window_length', 'aggregations'])
        Traceback (most recent call last):
        ...
        ValueError: Missing required parameters: ['aggregations']
    """
    missing = [key for key in required if key not in parameters]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")


def validate_timedelta_positive(td: pd.Timedelta, name: str):
    """
    Validate that a Timedelta is positive.

    Args:
        td: Timedelta to validate.
        name: Name of the parameter (for error message).

    Raises:
        ValueError: If Timedelta is not positive.

    Examples:
        >>> validate_timedelta_positive(pd.Timedelta('30s'), 'window_length')
        >>> validate_timedelta_positive(pd.Timedelta('0s'), 'window_length')
        Traceback (most recent call last):
        ...
        ValueError: window_length must be positive.
    """
    if td <= pd.Timedelta(0):
        raise ValueError(f"{name} must be positive.")
