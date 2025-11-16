"""Utility functions for signal processing."""

from .logging import setup_logging, get_logger, log_operation, OperationLogger
import pandas as pd
import logging
from typing import Dict, Type, Any, Optional
from enum import Enum
# Consider adding pytz or similar for validation if needed
# import pytz

def map_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename DataFrame columns according to the provided mapping.
    
    Args:
        df: Input DataFrame.
        column_mapping: Dictionary mapping target column names to source column names.
    
    Returns:
        DataFrame with renamed columns.
    """
    # Create a mapping from source columns to target columns
    rename_dict = {}
    for target_col, source_col in column_mapping.items():
        if source_col in df.columns:
            rename_dict[source_col] = target_col
    
    # Rename the columns
    result = df.rename(columns=rename_dict)
    
    # Only keep the columns that were mapped if there are any mapped columns
    mapped_cols = list(rename_dict.values())
    if mapped_cols:
        return result[mapped_cols]
    return result

def debug_multiindex(mi, logger=None) -> None:
    """
    Print detailed debug information about a MultiIndex.
    
    Args:
        mi: The MultiIndex to debug.
        logger: Optional logger to use instead of print.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Basic MultiIndex info
    logger.debug(f"MultiIndex type: {type(mi)}")
    logger.debug(f"MultiIndex names: {mi.names}")
    logger.debug(f"MultiIndex levels count: {mi.nlevels}")
    logger.debug(f"MultiIndex length: {len(mi)}")
    
    # Check if the MultiIndex is properly formed
    has_nan_values = any(pd.isna(mi.get_level_values(i)).any() for i in range(mi.nlevels))
    logger.debug(f"MultiIndex has NaN/None values: {has_nan_values}")
    
    # Check for empty strings
    has_empty_strings = any(
        (mi.get_level_values(i).astype(str) == "").any() 
        for i in range(mi.nlevels)
    )
    logger.debug(f"MultiIndex has empty strings: {has_empty_strings}")
    
    # Check level names for None/NaN
    if mi.names:
        none_names = [i for i, name in enumerate(mi.names) if name is None]
        if none_names:
            logger.debug(f"Levels with None names: {none_names}")
    
    # Print all unique values for each level
    for i in range(mi.nlevels):
        level_values = mi.get_level_values(i).unique().tolist()
        logger.debug(f"Level {i} ({mi.names[i] if i < len(mi.names) else 'unnamed'}) values: {level_values[:10]}")
    
    # Show complete tuples for first few indices
    sample_tuples = [mi[i] for i in range(min(5, len(mi)))]
    logger.debug(f"Complete tuples (first 5):")
    for i, tup in enumerate(sample_tuples):
        logger.debug(f"  Tuple {i}: {tup}")
    
    # Check for duplicates in the tuples
    has_duplicates = len(mi) != len(set(mi))
    logger.debug(f"MultiIndex has duplicate tuples: {has_duplicates}")
    
    # Check for correct tuple lengths
    incorrect_tuples = [i for i, tup in enumerate(mi) if len(tup) != mi.nlevels]
    if incorrect_tuples:
        logger.debug(f"Found {len(incorrect_tuples)} tuples with incorrect length (expected {mi.nlevels})")
        if incorrect_tuples:
            sample_bad_tuple = mi[incorrect_tuples[0]]
            logger.debug(f"Example bad tuple at index {incorrect_tuples[0]}: {sample_bad_tuple} (length {len(sample_bad_tuple)})")

def debug_csv_file(csv_path, logger=None):
    """
    Analyze a CSV file and provide detailed debugging information about its structure.
    
    Args:
        csv_path: Path to the CSV file to analyze
        logger: Optional logger to use instead of print
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    import os
    import pandas as pd
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    logger.debug(f"Analyzing CSV file: {csv_path}")
    
    # Read raw content
    with open(csv_path, 'r') as f:
        first_lines = [line.strip() for line in f if line.strip()][:10]
    
    logger.debug("CSV file content (first 10 non-empty lines):")
    for i, line in enumerate(first_lines):
        logger.debug(f"Line {i}: {line}")
    
    # Detailed analysis of CSV structure
    if len(first_lines) >= 2:
        # Check for potential MultiIndex header structure
        first_header = first_lines[0].split(',')
        second_header = first_lines[1].split(',')
        
        # Compare header rows
        logger.debug(f"First header row has {len(first_header)} elements")
        logger.debug(f"Second header row has {len(second_header)} elements")
        
        # Check for empty cells in header rows (common in MultiIndex CSVs)
        empty_cells_first = [i for i, cell in enumerate(first_header) if not cell]
        empty_cells_second = [i for i, cell in enumerate(second_header) if not cell]
        
        logger.debug(f"Empty cells in first header row: {empty_cells_first}")
        logger.debug(f"Empty cells in second header row: {empty_cells_second}")
        
        # Look for potential level names (first column in each header row)
        if len(first_header) > 0:
            logger.debug(f"Potential first level name: {first_header[0]}")
        if len(second_header) > 0:
            logger.debug(f"Potential second level name: {second_header[0]}")
    
    # Detect potential MultiIndex structure
    comma_counts = [line.count(',') for line in first_lines[:3]]
    logger.debug(f"Comma counts in first lines: {comma_counts}")
    
    # Attempt to read with different header options
    try:
        # Try reading with default parameters
        df_default = pd.read_csv(csv_path)
        logger.debug(f"Default read shape: {df_default.shape}, columns: {list(df_default.columns)}")
    except Exception as e:
        logger.error(f"Error with default read: {e}")
    
    try:
        # Try reading with index in first column
        df_with_index = pd.read_csv(csv_path, index_col=0)
        logger.debug(f"Read with index_col=0 shape: {df_with_index.shape}, columns: {list(df_with_index.columns)}")
    except Exception as e:
        logger.error(f"Error reading with index_col=0: {e}")
    
    # Try reading with MultiIndex columns
    for header_count in range(1, 7):  # Try up to 6 header rows
        try:
            # Try with different header row counts
            header_rows = list(range(header_count))
            df_multi = pd.read_csv(csv_path, header=header_rows, index_col=0)
            logger.debug(f"Read with header={header_rows} shape: {df_multi.shape}")
            if isinstance(df_multi.columns, pd.MultiIndex):
                logger.debug(f"Successfully detected MultiIndex columns with {header_count} levels")
                debug_multiindex(df_multi.columns, logger)
                
                # Check the first data row to make sure it was parsed correctly
                if len(df_multi) > 0:
                    logger.debug(f"First data row: {df_multi.iloc[0].to_dict()}")
                break
        except Exception as e:
            logger.debug(f"Error reading with header={header_rows}: {e}")
            
    # Try explicitly specifying header names to see if that helps
    try:
        # Try to infer appropriate column names from file
        with open(csv_path, 'r') as f:
            # Read first few lines to analyze header structure
            header_lines = [next(f) for _ in range(10) if f]
        
        # Count non-empty header lines at the beginning
        header_count = 0
        for line in header_lines:
            if ',' in line and not line.strip().split(',')[0].replace('.', '').isdigit():
                header_count += 1
            else:
                break
        
        if header_count > 1:
            logger.debug(f"Detected {header_count} header lines based on content analysis")
            df_explicit = pd.read_csv(csv_path, header=list(range(header_count)), index_col=0)
            logger.debug(f"Read with explicit header count {header_count}: shape={df_explicit.shape}")
            if isinstance(df_explicit.columns, pd.MultiIndex):
                logger.debug("Successfully created MultiIndex with explicit header count")
                debug_multiindex(df_explicit.columns, logger)
    except Exception as e:
        logger.debug(f"Error with explicit header detection: {e}")

def convert_timestamp_format(series: pd.Series, source_format: str = None, target_format: str = "%Y-%m-%d %H:%M:%S") -> pd.Series:
    """
    Convert a timestamp series to a standard format.
    
    Args:
        series: Series containing timestamps.
        source_format: Format of the input timestamps (optional).
        target_format: Desired output format (from CollectionMetadata).
    
    Returns:
        Series with standardized timestamps.
    
    Raises:
        ValueError: If conversion fails.
    """
    try:
        if source_format:
            return pd.to_datetime(series, format=source_format).dt.strftime(target_format)
        else:
            return pd.to_datetime(series).dt.strftime(target_format)
    except Exception as e:
        raise ValueError(f"Failed to convert timestamps: {e}")


def str_to_enum(value_str: str, enum_class: Type[Enum]) -> Enum:
    """
    Convert a string to an enum value using case-insensitive matching.
    
    Args:
        value_str: String representation of the enum value
        enum_class: The enum class to convert to
            
    Returns:
        The corresponding enum value
            
    Raises:
        ValueError: If the string does not match any enum value
    """
    # Normalize the input string (lowercase and standardize separators)
    value_str_normalized = value_str.lower().replace('_', ' ').replace('-', ' ')
    
    # First try direct match on enum name (case-insensitive)
    try:
        return enum_class[value_str.upper()]
    except KeyError:
        pass
            
    # Next, try matching on normalized enum values
    for member in enum_class:
        # Normalize the enum value string for comparison
        member_value_normalized = member.value.lower().replace('_', ' ').replace('-', ' ')
        if member_value_normalized == value_str_normalized:
            return member
                
    # If we get here, try fuzzy matching (helpful error message)
    close_matches = []
    for member in enum_class:
        # Check both member name and value
        member_name_normalized = member.name.lower().replace('_', ' ')
        member_value_normalized = member.value.lower().replace('_', ' ').replace('-', ' ')
            
        if (value_str_normalized in member_name_normalized or 
            member_name_normalized in value_str_normalized or
            value_str_normalized in member_value_normalized or
            member_value_normalized in value_str_normalized):
            close_matches.append(f"{member.name} ({member.value})")
                
    # Provide helpful error message with available options
    error_msg = f"No matching {enum_class.__name__} value for: '{value_str}'\n"
    error_msg += f"Available options: {', '.join([f'{m.name} ({m.value})' for m in enum_class])}"
        
    if close_matches:
        error_msg += f"\nDid you mean one of these? {', '.join(close_matches)}"
            
    raise ValueError(error_msg)


# Centralized timestamp standardization function (from revised plan)
def standardize_timestamp(
    df: pd.DataFrame,
    timestamp_col: str,
    origin_timezone: Optional[str], # Can be None if source is already aware
    target_timezone: str, # Resolved target timezone string
    set_index: bool = True
) -> pd.DataFrame:
    """
    Standardize the timestamp column, handling timezone localization and conversion.

    Args:
        df: Input DataFrame.
        timestamp_col: Name of the timestamp column.
        origin_timezone: Timezone of the source data if naive (e.g., 'America/New_York').
                         If None and data is naive, localization might be ambiguous.
        target_timezone: The target timezone for the final representation (e.g., 'UTC').
        set_index: Whether to set the timestamp column as the DataFrame index.

    Returns:
        DataFrame with standardized, timezone-aware timestamp (as index or column).

    Raises:
        ValueError: If timestamp parsing or timezone conversion fails.
    """
    logger = logging.getLogger(__name__)
    df = df.copy() # Work on a copy

    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found in DataFrame. Available columns: {list(df.columns)}")
        raise ValueError(f"Timestamp column '{timestamp_col}' not found. Available columns: {list(df.columns)}")

    # Optional: Validate timezone strings early
    # try:
    #     if origin_timezone: pytz.timezone(origin_timezone)
    #     pytz.timezone(target_timezone)
    # except pytz.UnknownTimeZoneError as e:
    #     logger.error(f"Invalid timezone provided: {e}")
    #     raise ValueError(f"Invalid timezone provided: {e}") from e

    # Parse timestamp column to datetime objects
    try:
        # Try converting existing column first, might already be datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
             df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        # Handle potential parsing errors if needed (e.g., check for NaT)
        if df[timestamp_col].isnull().any():
             logger.warning(f"Timestamp column '{timestamp_col}' contains null values after parsing.")
             # Decide on handling: dropna? fillna? For now, keep them.
    except Exception as parse_err:
        logger.error(f"Failed to parse timestamp column '{timestamp_col}': {parse_err}")
        raise ValueError(f"Failed to parse timestamp column '{timestamp_col}': {parse_err}") from parse_err


    # Localize naive timestamps to origin_timezone if provided
    if df[timestamp_col].dt.tz is None:
        if origin_timezone:
            try:
                logger.debug(f"Localizing naive timestamp to origin timezone: {origin_timezone}")
                # Use infer ambiguous time during DST transitions
                df[timestamp_col] = df[timestamp_col].dt.tz_localize(origin_timezone, ambiguous='infer', nonexistent='raise')
            except Exception as tz_err:
                # More specific error catching could be added (e.g., pytz.NonExistentTimeError)
                logger.warning(f"Could not localize timestamp to {origin_timezone}: {tz_err}. Proceeding as naive (will likely convert to target assuming UTC).")
        else:
            # If origin_timezone is None and timestamp is naive, assume UTC before converting.
            logger.warning(f"Timestamp column '{timestamp_col}' is naive but no origin_timezone was specified. Assuming UTC before converting to target.")
            try:
                df[timestamp_col] = df[timestamp_col].dt.tz_localize('UTC')
                logger.debug("Successfully localized naive timestamp to UTC as fallback.")
            except Exception as utc_loc_err:
                # This might happen if timestamps are ambiguous around DST changes even in UTC, though less likely.
                logger.error(f"Failed to localize naive timestamp to UTC: {utc_loc_err}")
                raise ValueError(f"Failed to localize naive timestamp to UTC: {utc_loc_err}") from utc_loc_err

    # Convert to target_timezone (handles both originally naive localized and already aware timestamps)
    # Only proceed if the timestamp column is now timezone-aware
    if df[timestamp_col].dt.tz is not None:
        try:
            logger.debug(f"Converting timestamp to target timezone: {target_timezone}")
            df[timestamp_col] = df[timestamp_col].dt.tz_convert(target_timezone)
        except Exception as tz_err:
            logger.error(f"Could not convert timestamp timezone to {target_timezone}: {tz_err}. Returning with original/localized timezone.")
            # Decide on error handling: raise error or return partially converted? Raising is safer.
            raise ValueError(f"Failed to convert timestamp to target timezone {target_timezone}: {tz_err}") from tz_err

    # Set as index if requested
    if set_index:
        df.set_index(df[timestamp_col], inplace=True)
        df.index.name = 'timestamp' # Ensure index is named
        # Check if original column still exists before dropping
        if timestamp_col in df.columns:
             df = df.drop(columns=[timestamp_col]) # Drop original column after setting index

    return df


__all__ = ['setup_logging', 'get_logger', 'log_operation', 'OperationLogger',
           'standardize_timestamp', 'map_columns', 'debug_multiindex', 'debug_csv_file',
           'str_to_enum', 'convert_timestamp_format']
