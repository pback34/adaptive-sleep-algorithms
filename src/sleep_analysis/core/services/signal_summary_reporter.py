"""
Signal summary reporter service for generating signal and feature summaries.

This module provides the SignalSummaryReporter class, which handles:
- Generating summary tables of signals and features
- Formatting metadata and calculated fields
- Storing and retrieving summary DataFrames
- Pretty-printing summaries
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import fields
from enum import Enum

# Third-party imports
import pandas as pd

# Local application imports
from ..metadata import TimeSeriesMetadata, FeatureMetadata
from ...signals.time_series_signal import TimeSeriesSignal
from ...features.feature import Feature

# Initialize logger for the module
logger = logging.getLogger(__name__)


class SignalSummaryReporter:
    """
    Service for generating summaries of signals and features.

    This service handles:
    - Creating summary DataFrames with metadata and calculated fields
    - Formatting cells for display (enums, timestamps, lists, etc.)
    - Storing summary DataFrames for retrieval
    - Pretty-printing summaries to console

    Example:
        >>> reporter = SignalSummaryReporter()
        >>> summary_df = reporter.summarize_signals(
        ...     time_series_signals={'hr_0': hr_signal},
        ...     features={'hr_features': hr_feature},
        ...     print_summary=True
        ... )
    """

    def __init__(self):
        """Initialize the SignalSummaryReporter."""
        self._summary_dataframe: Optional[pd.DataFrame] = None
        self._summary_dataframe_params: Optional[Dict[str, Any]] = None

    def summarize_signals(
        self,
        time_series_signals: Dict[str, TimeSeriesSignal],
        features: Dict[str, Feature],
        fields_to_include: Optional[List[str]] = None,
        print_summary: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Generates a summary table of TimeSeriesSignals and Features.

        Creates a DataFrame with metadata fields and calculated fields for all
        signals and features in the collection. Optionally prints a formatted
        version to the console.

        Args:
            time_series_signals: Dictionary of TimeSeriesSignal objects
            features: Dictionary of Feature objects
            fields_to_include: Optional list of specific fields to include
                             (default: all available fields)
            print_summary: Whether to print the summary to console (default: True)

        Returns:
            DataFrame with signal/feature summary information

        Example:
            >>> summary = reporter.summarize_signals(
            ...     time_series_signals={'hr_0': hr_sig},
            ...     features={'hr_feat': hr_feature},
            ...     fields_to_include=['name', 'signal_type', 'data_shape'],
            ...     print_summary=True
            ... )
        """
        # Combine both signal types for summary generation
        all_items = {**time_series_signals, **features}

        if not all_items:
            logger.info("Signal collection is empty. No summary to generate.")
            self._summary_dataframe = pd.DataFrame()
            self._summary_dataframe_params = None
            if print_summary:
                print("Signal collection is empty.")
            return self._summary_dataframe

        # Determine fields
        default_ts_fields = [f.name for f in fields(TimeSeriesMetadata)]
        default_feat_fields = [f.name for f in fields(FeatureMetadata)]

        # Combine and deduplicate
        common_fields = set(default_ts_fields) & set(default_feat_fields)
        ts_only_fields = set(default_ts_fields) - common_fields
        feat_only_fields = set(default_feat_fields) - common_fields

        # Add calculated fields
        calculated_fields = ['source_files_count', 'operations_count', 'feature_names_count', 'data_shape']
        default_fields_ordered = ['key', 'item_type'] + sorted(list(common_fields)) + \
                                 sorted(list(ts_only_fields)) + sorted(list(feat_only_fields)) + \
                                 sorted(calculated_fields)

        fields_for_summary = fields_to_include if fields_to_include is not None else default_fields_ordered
        logger.info(f"Generating summary. Fields requested: {'Default' if fields_to_include is None else fields_to_include}")
        logger.debug(f"Actual fields being processed: {fields_for_summary}")

        # Generate summary data
        summary_data = []
        valid_ts_meta_fields = {f.name for f in fields(TimeSeriesMetadata)}
        valid_feat_meta_fields = {f.name for f in fields(FeatureMetadata)}

        for key, item in all_items.items():
            row_data = {'key': key}
            metadata_obj = item.metadata
            is_feature = isinstance(item, Feature)
            row_data['item_type'] = 'Feature' if is_feature else 'TimeSeries'
            valid_meta_fields = valid_feat_meta_fields if is_feature else valid_ts_meta_fields

            for field in fields_for_summary:
                if field in ['key', 'item_type']:
                    continue

                value = None
                try:
                    # Calculated fields
                    if field == 'source_files_count':
                        value = len(metadata_obj.source_files) if hasattr(metadata_obj, 'source_files') and metadata_obj.source_files else 0
                    elif field == 'operations_count':
                        value = len(metadata_obj.operations) if hasattr(metadata_obj, 'operations') and metadata_obj.operations else 0
                    elif field == 'feature_names_count':
                        value = len(metadata_obj.feature_names) if hasattr(metadata_obj, 'feature_names') and metadata_obj.feature_names else (0 if is_feature else None)
                    elif field == 'data_shape':
                        try:
                            data = item.get_data()
                            value = data.shape if hasattr(data, 'shape') else None
                        except Exception:
                            value = None
                    # Metadata fields
                    elif field in valid_meta_fields:
                        value = getattr(metadata_obj, field, None)
                    else:
                        value = None

                    row_data[field] = value

                except Exception as e:
                    logger.warning(f"Error accessing field '{field}' for item '{key}': {e}")
                    row_data[field] = None

            summary_data.append(row_data)

        # Create raw DataFrame
        raw_summary_df = pd.DataFrame(summary_data)

        # Ensure all requested columns exist
        for col in fields_for_summary:
            if col not in raw_summary_df.columns:
                raw_summary_df[col] = None

        # Reorder columns
        raw_summary_df = raw_summary_df[['key'] + [f for f in fields_for_summary if f != 'key']]
        raw_summary_df = raw_summary_df.set_index('key').sort_index()

        # Store raw DataFrame
        self._summary_dataframe = raw_summary_df.copy()
        self._summary_dataframe_params = {
            'fields_to_include': fields_for_summary,
            'print_summary': print_summary
        }
        logger.info(f"Stored summary DataFrame with shape {self._summary_dataframe.shape}")

        # Handle printing
        if print_summary:
            formatted_summary_df = raw_summary_df.copy()
            # Apply formatting
            for col in formatted_summary_df.columns:
                logger.debug(f"Formatting summary column: '{col}'")
                try:
                    formatted_summary_df[col] = formatted_summary_df[col].apply(
                        lambda cell_value: self._format_summary_cell(cell_value, col)
                    )
                except ValueError as e:
                    if "The truth value of a Series is ambiguous" in str(e):
                        logger.error(
                            f"ValueError while formatting column '{col}'. "
                            f"Cell likely contains a Series."
                        )
                    raise e

            print("\n--- Signal Collection Summary ---")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(formatted_summary_df.to_string())
            print("-------------------------------\n")

        return self._summary_dataframe

    def get_summary_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the stored summary DataFrame.

        Returns:
            Stored summary DataFrame or None if no summary has been generated

        Example:
            >>> summary_df = reporter.get_summary_dataframe()
        """
        return self._summary_dataframe

    def get_summary_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the parameters used for the stored summary.

        Returns:
            Dictionary of summary parameters or None

        Example:
            >>> params = reporter.get_summary_params()
            >>> print(params['fields_to_include'])
        """
        return self._summary_dataframe_params

    def _format_summary_cell(self, x: Any, col_name: str) -> Any:
        """
        Helper function to format a single cell for summary DataFrame printout.

        Handles special formatting for:
        - Lists, tuples, dicts: Shows type and length
        - Enums: Shows enum name
        - Timestamps: Formatted datetime string
        - Timedeltas: String representation
        - None/NaN: Shows 'N/A'

        Args:
            x: The cell value to format
            col_name: Name of the column (for context-specific formatting)

        Returns:
            Formatted value suitable for display
        """
        logger.debug(f"_format_summary_cell received type: {type(x)} for column '{col_name}'")

        # Handle unexpected Series/DataFrame
        if isinstance(x, (pd.Series, pd.DataFrame)):
            logger.error(
                f"Unexpected Series/DataFrame in _format_summary_cell for column '{col_name}'. "
                f"Value:\n{x}"
            )
            return "<ERROR: Unexpected Series/DataFrame>"

        # Handle lists, tuples, dicts
        if isinstance(x, (list, tuple, dict)):
            try:
                if not x:
                    return f"<{type(x).__name__} len=0>"
                else:
                    return f"<{type(x).__name__} len={len(x)}>"
            except TypeError:
                return f"<{type(x).__name__}>"

        # Handle Enums
        elif isinstance(x, Enum):
            return x.name

        # Handle Timestamps/Datetimes
        elif isinstance(x, (pd.Timestamp, datetime)):
            try:
                if pd.isna(x):
                    return 'NaT'
                return x.strftime('%Y-%m-%d %H:%M:%S %Z')
            except ValueError:
                try:
                    return x.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return str(x)

        # Handle Timedeltas
        elif isinstance(x, pd.Timedelta):
            if pd.isna(x):
                return 'NaT'
            return str(x)

        # Handle specific column formatting
        elif col_name == 'data_shape' and isinstance(x, tuple):
            return str(x)

        # Handle None/NaN
        elif pd.isna(x):
            return 'N/A'

        # Fallback for other scalar types
        else:
            return x
