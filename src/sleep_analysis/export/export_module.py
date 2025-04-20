"""
Export module implementation.

This module contains the ExportModule class which provides functionality for exporting
signals and their metadata to various file formats.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict
import pickle
import warnings
import logging # Ensure logging is imported
from enum import Enum # Added import

# Updated core imports for refactored metadata and feature classes
from ..core import (
    SignalCollection, CollectionMetadata, TimeSeriesMetadata, FeatureMetadata
)
# Import specific signal/feature types needed
from ..signals.time_series_signal import TimeSeriesSignal
from ..features.feature import Feature
# Removed SignalData import as it's less directly used
# Removed FeatureSignal import as Feature is now used


class ExportModule:
    """
    Class for exporting signals and metadata to various formats.
    
    Supports exporting to Excel (.xlsx), CSV (.csv), Pickle (.pkl), and HDF5 (.h5).
    """
    
    SUPPORTED_FORMATS = ["excel", "csv", "pickle", "hdf5"]
    
    def __init__(self, collection: SignalCollection):
        """
        Initialize the ExportModule with a SignalCollection.
        
        Args:
            collection: The SignalCollection containing signals to export.
        """
        self.collection = collection

    # Update the export method signature and logic
    def export(self, formats: List[str], output_dir: str,
               content: Union[str, List[str]]) -> None:
        """
        Export signals, features, combined data, and summary based on content specification.

        Args:
            formats: List of formats to export to. Supported: "excel", "csv", "pickle", "hdf5".
            output_dir: Directory where exported files will be saved.
            content: Specifies what to export. Can be a string or list of strings:
                     - "all_ts": All individual non-temporary TimeSeriesSignals.
                     - "all_features": All individual Features.
                     - "combined_ts": The combined time-series dataframe.
                     - "combined_features": The combined feature matrix.
                     - "summary": The summary dataframe.
                     - <key>: A specific signal or feature key (e.g., "hr_0", "accel_stats_1").
                     If specific keys are provided, other keywords like "all_ts" in the
                     same list might be ignored for that specific export task (warning issued).

        Raises:
            ValueError: If an unsupported format or content keyword is specified,
                        or if specified signal/feature keys are not found.
            IOError: If there are issues creating output directory or writing files.
        """
        logger = logging.getLogger(__name__)

        # --- Validate Formats ---
        for fmt in formats:
            if fmt.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {fmt}. Supported formats: {self.SUPPORTED_FORMATS}")

        os.makedirs(output_dir, exist_ok=True)

        # --- Process Content Specification ---
        if isinstance(content, str):
            content_list = [content]
        elif isinstance(content, list):
            content_list = content
        else:
            raise ValueError(f"Invalid 'content' type: {type(content)}. Must be string or list.")

        # Initialize flags and lists
        export_individual_ts = False
        export_individual_features = False
        export_combined_ts = False
        export_combined_features = False
        export_summary = False
        specific_keys_to_export = []
        valid_content_keywords = {"all_ts", "all_features", "combined_ts", "combined_features", "summary"}

        # Parse the content list
        for item in content_list:
            item_lower = item.lower()
            if item_lower == "all_ts":
                export_individual_ts = True
            elif item_lower == "all_features":
                export_individual_features = True
            elif item_lower == "combined_ts":
                export_combined_ts = True
            elif item_lower == "combined_features":
                export_combined_features = True
            elif item_lower == "summary":
                export_summary = True
            elif item in self.collection.time_series_signals or item in self.collection.features:
                specific_keys_to_export.append(item)
            else:
                # Check if it's a base name before raising error
                is_base_name = False
                search_space = {**self.collection.time_series_signals, **self.collection.features}
                for existing_key in search_space:
                    if existing_key.startswith(f"{item}_") and existing_key[len(item)+1:].isdigit():
                        specific_keys_to_export.append(existing_key)
                        is_base_name = True
                if not is_base_name:
                    # If it's not a known keyword, existing key, or base name, raise error
                    raise ValueError(f"Invalid content item: '{item}'. Must be a valid keyword {valid_content_keywords}, an existing signal/feature key, or a base name.")

        # Deduplicate specific keys resolved from base names
        specific_keys_to_export = sorted(list(set(specific_keys_to_export)))

        # --- Determine What to Actually Export ---
        signals_to_process = {}
        metadata_keys_to_include = set() # Use a set for easy deduplication

        # If specific keys are requested, prioritize them
        if specific_keys_to_export:
            # Warn if keywords were mixed with specific keys, as keys take precedence
            if export_individual_ts or export_individual_features or export_combined_ts or export_combined_features or export_summary:
                 logger.warning(f"Export 'content' for '{output_dir}' includes specific keys ({specific_keys_to_export}) "
                                f"and keywords ('all_ts', 'all_features', etc.). Exporting *only* the specified keys/base names.")
                 # Reset flags if specific keys are given
                 export_individual_ts = False
                 export_individual_features = False
                 export_combined_ts = False
                 export_combined_features = False
                 export_summary = False

            logger.info(f"Exporting specific signals/features: {specific_keys_to_export}")
            search_space = {**self.collection.time_series_signals, **self.collection.features}
            missing_keys = [key for key in specific_keys_to_export if key not in search_space]
            if missing_keys:
                 raise ValueError(f"Specified signal/feature key(s) not found in collection for export: {missing_keys}")

            signals_to_process = {key: search_space[key] for key in specific_keys_to_export}
            metadata_keys_to_include.update(specific_keys_to_export)
            # Ensure flags reflect that we are exporting individuals (even if specific ones)
            export_individual_ts = any(isinstance(s, TimeSeriesSignal) for s in signals_to_process.values())
            export_individual_features = any(isinstance(s, Feature) for s in signals_to_process.values())


        else: # No specific keys, process flags
            logger.info(f"Processing export flags: individual_ts={export_individual_ts}, "
                        f"individual_features={export_individual_features}, combined_ts={export_combined_ts}, "
                        f"combined_features={export_combined_features}, summary={export_summary}")

            if export_individual_ts:
                ts_signals = {k: s for k, s in self.collection.time_series_signals.items() if not s.metadata.temporary}
                if not ts_signals:
                     logger.warning(f"Requested 'all_ts' export, but no non-temporary TimeSeriesSignals found in '{output_dir}'.")
                signals_to_process.update(ts_signals)
                metadata_keys_to_include.update(ts_signals.keys())

            if export_individual_features:
                features = self.collection.features # Features don't have 'temporary' flag currently
                if not features:
                     logger.warning(f"Requested 'all_features' export, but no Features found in '{output_dir}'.")
                signals_to_process.update(features)
                metadata_keys_to_include.update(features.keys())

            # Add metadata keys for combined data if requested (provides context)
            if export_combined_ts:
                 metadata_keys_to_include.update(k for k, s in self.collection.time_series_signals.items() if not s.metadata.temporary)
            if export_combined_features:
                 metadata_keys_to_include.update(self.collection.features.keys())
            # Summary doesn't add metadata keys itself

        # --- Retrieve Dataframes ---
        combined_ts_df = None
        if export_combined_ts:
            combined_ts_df = self.collection.get_stored_combined_dataframe()
            if combined_ts_df is None:
                logger.warning("Combined time-series dataframe requested ('combined_ts') but not found or generated. Skipping combined TS export.")
                export_combined_ts = False # Disable if not found
            elif combined_ts_df.empty:
                logger.warning("Combined time-series dataframe requested ('combined_ts') but is empty. Proceeding with export of empty file.")
                # Keep export_combined_ts = True

        combined_features_df = None
        if export_combined_features:
            combined_features_df = self.collection.get_stored_combined_feature_matrix()
            if combined_features_df is None:
                logger.warning("Combined feature matrix requested ('combined_features') but not found or generated. Skipping combined features export.")
                export_combined_features = False # Disable if not found
            elif combined_features_df.empty:
                logger.warning("Combined feature matrix requested ('combined_features') but is empty. Proceeding with export of empty file.")
                # Keep export_combined_features = True

        summary_df = None
        if export_summary:
            summary_df = getattr(self.collection, '_summary_dataframe', None)
            if summary_df is None:
                logger.warning("Summary export requested ('summary'), but summary dataframe not found in collection. Ensure 'summarize_signals' step was run.")
                export_summary = False # Disable summary export if not found
            elif summary_df.empty:
                 logger.warning("Summary export requested ('summary'), but stored summary dataframe is empty. Skipping summary export.")
                 export_summary = False # Disable summary export if empty

        # --- Serialize Metadata ---
        # Always serialize metadata for ALL signals in the collection, regardless of export content flags.
        # Pass None to _serialize_metadata to trigger inclusion of all signals.
        logger.debug("Serializing metadata for all signals in the collection.")
        serialized_metadata = self._serialize_metadata(signal_keys_to_include=None)

        # --- Perform Export for Each Format ---
        # Check if there's anything to export at all (based on data, not metadata)
        if not signals_to_process and combined_ts_df is None and combined_features_df is None and summary_df is None:
             logger.warning(f"Nothing to export for task targeting directory '{output_dir}' based on content specification: {content_list}. Skipping format processing.")
             return

        for fmt in formats:
            fmt = fmt.lower()
            # Pass the potentially filtered signals_to_process
            # Pass dataframes only if their corresponding flag is still True
            combined_ts_arg = combined_ts_df if export_combined_ts else None
            combined_features_arg = combined_features_df if export_combined_features else None
            summary_arg = summary_df if export_summary else None

            try:
                if fmt == "excel":
                    self._export_excel(output_dir, signals_to_process, combined_ts_arg, combined_features_arg, summary_arg, serialized_metadata)
                elif fmt == "csv":
                    self._export_csv(output_dir, signals_to_process, combined_ts_arg, combined_features_arg, summary_arg, serialized_metadata)
                elif fmt == "pickle":
                    self._export_pickle(output_dir, signals_to_process, combined_ts_arg, combined_features_arg, summary_arg, serialized_metadata)
                elif fmt == "hdf5":
                    self._export_hdf5(output_dir, signals_to_process, combined_ts_arg, combined_features_arg, summary_arg, serialized_metadata)
            except Exception as e:
                 logger.error(f"Failed to export to format '{fmt}' in directory '{output_dir}': {e}", exc_info=True)
                 # Re-raise or handle based on strictness? Re-raise for now.
                 raise IOError(f"Export to format '{fmt}' failed.") from e


    def _format_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp index to a column while preserving datetime objects.
        
        Args:
            df: DataFrame with timestamp index.
            
        Returns:
            DataFrame with timestamp as a column (not index).
        """
        import logging
        import pandas as pd
        
        logger = logging.getLogger(__name__)
        
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Handle DatetimeIndex 
        if isinstance(result_df.index, pd.DatetimeIndex):
            # Sort index first to avoid lexsort warning
            if isinstance(result_df.index, pd.MultiIndex):
                result_df = result_df.sort_index()
                
            # Check for duplicates before resetting index
            if result_df.index.duplicated().any():
                dupes = result_df.index.duplicated().sum()
                logger.warning(f"Found {dupes} duplicate timestamps in index. Keeping first occurrence.")
                result_df = result_df.loc[~result_df.index.duplicated(keep='first')]
                
            # Reset index to convert DatetimeIndex to column, preserving datetime type
            # Set the name of the index column to 'timestamp'
            result_df = result_df.reset_index(names='timestamp')
            logger.debug(f"Reset index to column named 'timestamp', dtype: {result_df['timestamp'].dtype}, TZ: {result_df['timestamp'].dt.tz}")

            # Ensure it's still datetime64[ns, TZ] type
            if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
                 logger.warning(f"Column 'timestamp' is not datetime after reset_index: {result_df['timestamp'].dtype}. Attempting conversion.")
                 try:
                     # Preserve original timezone if possible during conversion
                     original_tz = df.index.tz
                     result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                     if original_tz and result_df['timestamp'].dt.tz is None:
                          result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(original_tz)
                     elif original_tz and result_df['timestamp'].dt.tz != original_tz:
                          result_df['timestamp'] = result_df['timestamp'].dt.tz_convert(original_tz)
                     logger.debug(f"Successfully converted timestamp column back to datetime. New dtype: {result_df['timestamp'].dtype}, TZ: {result_df['timestamp'].dt.tz}")
                 except Exception as e:
                     logger.error(f"Could not convert timestamp column back to datetime: {e}")
                     # Decide how to handle - raise error or proceed with potentially incorrect type?
                     # raise ValueError(f"Failed to ensure timestamp column is datetime: {e}") from e

        logger.debug(f"Formatted dataframe for export has {len(result_df)} rows")
        return result_df

    def _serialize_metadata(self, signal_keys_to_include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Serialize collection and relevant signal metadata to a JSON-compatible format.

        Args:
            signal_keys_to_include: Optional list of signal keys. If provided, only metadata
                                    for these signals will be included. If None, metadata for
                                    all signals in the collection is included.

        Returns:
            Dictionary containing serialized metadata.
        """
        logger = logging.getLogger(__name__) # Keep logger init here for use within this method

        # Helper function to recursively convert non-serializable types
        def make_json_serializable(obj, _depth=0, _path="root"): # Add depth and path tracking
            # Limit recursion depth manually as an extra safeguard, though the root cause needs fixing
            if _depth > 50: # Adjust limit as needed
                logger.warning(f"Serialization depth limit exceeded at path: {_path}. Returning string representation.")
                return f"<Serialization Depth Limit Exceeded: {type(obj).__name__}>"

            # Removed per-field DEBUG log: logger.debug(f"{'  ' * _depth}Serializing path: {_path}, type: {type(obj).__name__}, value (preview): {str(obj)[:100]}")

            if isinstance(obj, dict):
                # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Entering dict at path: {_path}")
                return {key: make_json_serializable(value, _depth + 1, f"{_path}.{key}") for key, value in obj.items()}
            elif isinstance(obj, list):
                # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Entering list at path: {_path}")
                return [make_json_serializable(item, _depth + 1, f"{_path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, tuple):
                 # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Entering tuple at path: {_path}")
                 return [make_json_serializable(item, _depth + 1, f"{_path}[{i}]") for i, item in enumerate(obj)]
            elif hasattr(obj, 'isoformat') and callable(obj.isoformat):  # datetime-like objects
                # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Serializing datetime-like object at path: {_path}")
                return obj.isoformat()
            elif isinstance(obj, Enum): # More robust Enum check
                # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Serializing Enum object at path: {_path}")
                return obj.name
            elif isinstance(obj, type(dict.__dict__)): # Detect mappingproxy
                 # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Entering mappingproxy (converting to dict) at path: {_path}")
                 return {key: make_json_serializable(value, _depth + 1, f"{_path}.{key}") for key, value in obj.items()} # Convert to dict
            elif hasattr(obj, '__dict__') and not isinstance(obj, type): # Avoid serializing class objects themselves
                # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Entering object.__dict__ at path: {_path}")
                # Pass the object's class name into the path for clarity
                return make_json_serializable(obj.__dict__, _depth + 1, f"{_path}<{type(obj).__name__}>.__dict__")
            else:
                # Fallback: Convert unknown types to string to avoid JSON errors
                # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Attempting fallback serialization for path: {_path}, type: {type(obj).__name__}")
                try:
                    # Attempt direct JSON serialization first for basic types
                    json.dumps(obj)
                    # Removed duplicate json.dumps(obj)
                    # Removed DEBUG log: logger.debug(f"{'  ' * _depth}Fallback successful (direct JSON) for path: {_path}")
                    return obj
                except TypeError:
                    # If direct serialization fails, convert to string
                    logger.warning(f"{'  ' * _depth}Converting non-serializable type {type(obj).__name__} to string at path: {_path}.")
                    return str(obj)

        # Serialize collection metadata
        logger.debug("Starting metadata serialization for 'collection'")
        collection_metadata = make_json_serializable(asdict(self.collection.metadata), _path="collection")

        # Serialize metadata for TimeSeriesSignals and Features separately
        time_series_metadata = {}
        feature_metadata = {}
        keys_to_iterate = signal_keys_to_include if signal_keys_to_include is not None else list(self.collection.time_series_signals.keys()) + list(self.collection.features.keys())

        for key in keys_to_iterate:
            if key in self.collection.time_series_signals:
                signal = self.collection.time_series_signals[key]
                # Only include if key was requested or if all keys are included
                if signal_keys_to_include is None or key in signal_keys_to_include:
                    # Removed DEBUG log: logger.debug(f"Starting metadata serialization for time_series_signal: {key}")
                    time_series_metadata[key] = make_json_serializable(asdict(signal.metadata), _path=f"time_series_signals.{key}")
            elif key in self.collection.features:
                feature = self.collection.features[key]
                # Only include if key was requested or if all keys are included
                if signal_keys_to_include is None or key in signal_keys_to_include:
                    # Removed DEBUG log: logger.debug(f"Starting metadata serialization for feature: {key}")
                    feature_metadata[key] = make_json_serializable(asdict(feature.metadata), _path=f"features.{key}")
            else:
                # This case should ideally not happen if keys are validated before calling
                logger.warning(f"Key '{key}' requested for metadata serialization not found in time_series_signals or features.")
        logger.debug("Finished serializing individual signal/feature metadata.")

        # Prepare combined metadata structure, merging time series and features under 'signals'
        combined_signals_metadata = {**time_series_metadata, **feature_metadata}
        metadata = {
            "collection": collection_metadata,
            "signals": combined_signals_metadata
        }
        logger.debug(f"Final serialized metadata structure keys: {list(metadata.keys())}")
        if "signals" in metadata:
            logger.debug(f"Serialized 'signals' metadata keys: {list(metadata['signals'].keys())}")


        return metadata

    # Updated type hint for signals_to_export
    def _export_excel(self, output_dir: str, signals_to_export: Dict[str, Union[TimeSeriesSignal, Feature]],
                      combined_df: Optional[pd.DataFrame], combined_features_df: Optional[pd.DataFrame], # Added combined_features_df
                      summary_df: Optional[pd.DataFrame],
                      serialized_metadata: Dict[str, Any]) -> None:
        """
        Export signals, metadata, and optionally summary to Excel format.

        Args:
            output_dir: Directory where Excel file(s) will be saved.
            signals_to_export: Dictionary of individual signals/features {key: TimeSeriesSignal or Feature} to export.
            export_combined: Whether to export the combined dataframe.
            summary_df: Optional summary DataFrame to export.
            serialized_metadata: Pre-serialized metadata dictionary.
        """
        logger = logging.getLogger(__name__) # Get logger for this method
        signals_file = os.path.join(output_dir, "signals.xlsx")

        # --- Export Individual Signals to Sheets ---
        # This part writes to the 'signals.xlsx' file
        if signals_to_export:
            logger.info(f"Exporting {len(signals_to_export)} individual signals to Excel sheets in '{signals_file}'.")
            with pd.ExcelWriter(signals_file, engine="openpyxl") as writer:
                has_sheets = False
                for key, signal in signals_to_export.items():
                    try:
                        data = signal.get_data()
                        if data is None or data.empty:
                            logger.warning(f"Signal/Feature '{key}' has no data, skipping Excel sheet.")
                            continue

                        if not isinstance(data, pd.DataFrame):
                            warnings.warn(f"Item {key} does not have DataFrame data, skipping Excel sheet")
                            continue

                        # Limit sheet name length for Excel compatibility
                        sheet_name = key[:31]

                        # --- Conditional Export Logic ---
                        if isinstance(signal, Feature):
                            # Handle Feature export (MultiIndex columns, DatetimeIndex)
                            logger.debug(f"Exporting Feature '{key}' to Excel sheet (index=True).")
                            df_to_export = data.copy() # Work on a copy
                            # Remove timezone from DatetimeIndex for Excel compatibility
                            if isinstance(df_to_export.index, pd.DatetimeIndex) and df_to_export.index.tz is not None:
                                logger.debug("Removing timezone from Feature index for Excel export.")
                                df_to_export.index = df_to_export.index.tz_localize(None)
                            # Write with index=True for Features
                            df_to_export.to_excel(writer, sheet_name=sheet_name, index=True, na_rep='')
                            has_sheets = True
                        elif isinstance(signal, TimeSeriesSignal):
                            # Handle TimeSeriesSignal export (standard columns, timestamp column)
                            logger.debug(f"Exporting TimeSeriesSignal '{key}' to Excel sheet (index=False).")
                            data_formatted = self._format_timestamp(data) # Resets index to 'timestamp' column
                            # Excel doesn't support timezone-aware datetimes, remove timezone from column
                            if 'timestamp' in data_formatted.columns and pd.api.types.is_datetime64_any_dtype(data_formatted['timestamp']):
                                if data_formatted['timestamp'].dt.tz is not None:
                                    logger.debug("Removing timezone from TimeSeriesSignal timestamp column for Excel export.")
                                    data_formatted['timestamp'] = data_formatted['timestamp'].dt.tz_localize(None)
                            # Write with index=False for TimeSeriesSignals after formatting
                            data_formatted.to_excel(writer, sheet_name=sheet_name, index=False, na_rep='')
                            has_sheets = True
                        else:
                            # Should not happen if signals_to_export contains only TS or Feature
                            warnings.warn(f"Item '{key}' is neither TimeSeriesSignal nor Feature. Skipping Excel sheet.")

                    except Exception as e:
                        logger.error(f"Failed to export item '{key}' to Excel sheet: {e}", exc_info=True)
                        warnings.warn(f"Failed to export item '{key}' to Excel sheet.")

                # Add Metadata sheet to the *same* signals.xlsx file
                metadata_rows = []
                for section, section_data in serialized_metadata.items():
                    if section == 'signals':
                        for sig_key, sig_meta_dict in section_data.items():
                            for k, v in sig_meta_dict.items():
                                value = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                                metadata_rows.append({"key": f"{section}.{sig_key}.{k}", "value": value})
                    elif isinstance(section_data, dict):
                        for k, v in section_data.items():
                            value = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                            metadata_rows.append({"key": f"{section}.{k}", "value": value})

                metadata_df = pd.DataFrame(metadata_rows)
                metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

                # Add Info sheet if no data sheets were written
                if not has_sheets:
                    empty_df = pd.DataFrame({'Note': ['No individual signals with DataFrame data found or requested']})
                    empty_df.to_excel(writer, sheet_name="Info", index=False)

        # --- Export Combined Dataframe to Separate File ---
        # Check if combined_df was passed (controlled by include_combined flag in main export method)
        if combined_df is not None:
            logger.info("Exporting combined dataframe to separate Excel file.")
            combined_file = os.path.join(output_dir, "combined.xlsx")
            # combined_df is already passed as an argument, no need to retrieve again
            # combined_df = self.collection.get_stored_combined_dataframe() # REMOVED

            if not combined_df.empty: # Check if the passed dataframe is not empty
                logger.info(f"Retrieved stored combined dataframe (shape: {combined_df.shape}) for Excel export.")

                # Check if columns are MultiIndex
                if isinstance(combined_df.columns, pd.MultiIndex):
                    logger.debug("Handling combined export for MultiIndex columns.")
                    # Work on a copy
                    combined_df_excel = combined_df.copy()
                    # Remove timezone from index for Excel compatibility
                    if isinstance(combined_df_excel.index, pd.DatetimeIndex) and combined_df_excel.index.tz is not None:
                        logger.debug("Removing timezone from DatetimeIndex for Excel export.")
                        combined_df_excel.index = combined_df_excel.index.tz_localize(None)
                    # Write with index=True (default) for DatetimeIndex
                    combined_df_excel.to_excel(combined_file, na_rep='')
                    logger.info(f"Successfully exported combined data with MultiIndex columns to {combined_file}")
                else:
                    logger.debug("Handling combined export for standard columns.")
                    # Format timestamp to column for standard export
                    combined_df_formatted = self._format_timestamp(combined_df.copy()) # Work on copy

                    # Excel doesn't support timezone-aware datetimes, remove timezone from column
                    if 'timestamp' in combined_df_formatted.columns and pd.api.types.is_datetime64_any_dtype(combined_df_formatted['timestamp']):
                        if combined_df_formatted['timestamp'].dt.tz is not None:
                            logger.debug("Removing timezone information from timestamp column for Excel export.")
                            combined_df_formatted['timestamp'] = combined_df_formatted['timestamp'].dt.tz_localize(None)

                    # Write with index=False as timestamp is now a column
                    combined_df_formatted.to_excel(combined_file, na_rep='', index=False)
                    logger.info(f"Successfully exported combined data with standard columns to {combined_file}")

            elif combined_df is None:
                logger.warning("Stored combined dataframe not found or generation failed. Skipping combined Excel export.")
                warnings.warn("Stored combined dataframe not found or generation failed. Skipping combined Excel export.")
            else: # combined_df is empty
                logger.warning("Stored combined dataframe is empty. Skipping export of combined Excel file.")
                warnings.warn("Stored combined dataframe is empty. Skipping export of combined Excel file.")

        # --- Export Combined Feature Matrix to Separate File ---
        if combined_features_df is not None:
            logger.info("Exporting combined feature matrix to separate Excel file.")
            features_excel_path = os.path.join(output_dir, "combined_features.xlsx")
            try:
                if not isinstance(combined_features_df, pd.DataFrame):
                     logger.warning("Stored combined feature matrix is not a DataFrame. Skipping Excel export.")
                elif combined_features_df.empty:
                     logger.warning("Stored combined feature matrix is empty. Skipping Excel export.")
                else:
                     logger.info(f"Retrieved stored combined feature matrix (shape: {combined_features_df.shape}) for Excel export.")
                     df_to_export = combined_features_df.copy() # Use copy
                     # Remove timezone from DatetimeIndex for Excel compatibility
                     if isinstance(df_to_export.index, pd.DatetimeIndex) and df_to_export.index.tz is not None:
                          logger.debug("Removing timezone from DatetimeIndex for Excel export.")
                          df_to_export.index = df_to_export.index.tz_localize(None)
                     # Write with index=True for epoch grid index
                     df_to_export.to_excel(features_excel_path, sheet_name="Combined_Features", index=True, na_rep='')
                     logger.info(f"Successfully exported combined features data with {'MultiIndex' if isinstance(df_to_export.columns, pd.MultiIndex) else 'simple'} columns to {features_excel_path}")
            except Exception as e:
                logger.error(f"Failed to export combined feature matrix to Excel: {e}", exc_info=True)
                warnings.warn("Failed to export combined feature matrix to Excel.")

        # --- Export Summary Dataframe to Separate File ---
        if summary_df is not None:
            logger.info("Exporting summary dataframe to separate Excel file.")
            summary_file = os.path.join(output_dir, "summary.xlsx")
            try:
                # Create a copy for potential formatting
                summary_df_excel = summary_df.copy()
                # Convert complex types like lists/tuples/dicts/enums to strings for Excel
                for col in summary_df_excel.columns:
                    if summary_df_excel[col].apply(lambda x: isinstance(x, (list, tuple, dict, Enum))).any():
                        summary_df_excel[col] = summary_df_excel[col].astype(str)
                # Handle datetime columns (remove timezone for Excel compatibility)
                for col in summary_df_excel.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns:
                     if summary_df_excel[col].dt.tz is not None:
                          summary_df_excel[col] = summary_df_excel[col].dt.tz_localize(None)

                summary_df_excel.to_excel(summary_file, index=True, na_rep='') # index=True for summary
                logger.info(f"Successfully exported summary data to {summary_file}")
            except Exception as e:
                logger.error(f"Failed to export summary dataframe to Excel: {e}", exc_info=True)
                warnings.warn("Failed to export summary dataframe to Excel.")


    # Updated type hint for signals_to_export
    def _export_csv(self, output_dir: str, signals_to_export: Dict[str, Union[TimeSeriesSignal, Feature]],
                    combined_df: Optional[pd.DataFrame], combined_features_df: Optional[pd.DataFrame], # Added combined_features_df
                    summary_df: Optional[pd.DataFrame],
                    serialized_metadata: Dict[str, Any]) -> None:
        """
        Export signals, metadata, and optionally summary to CSV format.

        Args:
            output_dir: Directory where CSV file(s) will be saved.
            signals_to_export: Dictionary of individual signals/features {key: TimeSeriesSignal or Feature} to export.
            export_combined: Whether to export the combined dataframe.
            summary_df: Optional summary DataFrame to export.
            serialized_metadata: Pre-serialized metadata dictionary.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get timestamp format from collection metadata, with fallback
        timestamp_format = getattr(self.collection.metadata, 'timestamp_format', '%Y-%m-%d %H:%M:%S.%f')
        logger.debug(f"Using timestamp format: {timestamp_format}")

        # Export each individual signal if requested
        if signals_to_export:
            signals_dir = os.path.join(output_dir, "signals")
            os.makedirs(signals_dir, exist_ok=True)
            logger.info(f"Exporting {len(signals_to_export)} individual signals to CSV files in '{signals_dir}'.")

            for key, signal in signals_to_export.items():
                try:
                    data = signal.get_data()
                    if data is None or data.empty:
                        logger.warning(f"Signal '{key}' has no data, skipping CSV export.")
                        continue
                    if isinstance(data, pd.DataFrame):
                        logger.debug(f"Exporting signal {key} with {len(data)} rows to CSV")

                        # Check for duplicated timestamps
                        if isinstance(data.index, pd.DatetimeIndex) and data.index.duplicated().any():
                            dupes = data.index.duplicated().sum()
                            logger.warning(f"Signal {key} has {dupes} duplicate timestamps. Keeping first occurrences.")
                            data = data.loc[~data.index.duplicated(keep='first')]

                        # Format the timestamp to be a column not an index
                        data_formatted = self._format_timestamp(data)

                        # --- MultiIndex columns should now exist directly in the Feature's data ---
                        # --- if created by combine_features. No need to create it here. ---

                        file_path = os.path.join(signals_dir, f"{key}.csv")

                        # Let pandas handle the timestamp formatting natively (works for MultiIndex columns too)
                        logger.debug(f"Exporting with timestamp column dtype: {data_formatted['timestamp'].dtype}")
                        sample_time = data_formatted['timestamp'].iloc[0] if len(data_formatted) > 0 else None
                        logger.debug(f"Sample timestamp before export: {sample_time}")

                        # Use pandas native date_format to ensure milliseconds are included
                        data_formatted.to_csv(file_path, date_format='%Y-%m-%d %H:%M:%S.%f', index=False, na_rep='') # Use index=False

                        logger.debug(f"Exported {len(data_formatted)} rows to {file_path}")
                    else:
                        warnings.warn(f"Signal {key} does not have DataFrame data, skipping CSV export")
                except Exception as e:
                    logger.error(f"Failed to export signal '{key}' to CSV: {e}", exc_info=True)
                    warnings.warn(f"Failed to export signal '{key}' to CSV.")

        # Export metadata (always include metadata relevant to the export task)
        metadata_path = os.path.join(output_dir, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                # Use default=str for any remaining non-serializable types
                json.dump(serialized_metadata, f, indent=2, default=str)
            logger.info(f"Exported metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to export metadata to JSON: {e}", exc_info=True)
            warnings.warn("Failed to export metadata.")

        # Export combined dataframe if requested
        # Check if combined_df was passed (controlled by the 'combined_ts' content keyword)
        if combined_df is not None:
            # Use the filename corresponding to the content keyword
            combined_file = os.path.join(output_dir, "combined_ts.csv")
            logger.info(f"Attempting to export combined dataframe to {combined_file}")

            # combined_df is already passed as an argument
            # combined_df = self.collection.get_stored_combined_dataframe() # REMOVED

            if not combined_df.empty: # Check if the passed dataframe is not empty
                logger.info(f"Retrieved stored combined dataframe with {len(combined_df)} rows and {len(combined_df.columns)} columns for CSV export.")

                # Check if columns are MultiIndex
                if isinstance(combined_df.columns, pd.MultiIndex):
                    logger.debug("Exporting combined CSV with MultiIndex columns and DatetimeIndex.")
                    # Export directly: pandas handles MultiIndex columns and DatetimeIndex correctly
                    combined_df.to_csv(combined_file, date_format='%Y-%m-%d %H:%M:%S.%f', na_rep='')
                    logger.info(f"Exported combined dataframe with MultiIndex columns and {len(combined_df)} rows to {combined_file}")
                else:
                    logger.debug("Exporting combined CSV with standard columns.")
                    # Format the timestamp to be a column for standard export
                    combined_formatted = self._format_timestamp(combined_df.copy()) # Work on copy
                    logger.debug(f"Combined export timestamp column dtype: {combined_formatted['timestamp'].dtype}")
                    sample_time = combined_formatted['timestamp'].iloc[0] if len(combined_formatted) > 0 else None
                    logger.debug(f"Combined sample timestamp before export: {sample_time}")
                    # Use pandas native date_format
                    combined_formatted.to_csv(combined_file, date_format='%Y-%m-%d %H:%M:%S.%f', na_rep='', index=False) # index=False as timestamp is a column
                    logger.info(f"Exported combined dataframe with standard columns and {len(combined_formatted)} rows to {combined_file}")
            elif combined_df is None:
                logger.warning("Stored combined dataframe not found or generation failed. Skipping combined CSV export.")
                warnings.warn("Stored combined dataframe not found or generation failed. Skipping combined CSV export.")
            else: # combined_df is empty
                logger.warning("Stored combined dataframe is empty. Skipping export of combined CSV file.")
                warnings.warn("Stored combined dataframe is empty. Skipping export of combined CSV file.")

        # --- Export Combined Feature Matrix ---
        if combined_features_df is not None:
            features_csv_path = os.path.join(output_dir, "combined_features.csv")
            logger.info(f"Attempting to export combined feature matrix to {features_csv_path}")
            try:
                if not isinstance(combined_features_df, pd.DataFrame):
                     logger.warning("Stored combined feature matrix is not a DataFrame. Skipping export.")
                elif combined_features_df.empty:
                     logger.warning("Stored combined feature matrix is empty. Skipping export.")
                else:
                     logger.info(f"Retrieved stored combined feature matrix with {len(combined_features_df)} rows and {len(combined_features_df.columns)} columns for CSV export.")
                     # Features usually have DatetimeIndex from epoch grid
                     # Use pandas native date_format
                     combined_features_df.to_csv(features_csv_path, date_format='%Y-%m-%d %H:%M:%S.%f', na_rep='', index=True) # index=True for epoch grid index
                     logger.info(f"Exported combined feature matrix with {'MultiIndex' if isinstance(combined_features_df.columns, pd.MultiIndex) else 'simple'} columns and {len(combined_features_df)} rows to {features_csv_path}")
            except Exception as e:
                logger.error(f"Failed to export combined feature matrix to CSV: {e}", exc_info=True)
                warnings.warn(f"Failed to export combined feature matrix to CSV: {e}")

        # Export summary dataframe if requested
        if summary_df is not None:
            summary_file = os.path.join(output_dir, "summary.csv")
            logger.info(f"Attempting to export summary dataframe to {summary_file}")
            try:
                # Create a copy for potential formatting
                summary_df_csv = summary_df.copy()
                # Convert complex types like lists/tuples/dicts/enums to strings for CSV
                for col in summary_df_csv.columns:
                    if summary_df_csv[col].apply(lambda x: isinstance(x, (list, tuple, dict, Enum))).any():
                        summary_df_csv[col] = summary_df_csv[col].astype(str)

                # Export with index=True as index is the signal key
                summary_df_csv.to_csv(summary_file, index=True, date_format='%Y-%m-%d %H:%M:%S.%f', na_rep='')
                logger.info(f"Exported summary dataframe with {len(summary_df_csv)} rows to {summary_file}")
            except Exception as e:
                logger.error(f"Failed to export summary dataframe to CSV: {e}", exc_info=True)
                warnings.warn("Failed to export summary dataframe to CSV.")


    # Removed _create_multi_index_for_features method as it's no longer needed.
    # MultiIndex creation for combined features is handled by SignalCollection.combine_features.


    def _export_pickle(self, output_dir: str, signals_to_export: Dict[str, Union[TimeSeriesSignal, Feature]],
                       combined_df: Optional[pd.DataFrame], combined_features_df: Optional[pd.DataFrame], # Added combined_features_df
                       summary_df: Optional[pd.DataFrame],
                       serialized_metadata: Dict[str, Any]) -> None:
        """
        Export signals, metadata, combined dataframes, and optionally summary to Pickle format.

        Args:
            output_dir: Directory where Pickle file(s) will be saved.
            signals_to_export: Dictionary of individual signals {key: SignalData} to export.
            export_combined: Whether to export the combined dataframe.
            summary_df: Optional summary DataFrame to export.
            serialized_metadata: Pre-serialized metadata dictionary (used for consistency).
        """
        # Export selected signals and metadata in a single pickle file
        pickle_file = os.path.join(output_dir, "data.pkl") # Changed filename for clarity
        logger = logging.getLogger(__name__)

        # Prepare data structure
        export_data = {
            "metadata": serialized_metadata, # Store the serialized version for consistency
            "signals": {}
        }

        # Add individual signal data if requested
        if signals_to_export:
            logger.info(f"Adding {len(signals_to_export)} individual signals to Pickle export.")
            for key, signal in signals_to_export.items():
                try:
                    # Store the actual data object
                    export_data["signals"][key] = signal.get_data()
                except Exception as e:
                    logger.error(f"Failed to get data for signal '{key}' for Pickle export: {e}", exc_info=True)
                    warnings.warn(f"Failed to include signal '{key}' in Pickle export.")

        # Add combined time-series dataframe if available
        if combined_df is not None:
            logger.info("Including combined time-series dataframe in Pickle export.")
            logger.info(f"Adding stored combined time-series dataframe (shape: {combined_df.shape}) to Pickle export.")
            export_data["combined_time_series"] = combined_df # Use specific key
        else:
            # Only log if it was expected (i.e., export_combined_ts was True in calling function)
            # We don't have that flag here, so just store None if it's None
            export_data["combined_time_series"] = None

        # Add combined feature matrix if available
        if combined_features_df is not None:
            logger.info("Including combined feature matrix in Pickle export.")
            logger.info(f"Adding stored combined feature matrix (shape: {combined_features_df.shape}) to Pickle export.")
            export_data["combined_features"] = combined_features_df # Use specific key
        else:
            export_data["combined_features"] = None

        # Add summary dataframe if available
        if summary_df is not None:
            logger.info("Including summary dataframe in Pickle export.")
            export_data["summary"] = summary_df # Store the actual DataFrame

        # Save to pickle file
        logger.info(f"Saving export data to Pickle file: {pickle_file}")
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(export_data, f)
        except Exception as e:
            logger.error(f"Failed to save data to Pickle file '{pickle_file}': {e}", exc_info=True)
            raise IOError(f"Failed to save Pickle file.") from e

    # Updated type hint for signals_to_export
    def _export_hdf5(self, output_dir: str, signals_to_export: Dict[str, Union[TimeSeriesSignal, Feature]],
                     combined_df: Optional[pd.DataFrame], combined_features_df: Optional[pd.DataFrame], # Added combined_features_df
                     summary_df: Optional[pd.DataFrame],
                     serialized_metadata: Dict[str, Any]) -> None:
        """
        Export signals, metadata, combined dataframes, and optionally summary to HDF5 format.

        Args:
            output_dir: Directory where HDF5 file(s) will be saved.
            signals_to_export: Dictionary of individual signals/features {key: TimeSeriesSignal or Feature} to export.
            export_combined: Whether to export the combined dataframe.
            summary_df: Optional summary DataFrame to export.
            serialized_metadata: Pre-serialized metadata dictionary.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")
        
        # Export to HDF5 file
        h5_file = os.path.join(output_dir, "data.h5") # Changed filename for clarity
        logger = logging.getLogger(__name__)

        # Use pandas HDFStore for DataFrame data
        try:
            with pd.HDFStore(h5_file, mode='w', complevel=9, complib='zlib') as store: # Add compression
                # Store each individual signal if requested
                if signals_to_export:
                    logger.info(f"Storing {len(signals_to_export)} individual signals in HDF5 file.")
                    for key, signal in signals_to_export.items():
                        try:
                            data = signal.get_data()
                            if data is None or data.empty:
                                logger.warning(f"Signal '{key}' has no data, skipping HDF5 storage.")
                                continue
                            if isinstance(data, pd.DataFrame):
                                store.put(f"signals/{key}", data, format='table', data_columns=True) # Use put for more options
                            else:
                                warnings.warn(f"Signal {key} does not have DataFrame data, skipping HDF5 storage")
                        except Exception as e:
                            logger.error(f"Failed to store signal '{key}' in HDF5: {e}", exc_info=True)
                            warnings.warn(f"Failed to store signal '{key}' in HDF5.")

                # Store combined time-series dataframe if available
                if combined_df is not None and not combined_df.empty:
                    logger.info("Storing combined time-series dataframe in HDF5 export.")
                    logger.info(f"Storing combined time-series dataframe (shape: {combined_df.shape}) to HDF5 key 'combined_time_series'.")
                    store.put("combined_time_series", combined_df, format='table', data_columns=True) # Use specific key
                elif combined_df is None:
                    logger.debug("Combined time-series dataframe is None. Skipping HDF5 storage.")
                else: # combined_df is empty
                    logger.warning("Combined time-series dataframe is empty. Skipping storage in HDF5 file.")

                # Store combined feature matrix if available
                if combined_features_df is not None and not combined_features_df.empty:
                    logger.info("Storing combined feature matrix in HDF5 export.")
                    logger.info(f"Storing combined feature matrix (shape: {combined_features_df.shape}) to HDF5 key 'combined_features'.")
                    store.put("combined_features", combined_features_df, format='table', data_columns=True) # Use specific key
                elif combined_features_df is None:
                    logger.debug("Combined feature matrix is None. Skipping HDF5 storage.")
                else: # combined_features_df is empty
                    logger.warning("Combined feature matrix is empty. Skipping storage in HDF5 file.")

                # Store summary dataframe if available
                if summary_df is not None:
                    logger.info("Storing summary dataframe in HDF5 export.")
                    try:
                        # Create a copy for potential formatting
                        summary_df_hdf = summary_df.copy()
                        # Convert complex types like lists/tuples/dicts/enums to strings for HDF5 compatibility
                        for col in summary_df_hdf.columns:
                            if summary_df_hdf[col].apply(lambda x: isinstance(x, (list, tuple, dict, Enum))).any():
                                summary_df_hdf[col] = summary_df_hdf[col].astype(str)
                        # HDFStore handles datetime objects well, no need to remove timezone usually
                        store.put("summary", summary_df_hdf, format='table', data_columns=True) # Use put, index=True is implicit for table format
                        logger.info(f"Stored summary dataframe to HDF5 key 'summary'.")
                    except Exception as e:
                        logger.error(f"Failed to store summary dataframe in HDF5: {e}", exc_info=True)
                        warnings.warn("Failed to store summary dataframe in HDF5.")

                logger.info(f"DataFrames stored in HDF5 file: {h5_file}")
        except Exception as e:
            logger.error(f"Failed to write DataFrame data to HDF5 file '{h5_file}': {e}", exc_info=True)
            raise IOError(f"Failed to write HDF5 data.") from e

        # Store metadata separately using h5py (HDFStore is closed now)
        logger.debug(f"Storing metadata in HDF5 file: {h5_file}")
        try:
            with h5py.File(h5_file, 'a') as f:
                # Convert metadata to JSON string and store
                # Use the pre-serialized metadata passed as argument
                # Use dumps with default=str to handle any remaining non-serializable objects
                metadata_json = json.dumps(serialized_metadata, default=str)
                # Delete existing metadata dataset if it exists before creating new one
                if "metadata" in f:
                    del f["metadata"]
                f.create_dataset("metadata", data=metadata_json.encode('utf-8')) # Encode to bytes
            logger.info(f"Metadata stored in HDF5 file: {h5_file}")
        except Exception as e:
            logger.error(f"Failed to write metadata to HDF5 file '{h5_file}': {e}", exc_info=True)
            # Don't necessarily raise IOError here, as data might be saved
            warnings.warn(f"Failed to write metadata to HDF5 file.")
