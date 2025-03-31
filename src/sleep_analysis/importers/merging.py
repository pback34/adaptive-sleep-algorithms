"""
Merging importer for combining fragmented files.

This module defines the MergingImporter class that merges multiple fragmented
files with the same structure into a single cohesive signal.
"""

import glob
import os
import importlib # Added import
import pandas as pd
from typing import List, Dict, Any

from .base import SignalImporter
from ..core.signal_data import SignalData
from ..signal_types import SignalType
from ..utils import get_logger, standardize_timestamp

class MergingImporter(SignalImporter):
    """
    A specialized importer that merges multiple fragmented files into a single signal.
    
    This importer is useful when data is split across multiple files with the same
    structure, such as when recordings are segmented into hourly or daily files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration for file patterns and sorting.
        
        Args:
            config: Dict with keys:
                - file_pattern: Glob pattern to match files (e.g., "ppg_*.csv")
                - time_column: Column containing timestamps for sorting (e.g., "timestamp")
                - sort_by: How to sort files - "filename" (default) or "timestamp"
                - delimiter: CSV delimiter (default: ',')
                - header: CSV header row (default: 0)
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MergingImporter")
        
        # Store the entire config dictionary
        self.config = config
        
        # Required configuration
        if "file_pattern" not in config:
            raise ValueError("MergingImporter config must include 'file_pattern'")

        self.importer_name = config.get("importer_name") # Check if wrapping another importer

        # Timestamp column is required only if not wrapping another importer
        if not self.importer_name:
            if "timestamp_col" not in config and "time_column" not in config:
                raise ValueError("MergingImporter config must include 'timestamp_col' or 'time_column' if 'importer_name' is not specified")
            self.time_column = config.get("timestamp_col") or config.get("time_column") # Prefer new name
        else:
            self.time_column = None # Not used when wrapping

        self.file_pattern = config["file_pattern"]

        # Optional configuration with defaults
        # Disable timestamp sort if wrapping another importer, as it handles timestamps internally
        default_sort = "filename"
        if self.importer_name and config.get("sort_by") == "timestamp":
            self.logger.warning("Sorting by 'timestamp' is not supported when using 'importer_name'. Defaulting to 'filename' sort.")
            self.sort_by = "filename"
        else:
            self.sort_by = config.get("sort_by", default_sort)
        self.delimiter = config.get("delimiter", ",") # Used only if not wrapping
        self.header = config.get("header", 0) # Used only if not wrapping

        self.logger.debug(f"Configuration: file_pattern={self.file_pattern}, "
                         f"time_column={self.time_column}, sort_by={self.sort_by}, "
                         f"importer_name={self.importer_name}")

    def _get_underlying_importer_instance(self, importer_name: str, config: Dict[str, Any]):
        """Helper to get instance of the underlying importer."""
        # This logic might need refinement based on where importer classes are located
        try:
            # Try importing from sensors submodule first
            module_path = f"sleep_analysis.importers.sensors.{importer_name.lower()}"
            module = importlib.import_module(module_path)
            importer_class = getattr(module, importer_name)
            # Pass the *entire* MergingImporter config, the underlying importer can pick what it needs
            return importer_class(self.config)
        except (ImportError, AttributeError, ModuleNotFoundError):
             # Fallback or broader search if needed
             try:
                 # Example: try importing directly from importers package
                 module_path = f"sleep_analysis.importers"
                 importers_module = importlib.import_module(module_path)
                 importer_class = getattr(importers_module, importer_name)
                 return importer_class(self.config)
             except (ImportError, AttributeError, ModuleNotFoundError):
                 self.logger.error(f"Could not find or instantiate underlying importer: {importer_name}")
                 raise ImportError(f"Underlying importer class {importer_name} not found or failed to instantiate.")


    def import_signal(self, directory: str, signal_type: str) -> SignalData:
        """
        Import and merge signals from multiple files.
        
        Args:
            directory: Path to the directory containing fragmented files.
            signal_type: Type of signal (e.g., "ppg").
        
        Returns:
            SignalData: A merged signal instance.
            
        Raises:
            FileNotFoundError: If directory does not exist or no matching files found
            ValueError: If files cannot be merged or signal type is invalid
        """
        import time
        start_time = time.time()
        self.logger.info(f"Importing and merging {signal_type} signals from {directory}")
        
        # Check if directory exists
        if not os.path.isdir(directory):
            self.logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Locate all files matching the pattern
        pattern_path = os.path.join(directory, self.file_pattern)
        self.logger.debug(f"Looking for files matching pattern: {pattern_path}")
        files = glob.glob(pattern_path)
        
        if not files:
            self.logger.warning(f"No files found matching pattern: {pattern_path}")
            return None  # Return None instead of raising an exception
            
        self.logger.debug(f"Found {len(files)} files matching pattern")
        
        # Sort files based on configuration
        if self.sort_by == "timestamp":
            self.logger.debug("Sorting files by embedded timestamps")
            # Timestamp sorting is disabled when wrapping another importer
            if self.sort_by == "timestamp" and not self.importer_name:
                self.logger.debug("Sorting files by embedded timestamps")
                files = self._sort_by_embedded_timestamp(files)
            else:
                 # Default to filename sort or if wrapping
                 self.logger.debug("Sorting files by filename")
                 files = sorted(files)

        # --- Data Loading and Merging ---
        all_signals_data = []
        source_files_processed = []

        if self.importer_name:
            # --- Wrapping Logic ---
            self.logger.info(f"Using underlying importer '{self.importer_name}' to process files.")
            import importlib # Moved import here
            underlying_importer = self._get_underlying_importer_instance(self.importer_name, self.config)

            for file_path in files:
                try:
                    self.logger.debug(f"Processing file with underlying importer: {file_path}")
                    # Use the underlying importer's import_signals which returns a list
                    imported_signals = underlying_importer.import_signals(file_path, signal_type)
                    if imported_signals:
                        # Assume the underlying importer returns correctly processed SignalData
                        for signal in imported_signals:
                             if signal.get_data() is not None and not signal.get_data().empty:
                                 all_signals_data.append(signal.get_data())
                                 source_files_processed.append(file_path) # Track successfully processed files
                             else:
                                 self.logger.warning(f"Underlying importer returned empty data for {file_path}")
                    else:
                         self.logger.warning(f"Underlying importer returned no signals for {file_path}")

                except Exception as e:
                    self.logger.error(f"Error processing file {file_path} with underlying importer {self.importer_name}: {str(e)}")
                    # Decide whether to raise or continue based on strictness (consider adding)
                    # For now, we log and continue to potentially merge partial data
                    # raise ValueError(f"Error processing file {file_path} with {self.importer_name}: {str(e)}") from e

            if not all_signals_data:
                 self.logger.error(f"No data successfully processed by underlying importer '{self.importer_name}'")
                 # Return None as per original logic when no files are found/processed
                 return None

            # Merge the dataframes obtained from the underlying importer
            self.logger.debug(f"Concatenating {len(all_signals_data)} dataframes from underlying importer.")
            merged_df = pd.concat(all_signals_data)

            # Sort by the DatetimeIndex (which should be set by the underlying importer)
            if isinstance(merged_df.index, pd.DatetimeIndex):
                merged_df = merged_df.sort_index()
                # Remove duplicates after sorting, keeping the first occurrence
                if merged_df.index.duplicated().any():
                    self.logger.warning(f"Removing {merged_df.index.duplicated().sum()} duplicate timestamps after merging.")
                    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
            else:
                self.logger.warning("Merged DataFrame does not have a DatetimeIndex. Sorting may not be correct.")

        else:
            # --- Original Logic (Direct CSV Reading) ---
            self.logger.debug(f"Loading and merging {len(files)} files directly.")
            dfs = []
            for file_path in files:
                try:
                    self.logger.debug(f"Reading file: {file_path}")
                    df = pd.read_csv(file_path, delimiter=self.delimiter, header=self.header)
                    dfs.append(df)
                    source_files_processed.append(file_path)
                except Exception as e:
                    self.logger.error(f"Error reading file {file_path}: {str(e)}")
                    raise ValueError(f"Error reading file {file_path}: {str(e)}")

            if not dfs:
                self.logger.error("No data frames were successfully loaded")
                raise ValueError("No data frames were successfully loaded")

            # Merge the dataframes and sort by time column
            try:
                self.logger.debug("Concatenating dataframes and sorting by time column")
                merged_df = pd.concat(dfs, ignore_index=True)

                # Ensure timestamp column exists
                if self.time_column not in merged_df.columns:
                    self.logger.error(f"Timestamp column '{self.time_column}' not found in merged data")
                    raise ValueError(f"Timestamp column '{self.time_column}' not found in merged data")

                # Apply column mapping if specified
                if "column_mapping" in self.config:
                    column_mapping = self.config.get("column_mapping", {})
                    self.logger.debug(f"Applying column mapping: {column_mapping}")

                    # Create a mapping from source columns to standard columns
                    rename_dict = {}
                    for std_col, src_col in column_mapping.items():
                        if src_col in merged_df.columns:
                            rename_dict[src_col] = std_col

                    self.logger.debug(f"Renaming columns: {rename_dict}")
                    merged_df = merged_df.rename(columns=rename_dict)

                    # Only keep the columns that were mapped (drop others)
                    mapped_cols = list(rename_dict.values())
                    self.logger.debug(f"Keeping only mapped columns: {mapped_cols}")
                    merged_df = merged_df[mapped_cols]
                else:
                    # If no column mapping, just rename the timestamp column
                    merged_df = merged_df.rename(columns={self.time_column: "timestamp"})

                # Check if timestamp column exists after mapping
                if "timestamp" not in merged_df.columns:
                    self.logger.error(f"Timestamp column 'timestamp' not found after column mapping")
                    raise ValueError(f"Timestamp column 'timestamp' not found after column mapping")

                # Standardize timestamp column before sorting
                try:
                    # Make sure we preserve millisecond precision
                    import datetime as dt

                    # Check current timestamp format
                    sample_time = merged_df["timestamp"].iloc[0] if not merged_df.empty else None
                    if sample_time:
                        self.logger.debug(f"Sample timestamp before standardization: {sample_time}")

                    # Ensure we retain millisecond precision and get naive timestamps
                    # Use the enhanced standardize_timestamp, ensuring tz stripping
                    merged_df = standardize_timestamp(
                        merged_df.copy(), # Pass copy to avoid SettingWithCopyWarning
                        "timestamp",
                        output_format="%Y-%m-%d %H:%M:%S.%f", # Retain precision
                        set_index=True, # Set index directly
                        tz_convert='UTC', # Convert to UTC first (optional but good practice)
                        tz_strip=True # Ensure output is naive
                    )

                    # Log sample after conversion (index is now the timestamp)
                    if not merged_df.empty:
                        self.logger.debug(f"Sample timestamp after standardization (index): {merged_df.index[0]}")

                    # Index is already set and sorted by standardize_timestamp if set_index=True
                    # Remove duplicates after sorting (which happens inside standardize if index is set)
                    if merged_df.index.duplicated().any():
                        self.logger.warning(f"Removing {merged_df.index.duplicated().sum()} duplicate timestamps after merging.")
                        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

                except Exception as e:
                    self.logger.error(f"Failed to standardize timestamp column: {e}")
                    raise ValueError(f"Failed to standardize timestamp column: {e}")

                self.logger.debug(f"Merged dataframe has {len(merged_df)} rows")
            except Exception as e: # This except now correctly corresponds to the try block starting line 216
                self.logger.error(f"Error merging or processing dataframes: {str(e)}")
                raise ValueError(f"Error merging or processing dataframes: {str(e)}")

        # --- Signal Creation ---
        # Get appropriate signal class for the signal type
        signal_class = self._get_signal_class(signal_type)

        # Prepare metadata
        metadata = {
            "signal_type": SignalType[signal_type.upper()],
            "source_files": source_files_processed, # Use only files that were successfully processed
            "merged": True,  # Set merged flag
            "source": directory,
            "sample_count": len(merged_df)
        }
        
        # Get required columns from signal class
        required_columns = signal_class.required_columns
        self.logger.debug(f"Required columns for {signal_type}: {required_columns}")
            
        # Validate all required data columns are present
        missing_columns = [col for col in required_columns if col not in merged_df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns for {signal_type}: {missing_columns}")
            raise ValueError(f"Missing required columns for {signal_type}: {missing_columns}")
        
        # Validate that we have a DatetimeIndex for the timestamp
        if not isinstance(merged_df.index, pd.DatetimeIndex):
            self.logger.error(f"DataFrame must have DatetimeIndex for timestamp")
            raise ValueError(f"DataFrame must have DatetimeIndex for timestamp")
        
        # Now create and return the signal instance
        creation_start = time.time()
        self.logger.debug(f"Creating signal class instance of type {signal_class.__name__}")
        signal_class_instance = signal_class(data=merged_df, metadata=metadata)
        signal_class_instance.metadata.merged = True
        self.logger.info(f"Signal import and merge completed in {time.time() - start_time:.2f} seconds")
        self.logger.debug(f"Signal creation took {time.time() - creation_start:.2f} seconds")
        return signal_class_instance
    
    def import_signals(self, directory: str, signal_type: str) -> List[SignalData]:
        """
        Import and merge signals from multiple files.
        
        For MergingImporter, this returns a single merged signal in a list.
        If no matching files are found, returns an empty list.
        
        Args:
            directory: Path to the directory containing fragmented files.
            signal_type: Type of signal (e.g., "ppg").
        
        Returns:
            List containing a single merged SignalData instance, or empty list if no files found.
        """
        self.logger.info(f"Importing merged {signal_type} signals from {directory}")
        signal = self.import_signal(directory, signal_type)
        
        # If import_signal returned None (no files found), return empty list
        if signal is None:
            self.logger.warning(f"No {signal_type} signals found in {directory}, skipping")
            return []
            
        return [signal]
    
    def _sort_by_embedded_timestamp(self, files: List[str]) -> List[str]:
        """
        Sort files by the earliest timestamp in each file.
        
        Args:
            files: List of file paths.
        
        Returns:
            List of file paths sorted by timestamp.
            
        Raises:
            ValueError: If timestamp column is not found or cannot be parsed
        """
        self.logger.debug(f"Sorting {len(files)} files by embedded timestamps")
        timestamps = []
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, delimiter=self.delimiter, header=self.header)
                
                if self.time_column not in df.columns:
                    self.logger.error(f"Time column '{self.time_column}' not found in {file_path}")
                    raise ValueError(f"Time column '{self.time_column}' not found in {file_path}")
                
                earliest = df[self.time_column].min()
                timestamps.append((earliest, file_path))
                self.logger.debug(f"File {file_path} earliest timestamp: {earliest}")
            except Exception as e:
                self.logger.error(f"Error extracting timestamp from {file_path}: {str(e)}")
                raise ValueError(f"Error extracting timestamp from {file_path}: {str(e)}")
        
        # Sort by timestamp and extract only the file paths
        sorted_files = [f for _, f in sorted(timestamps)]
        self.logger.debug(f"Files sorted by timestamp: {sorted_files}")
        return sorted_files
    
    def _get_signal_class(self, signal_type: str):
        """
        Get the appropriate SignalData subclass for the signal type.
        
        Args:
            signal_type: Type of signal (e.g., "ppg").
            
        Returns:
            The SignalData subclass for the given signal type.
            
        Raises:
            ValueError: If signal type is not recognized.
        """
        # Import here to avoid circular imports
        from ..signals import PPGSignal, AccelerometerSignal, HeartRateSignal, EEGSleepStageSignal # Added EEGSleepStageSignal
        
        try:
            enum_type = SignalType[signal_type.upper()]
            
            if enum_type == SignalType.PPG:
                return PPGSignal
            elif enum_type == SignalType.ACCELEROMETER:
                return AccelerometerSignal
            elif enum_type == SignalType.HEART_RATE:
                return HeartRateSignal
            elif enum_type == SignalType.EEG_SLEEP_STAGE: # Added case for EEG Sleep Stage
                return EEGSleepStageSignal
            else:
                self.logger.error(f"No signal class defined for type: {signal_type}")
                raise ValueError(f"No signal class defined for type: {signal_type}")
        except KeyError:
            self.logger.error(f"Unknown signal type: {signal_type}")
            raise ValueError(f"Unknown signal type: {signal_type}")
