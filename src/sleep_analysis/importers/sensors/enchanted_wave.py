"""
Enchanted Wave EEG headband sleep staging data importer.
"""

import pandas as pd
import re
from typing import Dict, Any, List, Optional
import os
import logging

from ..formats.csv import CSVImporterBase
from ...core.signal_data import SignalData
from ...signals.eeg_sleep_stage_signal import EEGSleepStageSignal
from ...signal_types import SensorModel, BodyPosition, SignalType, SensorType
from ...utils import get_logger, str_to_enum # Removed standardize_timestamp import

class EnchantedWaveImporter(CSVImporterBase):
    """
    Concrete importer for Enchanted Wave EEG headband sleep staging CSV files.

    Parses the specific format of Enchanted Wave Session.csv files,
    extracting sleep stages, EEG quality, and spectral power sum.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the importer with configuration.

        Args:
            config: Dictionary containing configuration parameters:
                - column_mapping: Maps standard names (sleep_stage, sum_power, eeg_quality)
                                  to source column names.
                - time_format: Format string for parsing timestamps (if needed).
                - filename_pattern: Regex pattern for extracting metadata from filenames.
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing EnchantedWaveImporter")

        self.config = config or {}
        self.logger.debug(f"Configuration: {self.config}")

        # Set default configuration values if not provided
        if "column_mapping" not in self.config:
            self.logger.debug("Using default column mapping for Enchanted Wave (lowercase)")
            # Default to lowercase names apparently found in the file based on logs
            self.config["column_mapping"] = {
                # Standard Name : Source Name (from file)
                "timestamp": "Time", # Keep 'Time' as potential primary, fallback handled in _parse_csv
                "seconds": "Seconds", # Corrected case
                "sleep_stage": "SleepStage", # Corrected case
                "sum_power": "SumPower", # Corrected case
                "eeg_quality": "EegQuality" # Corrected case
            }
        # Enchanted Wave uses a specific header structure, not a simple CSV header row
        self.config["header"] = None # Header is determined dynamically
        self.config["delimiter"] = "," # Standard CSV delimiter

    def _parse_csv(self, source: str) -> pd.DataFrame:
        """
        Parse an Enchanted Wave Session.csv file.

        Handles the specific structure with metadata header, "Data Begin:",
        and data lines. Extracts timestamps and relevant columns.

        Args:
            source: Path to the Session.csv file.

        Returns:
            DataFrame containing the parsed sleep stage data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the file format is invalid or essential markers are missing.
        """
        self.logger.info(f"Parsing Enchanted Wave file: {source}")

        try:
            with open(source, 'r') as f:
                file_content = f.readlines()
        except FileNotFoundError:
            self.logger.error(f"File not found: {source}")
            raise FileNotFoundError(f"File not found: {source}")
        except Exception as e:
            self.logger.error(f"Error reading file {source}: {str(e)}")
            raise ValueError(f"Error reading file {source}: {e}")

        data_begin_idx = -1
        header_line_idx = -1
        data_lines = []
        base_timestamp_utc = None # Store base timestamp as aware UTC
        parsed_timezone_offset = None # Store parsed offset string like "-04:00"

        # --- Stage 1: Find Markers and Base Timestamp ---
        for i, line in enumerate(file_content):
            stripped_line = line.strip()
            if "Data Begin:" in stripped_line:
                data_begin_idx = i
                self.logger.debug(f"Found 'Data Begin:' at line {i+1}")
                try:
                    # Extract timestamp like "Data Begin: 2025-03-17T02:41:52.1866600-04:00;"
                    # Use regex to capture timestamp and offset separately
                    match = re.search(r"Data Begin:\s*([\d\-T:]+\.?\d*)(\s*[+\-]\d{2}:\d{2})?;?", stripped_line)
                    if match:
                        time_str = match.group(1)
                        offset_str = match.group(2) # Might be None if no offset found
                        if offset_str:
                             parsed_timezone_offset = offset_str.strip()
                             self.logger.debug(f"Parsed timezone offset from file: {parsed_timezone_offset}")
                             # Combine time string and offset for parsing
                             full_time_str = f"{time_str}{parsed_timezone_offset}"
                             base_timestamp_aware = pd.to_datetime(full_time_str)
                        else:
                             # No offset found, parse as naive
                             self.logger.warning("No timezone offset found in 'Data Begin:' line. Parsing as naive.")
                             base_timestamp_aware = pd.to_datetime(time_str) # Naive

                        # Convert to UTC for internal consistency
                        if base_timestamp_aware.tz is not None:
                             base_timestamp_utc = base_timestamp_aware.tz_convert('UTC')
                             self.logger.debug(f"Extracted base timestamp (aware UTC): {base_timestamp_utc}")
                        else:
                             # If naive, try localizing using config or parsed offset before converting to UTC
                             origin_tz_for_base = self.config.get("origin_timezone", parsed_timezone_offset)
                             if origin_tz_for_base:
                                 try:
                                     base_timestamp_utc = base_timestamp_aware.tz_localize(origin_tz_for_base).tz_convert('UTC')
                                     self.logger.debug(f"Localized naive base timestamp to {origin_tz_for_base} and converted to UTC: {base_timestamp_utc}")
                                 except Exception as loc_err:
                                     self.logger.warning(f"Failed to localize naive base timestamp using '{origin_tz_for_base}': {loc_err}. Assuming UTC.")
                                     base_timestamp_utc = base_timestamp_aware.tz_localize('UTC') # Fallback: assume UTC
                             else:
                                 self.logger.warning("Base timestamp is naive and no origin timezone specified/parsed. Assuming UTC.")
                                 base_timestamp_utc = base_timestamp_aware.tz_localize('UTC') # Fallback: assume UTC

                    else:
                         self.logger.warning("Could not parse timestamp from 'Data Begin:' line using regex.")

                except Exception as e:
                    self.logger.warning(f"Failed to extract or process base timestamp from 'Data Begin:' line: {str(e)}")
                    base_timestamp_utc = None # Ensure base_timestamp is None on failure
                # Don't assume header position here, break after finding Data Begin
                break # Exit loop once Data Begin is found

        if data_begin_idx == -1:
            self.logger.error(f"Could not find 'Data Begin:' marker in {source}")
            raise ValueError(f"Invalid format: 'Data Begin:' marker not found in {source}")

        # --- Stage 2: Find Header and Data Lines ---
        expected_headers = ["Time", "Seconds", "SleepStage", "SumPower", "EegQuality"] # Add more if needed
        for i in range(data_begin_idx + 1, len(file_content)):
            line = file_content[i].strip()
            if not line: # Skip empty lines
                continue

            # Check if this line looks like the header
            potential_headers = [h.strip() for h in line.split(',')]
            if any(eh in potential_headers for eh in expected_headers):
                header_line_idx = i
                column_names = potential_headers
                self.logger.debug(f"Found header line at index {header_line_idx}: {column_names}")
                break # Header found

        if header_line_idx == -1:
            self.logger.error(f"Could not find header line after 'Data Begin:' in {source}")
            raise ValueError(f"Invalid format: Header line not found after 'Data Begin:' in {source}")

        # Collect data lines *after* the header
        for i in range(header_line_idx + 1, len(file_content)):
             stripped_line = file_content[i].strip()
             # Stop if end markers are found
             if "Data Completed:" in stripped_line or "SleepSummary" in stripped_line:
                 break
             # Check if it looks like a data line (contains commas and is not empty)
             if ',' in stripped_line and stripped_line:
                 data_lines.append(stripped_line)

        if not data_lines:
            self.logger.warning(f"No data lines found after header line in {source}")
            return pd.DataFrame() # Return empty DataFrame if no data

        # --- Stage 3: Create DataFrame ---
        # Split data lines, ensuring consistent number of columns
        parsed_data = []
        num_header_cols = len(column_names)
        for idx, line in enumerate(data_lines):
            split_line = line.split(',')
            # Pad rows with fewer columns than header, log warning for rows with more
            if len(split_line) < num_header_cols:
                 self.logger.warning(f"Data line {header_line_idx + 1 + idx + 1} has fewer columns ({len(split_line)}) than header ({num_header_cols}). Padding with NaN.")
                 split_line.extend([None] * (num_header_cols - len(split_line)))
            elif len(split_line) > num_header_cols:
                 self.logger.warning(f"Data line {header_line_idx + 1 + idx + 1} has more columns ({len(split_line)}) than header ({num_header_cols}). Truncating.")
                 split_line = split_line[:num_header_cols]
            parsed_data.append(split_line)

        try:
            df = pd.DataFrame(parsed_data, columns=column_names)
            self.logger.debug(f"Created DataFrame with shape: {df.shape}")
        except Exception as e:
             self.logger.error(f"Error creating DataFrame from data lines: {str(e)}")
             self.logger.error(f"Number of columns in header: {len(column_names)}")
             if parsed_data:
                 self.logger.error(f"Number of columns in first data row: {len(parsed_data[0])}")
             raise ValueError(f"Error creating DataFrame: {e}")


        # --- Timestamp Handling ---
        timestamp_col_name = self.config.get("column_mapping", {}).get("timestamp", "Time")
        seconds_col_name = self.config.get("column_mapping", {}).get("seconds", "Seconds")

        # Determine origin and target timezones from config (injected by WorkflowExecutor)
        origin_timezone = self.config.get("origin_timezone", parsed_timezone_offset) # Use parsed offset as fallback
        target_timezone = self.config.get("target_timezone")

        if target_timezone is None:
            self.logger.error("Target timezone not found in importer configuration. Cannot standardize timestamps.")
            raise ValueError("Target timezone is required for timestamp standardization.")

        self.logger.debug(f"Attempting timestamp standardization with: col='{timestamp_col_name}', origin='{origin_timezone}', target='{target_timezone}'")

        timestamp_standardized = False
        if timestamp_col_name in df.columns:
            self.logger.debug(f"Using '{timestamp_col_name}' column for timestamps.")
            try:
                # Use the base class helper method
                df = self._standardize_timestamp(df, timestamp_col_name, origin_timezone, target_timezone, set_index=True)
                timestamp_standardized = True
                self.logger.debug(f"Timestamp standardization successful using '{timestamp_col_name}' column.")
            except Exception as e:
                self.logger.warning(f"Failed to standardize timestamp using '{timestamp_col_name}' column: {str(e)}. Trying 'Seconds'.")
                # Don't reset index here, let the next block handle it if needed
        else:
             self.logger.debug(f"Primary timestamp column '{timestamp_col_name}' not found.")

        # Fallback to 'Seconds' column if primary timestamp failed or wasn't present
        if not timestamp_standardized:
            if seconds_col_name in df.columns and base_timestamp_utc is not None:
                self.logger.debug(f"Using '{seconds_col_name}' column and base timestamp (UTC) for fallback.")
                try:
                    seconds = pd.to_numeric(df[seconds_col_name], errors='coerce')
                    valid_seconds_mask = seconds.notna()
                    df = df[valid_seconds_mask]
                    seconds = seconds[valid_seconds_mask]

                    if not df.empty:
                        # Calculate timestamps relative to base_timestamp_utc (which is aware UTC)
                        timestamps = base_timestamp_utc + pd.to_timedelta(seconds, unit='s')
                        # Convert the calculated UTC timestamps to the target timezone
                        timestamps = timestamps.tz_convert(target_timezone)
                        df.index = pd.DatetimeIndex(timestamps)
                        df.index.name = 'timestamp' # Use standard name
                        if seconds_col_name in df.columns and seconds_col_name != 'timestamp':
                            df = df.drop(columns=[seconds_col_name])
                        timestamp_standardized = True
                        self.logger.debug(f"Timestamp creation successful using 'Seconds' column (result is aware {target_timezone}).")
                    else:
                        self.logger.warning("No valid 'Seconds' values found after coercion.")
                        df = pd.DataFrame() # Return empty if no valid seconds

                except Exception as e:
                    self.logger.warning(f"Failed to create timestamps using '{seconds_col_name}' column: {str(e)}. Trying synthetic.")
            else:
                self.logger.warning(f"Neither '{timestamp_col_name}' nor '{seconds_col_name}' column available or base timestamp missing. Trying synthetic.")

        # Final fallback: Synthetic timestamps (make them target_timezone aware)
        if not timestamp_standardized:
             if not df.empty:
                 self.logger.warning(f"Generating synthetic timestamps (4-second interval, {target_timezone}).")
                 # Use a 4-second frequency, make target_timezone aware
                 df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='4S', tz=target_timezone)
                 df.index.name = 'timestamp' # Use standard name
                 timestamp_standardized = True
             else:
                 self.logger.warning("DataFrame is empty, cannot generate synthetic timestamps.")
                 return pd.DataFrame() # Return empty if no data and no timestamps

        # Ensure index is set if standardization succeeded
        if timestamp_standardized and not isinstance(df.index, pd.DatetimeIndex):
             self.logger.error("Timestamp standardization failed to set a DatetimeIndex.")
             raise ValueError("Timestamp standardization failed to produce a DatetimeIndex.")


        # --- Column Mapping and Data Cleaning ---
        column_mapping = self.config.get("column_mapping", {})
        rename_dict = {}
        final_columns = []

        # Map standard names to source names found in the DataFrame
        for std_col, src_col in column_mapping.items():
            if src_col in df.columns:
                rename_dict[src_col] = std_col
                final_columns.append(std_col) # Keep track of successfully mapped columns

        self.logger.debug(f"Applying column renaming: {rename_dict}")
        df = df.rename(columns=rename_dict)

        # Keep only the columns that were successfully mapped (plus the index)
        # Ensure we don't try to select the index as a column
        columns_to_keep = [col for col in final_columns if col in df.columns]
        if columns_to_keep:
             self.logger.debug(f"Keeping columns: {columns_to_keep}")
             df = df[columns_to_keep]
        else:
             self.logger.warning("No columns were successfully mapped. Returning empty DataFrame.")
             return pd.DataFrame()


        # --- Data Type Conversion ---
        if 'sum_power' in df.columns:
            df['sum_power'] = pd.to_numeric(df['sum_power'], errors='coerce')
        if 'eeg_quality' in df.columns:
            df['eeg_quality'] = pd.to_numeric(df['eeg_quality'], errors='coerce')
        if 'sleep_stage' in df.columns:
            # Keep sleep stage as object/string, handle potential variations
            df['sleep_stage'] = df['sleep_stage'].astype(str).str.strip()
            # Map common variations if necessary (e.g., 'Wake' -> 'Awake')
            stage_map = {'Wake': 'Awake', 'wake': 'Awake'}
            df['sleep_stage'] = df['sleep_stage'].replace(stage_map)


        # --- Final Validation and Cleanup ---
        # Drop rows where the timestamp index is NaT (shouldn't happen with fallbacks, but good practice)
        df = df[df.index.notna()]

        # Remove duplicate timestamps, keeping the first occurrence
        if df.index.duplicated().any():
            duplicates = df.index.duplicated().sum()
            self.logger.warning(f"Removing {duplicates} duplicate timestamps, keeping first.")
            df = df[~df.index.duplicated(keep='first')]

        # Sort by timestamp index again after potential duplicate removal
        df = df.sort_index()

        self.logger.info(f"Successfully parsed Enchanted Wave file. Final shape: {df.shape}")
        return df

    def _extract_metadata(self, data: pd.DataFrame, source: str, signal_type: str) -> Dict[str, Any]:
        """
        Extract metadata for Enchanted Wave signals.

        Args:
            data: The DataFrame containing the signal data.
            source: The source path (directory or file).
            signal_type: The type of the signal (should be EEG_SLEEP_STAGE).

        Returns:
            A dictionary of metadata key-value pairs.
        """
        self.logger.debug(f"Extracting metadata for {signal_type} from {source}")

        # Get base metadata (includes source_files if available from MergingImporter)
        metadata = super()._extract_metadata(data, source, signal_type)

        # Add Enchanted Wave specific metadata
        metadata["sensor_type"] = SensorType.EEG # Explicitly EEG sensor
        metadata["sensor_model"] = SensorModel.ENCHANTED_WAVE
        metadata["body_position"] = BodyPosition.HEAD # EEG is always head-worn

        # --- Remove Units Logic ---
        # Units are now handled by the Signal class __init__ using _default_units
        # --- End Remove Units Logic ---

        # Extract additional metadata from filename if pattern is specified in config
        filename_pattern = self.config.get("filename_pattern")
        # Use the first source file if multiple were merged
        file_to_parse = metadata.get("source_files", [source])[0]
        if filename_pattern:
            filename = os.path.basename(file_to_parse)
            self.logger.debug(f"Attempting to extract metadata from filename '{filename}' using pattern '{filename_pattern}'")
            match = re.match(filename_pattern, filename)
            if match:
                extracted_info = match.groupdict()
                self.logger.debug(f"Extracted metadata from filename: {extracted_info}")
                if "sensor_info" not in metadata:
                    metadata["sensor_info"] = {}
                metadata["sensor_info"].update(extracted_info)
            else:
                self.logger.warning(f"Filename '{filename}' did not match pattern '{filename_pattern}'")

        # Sample rate is now handled automatically by the TimeSeriesSignal constructor
        # The calculation block previously here has been removed.

        self.logger.debug(f"Final metadata (sample rate will be set by constructor): {metadata}")
        return metadata

    # Override import_signals to use the base class logic directly,
    # as EnchantedWaveImporter doesn't need the directory merging logic
    # from PolarCSVImporter's override. It relies on MergingImporter if needed.
    def import_signals(self, source: str, signal_type: str) -> List[SignalData]:
        """
        Import signals from the specified source.

        Uses the base CSVImporterBase implementation which handles single files.
        If merging is required, the WorkflowExecutor should use MergingImporter
        which wraps this importer.

        Args:
            source: Path to the CSV file.
            signal_type: Type of the signals to import (e.g., "EEG_SLEEP_STAGE").

        Returns:
            A list containing one SignalData instance.
        """
        self.logger.debug(f"Using CSVImporterBase.import_signals for source: {source}")
        return super().import_signals(source, signal_type)

    # Override _get_signal_class for EnchantedWave specific type
    def _get_signal_class(self, signal_type: str) -> type[SignalData]:
        """
        Get the SignalData subclass for Enchanted Wave data.
        """
        self.logger.debug(f"Getting signal class for type: {signal_type}")
        try:
            enum_type = SignalType[signal_type.upper()]
            if enum_type == SignalType.EEG_SLEEP_STAGE:
                return EEGSleepStageSignal
            else:
                self.logger.error(f"Unsupported signal type for EnchantedWaveImporter: {signal_type}")
                raise ValueError(f"Unsupported signal type for EnchantedWaveImporter: {signal_type}")
        except KeyError:
            self.logger.error(f"Unknown signal type: {signal_type}")
            raise ValueError(f"Unknown signal type: {signal_type}")
