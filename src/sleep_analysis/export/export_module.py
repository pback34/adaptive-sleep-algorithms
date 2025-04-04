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

from ..core import SignalCollection, SignalData, SignalMetadata, CollectionMetadata


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
    
    def export(self, formats: List[str], output_dir: str, include_combined: bool = False) -> None:
        """
        Export signals and metadata to specified formats.
        
        Args:
            formats: List of formats to export to. Supported: "excel", "csv", "pickle", "hdf5".
            output_dir: Directory where exported files will be saved.
            include_combined: Whether to include a combined dataframe of all non-temporary signals.
            
        Raises:
            ValueError: If an unsupported format is specified.
            IOError: If there are issues creating output directory or writing files.
        """
        # Validate formats
        for fmt in formats:
            if fmt.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {fmt}. Supported formats: {self.SUPPORTED_FORMATS}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export in each requested format
        for fmt in formats:
            fmt = fmt.lower()
            if fmt == "excel":
                self._export_excel(output_dir, include_combined)
            elif fmt == "csv":
                self._export_csv(output_dir, include_combined)
            elif fmt == "pickle":
                self._export_pickle(output_dir, include_combined)
            elif fmt == "hdf5":
                self._export_hdf5(output_dir, include_combined)
    
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
        
    def _serialize_metadata(self) -> Dict[str, Any]:
        """
        Serialize collection and signal metadata to a JSON-compatible format.
        
        Returns:
            Dictionary containing serialized metadata.
        """
        # Helper function to recursively convert non-serializable types
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'isoformat') and callable(obj.isoformat):  # datetime-like objects
                return obj.isoformat()
            elif hasattr(obj, 'name') and hasattr(obj, 'value') and isinstance(obj.value, str):  # Enum objects
                return obj.name
            elif hasattr(obj, '__dict__'):  # Other custom objects
                return make_json_serializable(obj.__dict__)
            else:
                return obj
                
        # Serialize collection metadata
        collection_metadata = make_json_serializable(asdict(self.collection.metadata))
        
        # Serialize signal metadata
        signals_metadata = {}
        for key, signal in self.collection.signals.items():
            signals_metadata[key] = make_json_serializable(asdict(signal.metadata))
        
        # Prepare combined metadata
        metadata = {
            "collection": collection_metadata,
            "signals": signals_metadata
        }
        
        return metadata

    def _export_excel(self, output_dir: str, include_combined: bool) -> None:
        """
        Export signals and metadata to Excel format.
        
        Args:
            output_dir: Directory where Excel file(s) will be saved.
            include_combined: Whether to include a combined dataframe.
        """
        signals_file = os.path.join(output_dir, "signals.xlsx")
        with pd.ExcelWriter(signals_file, engine="openpyxl") as writer:
            has_sheets = False  # Track if we've written any sheets
            
            # Export each signal
            for key, signal in self.collection.signals.items():
                data = signal.get_data()
                if isinstance(data, pd.DataFrame):
                    data = self._format_timestamp(data)
                    # Excel doesn't support timezone-aware datetimes, remove timezone
                    if 'timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                         if data['timestamp'].dt.tz is not None:
                              data['timestamp'] = data['timestamp'].dt.tz_localize(None)
                    data.to_excel(writer, sheet_name=key[:31], index=False) # Use index=False as timestamp is a column
                    has_sheets = True
                else:
                    warnings.warn(f"Signal {key} does not have DataFrame data, skipping")
            
            # Ensure at least one sheet is created
            if not has_sheets:
                # Add an empty sheet with metadata if no signal data
                empty_df = pd.DataFrame({'Note': ['No signals with DataFrame data found']})
                empty_df.to_excel(writer, sheet_name="Info")
                
            # Metadata export
            metadata = self._serialize_metadata()
            metadata_rows = []
            for section, section_data in metadata.items():
                for k, v in section_data.items():
                    if isinstance(v, (dict, list)):
                        value = json.dumps(v)
                    else:
                        value = str(v)
                    metadata_rows.append({"key": f"{section}.{k}", "value": value})
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_excel(writer, sheet_name="Metadata")
        
        if include_combined:
            logger = logging.getLogger(__name__)
            logger.info("Including combined dataframe in Excel export.")
            combined_file = os.path.join(output_dir, "combined.xlsx")
            # Retrieve the stored combined dataframe
            combined_df = self.collection.get_stored_combined_dataframe()

            if combined_df is not None and not combined_df.empty:
                logger.info(f"Retrieved stored combined dataframe (shape: {combined_df.shape}) for Excel export.")
                # Format timestamp before writing
                combined_df_formatted = self._format_timestamp(combined_df.copy()) # Work on copy

                # Excel doesn't support timezone-aware datetimes, remove timezone
                if 'timestamp' in combined_df_formatted.columns and pd.api.types.is_datetime64_any_dtype(combined_df_formatted['timestamp']):
                    if combined_df_formatted['timestamp'].dt.tz is not None:
                        logger.debug("Removing timezone information for Excel export.")
                        combined_df_formatted['timestamp'] = combined_df_formatted['timestamp'].dt.tz_localize(None)

                # Ensure no NaN rows are included and index is properly handled
                combined_df_formatted.to_excel(combined_file, na_rep='', index=False) # Use index=False as timestamp is a column
                logger.info(f"Successfully exported combined data to {combined_file}")
            elif combined_df is None:
                 logger.warning("Stored combined dataframe not found or generation failed. Skipping combined Excel export.")
                 warnings.warn("Stored combined dataframe not found or generation failed. Skipping combined Excel export.")
            else: # combined_df is empty
                 logger.warning("Stored combined dataframe is empty. Skipping export of combined Excel file.")
                 warnings.warn("Stored combined dataframe is empty. Skipping export of combined Excel file.")

    def _export_csv(self, output_dir: str, include_combined: bool) -> None:
        """
        Export signals and metadata to CSV format.
        
        Args:
            output_dir: Directory where CSV file(s) will be saved.
            include_combined: Whether to include a combined dataframe.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        signals_dir = os.path.join(output_dir, "signals")
        os.makedirs(signals_dir, exist_ok=True)
        
        # Get timestamp format from collection metadata, with fallback
        timestamp_format = getattr(self.collection.metadata, 'timestamp_format', '%Y-%m-%d %H:%M:%S.%f')
        logger.debug(f"Using timestamp format: {timestamp_format}")
        
        # Export each signal
        for key, signal in self.collection.signals.items():
            data = signal.get_data()
            if isinstance(data, pd.DataFrame):
                logger.info(f"Exporting signal {key} with {len(data)} rows to CSV")
                
                # Check for duplicated timestamps
                if isinstance(data.index, pd.DatetimeIndex) and data.index.duplicated().any():
                    dupes = data.index.duplicated().sum()
                    logger.warning(f"Signal {key} has {dupes} duplicate timestamps. Keeping first occurrences.")
                    data = data.loc[~data.index.duplicated(keep='first')]
                
                # Format the timestamp to be a column not an index
                data_formatted = self._format_timestamp(data)
                
                file_path = os.path.join(signals_dir, f"{key}.csv")
                
                # Let pandas handle the timestamp formatting natively
                # This preserves all precision in the original datetime objects
                logger.debug(f"Exporting with timestamp column dtype: {data_formatted['timestamp'].dtype}")
                sample_time = data_formatted['timestamp'].iloc[0] if len(data_formatted) > 0 else None
                logger.debug(f"Sample timestamp before export: {sample_time}")
                
                # Use pandas native date_format to ensure milliseconds are included
                data_formatted.to_csv(file_path, date_format='%Y-%m-%d %H:%M:%S.%f')
                    
                logger.debug(f"Exported {len(data_formatted)} rows to {file_path}")
            else:
                warnings.warn(f"Signal {key} does not have DataFrame data, skipping")
        
        # Export metadata
        metadata = self._serialize_metadata()
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Export combined dataframe
        if include_combined:
            combined_file = os.path.join(output_dir, "combined.csv")
            logger.info(f"Attempting to export combined dataframe to {combined_file}")

            # Retrieve the stored combined dataframe
            combined_df = self.collection.get_stored_combined_dataframe()

            if combined_df is not None and not combined_df.empty:
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

    def _export_pickle(self, output_dir: str, include_combined: bool) -> None:
        """
        Export signals and metadata to Pickle format.
        
        Args:
            output_dir: Directory where Pickle file(s) will be saved.
            include_combined: Whether to include a combined dataframe.
        """
        # Export all signals and metadata in a single pickle file
        pickle_file = os.path.join(output_dir, "signals.pkl")
        
        # Prepare data structure
        export_data = {
            "metadata": self._serialize_metadata(),
            "signals": {}
        }
        
        # Add signal data
        for key, signal in self.collection.signals.items():
            export_data["signals"][key] = signal.get_data()
        
        # Add combined dataframe if requested
        if include_combined:
            logger = logging.getLogger(__name__)
            logger.info("Including combined dataframe in Pickle export.")
            # Retrieve the stored combined dataframe
            combined_df = self.collection.get_stored_combined_dataframe()
            if combined_df is not None: # Include even if empty, as None indicates failure
                 logger.info(f"Adding stored combined dataframe (shape: {combined_df.shape}) to Pickle export.")
                 export_data["combined"] = combined_df
            else:
                 logger.warning("Stored combined dataframe not found or generation failed. Storing None for 'combined' in Pickle.")
                 export_data["combined"] = None # Explicitly set to None if failed

        # Save to pickle file
        logger.info(f"Saving export data to Pickle file: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(export_data, f)
    
    def _export_hdf5(self, output_dir: str, include_combined: bool) -> None:
        """
        Export signals and metadata to HDF5 format.
        
        Args:
            output_dir: Directory where HDF5 file(s) will be saved.
            include_combined: Whether to include a combined dataframe.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")
        
        # Export to HDF5 file
        h5_file = os.path.join(output_dir, "signals.h5")
        
        # Use pandas HDFStore for DataFrame data
        with pd.HDFStore(h5_file, mode='w') as store:
            # Store each signal
            for key, signal in self.collection.signals.items():
                data = signal.get_data()
                if isinstance(data, pd.DataFrame):
                    store[f"signals/{key}"] = data
                else:
                    warnings.warn(f"Signal {key} does not have DataFrame data, skipping")
            
            # Store combined dataframe if requested
            if include_combined:
                logger = logging.getLogger(__name__) # Get logger
                logger.info("Including combined dataframe in HDF5 export.")
                # Use the new helper method to get the dataframe
                combined_df = self._get_combined_dataframe_for_export()
                if combined_df is not None and not combined_df.empty:
                    logger.debug(f"Storing combined dataframe (shape: {combined_df.shape}) to HDF5 key 'combined'.")
                    store["combined"] = combined_df
                elif combined_df is None:
                     logger.error("Failed to retrieve or generate combined dataframe for HDF5 export. Skipping.")
                else: # combined_df is empty
                     logger.warning("Combined dataframe is empty. Skipping storage in HDF5 file.")
                     # Optionally store an empty dataframe? store["combined"] = pd.DataFrame()
            logger.info(f"DataFrames stored in HDF5 file: {h5_file}")

        # Store metadata separately (since HDFStore is only for DataFrames)
        logger.debug(f"Storing metadata in HDF5 file: {h5_file}")
        with h5py.File(h5_file, 'a') as f:
            # Convert metadata to JSON string and store
            # metadata is already serialized by _serialize_metadata
            metadata = self._serialize_metadata()
            # Use dumps with default=str to handle any remaining non-serializable objects
            metadata_json = json.dumps(metadata, default=str)
            f.create_dataset("metadata", data=metadata_json)

    # This method might be redundant now if combined export relies solely on get_stored_combined_dataframe
    # Keeping it for now in case it's used elsewhere, but review if it can be removed.
    def _get_filtered_combined_dataframe(self) -> pd.DataFrame:
        """
        Get a combined dataframe with temporary signals filtered out.

        Returns:
            A DataFrame with all non-temporary signals combined.
        """
        import logging
        from ..core import SignalCollection # Ensure SignalCollection is imported
        from ..core.metadata_handler import MetadataHandler # Import MetadataHandler

        logger = logging.getLogger(__name__)

        # Determine the handler to use for the temporary collection
        # Prefer the handler from the original collection if it exists
        # Otherwise, try getting from the first signal, or create a new default
        handler_to_use = None
        if hasattr(self.collection, 'handler') and self.collection.handler:
            handler_to_use = self.collection.handler
            logger.debug("Using handler from original collection for temp collection.")
        elif self.collection.signals:
            # Try getting handler from the first signal
            first_signal = next(iter(self.collection.signals.values()), None)
            if first_signal and hasattr(first_signal, 'handler') and first_signal.handler:
                handler_to_use = first_signal.handler
                logger.debug("Using handler from first signal for temp collection.")
            else:
                logger.warning("Could not get handler from first signal, creating new default handler.")
                handler_to_use = MetadataHandler()
        else:
            # No signals in collection, create a new default handler
            logger.debug("Original collection has no signals, creating new default handler for temp collection.")
            handler_to_use = MetadataHandler()

        # Create a temporary collection containing only non-temporary signals
        # Initialize with copies of essential attributes from the original collection
        logger.debug(f"Initializing temporary collection for export with handler: {type(handler_to_use).__name__}")
        temp_collection = SignalCollection(metadata=asdict(self.collection.metadata), # Use asdict for clean copy
                                           metadata_handler=handler_to_use) # Use determined handler

        # Explicitly copy alignment parameters if they exist on the original collection
        # This ensures the temp collection uses the same grid as calculated previously
        if hasattr(self.collection, 'target_rate'):
            temp_collection.target_rate = self.collection.target_rate
            logger.debug(f"Copied target_rate ({temp_collection.target_rate}) to temp collection.")
        if hasattr(self.collection, 'ref_time'):
            temp_collection.ref_time = self.collection.ref_time
            logger.debug(f"Copied ref_time ({temp_collection.ref_time}) to temp collection.")
        if hasattr(self.collection, 'grid_index'):
            temp_collection.grid_index = self.collection.grid_index
            logger.debug(f"Copied grid_index ({len(temp_collection.grid_index)} points) to temp collection.")

        # Ensure index config is copied (already handled by asdict if part of CollectionMetadata)
        # Redundant check for safety:
        if hasattr(self.collection.metadata, 'index_config') and not temp_collection.metadata.index_config:
             temp_collection.metadata.index_config = list(self.collection.metadata.index_config)
             logger.debug(f"Copied index_config ({temp_collection.metadata.index_config}) to temp collection metadata.")
        else:
             temp_collection.metadata.index_config = [] # Default if not set

        non_temporary_signals_count = 0
        for key, signal in self.collection.signals.items():
            # Check if 'temporary' attribute exists and is True
            is_temporary = getattr(signal.metadata, 'temporary', False)
            if not is_temporary:
                # Add a reference to the non-temporary signal
                # Using reference is fine as get_combined_dataframe doesn't modify signals
                temp_collection.add_signal(key, signal)
                non_temporary_signals_count += 1
            else:
                 logger.debug(f"Excluding temporary signal '{key}' from combined export.")

        if non_temporary_signals_count == 0:
            logger.warning("No non-temporary signals found for combined export.")
            return pd.DataFrame() # Return empty DataFrame if no signals to combine

        # Generate the combined dataframe from the temporary collection
        # This inherently excludes temporary signals because they were never added
        logger.info(f"Calling get_combined_dataframe on temporary collection with {non_temporary_signals_count} signals.")
        combined_df = temp_collection.get_combined_dataframe()

        # Check if the combined dataframe is valid before logging dimensions
        # get_combined_dataframe should now always return a DataFrame
        if combined_df.empty:
            logger.warning("get_combined_dataframe returned an empty DataFrame.")
            # Proceed with the empty DataFrame, export functions will handle it
        else:
            logger.debug(f"Generated combined dataframe for export with {len(combined_df)} rows and {len(combined_df.columns)} columns.")


        return combined_df
