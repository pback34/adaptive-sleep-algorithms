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
            # Set the name of the index column to 'timestamp' instead of default 'index'
            result_df = result_df.reset_index(names='timestamp')
            logger.debug(f"Reset index to column named 'timestamp', dtype: {result_df['timestamp'].dtype}")
            
            # Double-check that it's still a datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
                logger.warning(f"Column 'timestamp' is not datetime: {result_df['timestamp'].dtype}")
                try:
                    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                    logger.debug("Successfully converted timestamp column to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert timestamp to datetime: {e}")
            
        logger.debug(f"Formatted dataframe has {len(result_df)} rows")
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
                    data.to_excel(writer, sheet_name=key[:31])
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
            combined_file = os.path.join(output_dir, "combined.xlsx")
            combined_df = self.collection.get_combined_dataframe()
            if not combined_df.empty:
                combined_df = self._format_timestamp(combined_df)
                # Ensure no NaN rows are included and index is properly handled
                combined_df.to_excel(combined_file, na_rep='', index=True)
            else:
                warnings.warn("No data available for combined export")
    
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
            logger.info(f"Generating combined dataframe for export to {combined_file}")
            
            combined_df = self.collection.get_combined_dataframe()
            
            if not combined_df.empty:
                logger.info(f"Combined dataframe has {len(combined_df)} rows and {len(combined_df.columns)} columns")
                
                # Format the dataframe for output (convert index to column but preserve datetime objects)
                combined_formatted = self._format_timestamp(combined_df)
                
                # For combined CSV, let pandas handle timestamp formatting natively
                logger.debug(f"Combined export timestamp column dtype: {combined_formatted.iloc[:, 0].dtype}")
                sample_time = combined_formatted.iloc[0, 0] if len(combined_formatted) > 0 else None
                logger.debug(f"Combined sample timestamp before export: {sample_time}")
                
                # Use pandas native date_format to ensure milliseconds are included
                combined_formatted.to_csv(combined_file, date_format='%Y-%m-%d %H:%M:%S.%f', na_rep='')
                logger.info(f"Exported combined dataframe with {len(combined_formatted)} rows to {combined_file}")
            else:
                logger.warning("No data available for combined export")
                warnings.warn("No data available for combined export")
    
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
            export_data["combined"] = self.collection.get_combined_dataframe()
        
        # Save to pickle file
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
                combined_df = self.collection.get_combined_dataframe()
                if not combined_df.empty:
                    store["combined"] = combined_df
        
        # Store metadata separately (since HDFStore is only for DataFrames)
        with h5py.File(h5_file, 'a') as f:
            # Convert metadata to JSON string and store
            # metadata is already serialized by _serialize_metadata
            metadata = self._serialize_metadata()
            # Use dumps with default=str to handle any remaining non-serializable objects
            metadata_json = json.dumps(metadata, default=str)
            f.create_dataset("metadata", data=metadata_json)
