"""
CSV format importer base class.

This module defines the CSVImporterBase abstract class that provides
format-specific functionality for importing CSV data.
"""

from abc import abstractmethod
import pandas as pd
from typing import Dict, Any, List, Type
import logging

from ..base import SignalImporter
from ...core.signal_data import SignalData
from ...signal_types import SignalType
from ...signals import PPGSignal, AccelerometerSignal, HeartRateSignal
from ...utils import get_logger

class CSVImporterBase(SignalImporter):
    """
    Abstract base class for CSV format importers.
    
    Provides common functionality for importing signal data from CSV files,
    while allowing concrete subclasses to customize parsing behavior through
    declarative configurations.
    """
    
    def __init__(self):
        """Initialize with a logger."""
        super().__init__()
        self.logger = get_logger(__name__)
    
    # Required columns are now defined in signal classes
    
    @abstractmethod
    def _parse_csv(self, source: str) -> pd.DataFrame:
        """
        Parse CSV data into a DataFrame based on implementation-specific logic.
        
        Args:
            source: Path to the CSV file.
            
        Returns:
            DataFrame containing the parsed CSV data.
            
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is malformed.
        """
        pass
    
    def import_signal(self, source: str, signal_type: str) -> SignalData:
        """
        Import a single signal from a CSV file.
        
        Args:
            source: Path to the CSV file.
            signal_type: Type of the signal (e.g., "PPG").
            
        Returns:
            An instance of the appropriate SignalData subclass.
            
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is malformed, signal type is unrecognized,
                       or required columns are missing.
        """
        self.logger.info(f"Importing {signal_type} signal from {source}")
        
        # Get the signal class based on signal_type
        signal_class = self._get_signal_class(signal_type)
        self.logger.debug(f"Using signal class: {signal_class.__name__}")
        
        # Parse the CSV data using the implementation-specific method
        try:
            self.logger.debug(f"Parsing CSV file: {source}")
            data = self._parse_csv(source)
            self.logger.debug(f"Parsed CSV data with shape: {data.shape}")
        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {source}")
            raise FileNotFoundError(f"CSV file not found: {source}")
        except pd.errors.ParserError as e:
            self.logger.error(f"Invalid CSV format in file: {source} - {str(e)}")
            raise ValueError(f"Invalid CSV format in file: {source}")
        
        # Validate the parsed data contains required columns
        self._validate_columns(data, signal_type)
    
        # Filter to required columns only
        try:
            signal_class = self._get_signal_class(signal_type)
            required_cols = signal_class.required_columns
            self.logger.debug(f"Filtering to required columns: {required_cols}")
            
            # Check if we need to handle timestamp column
            if 'timestamp' in required_cols and 'timestamp' not in data.columns:
                # If DataFrame has DatetimeIndex, create a copy with timestamp as column
                if isinstance(data.index, pd.DatetimeIndex):
                    self.logger.debug("Converting DatetimeIndex to timestamp column")
                    # Create a copy with timestamp column for filtering
                    data_with_timestamp = data.copy()
                    data_with_timestamp['timestamp'] = data.index
                    data = data_with_timestamp
                else:
                    self.logger.error(f"Timestamp column required but not found in data or index")
                    raise ValueError(f"Required timestamp column not found in data")
            
            # Now filter to required columns
            data = data[required_cols]
        except KeyError as e:
            self.logger.error(f"Required column(s) missing: {str(e)}")
            raise ValueError(f"Required column(s) missing for {signal_type}: {str(e)}")
    
        # Extract metadata from parsed data and source
        self.logger.debug(f"Extracting metadata from {source}")
        metadata = self._extract_metadata(data, source, signal_type)
        
        # Create and return the signal instance
        self.logger.info(f"Successfully created {signal_type} signal from {source}")
        self.logger.debug(f"Signal metadata: {metadata}")
        return signal_class(data=data, metadata=metadata)
    
    def import_signals(self, source: str, signal_type: str) -> List[SignalData]:
        """
        Import multiple signals from a CSV file or directory.
        
        Default implementation returns a single signal as a list.
        Subclasses can override to handle multiple signals per file or directory.
        
        Args:
            source: Path to the CSV file or directory.
            signal_type: Type of the signals (e.g., "PPG").
            
        Returns:
            A list containing one or more SignalData instances.
        """
        self.logger.info(f"Importing multiple {signal_type} signals from {source}")
        # Default implementation treats as a single signal import
        signal = self.import_signal(source, signal_type)
        self.logger.debug(f"Returning single signal as list from {source}")
        return [signal]
    
    def _get_signal_class(self, signal_type: str) -> Type[SignalData]:
        """
        Get the SignalData subclass based on the signal type.
        
        Note: Type safety is ensured by converting the input string to a SignalType enum value.
        
        Args:
            signal_type: String representation of the signal type (e.g., "PPG").
            
        Returns:
            The corresponding SignalData subclass.
            
        Raises:
            ValueError: If the signal type is not recognized.
        """
        self.logger.debug(f"Getting signal class for type: {signal_type}")
        try:
            # Convert string to enum value (case-insensitive)
            enum_type = SignalType[signal_type.upper()]
            
            # Return the appropriate signal class
            if enum_type == SignalType.PPG:
                return PPGSignal
            elif enum_type == SignalType.ACCELEROMETER:
                return AccelerometerSignal
            elif enum_type == SignalType.HEART_RATE:
                return HeartRateSignal
            # Add mappings for other signal types as needed
            else:
                self.logger.error(f"No signal class defined for type: {signal_type}")
                raise ValueError(f"No signal class defined for type: {signal_type}")
        except KeyError:
            self.logger.error(f"Unknown signal type: {signal_type}")
            raise ValueError(f"Unknown signal type: {signal_type}")
    
    def _validate_columns(self, data: pd.DataFrame, signal_type: str) -> None:
        """
        Validate that the CSV data contains the required columns for the signal type.
        
        Args:
            data: The DataFrame containing the signal data.
            signal_type: The type of the signal.
            
        Raises:
            ValueError: If required columns are missing or signal type is unknown.
        """
        self.logger.debug(f"Validating columns for {signal_type} signal")
        self.logger.debug(f"DataFrame columns: {list(data.columns)}")
        
        signal_class = self._get_signal_class(signal_type)
        required_columns = signal_class.required_columns
        self.logger.debug(f"Required columns for {signal_type}: {required_columns}")
            
        # Validate that all required data columns are present
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns for {signal_type}: {missing_columns}")
            raise ValueError(f"Missing required columns for {signal_type}: {missing_columns}")
        
        # For timestamp, we check for DatetimeIndex
        if 'timestamp' in required_columns and not isinstance(data.index, pd.DatetimeIndex):
            # Only validate timestamp as column if it's still in required_columns
            if 'timestamp' not in data.columns:
                self.logger.error("Required timestamp column not found in data")
                raise ValueError("Required timestamp column not found in data")
        
        self.logger.debug(f"All required columns present for {signal_type}")
    
    def _extract_metadata(self, data: pd.DataFrame, source: str, signal_type: str) -> Dict[str, Any]:
        """
        Extract metadata from the CSV data and source path.
        
        Base implementation provides minimal metadata. Subclasses should override
        to provide format-specific metadata extraction.
        
        Args:
            data: The DataFrame containing the signal data.
            source: The source path or identifier.
            signal_type: The type of the signal.
            
        Returns:
            A dictionary of metadata key-value pairs.
        """
        # Basic metadata with defaults - subclasses should enhance this
        return {
            "signal_type": SignalType[signal_type.upper()],
            "sample_rate": "100Hz",  # Default value
            "units": "bpm" if signal_type.upper() == "PPG" else "m/s^2",  # Signal-specific default
            "source": source,  # Store the source path for traceability
        }
