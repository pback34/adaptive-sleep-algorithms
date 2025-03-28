"""
Metadata structures for the sleep analysis framework.

This module defines the metadata classes used to store information about signals
and collections, including signal types, operation history, and provenance.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import datetime

from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from sleep_analysis import __version__

@dataclass
class OperationInfo:
    """Class for storing information about an operation applied to a signal."""
    operation_name: str
    parameters: Dict[str, Any]

@dataclass
class SignalMetadata:
    """Class for storing metadata about a signal."""
    signal_id: str  # Unique identifier (immutable)
    name: Optional[str] = None  # User-friendly name for reference
    signal_type: Optional[SignalType] = None  # Type of signal (e.g., PPG, ACCELEROMETER)
    sample_rate: Optional[str] = None  # e.g., "100Hz"
    units: Optional[Unit] = None  # Physical units (e.g., G, BPM)
    start_time: Optional[datetime.datetime] = None  # Signal start time
    end_time: Optional[datetime.datetime] = None  # Signal end time
    derived_from: List[Tuple[str, int]] = field(default_factory=list)  # List of (signal_id, operation_index) tuples
    operations: List[OperationInfo] = field(default_factory=list)  # Processing history
    temporary: bool = False  # Flag for memory optimization
    sensor_type: Optional[SensorType] = None  # Type of sensor
    sensor_model: Optional[SensorModel] = None  # Model of sensor
    body_position: Optional[BodyPosition] = None  # Position on body
    sensor_info: Optional[Dict[str, Any]] = None  # Additional sensor details
    source_files: List[str] = field(default_factory=list)  # List of file paths contributing to the signal
    merged: bool = False  # Flag indicating if this signal was merged from multiple sources
    framework_version: str = __version__  # Framework version used to process the signal

@dataclass
class CollectionMetadata:
    """Class for storing metadata about a collection of signals."""
    collection_id: str  # Unique identifier
    subject_id: str  # Subject identifier
    session_id: Optional[str] = None  # Session identifier
    start_datetime: Optional[datetime.datetime] = None  # Collection start time
    end_datetime: Optional[datetime.datetime] = None  # Collection end time
    timezone: str = "UTC"  # Timezone for timestamps
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"  # Format for timestamps in export
    study_info: Dict[str, Any] = field(default_factory=dict)  # Study details
    device_info: Dict[str, Any] = field(default_factory=dict)  # Summary of all devices used
    notes: str = ""  # Additional notes
    protocol_id: Optional[str] = None  # Optional: links to study protocol
    data_acquisition_notes: Optional[str] = None  # Optional: notes on data collection process
    index_config: List[str] = field(default_factory=list)  # Multi-index configuration for dataframe exports
    framework_version: str = __version__  # Framework version used to process the collection
