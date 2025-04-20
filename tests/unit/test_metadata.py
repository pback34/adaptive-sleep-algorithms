"""Tests for the metadata module."""

import pytest
from datetime import datetime
# Updated import to use TimeSeriesMetadata
from sleep_analysis.core.metadata import TimeSeriesMetadata, CollectionMetadata, OperationInfo, FeatureMetadata # Added FeatureMetadata for potential future use
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from sleep_analysis import __version__

def test_operation_info():
    """Test the OperationInfo class."""
    op = OperationInfo(operation_name="filter_lowpass", parameters={"cutoff": 5.0})
    assert op.operation_name == "filter_lowpass"
    assert op.parameters == {"cutoff": 5.0}

# Updated test name and class usage
def test_time_series_metadata(sample_metadata):
    """Test the TimeSeriesMetadata class."""
    # Create with minimal required fields
    metadata = TimeSeriesMetadata(signal_id="test_signal_002")
    assert metadata.signal_id == "test_signal_002"
    assert metadata.framework_version == __version__

    # Create with full sample metadata
    # Create with sample metadata, excluding sample_rate which is auto-calculated
    metadata = TimeSeriesMetadata(
        signal_id=sample_metadata["signal_id"],
        name=sample_metadata["name"],
        # sample_rate=sample_metadata["sample_rate"], # Removed - sample_rate is now derived
        units=sample_metadata["units"],
        start_time=sample_metadata["start_time"],
        end_time=sample_metadata["end_time"],
        sensor_type=sample_metadata["sensor_type"],
        sensor_model=sample_metadata["sensor_model"],
        body_position=sample_metadata["body_position"]
    )
    
    assert metadata.signal_id == "test_signal_001"
    assert metadata.name == "Test Signal"
    # assert metadata.sample_rate == "100Hz" # Removed check - depends on data now
    assert metadata.units == Unit.BPM
    assert metadata.sensor_type == SensorType.PPG
    assert metadata.framework_version == __version__

def test_collection_metadata():
    """Test the CollectionMetadata class."""
    metadata = CollectionMetadata(
        collection_id="test_collection_001",
        subject_id="test_subject_001"
    )
    
    assert metadata.collection_id == "test_collection_001"
    assert metadata.subject_id == "test_subject_001"
    assert metadata.framework_version == __version__
    assert metadata.timezone == "UTC"  # Default value
