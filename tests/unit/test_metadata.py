"""Tests for the metadata module."""

import pytest
import pandas as pd # Added import
from datetime import datetime
# Updated import to use TimeSeriesMetadata
from sleep_analysis.core.metadata import TimeSeriesMetadata, CollectionMetadata, OperationInfo, FeatureMetadata, FeatureType # Added FeatureMetadata and FeatureType
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

def test_feature_metadata():
    """Test the FeatureMetadata class."""
    # Test minimal creation (feature_id is auto-generated)
    min_metadata = FeatureMetadata(
        epoch_window_length=pd.Timedelta("30s"),
        epoch_step_size=pd.Timedelta("5s"),
        feature_names=["mean", "std"],
        source_signal_keys=["ppg_0"],
        source_signal_ids=["uuid-ppg-0"]
    )
    assert isinstance(min_metadata.feature_id, str)
    assert len(min_metadata.feature_id) > 0 # Check it's not empty
    assert min_metadata.framework_version == __version__
    assert min_metadata.epoch_window_length == pd.Timedelta("30s")
    assert min_metadata.feature_names == ["mean", "std"]
    assert min_metadata.source_signal_keys == ["ppg_0"]
    assert min_metadata.source_signal_ids == ["uuid-ppg-0"]
    assert min_metadata.name is None # Default
    assert min_metadata.feature_type is None # Default

    # Test creation with more fields
    full_metadata = FeatureMetadata(
        feature_id="test_feature_001",
        name="PPG Features",
        feature_type=FeatureType.STATISTICAL,
        epoch_window_length=pd.Timedelta("60s"),
        epoch_step_size=pd.Timedelta("10s"),
        feature_names=["mean", "std", "min", "max"],
        source_signal_keys=["ppg_1", "accel_x"],
        source_signal_ids=["uuid-ppg-1", "uuid-accel-x"],
        sensor_type=SensorType.PPG, # Example propagated field
        sensor_model="Mixed", # Example propagated field
        body_position=BodyPosition.WRIST # Example propagated field
    )
    assert full_metadata.feature_id == "test_feature_001"
    assert full_metadata.name == "PPG Features"
    assert full_metadata.feature_type == FeatureType.STATISTICAL
    assert full_metadata.epoch_window_length == pd.Timedelta("60s")
    assert full_metadata.epoch_step_size == pd.Timedelta("10s")
    assert full_metadata.feature_names == ["mean", "std", "min", "max"]
    assert full_metadata.source_signal_keys == ["ppg_1", "accel_x"]
    assert full_metadata.source_signal_ids == ["uuid-ppg-1", "uuid-accel-x"]
    assert full_metadata.sensor_type == SensorType.PPG
    assert full_metadata.sensor_model == "Mixed"
    assert full_metadata.body_position == BodyPosition.WRIST
    assert full_metadata.framework_version == __version__
    assert full_metadata.operations == [] # Default

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
