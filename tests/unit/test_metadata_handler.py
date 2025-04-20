"""Tests for the MetadataHandler class."""

import pytest
import pandas as pd # Added import
from datetime import datetime

from sleep_analysis.core.metadata_handler import MetadataHandler
# Updated imports for specific metadata types
from sleep_analysis.core.metadata import TimeSeriesMetadata, FeatureMetadata, FeatureType
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit

# Renamed test to be specific to TimeSeriesMetadata
def test_initialize_time_series_metadata():
    """Test initializing metadata with the handler."""
    handler = MetadataHandler()

    # Test with minimal fields (using the correct method)
    metadata = handler.initialize_time_series_metadata(signal_id="test_signal")
    assert isinstance(metadata, TimeSeriesMetadata)
    assert metadata.signal_id == "test_signal"
    assert metadata.name is None
    assert metadata.operations == []
    assert metadata.framework_version is not None # Check default framework version
    
    # Test with default values provided to handler
    handler = MetadataHandler(default_values={"name": "Default Name", "units": Unit.BPM})
    metadata = handler.initialize_time_series_metadata(signal_id="test_signal_2")
    assert isinstance(metadata, TimeSeriesMetadata)
    assert metadata.signal_id == "test_signal_2"
    assert metadata.name == "Default Name"
    assert metadata.units == Unit.BPM
    
    # Test with overrides provided to method
    metadata = handler.initialize_time_series_metadata(signal_id="test_signal_3",
                                                      name="Custom Name",
                                                      units=Unit.G)
    assert isinstance(metadata, TimeSeriesMetadata)
    assert metadata.signal_id == "test_signal_3"
    assert metadata.name == "Custom Name"
    assert metadata.units == Unit.G # Overrides default

    # Note: Removed pytest.raises check for missing signal_id,
    # as the handler now always auto-generates one if not provided.
    # Testing the absence of signal_id would require mocking uuid.uuid4 or altering the handler's core logic.

# Renamed test
def test_auto_generate_time_series_signal_id():
    """Test auto-generation of signal_id for TimeSeriesMetadata."""
    handler = MetadataHandler()
    # Initialize without providing signal_id
    metadata = handler.initialize_time_series_metadata()
    assert isinstance(metadata, TimeSeriesMetadata)
    assert metadata.signal_id is not None
    assert isinstance(metadata.signal_id, str)
    assert len(metadata.signal_id) > 0 # Basic check for non-empty UUID

# Renamed test
def test_filter_invalid_time_series_fields():
    """Test that invalid fields are filtered out for TimeSeriesMetadata."""
    handler = MetadataHandler()
    # Include fields not defined in TimeSeriesMetadata
    metadata = handler.initialize_time_series_metadata(signal_id="test_signal",
                                                      invalid_field="This should be ignored",
                                                      source="This should also be ignored")
    
    assert isinstance(metadata, TimeSeriesMetadata)
    # Verify the valid fields were set
    assert metadata.signal_id == "test_signal"

    # Verify invalid fields were filtered out
    assert not hasattr(metadata, "invalid_field")
    assert not hasattr(metadata, "source")

# --- Add Tests for FeatureMetadata Initialization ---

def test_initialize_feature_metadata():
    """Test initializing FeatureMetadata with the handler."""
    handler = MetadataHandler()
    required_args = {
        "epoch_window_length": pd.Timedelta("30s"),
        "epoch_step_size": pd.Timedelta("5s"),
        "feature_names": ["mean", "std"],
        "source_signal_keys": ["ppg_0"],
        "source_signal_ids": ["uuid-ppg-0"]
    }

    # Test with minimal required fields (feature_id auto-generated)
    metadata = handler.initialize_feature_metadata(**required_args)
    assert isinstance(metadata, FeatureMetadata)
    assert metadata.feature_id is not None and len(metadata.feature_id) > 0
    assert metadata.epoch_window_length == pd.Timedelta("30s")
    assert metadata.feature_names == ["mean", "std"]
    assert metadata.source_signal_keys == ["ppg_0"]
    assert metadata.source_signal_ids == ["uuid-ppg-0"]
    assert metadata.name is None # Default
    assert metadata.feature_type is None # Default
    assert metadata.framework_version is not None

    # Test with overrides and explicit feature_id
    metadata_override = handler.initialize_feature_metadata(
        feature_id="feat_abc",
        name="My Features",
        feature_type=FeatureType.STATISTICAL,
        sensor_type=SensorType.PPG, # Propagated field example
        **required_args
    )
    assert isinstance(metadata_override, FeatureMetadata)
    assert metadata_override.feature_id == "feat_abc"
    assert metadata_override.name == "My Features"
    assert metadata_override.feature_type == FeatureType.STATISTICAL
    assert metadata_override.sensor_type == SensorType.PPG

    # Test required field validation
    incomplete_args = required_args.copy()
    del incomplete_args["feature_names"]
    with pytest.raises(ValueError, match="Required FeatureMetadata field 'feature_names' is missing or empty"):
        handler.initialize_feature_metadata(**incomplete_args)

    # Test Timedelta string conversion
    metadata_str_delta = handler.initialize_feature_metadata(
        epoch_window_length="60s", # Pass as string
        epoch_step_size="10s",     # Pass as string
        feature_names=["min"],
        source_signal_keys=["accel_x"],
        source_signal_ids=["uuid-accel-x"]
    )
    assert metadata_str_delta.epoch_window_length == pd.Timedelta("60s")
    assert metadata_str_delta.epoch_step_size == pd.Timedelta("10s")

    # Test FeatureType string conversion
    metadata_str_enum = handler.initialize_feature_metadata(
        feature_type="spectral", # Pass as string
        **required_args
    )
    assert metadata_str_enum.feature_type == FeatureType.SPECTRAL

    # Test invalid FeatureType string
    with pytest.raises(ValueError, match="Invalid FeatureType value: 'invalid_type'"):
         handler.initialize_feature_metadata(feature_type="invalid_type", **required_args)


def test_auto_generate_feature_id():
    """Test auto-generation of feature_id for FeatureMetadata."""
    handler = MetadataHandler()
    required_args = {
        "epoch_window_length": pd.Timedelta("30s"),
        "epoch_step_size": pd.Timedelta("5s"),
        "feature_names": ["mean"],
        "source_signal_keys": ["ppg_0"],
        "source_signal_ids": ["uuid-ppg-0"]
    }
    metadata = handler.initialize_feature_metadata(**required_args)
    assert isinstance(metadata, FeatureMetadata)
    assert metadata.feature_id is not None
    assert isinstance(metadata.feature_id, str)
    assert len(metadata.feature_id) > 0

def test_filter_invalid_feature_fields():
    """Test that invalid fields are filtered out for FeatureMetadata."""
    handler = MetadataHandler()
    required_args = {
        "epoch_window_length": pd.Timedelta("30s"),
        "epoch_step_size": pd.Timedelta("5s"),
        "feature_names": ["mean"],
        "source_signal_keys": ["ppg_0"],
        "source_signal_ids": ["uuid-ppg-0"]
    }
    metadata = handler.initialize_feature_metadata(
        **required_args,
        invalid_field="This should be ignored",
        another_bad_one=123
    )
    assert isinstance(metadata, FeatureMetadata)
    # Verify required fields were set
    assert metadata.epoch_window_length == pd.Timedelta("30s")
    # Verify invalid fields were filtered out
    assert not hasattr(metadata, "invalid_field")
    assert not hasattr(metadata, "another_bad_one")

# --- Update Tests for Common Methods ---

def test_update_metadata():
    """Test updating existing metadata (both TimeSeries and Feature)."""
    handler = MetadataHandler()

    # Test TimeSeriesMetadata update
    ts_metadata = handler.initialize_time_series_metadata(signal_id="ts_test", name="Original TS Name")
    handler.update_metadata(ts_metadata, name="Updated TS Name", units=Unit.BPM)
    assert ts_metadata.name == "Updated TS Name"
    assert ts_metadata.units == Unit.BPM
    assert ts_metadata.signal_id == "ts_test" # Should not change
    handler.update_metadata(ts_metadata, invalid_field="Value") # Ignored
    assert not hasattr(ts_metadata, "invalid_field")

    # Test FeatureMetadata update
    required_args = {
        "epoch_window_length": pd.Timedelta("30s"), "epoch_step_size": pd.Timedelta("5s"),
        "feature_names": ["mean"], "source_signal_keys": ["ppg_0"], "source_signal_ids": ["uuid-ppg-0"]
    }
    feature_metadata = handler.initialize_feature_metadata(feature_id="feat_test", name="Original Feature Name", **required_args)
    handler.update_metadata(feature_metadata, name="Updated Feature Name", feature_type=FeatureType.HRV)
    assert feature_metadata.name == "Updated Feature Name"
    assert feature_metadata.feature_type == FeatureType.HRV
    assert feature_metadata.feature_id == "feat_test" # Should not change
    handler.update_metadata(feature_metadata, invalid_field="Value") # Ignored
    assert not hasattr(feature_metadata, "invalid_field")


def test_set_name():
    """Test the name setting logic for both metadata types."""
    handler = MetadataHandler()

    # --- TimeSeriesMetadata ---
    ts_metadata = handler.initialize_time_series_metadata(signal_id="ts_abc123xyz")
    # Test with key
    handler.set_name(ts_metadata, key="ts_key")
    assert ts_metadata.name == "ts_key"
    # Test with explicit name (no key)
    ts_metadata.name = None # Reset
    handler.set_name(ts_metadata, name="Explicit TS Name")
    assert ts_metadata.name == "Explicit TS Name"
    # Test fallback (no key, no name)
    ts_metadata.name = None # Reset
    handler.set_name(ts_metadata)
    assert ts_metadata.name == "signal_ts_abc12" # Uses signal_id prefix
    # Test precedence (key over name)
    ts_metadata.name = None # Reset
    handler.set_name(ts_metadata, name="Explicit TS Name", key="ts_key_priority")
    assert ts_metadata.name == "ts_key_priority"
    # Test when name is already set (should not override if key/name not provided)
    ts_metadata.name = "Original TS Name"
    handler.set_name(ts_metadata) # No key, no name passed
    assert ts_metadata.name == "Original TS Name"
    # Test when name is already set but key is provided (key should override)
    ts_metadata.name = "Original TS Name"
    handler.set_name(ts_metadata, key="new_ts_key")
    assert ts_metadata.name == "new_ts_key"


    # --- FeatureMetadata ---
    required_args = {
        "epoch_window_length": pd.Timedelta("30s"), "epoch_step_size": pd.Timedelta("5s"),
        "feature_names": ["mean"], "source_signal_keys": ["ppg_0"], "source_signal_ids": ["uuid-ppg-0"]
    }
    feature_metadata = handler.initialize_feature_metadata(feature_id="feat_def456uvw", **required_args)
    # Test with key
    handler.set_name(feature_metadata, key="feature_key")
    assert feature_metadata.name == "feature_key"
    # Test with explicit name (no key)
    feature_metadata.name = None # Reset
    handler.set_name(feature_metadata, name="Explicit Feature Name")
    assert feature_metadata.name == "Explicit Feature Name"
    # Test fallback (no key, no name)
    feature_metadata.name = None # Reset
    handler.set_name(feature_metadata)
    assert feature_metadata.name == "feature_feat_def" # Uses feature_id prefix [:8]
    # Test precedence (key over name)
    feature_metadata.name = None # Reset
    handler.set_name(feature_metadata, name="Explicit Feature Name", key="feature_key_priority")
    assert feature_metadata.name == "feature_key_priority"
    # Test when name is already set (should not override if key/name not provided)
    feature_metadata.name = "Original Feature Name"
    handler.set_name(feature_metadata) # No key, no name passed
    assert feature_metadata.name == "Original Feature Name"
     # Test when name is already set but key is provided (key should override)
    feature_metadata.name = "Original Feature Name"
    handler.set_name(feature_metadata, key="new_feature_key")
    assert feature_metadata.name == "new_feature_key"


def test_record_operation():
    """Test recording operations in both metadata types."""
    handler = MetadataHandler()

    # --- TimeSeriesMetadata ---
    ts_metadata = handler.initialize_time_series_metadata(signal_id="ts_op_test")
    handler.record_operation(ts_metadata, "filter_lowpass", {"cutoff": 5.0})
    assert len(ts_metadata.operations) == 1
    assert ts_metadata.operations[0].operation_name == "filter_lowpass"
    assert ts_metadata.operations[0].parameters == {"cutoff": 5.0}
    handler.record_operation(ts_metadata, "normalize", {})
    assert len(ts_metadata.operations) == 2
    assert ts_metadata.operations[1].operation_name == "normalize"

    # --- FeatureMetadata ---
    required_args = {
        "epoch_window_length": pd.Timedelta("30s"), "epoch_step_size": pd.Timedelta("5s"),
        "feature_names": ["mean"], "source_signal_keys": ["ppg_0"], "source_signal_ids": ["uuid-ppg-0"]
    }
    feature_metadata = handler.initialize_feature_metadata(feature_id="feat_op_test", **required_args)
    handler.record_operation(feature_metadata, "calculate_stats", {"window": "30s"})
    assert len(feature_metadata.operations) == 1
    assert feature_metadata.operations[0].operation_name == "calculate_stats"
    assert feature_metadata.operations[0].parameters == {"window": "30s"}
    handler.record_operation(feature_metadata, "select_features", {"names": ["mean"]})
    assert len(feature_metadata.operations) == 2
    assert feature_metadata.operations[1].operation_name == "select_features"


# --- Test Sanitization ---

def test_sanitize_parameters():
    """Test the _sanitize_parameters method."""
    handler = MetadataHandler()
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    series = pd.Series([10, 20, 30])
    index = pd.Index([1, 2, 3])
    dt_index = pd.date_range("2023-01-01", periods=3, freq='D')

    params = {
        "dataframe": df,
        "series": series,
        "index": index,
        "datetime_index": dt_index,
        "simple_value": 123,
        "list_value": [1, 2, 3],
        "none_value": None
    }

    sanitized = handler._sanitize_parameters(params)

    assert sanitized["dataframe"] == "<DataFrame shape=(2, 2)>"
    assert sanitized["series"] == "<Series size=3>"
    assert sanitized["index"] == "<Index size=3>"
    # Updated assertion to match pandas DatetimeIndex freq string representation
    assert sanitized["datetime_index"] == "<DatetimeIndex size=3 freq=<Day>>"
    assert sanitized["simple_value"] == 123
    assert sanitized["list_value"] == [1, 2, 3]
    assert sanitized["none_value"] is None

    # Test with non-dict input (should return as is, though unlikely in practice)
    assert handler._sanitize_parameters(None) is None
    assert handler._sanitize_parameters(123) == 123
# (This section is removed as the tests were updated and moved above)
