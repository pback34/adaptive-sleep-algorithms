"""Tests for MetadataManager class."""

import pytest
import pandas as pd

from src.sleep_analysis.core.services.metadata_manager import MetadataManager
from src.sleep_analysis.core.metadata_handler import MetadataHandler
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.features.feature import Feature
from src.sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from src.sleep_analysis.core.metadata import FeatureType


@pytest.fixture
def metadata_handler():
    """Create a metadata handler for testing."""
    return MetadataHandler()


@pytest.fixture
def metadata_manager(metadata_handler):
    """Create a metadata manager for testing."""
    return MetadataManager(metadata_handler)


@pytest.fixture
def sample_signal():
    """Create a sample HeartRateSignal for testing."""
    data = pd.DataFrame(
        {"hr": [70.0, 72.0, 68.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="1s", tz="UTC")
    )
    return HeartRateSignal(
        data=data,
        metadata={
            "signal_id": "test_signal_001",
            "signal_type": SignalType.HEART_RATE,
            "sensor_type": SensorType.PPG
        }
    )


@pytest.fixture
def sample_feature():
    """Create a sample Feature for testing."""
    data = pd.DataFrame(
        {"mean_hr": [70.0, 72.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="30s", tz="UTC")
    )
    return Feature(
        data=data,
        metadata={
            "feature_id": "test_feature_001",
            "sensor_type": SensorType.PPG,
            "feature_type": FeatureType.HRV,
            "epoch_window_length": pd.Timedelta(seconds=30),
            "epoch_step_size": pd.Timedelta(seconds=30),
            "feature_names": ["mean_hr"],
            "source_signal_keys": ["hr_0"],
            "source_signal_ids": ["test_signal_001"]
        }
    )


class TestMetadataManagerBasics:
    """Test basic metadata manager initialization."""

    def test_init_sets_handler(self, metadata_handler):
        """Test that initialization sets the metadata handler."""
        manager = MetadataManager(metadata_handler)
        assert manager.metadata_handler is metadata_handler


class TestUpdateTimeSeriesMetadata:
    """Test updating time-series signal metadata."""

    def test_update_simple_field(self, metadata_manager, sample_signal):
        """Test updating a simple metadata field."""
        original_rate = sample_signal.metadata.sample_rate
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {"sample_rate": "100Hz"}
        )
        assert sample_signal.metadata.sample_rate == "100Hz"
        assert sample_signal.metadata.sample_rate != original_rate

    def test_update_enum_field_with_enum(self, metadata_manager, sample_signal):
        """Test updating an enum field with an enum value."""
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {"sensor_model": SensorModel.POLAR_H10}
        )
        assert sample_signal.metadata.sensor_model == SensorModel.POLAR_H10

    def test_update_enum_field_with_string(self, metadata_manager, sample_signal):
        """Test updating an enum field with a string value."""
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {"sensor_model": "POLAR_H10"}
        )
        assert sample_signal.metadata.sensor_model == SensorModel.POLAR_H10

    def test_update_multiple_fields(self, metadata_manager, sample_signal):
        """Test updating multiple fields at once."""
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {
                "sensor_model": SensorModel.POLAR_H10,
                "sample_rate": "100Hz",
                "body_position": BodyPosition.CHEST
            }
        )
        assert sample_signal.metadata.sensor_model == SensorModel.POLAR_H10
        assert sample_signal.metadata.sample_rate == "100Hz"
        assert sample_signal.metadata.body_position == BodyPosition.CHEST

    def test_update_sensor_info(self, metadata_manager, sample_signal):
        """Test updating sensor_info dictionary."""
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {"sensor_info": {"device_id": "ABC123", "firmware": "v1.2.3"}}
        )
        assert sample_signal.metadata.sensor_info["device_id"] == "ABC123"
        assert sample_signal.metadata.sensor_info["firmware"] == "v1.2.3"

    def test_update_sensor_info_merges_with_existing(self, metadata_manager, sample_signal):
        """Test that sensor_info updates merge with existing values."""
        # First update
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {"sensor_info": {"device_id": "ABC123"}}
        )
        # Second update with different field
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {"sensor_info": {"firmware": "v1.2.3"}}
        )
        # Both should be present
        assert sample_signal.metadata.sensor_info["device_id"] == "ABC123"
        assert sample_signal.metadata.sensor_info["firmware"] == "v1.2.3"

    def test_update_invalid_enum_value_logs_warning_but_accepts_string(
        self, metadata_manager, sample_signal, caplog
    ):
        """Test that invalid enum values log a warning but are still accepted as strings."""
        # sensor_model appears to accept Union[SensorModel, str]
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {"sensor_model": "INVALID_MODEL"}
        )
        assert "Invalid enum value" in caplog.text
        # The string value is accepted even though it's not a valid enum
        assert sample_signal.metadata.sensor_model == "INVALID_MODEL"

    def test_update_wrong_signal_type_raises(self, metadata_manager, sample_feature):
        """Test that passing a Feature raises TypeError."""
        with pytest.raises(TypeError, match="Expected TimeSeriesSignal"):
            metadata_manager.update_time_series_metadata(
                sample_feature,
                {"quality_score": 0.95}
            )


class TestUpdateFeatureMetadata:
    """Test updating feature metadata."""

    def test_update_simple_field(self, metadata_manager, sample_feature):
        """Test updating a simple metadata field."""
        original_name = sample_feature.metadata.name
        metadata_manager.update_feature_metadata(
            sample_feature,
            {"name": "updated_feature_name"}
        )
        assert sample_feature.metadata.name == "updated_feature_name"
        assert sample_feature.metadata.name != original_name

    def test_update_feature_type_with_enum(self, metadata_manager, sample_feature):
        """Test updating feature_type with an enum value."""
        metadata_manager.update_feature_metadata(
            sample_feature,
            {"feature_type": FeatureType.SLEEP_STAGE}
        )
        assert sample_feature.metadata.feature_type == FeatureType.SLEEP_STAGE

    def test_update_feature_type_with_string(self, metadata_manager, sample_feature):
        """Test updating feature_type with a string value."""
        metadata_manager.update_feature_metadata(
            sample_feature,
            {"feature_type": "SLEEP_STAGE"}
        )
        assert sample_feature.metadata.feature_type == FeatureType.SLEEP_STAGE

    def test_update_timedelta_field_with_timedelta(self, metadata_manager, sample_feature):
        """Test updating timedelta field with a Timedelta object."""
        new_window = pd.Timedelta(seconds=60)
        metadata_manager.update_feature_metadata(
            sample_feature,
            {"epoch_window_length": new_window}
        )
        assert sample_feature.metadata.epoch_window_length == new_window

    def test_update_timedelta_field_with_string(self, metadata_manager, sample_feature):
        """Test updating timedelta field with a string."""
        metadata_manager.update_feature_metadata(
            sample_feature,
            {"epoch_window_length": "60s"}
        )
        assert sample_feature.metadata.epoch_window_length == pd.Timedelta(seconds=60)

    def test_update_multiple_timedelta_fields(self, metadata_manager, sample_feature):
        """Test updating multiple timedelta fields."""
        metadata_manager.update_feature_metadata(
            sample_feature,
            {
                "epoch_window_length": "60s",
                "epoch_step_size": "30s"
            }
        )
        assert sample_feature.metadata.epoch_window_length == pd.Timedelta(seconds=60)
        assert sample_feature.metadata.epoch_step_size == pd.Timedelta(seconds=30)

    def test_update_invalid_timedelta_string_logs_warning(
        self, metadata_manager, sample_feature, caplog
    ):
        """Test that invalid timedelta strings log a warning and are skipped."""
        original_window = sample_feature.metadata.epoch_window_length
        metadata_manager.update_feature_metadata(
            sample_feature,
            {"epoch_window_length": "invalid_timedelta"}
        )
        assert "Invalid timedelta format" in caplog.text
        # Original value should remain unchanged
        assert sample_feature.metadata.epoch_window_length == original_window

    def test_update_feature_type_string_not_in_enum_accepts_anyway(
        self, metadata_manager, sample_feature
    ):
        """Test that feature_type accepts Union[FeatureType, str] so any string works."""
        # FeatureMetadata.feature_type is Optional[Union[FeatureType, str]]
        # so it will accept any string value without error
        metadata_manager.update_feature_metadata(
            sample_feature,
            {"feature_type": "CUSTOM_TYPE"}
        )
        # The string should be accepted (not converted to enum)
        assert sample_feature.metadata.feature_type == "CUSTOM_TYPE"

    def test_update_wrong_feature_type_raises(self, metadata_manager, sample_signal):
        """Test that passing a TimeSeriesSignal raises TypeError."""
        with pytest.raises(TypeError, match="Expected Feature"):
            metadata_manager.update_feature_metadata(
                sample_signal,
                {"feature_type": FeatureType.HRV}
            )


class TestGetValidFields:
    """Test getting valid metadata fields."""

    def test_get_valid_time_series_fields(self, metadata_manager):
        """Test getting valid TimeSeriesMetadata fields."""
        fields = metadata_manager.get_valid_time_series_fields()
        assert "signal_id" in fields
        assert "signal_type" in fields
        assert "sensor_type" in fields
        assert "sensor_model" in fields
        assert "sample_rate" in fields
        assert "body_position" in fields
        assert isinstance(fields, set)

    def test_get_valid_feature_fields(self, metadata_manager):
        """Test getting valid FeatureMetadata fields."""
        fields = metadata_manager.get_valid_feature_fields()
        assert "feature_id" in fields
        assert "feature_type" in fields
        assert "epoch_window_length" in fields
        assert "epoch_step_size" in fields
        assert "feature_names" in fields
        assert isinstance(fields, set)


class TestValidateMetadataSpec:
    """Test metadata spec validation."""

    def test_validate_time_series_metadata_spec_valid(self, metadata_manager):
        """Test validation with valid TimeSeriesMetadata spec."""
        # Should not raise
        metadata_manager.validate_time_series_metadata_spec({
            "signal_type": SignalType.HEART_RATE,
            "sensor_type": SensorType.PPG,
            "sample_rate": "100Hz"
        })

    def test_validate_time_series_metadata_spec_invalid(self, metadata_manager):
        """Test validation with invalid TimeSeriesMetadata spec."""
        with pytest.raises(ValueError, match="Invalid TimeSeriesMetadata fields"):
            metadata_manager.validate_time_series_metadata_spec({
                "invalid_field": "value",
                "another_invalid": 123
            })

    def test_validate_time_series_metadata_spec_partial_invalid(self, metadata_manager):
        """Test validation with mix of valid and invalid fields."""
        with pytest.raises(ValueError, match="Invalid TimeSeriesMetadata fields"):
            metadata_manager.validate_time_series_metadata_spec({
                "signal_type": SignalType.HEART_RATE,  # Valid
                "quality_score": 0.95,  # Invalid - not a real field
                "invalid_field": "value"  # Invalid
            })

    def test_validate_feature_metadata_spec_valid(self, metadata_manager):
        """Test validation with valid FeatureMetadata spec."""
        # Should not raise
        metadata_manager.validate_feature_metadata_spec({
            "feature_type": FeatureType.HRV,
            "epoch_window_length": pd.Timedelta(seconds=30)
        })

    def test_validate_feature_metadata_spec_invalid(self, metadata_manager):
        """Test validation with invalid FeatureMetadata spec."""
        with pytest.raises(ValueError, match="Invalid FeatureMetadata fields"):
            metadata_manager.validate_feature_metadata_spec({
                "invalid_field": "value"
            })

    def test_validate_feature_metadata_spec_partial_invalid(self, metadata_manager):
        """Test validation with mix of valid and invalid fields."""
        with pytest.raises(ValueError, match="Invalid FeatureMetadata fields"):
            metadata_manager.validate_feature_metadata_spec({
                "feature_type": FeatureType.HRV,  # Valid
                "invalid_field": "value"  # Invalid
            })


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_update_empty_metadata_spec(self, metadata_manager, sample_signal):
        """Test updating with empty metadata spec."""
        original_id = sample_signal.metadata.signal_id
        metadata_manager.update_time_series_metadata(sample_signal, {})
        # Signal ID should remain unchanged
        assert sample_signal.metadata.signal_id == original_id

    def test_update_all_enum_fields(self, metadata_manager, sample_signal):
        """Test updating all enum fields at once."""
        metadata_manager.update_time_series_metadata(
            sample_signal,
            {
                "signal_type": "PPG",
                "sensor_type": "PPG",
                "sensor_model": "POLAR_H10",
                "body_position": "CHEST",
                "units": "bpm"
            }
        )
        assert sample_signal.metadata.signal_type == SignalType.PPG
        assert sample_signal.metadata.sensor_type == SensorType.PPG
        assert sample_signal.metadata.sensor_model == SensorModel.POLAR_H10
        assert sample_signal.metadata.body_position == BodyPosition.CHEST
        assert sample_signal.metadata.units == Unit.BPM
