"""Tests for SignalRepository class."""

import pytest
import pandas as pd
import uuid
from unittest.mock import Mock

from src.sleep_analysis.core.repositories.signal_repository import SignalRepository
from src.sleep_analysis.core.metadata_handler import MetadataHandler
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.features.feature import Feature
from src.sleep_analysis.core.metadata import TimeSeriesMetadata, FeatureMetadata
from src.sleep_analysis.signal_types import SignalType, SensorType


@pytest.fixture
def metadata_handler():
    """Create a metadata handler for testing."""
    return MetadataHandler()


@pytest.fixture
def signal_repository(metadata_handler):
    """Create a signal repository for testing."""
    return SignalRepository(metadata_handler, collection_timezone="UTC")


@pytest.fixture
def sample_time_series_signal():
    """Create a sample HeartRateSignal for testing."""
    data = pd.DataFrame(
        {"hr": [70.0, 72.0, 68.0]},  # HeartRateSignal requires 'hr' column
        index=pd.date_range("2025-01-01", periods=3, freq="1s", tz="UTC")
    )
    # Pass metadata as dictionary
    return HeartRateSignal(
        data=data,
        metadata={"signal_id": "test_signal_001", "signal_type": SignalType.HEART_RATE}
    )


@pytest.fixture
def sample_feature():
    """Create a sample Feature for testing."""
    data = pd.DataFrame(
        {"mean_hr": [70.0, 72.0, 68.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="30s", tz="UTC")
    )
    # Pass metadata as dictionary
    return Feature(
        data=data,
        metadata={"feature_id": "test_feature_001", "sensor_type": SensorType.PPG}
    )


class TestSignalRepositoryBasics:
    """Test basic repository initialization and properties."""

    def test_init_creates_empty_dictionaries(self, signal_repository):
        """Test that initialization creates empty signal and feature dictionaries."""
        assert len(signal_repository.time_series_signals) == 0
        assert len(signal_repository.features) == 0

    def test_init_sets_metadata_handler(self, metadata_handler):
        """Test that initialization sets the metadata handler."""
        repo = SignalRepository(metadata_handler)
        assert repo.metadata_handler is metadata_handler

    def test_init_sets_collection_timezone(self):
        """Test that initialization sets the collection timezone."""
        repo = SignalRepository(MetadataHandler(), collection_timezone="America/New_York")
        assert repo.collection_timezone == "America/New_York"

    def test_init_defaults_to_utc(self):
        """Test that timezone defaults to UTC."""
        repo = SignalRepository(MetadataHandler())
        assert repo.collection_timezone == "UTC"


class TestAddTimeSeriesSignal:
    """Test adding TimeSeriesSignals to the repository."""

    def test_add_time_series_signal_success(self, signal_repository, sample_time_series_signal):
        """Test successfully adding a TimeSeriesSignal."""
        signal_repository.add_time_series_signal("hr_0", sample_time_series_signal)
        assert "hr_0" in signal_repository.time_series_signals
        assert signal_repository.time_series_signals["hr_0"] is sample_time_series_signal

    def test_add_time_series_signal_sets_name(self, signal_repository, sample_time_series_signal):
        """Test that adding a signal sets its name."""
        signal_repository.add_time_series_signal("hr_sensor", sample_time_series_signal)
        # Name should be set by the metadata handler
        assert sample_time_series_signal.metadata.name == "hr_sensor"

    def test_add_time_series_signal_type_error(self, signal_repository, sample_feature):
        """Test that adding a non-TimeSeriesSignal raises TypeError."""
        with pytest.raises(TypeError, match="not a TimeSeriesSignal"):
            signal_repository.add_time_series_signal("wrong", sample_feature)

    def test_add_time_series_signal_duplicate_key(self, signal_repository, sample_time_series_signal):
        """Test that adding a signal with duplicate key raises ValueError."""
        signal_repository.add_time_series_signal("hr_0", sample_time_series_signal)

        # Create another signal
        data = pd.DataFrame(
            {"hr": [75.0, 73.0, 71.0]},
            index=pd.date_range("2025-01-02", periods=3, freq="1s", tz="UTC")
        )
        metadata = TimeSeriesMetadata(
            signal_id="test_signal_002",
            signal_type=SignalType.HEART_RATE
        )
        signal2 = HeartRateSignal(data=data, metadata=metadata)

        with pytest.raises(ValueError, match="already exists"):
            signal_repository.add_time_series_signal("hr_0", signal2)

    def test_add_time_series_signal_duplicate_id_assigns_new_id(
        self, signal_repository, sample_time_series_signal
    ):
        """Test that duplicate signal IDs are reassigned."""
        # Add first signal
        signal_repository.add_time_series_signal("hr_0", sample_time_series_signal)
        original_id = sample_time_series_signal.metadata.signal_id

        # Create second signal with same ID
        data = pd.DataFrame(
            {"hr": [75.0, 73.0, 71.0]},
            index=pd.date_range("2025-01-02", periods=3, freq="1s", tz="UTC")
        )
        metadata = TimeSeriesMetadata(
            signal_id=original_id,  # Same ID!
            signal_type=SignalType.HEART_RATE
        )
        signal2 = HeartRateSignal(data=data, metadata=metadata)

        # Add second signal - ID should be reassigned
        signal_repository.add_time_series_signal("hr_1", signal2)
        assert signal2.metadata.signal_id != original_id

    def test_add_time_series_signal_invalid_index(self, signal_repository):
        """Test that signals without DatetimeIndex raise ValueError during construction."""
        data = pd.DataFrame(
            {"hr": [70.0, 72.0, 68.0]},
            index=[0, 1, 2]  # Regular index, not DatetimeIndex
        )
        metadata = TimeSeriesMetadata(
            signal_id="test_signal_003",
            signal_type=SignalType.HEART_RATE
        )
        # HeartRateSignal validates DatetimeIndex during __init__
        with pytest.raises(ValueError, match="must have a DatetimeIndex"):
            signal = HeartRateSignal(data=data, metadata=metadata)


class TestAddFeature:
    """Test adding Features to the repository."""

    def test_add_feature_success(self, signal_repository, sample_feature):
        """Test successfully adding a Feature."""
        signal_repository.add_feature("hrv_0", sample_feature)
        assert "hrv_0" in signal_repository.features
        assert signal_repository.features["hrv_0"] is sample_feature

    def test_add_feature_sets_name(self, signal_repository, sample_feature):
        """Test that adding a feature sets its name."""
        signal_repository.add_feature("hrv_features", sample_feature)
        assert sample_feature.metadata.name == "hrv_features"

    def test_add_feature_type_error(self, signal_repository, sample_time_series_signal):
        """Test that adding a non-Feature raises TypeError."""
        with pytest.raises(TypeError, match="not a Feature"):
            signal_repository.add_feature("wrong", sample_time_series_signal)

    def test_add_feature_duplicate_key(self, signal_repository, sample_feature):
        """Test that adding a feature with duplicate key raises ValueError."""
        signal_repository.add_feature("hrv_0", sample_feature)

        # Create another feature
        data = pd.DataFrame(
            {"mean_hr": [75.0, 73.0, 71.0]},
            index=pd.date_range("2025-01-02", periods=3, freq="30s", tz="UTC")
        )
        metadata = FeatureMetadata(
            feature_id="test_feature_002",
            sensor_type=SensorType.PPG
        )
        feature2 = Feature(data=data, metadata=metadata)

        with pytest.raises(ValueError, match="already exists"):
            signal_repository.add_feature("hrv_0", feature2)

    def test_add_feature_duplicate_id_assigns_new_id(
        self, signal_repository, sample_feature
    ):
        """Test that duplicate feature IDs are reassigned."""
        signal_repository.add_feature("hrv_0", sample_feature)
        original_id = sample_feature.metadata.feature_id

        # Create second feature with same ID
        data = pd.DataFrame(
            {"mean_hr": [75.0, 73.0, 71.0]},
            index=pd.date_range("2025-01-02", periods=3, freq="30s", tz="UTC")
        )
        metadata = FeatureMetadata(
            feature_id=original_id,  # Same ID!
            sensor_type=SensorType.PPG
        )
        feature2 = Feature(data=data, metadata=metadata)

        signal_repository.add_feature("hrv_1", feature2)
        assert feature2.metadata.feature_id != original_id


class TestAddSignalWithBaseName:
    """Test adding signals with auto-incremented base names."""

    def test_add_signal_with_base_name_first_signal(
        self, signal_repository, sample_time_series_signal
    ):
        """Test that first signal gets index 0."""
        key = signal_repository.add_signal_with_base_name("hr", sample_time_series_signal)
        assert key == "hr_0"
        assert "hr_0" in signal_repository.time_series_signals

    def test_add_signal_with_base_name_increments(self, signal_repository):
        """Test that subsequent signals get incremented indices."""
        # Create multiple signals
        signals = []
        for i in range(3):
            data = pd.DataFrame(
                {"hr": [float(i+70), float(i+71), float(i+72)]},
                index=pd.date_range("2025-01-01", periods=3, freq="1s", tz="UTC")
            )
            metadata = TimeSeriesMetadata(
                signal_id=f"test_signal_{i:03d}",
                signal_type=SignalType.HEART_RATE
            )
            signals.append(HeartRateSignal(data=data, metadata=metadata))

        key0 = signal_repository.add_signal_with_base_name("hr", signals[0])
        key1 = signal_repository.add_signal_with_base_name("hr", signals[1])
        key2 = signal_repository.add_signal_with_base_name("hr", signals[2])

        assert key0 == "hr_0"
        assert key1 == "hr_1"
        assert key2 == "hr_2"

    def test_add_signal_with_base_name_feature(self, signal_repository, sample_feature):
        """Test adding a feature with base name."""
        key = signal_repository.add_signal_with_base_name("hrv", sample_feature)
        assert key == "hrv_0"
        assert "hrv_0" in signal_repository.features

    def test_add_signal_with_base_name_empty_name_raises(
        self, signal_repository, sample_time_series_signal
    ):
        """Test that empty base name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            signal_repository.add_signal_with_base_name("", sample_time_series_signal)

    def test_add_signal_with_base_name_invalid_type_raises(self, signal_repository):
        """Test that invalid signal type raises TypeError."""
        with pytest.raises(TypeError, match="must be a TimeSeriesSignal or Feature"):
            signal_repository.add_signal_with_base_name("test", "not a signal")


class TestAddImportedSignals:
    """Test batch import of signals."""

    def test_add_imported_signals_success(self, signal_repository):
        """Test successfully importing multiple signals."""
        signals = []
        for i in range(3):
            data = pd.DataFrame(
                {"hr": [float(i+70), float(i+71), float(i+72)]},
                index=pd.date_range("2025-01-01", periods=3, freq="1s", tz="UTC")
            )
            metadata = TimeSeriesMetadata(
                signal_id=f"import_{i:03d}",
                signal_type=SignalType.HEART_RATE
            )
            signals.append(HeartRateSignal(data=data, metadata=metadata))

        keys = signal_repository.add_imported_signals(signals, "polar")

        assert len(keys) == 3
        assert keys == ["polar_0", "polar_1", "polar_2"]
        assert len(signal_repository.time_series_signals) == 3

    def test_add_imported_signals_with_start_index(self, signal_repository):
        """Test importing signals with custom start index."""
        signals = []
        for i in range(2):
            data = pd.DataFrame(
                {"hr": [float(i+70)]},
                index=pd.date_range("2025-01-01", periods=1, freq="1s", tz="UTC")
            )
            metadata = TimeSeriesMetadata(
                signal_id=f"import_{i:03d}",
                signal_type=SignalType.HEART_RATE
            )
            signals.append(HeartRateSignal(data=data, metadata=metadata))

        keys = signal_repository.add_imported_signals(signals, "polar", start_index=5)
        assert keys == ["polar_5", "polar_6"]

    def test_add_imported_signals_skips_non_signals(self, signal_repository, caplog):
        """Test that non-TimeSeriesSignal objects are skipped."""
        signals = [
            "not a signal",
            123,
            None
        ]

        keys = signal_repository.add_imported_signals(signals, "polar")
        assert len(keys) == 0
        assert "Skipping object" in caplog.text


class TestGetMethods:
    """Test retrieval methods."""

    def test_get_time_series_signal(self, signal_repository, sample_time_series_signal):
        """Test retrieving a TimeSeriesSignal by key."""
        signal_repository.add_time_series_signal("hr_0", sample_time_series_signal)
        retrieved = signal_repository.get_time_series_signal("hr_0")
        assert retrieved is sample_time_series_signal

    def test_get_time_series_signal_not_found(self, signal_repository):
        """Test that getting non-existent signal raises KeyError."""
        with pytest.raises(KeyError, match="No TimeSeriesSignal"):
            signal_repository.get_time_series_signal("nonexistent")

    def test_get_feature(self, signal_repository, sample_feature):
        """Test retrieving a Feature by key."""
        signal_repository.add_feature("hrv_0", sample_feature)
        retrieved = signal_repository.get_feature("hrv_0")
        assert retrieved is sample_feature

    def test_get_feature_not_found(self, signal_repository):
        """Test that getting non-existent feature raises KeyError."""
        with pytest.raises(KeyError, match="No Feature"):
            signal_repository.get_feature("nonexistent")

    def test_get_by_key_signal(self, signal_repository, sample_time_series_signal):
        """Test getting a signal by key using get_by_key."""
        signal_repository.add_time_series_signal("hr_0", sample_time_series_signal)
        retrieved = signal_repository.get_by_key("hr_0")
        assert retrieved is sample_time_series_signal

    def test_get_by_key_feature(self, signal_repository, sample_feature):
        """Test getting a feature by key using get_by_key."""
        signal_repository.add_feature("hrv_0", sample_feature)
        retrieved = signal_repository.get_by_key("hrv_0")
        assert retrieved is sample_feature

    def test_get_by_key_not_found(self, signal_repository):
        """Test that get_by_key raises KeyError for non-existent key."""
        with pytest.raises(KeyError, match="No TimeSeriesSignal or Feature"):
            signal_repository.get_by_key("nonexistent")

    def test_get_all_time_series(self, signal_repository, sample_time_series_signal):
        """Test getting all time-series signals."""
        signal_repository.add_time_series_signal("hr_0", sample_time_series_signal)
        all_signals = signal_repository.get_all_time_series()
        assert len(all_signals) == 1
        assert "hr_0" in all_signals

    def test_get_all_features(self, signal_repository, sample_feature):
        """Test getting all features."""
        signal_repository.add_feature("hrv_0", sample_feature)
        all_features = signal_repository.get_all_features()
        assert len(all_features) == 1
        assert "hrv_0" in all_features


class TestValidation:
    """Test validation methods."""

    def test_validate_timestamp_index_invalid(self, signal_repository):
        """Test validation of timestamp index happens during signal construction."""
        # HeartRateSignal validates DatetimeIndex during construction
        # So this test just verifies that the validation exists
        data = pd.DataFrame(
            {"hr": [70.0, 72.0, 68.0]},
            index=[0, 1, 2]  # Not a DatetimeIndex
        )
        metadata = TimeSeriesMetadata(
            signal_id="test",
            signal_type=SignalType.HEART_RATE
        )
        # Signal construction itself will fail
        with pytest.raises(ValueError, match="must have a DatetimeIndex"):
            signal = HeartRateSignal(data=data, metadata=metadata)

    def test_validate_timezone_logs_warning_for_mismatch(
        self, signal_repository, caplog, sample_time_series_signal
    ):
        """Test that timezone mismatch logs a warning."""
        # Repository is UTC, signal is UTC - should be fine
        signal_repository._validate_timezone("hr_0", sample_time_series_signal)

        # Create signal with different timezone
        data = pd.DataFrame(
            {"hr": [70.0, 72.0, 68.0]},
            index=pd.date_range("2025-01-01", periods=3, freq="1s", tz="America/New_York")
        )
        metadata = TimeSeriesMetadata(
            signal_id="test_ny",
            signal_type=SignalType.HEART_RATE
        )
        signal_ny = HeartRateSignal(data=data, metadata=metadata)

        signal_repository._validate_timezone("hr_ny", signal_ny)
        # Should log warning about timezone mismatch
        assert "Potential inconsistency" in caplog.text or "does not match" in caplog.text
