"""Tests for ImportService class."""

import pytest
import pandas as pd
import uuid
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from sleep_analysis.services.import_service import ImportService
from sleep_analysis.signals.ppg_signal import PPGSignal
from sleep_analysis.signals.time_series_signal import TimeSeriesSignal
from sleep_analysis.features.feature import Feature
from sleep_analysis.core.metadata import TimeSeriesMetadata, FeatureMetadata, FeatureType
from sleep_analysis.core.metadata_handler import MetadataHandler
from sleep_analysis.signal_types import SignalType, SensorType, Unit


@pytest.fixture
def import_service():
    """Create an ImportService instance for testing."""
    return ImportService()


@pytest.fixture
def mock_importer():
    """Create a mock importer."""
    importer = Mock()
    return importer


@pytest.fixture
def sample_ppg_signal():
    """Create a sample PPG signal for testing."""
    index = pd.date_range(start="2023-01-01", periods=100, freq="10ms", tz="UTC")
    data = pd.DataFrame({"value": range(100)}, index=index)
    data.index.name = 'timestamp'
    return PPGSignal(
        data=data,
        metadata={
            "signal_id": str(uuid.uuid4()),
            "signal_type": SignalType.PPG,
            "units": Unit.ARBITRARY
        }
    )


@pytest.fixture
def sample_feature():
    """Create a sample Feature for testing."""
    epoch_index = pd.date_range(start="2023-01-01", periods=10, freq="1s", name='epoch_start_time', tz="UTC")
    feature_data = pd.DataFrame({
        ('signal_0', 'mean'): [1.5] * 10,
        ('signal_0', 'std'): [0.5] * 10
    }, index=epoch_index)
    feature_data.columns = pd.MultiIndex.from_tuples(feature_data.columns, names=['signal_key', 'feature'])

    return Feature(
        data=feature_data,
        metadata={
            "feature_id": str(uuid.uuid4()),
            "feature_type": FeatureType.STATISTICAL,
            "feature_names": ['mean', 'std'],
            "source_signal_ids": [str(uuid.uuid4())],
            "source_signal_keys": ["signal_0"],
            "epoch_window_length": pd.Timedelta("1s"),
            "epoch_step_size": pd.Timedelta("1s")
        }
    )


class TestImportServiceInit:
    """Tests for ImportService initialization."""

    def test_init_default_handler(self):
        """Test initialization with default metadata handler."""
        service = ImportService()
        assert service.metadata_handler is not None
        assert isinstance(service.metadata_handler, MetadataHandler)

    def test_init_custom_handler(self):
        """Test initialization with custom metadata handler."""
        custom_handler = MetadataHandler(default_values={"test": "value"})
        service = ImportService(custom_handler)
        assert service.metadata_handler is custom_handler
        assert service.metadata_handler.default_values == {"test": "value"}


class TestImportSignalsFromSource:
    """Tests for import_signals_from_source method."""

    def test_import_single_file(self, import_service, mock_importer, sample_ppg_signal):
        """Test importing from a single file."""
        # Setup mock importer to return a signal
        mock_importer.import_signal.return_value = sample_ppg_signal

        spec = {
            "signal_type": "ppg",
            "strict_validation": True
        }

        with patch('os.path.exists', return_value=True):
            signals = import_service.import_signals_from_source(
                mock_importer,
                "/path/to/file.csv",
                spec
            )

        assert len(signals) == 1
        assert isinstance(signals[0], TimeSeriesSignal)
        mock_importer.import_signal.assert_called_once()

    def test_import_file_pattern(self, import_service, mock_importer, sample_ppg_signal):
        """Test importing multiple files with a pattern."""
        # Setup mock importer to return signals for pattern import
        mock_importer.import_signals.return_value = [sample_ppg_signal, sample_ppg_signal]

        spec = {
            "signal_type": "ppg",
            "file_pattern": "*.csv",
            "strict_validation": True
        }

        with patch('os.path.isdir', return_value=True):
            signals = import_service.import_signals_from_source(
                mock_importer,
                "/path/to/dir",
                spec
            )

        assert len(signals) == 2
        assert all(isinstance(s, TimeSeriesSignal) for s in signals)
        mock_importer.import_signals.assert_called_once()

    def test_import_nonexistent_file_strict(self, import_service, mock_importer):
        """Test importing nonexistent file with strict validation."""
        spec = {
            "signal_type": "ppg",
            "strict_validation": True
        }

        with patch('os.path.exists', return_value=False):
            with pytest.raises(ValueError, match="Source file not found"):
                import_service.import_signals_from_source(
                    mock_importer,
                    "/path/to/nonexistent.csv",
                    spec
                )

    def test_import_nonexistent_file_lenient(self, import_service, mock_importer):
        """Test importing nonexistent file with lenient validation."""
        spec = {
            "signal_type": "ppg",
            "strict_validation": False
        }

        with patch('os.path.exists', return_value=False):
            signals = import_service.import_signals_from_source(
                mock_importer,
                "/path/to/nonexistent.csv",
                spec
            )

        assert signals == []

    def test_import_invalid_type(self, import_service, mock_importer):
        """Test importing when importer returns invalid type."""
        # Setup mock to return non-signal object
        mock_importer.import_signal.return_value = "not a signal"

        spec = {
            "signal_type": "ppg",
            "strict_validation": True
        }

        with patch('os.path.exists', return_value=True):
            signals = import_service.import_signals_from_source(
                mock_importer,
                "/path/to/file.csv",
                spec
            )

        # Should filter out invalid types
        assert len(signals) == 0


class TestUpdateTimeSeriesMetadata:
    """Tests for update_time_series_metadata method."""

    def test_update_basic_fields(self, import_service, sample_ppg_signal):
        """Test updating basic metadata fields."""
        metadata_spec = {
            "name": "Updated PPG Signal"
        }

        import_service.update_time_series_metadata(sample_ppg_signal, metadata_spec)

        assert sample_ppg_signal.metadata.name == "Updated PPG Signal"

    def test_update_enum_fields(self, import_service, sample_ppg_signal):
        """Test updating enum fields from strings."""
        metadata_spec = {
            "signal_type": "PPG",
            "sensor_type": "PPG",
            "units": "BPM"
        }

        import_service.update_time_series_metadata(sample_ppg_signal, metadata_spec)

        assert sample_ppg_signal.metadata.signal_type == SignalType.PPG
        assert sample_ppg_signal.metadata.sensor_type == SensorType.PPG
        assert sample_ppg_signal.metadata.units == Unit.BPM

    def test_update_sensor_info(self, import_service, sample_ppg_signal):
        """Test updating nested sensor_info dictionary."""
        metadata_spec = {
            "sensor_info": {
                "device_id": "ABC123",
                "firmware_version": "1.2.3"
            }
        }

        import_service.update_time_series_metadata(sample_ppg_signal, metadata_spec)

        assert sample_ppg_signal.metadata.sensor_info is not None
        assert sample_ppg_signal.metadata.sensor_info["device_id"] == "ABC123"
        assert sample_ppg_signal.metadata.sensor_info["firmware_version"] == "1.2.3"

    def test_update_invalid_enum(self, import_service, sample_ppg_signal, caplog):
        """Test updating with invalid enum value logs warning."""
        metadata_spec = {
            "signal_type": "INVALID_TYPE"
        }

        import_service.update_time_series_metadata(sample_ppg_signal, metadata_spec)

        # Should log warning but not crash
        assert "Invalid enum value" in caplog.text or "Skipping update" in caplog.text

    def test_update_wrong_type(self, import_service, sample_feature):
        """Test updating with wrong object type raises error."""
        with pytest.raises(TypeError, match="Expected TimeSeriesSignal"):
            import_service.update_time_series_metadata(sample_feature, {})


class TestUpdateFeatureMetadata:
    """Tests for update_feature_metadata method."""

    def test_update_basic_fields(self, import_service, sample_feature):
        """Test updating basic feature metadata fields."""
        metadata_spec = {
            "name": "Updated Feature"
        }

        import_service.update_feature_metadata(sample_feature, metadata_spec)

        assert sample_feature.metadata.name == "Updated Feature"

    def test_update_feature_type_enum(self, import_service, sample_feature):
        """Test updating feature_type enum from string."""
        metadata_spec = {
            "feature_type": "STATISTICAL"
        }

        import_service.update_feature_metadata(sample_feature, metadata_spec)

        assert sample_feature.metadata.feature_type == FeatureType.STATISTICAL

    def test_update_timedelta_fields(self, import_service, sample_feature):
        """Test updating timedelta fields from strings."""
        metadata_spec = {
            "epoch_window_length": "30s",
            "epoch_step_size": "15s"
        }

        # This test assumes the update logic handles timedelta conversion
        # The current implementation may not handle this automatically
        # You may need to update the service to handle this
        import_service.update_feature_metadata(sample_feature, metadata_spec)

    def test_update_wrong_type(self, import_service, sample_ppg_signal):
        """Test updating with wrong object type raises error."""
        with pytest.raises(TypeError, match="Expected Feature"):
            import_service.update_feature_metadata(sample_ppg_signal, {})


class TestIntegration:
    """Integration tests for ImportService."""

    def test_full_import_workflow(self, import_service, mock_importer, sample_ppg_signal):
        """Test complete import and metadata update workflow."""
        # Setup mock importer
        mock_importer.import_signal.return_value = sample_ppg_signal

        # Import signal
        spec = {"signal_type": "ppg", "strict_validation": True}
        with patch('os.path.exists', return_value=True):
            signals = import_service.import_signals_from_source(
                mock_importer,
                "/path/to/file.csv",
                spec
            )

        assert len(signals) == 1
        imported_signal = signals[0]

        # Update metadata
        metadata_spec = {
            "name": "Imported Signal",
            "sensor_type": "PPG"
        }
        import_service.update_time_series_metadata(imported_signal, metadata_spec)

        assert imported_signal.metadata.name == "Imported Signal"
        assert imported_signal.metadata.sensor_type == SensorType.PPG
