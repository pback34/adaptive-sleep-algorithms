"""
Unit tests for DataImportService.

Tests cover:
- Importing signals from files
- File pattern matching with glob
- Adding imported signals with sequential naming
- Strict vs non-strict validation
- Error handling
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from datetime import timezone

from src.sleep_analysis.core.services import DataImportService
from src.sleep_analysis.signals.time_series_signal import TimeSeriesSignal


class TestDataImportServiceInitialization:
    """Tests for DataImportService initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        add_ts = Mock()
        service = DataImportService(add_time_series_signal=add_ts)

        assert service.add_time_series_signal == add_ts


class TestImportSignalsFromSource:
    """Tests for import_signals_from_source method."""

    @pytest.fixture
    def mock_signal(self):
        """Create a mock time-series signal."""
        index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        data = pd.DataFrame({'hr': [70] * 10}, index=index)
        return TimeSeriesSignal(data, metadata={'name': 'test_signal'})

    @pytest.fixture
    def mock_importer(self, mock_signal):
        """Create a mock importer instance."""
        importer = Mock()
        importer.import_signal = Mock(return_value=mock_signal)
        importer.import_signals = Mock(return_value=[mock_signal, mock_signal])
        return importer

    def test_import_single_file_success(self, mock_importer, mock_signal):
        """Test successful import of single file."""
        service = DataImportService(add_time_series_signal=Mock())

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name

        try:
            spec = {'signal_type': 'hr'}
            signals = service.import_signals_from_source(mock_importer, tmp_path, spec)

            assert len(signals) == 1
            assert isinstance(signals[0], TimeSeriesSignal)
            mock_importer.import_signal.assert_called_once_with(tmp_path, 'hr')
        finally:
            os.unlink(tmp_path)

    def test_import_file_not_found_strict(self, mock_importer):
        """Test that missing file raises ValueError in strict mode."""
        service = DataImportService(add_time_series_signal=Mock())

        spec = {'signal_type': 'hr', 'strict_validation': True}

        with pytest.raises(ValueError, match="Source file not found"):
            service.import_signals_from_source(mock_importer, '/nonexistent/file.csv', spec)

    def test_import_file_not_found_non_strict(self, mock_importer):
        """Test that missing file returns empty list in non-strict mode."""
        service = DataImportService(add_time_series_signal=Mock())

        spec = {'signal_type': 'hr', 'strict_validation': False}

        with pytest.warns(UserWarning, match="Source file not found"):
            signals = service.import_signals_from_source(mock_importer, '/nonexistent/file.csv', spec)

        assert signals == []

    def test_import_with_file_pattern_using_importer(self, mock_importer):
        """Test import with file pattern when importer supports import_signals."""
        service = DataImportService(add_time_series_signal=Mock())

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = {'signal_type': 'hr', 'file_pattern': '*.csv'}
            signals = service.import_signals_from_source(mock_importer, tmpdir, spec)

            assert len(signals) == 2
            mock_importer.import_signals.assert_called_once_with(tmpdir, 'hr')

    @patch('glob.glob')
    def test_import_with_file_pattern_manual_glob(self, mock_glob, mock_signal):
        """Test import with file pattern using manual globbing."""
        # Create importer without import_signals method
        importer = Mock(spec=['import_signal'])
        importer.import_signal = Mock(return_value=mock_signal)

        service = DataImportService(add_time_series_signal=Mock())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock glob to return fake file paths
            mock_glob.return_value = [
                os.path.join(tmpdir, 'file1.csv'),
                os.path.join(tmpdir, 'file2.csv')
            ]

            spec = {'signal_type': 'hr', 'file_pattern': '*.csv'}
            signals = service.import_signals_from_source(importer, tmpdir, spec)

            assert len(signals) == 2
            assert importer.import_signal.call_count == 2

    def test_import_directory_not_found_strict(self, mock_importer):
        """Test that missing directory raises ValueError with file pattern in strict mode."""
        service = DataImportService(add_time_series_signal=Mock())

        spec = {'signal_type': 'hr', 'file_pattern': '*.csv', 'strict_validation': True}

        with pytest.raises(ValueError, match="Source directory not found"):
            service.import_signals_from_source(mock_importer, '/nonexistent/dir', spec)

    @patch('glob.glob')
    def test_import_no_matching_files_strict(self, mock_glob, mock_importer):
        """Test that no matching files raises ValueError in strict mode."""
        # Remove import_signals method to force manual globbing
        importer = Mock(spec=['import_signal'])
        mock_glob.return_value = []

        service = DataImportService(add_time_series_signal=Mock())

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = {'signal_type': 'hr', 'file_pattern': '*.csv', 'strict_validation': True}

            with pytest.raises(ValueError, match="No files found matching pattern"):
                service.import_signals_from_source(importer, tmpdir, spec)

    def test_import_error_in_file_strict(self, mock_importer):
        """Test that import error raises exception in strict mode."""
        mock_importer.import_signal = Mock(side_effect=Exception("Import error"))

        service = DataImportService(add_time_series_signal=Mock())

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name

        try:
            spec = {'signal_type': 'hr', 'strict_validation': True}

            with pytest.raises(Exception, match="Import error"):
                service.import_signals_from_source(mock_importer, tmp_path, spec)
        finally:
            os.unlink(tmp_path)

    def test_import_error_in_file_non_strict(self, mock_importer):
        """Test that import error returns empty list in non-strict mode."""
        mock_importer.import_signal = Mock(side_effect=Exception("Import error"))

        service = DataImportService(add_time_series_signal=Mock())

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name

        try:
            spec = {'signal_type': 'hr', 'strict_validation': False}

            with pytest.warns(UserWarning, match="Error importing"):
                signals = service.import_signals_from_source(mock_importer, tmp_path, spec)

            assert signals == []
        finally:
            os.unlink(tmp_path)

    def test_import_filters_non_timeseries_objects(self, mock_importer):
        """Test that non-TimeSeriesSignal objects are filtered out."""
        # Importer returns a mix of valid and invalid objects
        mock_importer.import_signal = Mock(return_value="not_a_signal")

        service = DataImportService(add_time_series_signal=Mock())

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name

        try:
            spec = {'signal_type': 'hr'}
            signals = service.import_signals_from_source(mock_importer, tmp_path, spec)

            # Should filter out the invalid object
            assert len(signals) == 0
        finally:
            os.unlink(tmp_path)


class TestAddImportedSignals:
    """Tests for add_imported_signals method."""

    @pytest.fixture
    def mock_signals(self):
        """Create mock signals for testing."""
        index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        signals = []
        for i in range(3):
            data = pd.DataFrame({f'signal{i}': [i] * 10}, index=index)
            signal = TimeSeriesSignal(data, metadata={'name': f'signal{i}'})
            signals.append(signal)
        return signals

    def test_add_imported_signals_success(self, mock_signals):
        """Test successful addition of imported signals."""
        add_ts = Mock()
        service = DataImportService(add_time_series_signal=add_ts)

        keys = service.add_imported_signals(mock_signals, 'hr', start_index=0)

        assert keys == ['hr_0', 'hr_1', 'hr_2']
        assert add_ts.call_count == 3

    def test_add_imported_signals_custom_start_index(self, mock_signals):
        """Test adding signals with custom start index."""
        add_ts = Mock()
        service = DataImportService(add_time_series_signal=add_ts)

        keys = service.add_imported_signals(mock_signals, 'ppg', start_index=5)

        assert keys == ['ppg_5', 'ppg_6', 'ppg_7']

    def test_add_imported_signals_skip_non_timeseries(self, mock_signals):
        """Test that non-TimeSeriesSignal objects are skipped."""
        add_ts = Mock()
        service = DataImportService(add_time_series_signal=add_ts)

        # Mix in a non-signal object
        mixed_list = [mock_signals[0], "not_a_signal", mock_signals[1]]

        keys = service.add_imported_signals(mixed_list, 'hr', start_index=0)

        # Should only add the 2 valid signals
        assert len(keys) == 2
        assert add_ts.call_count == 2

    def test_add_imported_signals_handles_key_conflict(self, mock_signals):
        """Test handling of key conflicts during addition."""
        # Mock add_ts to fail on first attempt, succeed on second
        add_ts = Mock()
        add_ts.side_effect = [
            ValueError("Key exists"),  # First call fails
            None,  # Second call with incremented index succeeds
            None,  # Third signal
            None   # Fourth signal (after skip)
        ]

        service = DataImportService(add_time_series_signal=add_ts)

        keys = service.add_imported_signals(mock_signals, 'hr', start_index=0)

        # First signal gets hr_1 (after hr_0 fails), then hr_2, hr_3
        assert len(keys) == 3
        assert add_ts.call_count == 4  # 1 failure + 3 successes

    def test_add_imported_signals_empty_list(self):
        """Test adding empty list of signals."""
        add_ts = Mock()
        service = DataImportService(add_time_series_signal=add_ts)

        keys = service.add_imported_signals([], 'hr', start_index=0)

        assert keys == []
        add_ts.assert_not_called()
