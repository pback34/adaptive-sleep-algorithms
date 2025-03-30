"""Tests for the importers module."""

import pytest
import pandas as pd
import os
import re
from datetime import datetime

from sleep_analysis.importers import SignalImporter, CSVImporterBase, PolarCSVImporter, MergingImporter
from sleep_analysis.signals import PPGSignal, AccelerometerSignal
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition

# ===== Fixtures =====

@pytest.fixture
def sample_csv(tmp_path):
    """
    Fixture to create a sample CSV file for testing.
    
    Returns:
        str: Path to the temporary CSV file.
    """
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
        "value": list(range(100))
    })
    csv_path = tmp_path / "sample.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def accelerometer_csv(tmp_path):
    """
    Fixture to create a sample accelerometer CSV file for testing.
    
    Returns:
        str: Path to the temporary CSV file.
    """
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
        "x": list(range(100)),
        "y": list(range(100, 200)),
        "z": list(range(200, 300)),
    })
    csv_path = tmp_path / "accelerometer.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def polar_csv(tmp_path):
    """
    Fixture to create a sample Polar CSV file for testing.
    
    Returns:
        str: Path to the temporary CSV file.
    """
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
        "ppg_value": list(range(100))
    })
    csv_path = tmp_path / "polar_subject123_01.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def polar_csv_directory(tmp_path):
    """
    Fixture to create a directory with multiple Polar CSV files.
    
    Returns:
        str: Path to the temporary directory.
    """
    # Create directory
    polar_dir = tmp_path / "polar_data"
    polar_dir.mkdir()
    
    # Create 3 CSV files
    for i in range(3):
        data = pd.DataFrame({
            "timestamp": pd.date_range(start=f"2023-01-0{i+1}", periods=50, freq="1s"),
            "ppg_value": list(range(i*50, (i+1)*50))
        })
        csv_path = polar_dir / f"polar_subject123_{i+1:02d}.csv"
        data.to_csv(csv_path, index=False)
    
    return str(polar_dir)


# ===== New Importer Hierarchy Tests =====

class TestCSVImporterBase:
    """Test suite for the CSVImporterBase class."""
    
    class ConcreteCSVImporter(CSVImporterBase):
        """Concrete implementation of CSVImporterBase for testing."""
        
        def _parse_csv(self, source: str) -> pd.DataFrame:
            """Implementation of abstract method."""
            df = pd.read_csv(source)
            # Convert timestamp column to datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            return df
    
    def test_base_class_abstract(self):
        """Test that CSVImporterBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CSVImporterBase()
    
    def test_concrete_subclass(self, sample_csv):
        """Test that a concrete subclass can be instantiated and used."""
        importer = self.ConcreteCSVImporter()
        signal = importer.import_signal(sample_csv, "PPG")
        assert isinstance(signal, PPGSignal)
        assert signal.metadata.signal_type == SignalType.PPG
        # Verify data has proper DatetimeIndex
        assert isinstance(signal.get_data().index, pd.DatetimeIndex)
    
    def test_validate_columns(self, tmp_path):
        """Test column validation in CSVImporterBase."""
        importer = self.ConcreteCSVImporter()
        
        # Create invalid CSV with timestamp but missing value column
        timestamp_data = pd.date_range(start="2023-01-01", periods=3, freq="1s")
        invalid_data = pd.DataFrame({"timestamp": timestamp_data})
        invalid_csv = tmp_path / "invalid.csv"
        invalid_data.to_csv(invalid_csv, index=False)
        
        with pytest.raises(ValueError, match="Missing required columns for PPG"):
            importer.import_signal(str(invalid_csv), "PPG")
            
    def test_timestamp_handling(self, tmp_path):
        """Test handling of timestamp columns and DatetimeIndex."""
        importer = self.ConcreteCSVImporter()
        
        # Create test data with timestamp and value columns
        timestamp_data = pd.date_range(start="2023-01-01", periods=10, freq="1s")
        test_data = pd.DataFrame({
            "timestamp": timestamp_data,
            "value": list(range(10))
        })
        test_csv = tmp_path / "timestamp_test.csv"
        test_data.to_csv(test_csv, index=False)
        
        # Import the signal
        signal = importer.import_signal(str(test_csv), "PPG")
        
        # Check that the timestamp was properly converted to DatetimeIndex
        assert isinstance(signal.get_data().index, pd.DatetimeIndex)
        assert len(signal.get_data()) == 10
        assert 'value' in signal.get_data().columns

class TestPolarCSVImporter:
    """Test suite for the PolarCSVImporter class."""
    
    def test_initialization_with_config(self):
        """Test initialization with a configuration dictionary."""
        config = {
            "column_mapping": {"timestamp": "time", "value": "ppg_value"},
            "time_format": "%Y-%m-%d %H:%M:%S",
            "filename_pattern": r"polar_(?P<subject_id>\w+)_(?P<session>\d+)\.csv",
            "sensor_model": "POLAR_H10",
            "body_position": "LEFT_WRIST"
        }
        importer = PolarCSVImporter(config)
        assert importer.config["column_mapping"] == {"timestamp": "time", "value": "ppg_value"}
        assert importer.config["time_format"] == "%Y-%m-%d %H:%M:%S"
    
    def test_initialization_without_config(self):
        """Test initialization without a configuration (using defaults)."""
        importer = PolarCSVImporter()
        assert "column_mapping" in importer.config
        assert "time_format" in importer.config
    
    def test_parse_csv(self, tmp_path):
        """Test CSV parsing with column mapping."""
        # Create CSV with custom column names
        data = pd.DataFrame({
            "time": pd.date_range(start="2023-01-01", periods=10, freq="1s"),
            "ppg_value": list(range(10))
        })
        csv_path = tmp_path / "test.csv"
        data.to_csv(csv_path, index=False, sep=",")
        
        # Configure importer with column mapping
        config = {
            "column_mapping": {"timestamp": "time", "value": "ppg_value"},
            "delimiter": ","
        }
        importer = PolarCSVImporter(config)
        
        # Parse the CSV and check column renaming
        df = importer._parse_csv(str(csv_path))
        # After processing, timestamp is set as index, so check 'value' column
        assert "value" in df.columns
        # Check that the index is a DatetimeIndex (timestamp was processed correctly)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "value" in df.columns
        assert len(df) == 10
    
    def test_import_signal_with_metadata(self, polar_csv):
        """Test importing a signal with metadata from filename and config."""
        config = {
            "column_mapping": {"timestamp": "timestamp", "value": "ppg_value"},
            "filename_pattern": r"polar_(?P<subject_id>\w+)_(?P<session>\d+)\.csv",
            "sensor_model": "POLAR_H10",
            "body_position": "LEFT_WRIST",
            "delimiter": ",",
            # Add flag to prevent timestamp from being set as index
            "preserve_timestamp_column": True
        }
        importer = PolarCSVImporter(config)
        signal = importer.import_signal(polar_csv, "PPG")
        
        # Verify signal data
        assert isinstance(signal, PPGSignal)
        assert signal.metadata.signal_type == SignalType.PPG
        
        # Verify extracted metadata
        assert "subject_id" in signal.metadata.sensor_info
        assert signal.metadata.sensor_info["subject_id"] == "subject123"
        assert signal.metadata.sensor_model == SensorModel.POLAR_H10
        assert signal.metadata.body_position == BodyPosition.LEFT_WRIST
    
    def test_import_signals_from_directory(self, polar_csv_directory):
        """Test importing multiple signals from a directory."""
        config = {
            "column_mapping": {"timestamp": "timestamp", "value": "ppg_value"},
            "filename_pattern": r"polar_(?P<subject_id>\w+)_(?P<session>\d+)\.csv",
            "delimiter": ","
        }
        importer = PolarCSVImporter(config)
        signals = importer.import_signals(polar_csv_directory, "PPG")
        
        # Should import 3 signals from the directory
        assert len(signals) == 3
        assert all(isinstance(signal, PPGSignal) for signal in signals)
        
        # Each signal should have metadata from its filename
        for signal in signals:
            assert "subject_id" in signal.metadata.sensor_info
            assert signal.metadata.sensor_info["subject_id"] == "subject123"
            assert "session" in signal.metadata.sensor_info
            
    def test_import_with_custom_time_format(self, tmp_path):
        """Test importing with a custom time format."""
        # Create CSV with custom date format
        custom_df = pd.DataFrame({
            "time": ["2023/01/01 08:00:00", "2023/01/01 08:00:01", "2023/01/01 08:00:02"],
            "ppg_value": [1, 2, 3]
        })
        custom_csv = tmp_path / "custom_time.csv"
        custom_df.to_csv(custom_csv, index=False, sep=",")
        
        # Configure importer with custom time format
        config = {
            "column_mapping": {"timestamp": "time", "value": "ppg_value"},
            "time_format": "%Y/%m/%d %H:%M:%S",
            "delimiter": ","
        }
        
        importer = PolarCSVImporter(config)
        signal = importer.import_signal(str(custom_csv), "PPG")
        
        # Verify timestamp parsing
        assert isinstance(signal.get_data().index, pd.DatetimeIndex)
        assert len(signal.get_data()) == 3
        # First timestamp should be 2023-01-01 08:00:00
        assert signal.get_data().index[0].strftime("%Y-%m-%d %H:%M:%S") == "2023-01-01 08:00:00"
        
    def test_error_handling_invalid_csv(self, tmp_path):
        """Test error handling with invalid CSV files."""
        # Create an empty file
        empty_file = tmp_path / "empty.csv"
        empty_file.touch()
        
        importer = PolarCSVImporter()
        
        # Empty file should raise ValueError
        with pytest.raises(ValueError, match="Error parsing CSV file"):
            importer.import_signal(str(empty_file), "PPG")
            
        # Non-existent file should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            importer.import_signal(str(tmp_path / "nonexistent.csv"), "PPG")

# ===== MergingImporter Tests =====

class TestMergingImporter:
    """Test suite for the MergingImporter class."""
    
    @pytest.fixture
    def merging_config(self):
        """Fixture to create a sample config for MergingImporter."""
        return {
            "file_pattern": "*.csv",
            "timestamp_col": "timestamp",
            "sort_by": "filename",
            "delimiter": ",",
            "header": 0
        }
    
    @pytest.fixture
    def fragmented_data_dir(self, tmp_path):
        """Create a directory with multiple CSV files with timestamped data."""
        data_dir = tmp_path / "fragmented_data"
        data_dir.mkdir()
        
        # Create three files with consecutive timestamps
        for i in range(3):
            start_time = pd.Timestamp(f"2023-01-0{i+1}")
            data = pd.DataFrame({
                "timestamp": pd.date_range(start=start_time, periods=10, freq="1s"),
                "value": list(range(i*10, (i+1)*10))
            })
            file_path = data_dir / f"data_part_{i+1}.csv"
            data.to_csv(file_path, index=False)
            
        return str(data_dir)
    
    def test_initialization(self, merging_config):
        """Test initialization with configuration."""
        importer = MergingImporter(merging_config)
        assert importer.file_pattern == "*.csv"
        assert importer.time_column == "timestamp"
        assert importer.sort_by == "filename"
        
    def test_initialization_missing_config(self):
        """Test initialization with missing required config."""
        with pytest.raises(ValueError, match="must include 'file_pattern'"):
            MergingImporter({})
            
        with pytest.raises(ValueError, match="must include 'timestamp_col'"):
            MergingImporter({"file_pattern": "*.csv"})
    
    def test_import_signal(self, fragmented_data_dir, merging_config):
        """Test importing and merging signals from multiple files."""
        importer = MergingImporter(merging_config)
        signal = importer.import_signal(fragmented_data_dir, "PPG")
        
        # Check that it's the right type and has merged data
        assert isinstance(signal, PPGSignal)
        assert len(signal.get_data()) == 30  # 3 files × 10 rows each
        assert signal.metadata.merged is True
        
    def test_sort_by_timestamp(self, fragmented_data_dir, merging_config):
        """Test sorting files by timestamp in the data."""
        # Modify config to sort by timestamp
        config = merging_config.copy()
        config["sort_by"] = "timestamp"
        
        importer = MergingImporter(config)
        signal = importer.import_signal(fragmented_data_dir, "PPG")
        
        # Check that the data is sorted by timestamp (should be anyway)
        data = signal.get_data()
        timestamps = data.index
        assert timestamps.is_monotonic_increasing
        
    def test_import_nonexistent_directory(self, merging_config):
        """Test importing from a nonexistent directory."""
        importer = MergingImporter(merging_config)
        with pytest.raises(FileNotFoundError):
            importer.import_signal("/nonexistent/directory", "PPG")
            
    def test_import_empty_directory(self, tmp_path, merging_config):
        """Test importing from an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        importer = MergingImporter(merging_config)
        # MergingImporter now returns None instead of raising an exception when no files found
        assert importer.import_signal(str(empty_dir), "PPG") is None

# ===== SignalImporter Interface Tests =====

def test_abstract_signal_importer():
    """Test that SignalImporter is an abstract class."""
    with pytest.raises(TypeError):
        SignalImporter()
