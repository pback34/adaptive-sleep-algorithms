"""Tests for the export module."""

import pytest
import os
import pandas as pd
import json
import pickle
import tempfile
import shutil
from datetime import datetime

from sleep_analysis.export import ExportModule
from sleep_analysis.core import SignalCollection
from sleep_analysis.signals import PPGSignal, AccelerometerSignal
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from sleep_analysis import __version__

@pytest.fixture
def sample_signal_collection():
    """Fixture for a sample signal collection with test signals."""
    collection = SignalCollection({
        "collection_id": "test_collection",
        "subject_id": "test_subject",
        "start_datetime": datetime(2023, 1, 1),
        "end_datetime": datetime(2023, 1, 2),
    })
    
    # Add PPG signal
    ppg_data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
    
    ppg_signal = PPGSignal(data=ppg_data, metadata={
        "signal_id": "ppg_test_id",
        "name": "PPG Signal",
        "sample_rate": "100Hz",
        "units": Unit.BPM,
        "start_time": datetime(2023, 1, 1),
        "end_time": datetime(2023, 1, 1, 0, 0, 5),
        "sensor_type": SensorType.PPG,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.LEFT_WRIST
    })
    
    # Add accelerometer signal
    accel_data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [6, 7, 8, 9, 10],
        "z": [11, 12, 13, 14, 15]
    }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
    
    accel_signal = AccelerometerSignal(data=accel_data, metadata={
        "signal_id": "accel_test_id",
        "name": "Accelerometer Signal",
        "sample_rate": "50Hz",
        "units": Unit.G,
        "start_time": datetime(2023, 1, 1),
        "end_time": datetime(2023, 1, 1, 0, 0, 5),
        "sensor_type": SensorType.ACCEL,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.CHEST
    })
    
    # Add temporary signal that should be excluded from combined export
    temp_data = pd.DataFrame({
        "value": [100, 200, 300]
    }, index=pd.date_range("2023-01-01", periods=3, freq="1s"))
    
    temp_signal = PPGSignal(data=temp_data, metadata={
        "signal_id": "temp_test_id",
        "name": "Temporary Signal",
        "temporary": True,
        "signal_type": SignalType.PPG
    })
    
    collection.add_signal("ppg_0", ppg_signal)
    collection.add_signal("accelerometer_0", accel_signal)
    collection.add_signal("temp_0", temp_signal)
    
    return collection

@pytest.fixture
def temp_output_dir():
    """Fixture for a temporary directory to store export outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)  # Cleanup after test

def test_export_module_initialization(sample_signal_collection):
    """Test initialization of ExportModule."""
    exporter = ExportModule(sample_signal_collection)
    assert exporter.collection == sample_signal_collection

def test_serialize_metadata(sample_signal_collection):
    """Test metadata serialization."""
    exporter = ExportModule(sample_signal_collection)
    metadata = exporter._serialize_metadata()
    
    # Check structure
    assert "collection" in metadata
    assert "signals" in metadata
    
    # Check collection metadata
    assert metadata["collection"]["collection_id"] == "test_collection"
    assert metadata["collection"]["subject_id"] == "test_subject"
    assert metadata["collection"]["framework_version"] == __version__
    
    # Check signal metadata
    assert "ppg_0" in metadata["signals"]
    assert "accelerometer_0" in metadata["signals"]
    assert metadata["signals"]["ppg_0"]["signal_id"] == "ppg_test_id"
    assert metadata["signals"]["ppg_0"]["signal_type"] == SignalType.PPG.name

def test_export_excel(sample_signal_collection, temp_output_dir):
    """Test Excel export functionality."""
    # Generate and store the combined dataframe within the collection first
    sample_signal_collection.generate_alignment_grid()
    sample_signal_collection.apply_grid_alignment()
    sample_signal_collection.combine_aligned_signals()

    exporter = ExportModule(sample_signal_collection)
    exporter.export(formats=["excel"], output_dir=temp_output_dir, include_combined=True)

    # Check files were created
    assert os.path.exists(os.path.join(temp_output_dir, "signals.xlsx"))
    assert os.path.exists(os.path.join(temp_output_dir, "combined.xlsx"))
    
    # Verify Excel content
    signals_df = pd.read_excel(os.path.join(temp_output_dir, "signals.xlsx"), sheet_name="ppg_0")
    assert "value" in signals_df.columns
    assert len(signals_df) == 5  # 5 data points
    
    combined_df = pd.read_excel(os.path.join(temp_output_dir, "combined.xlsx"))
    # The columns might be in a MultiIndex or flat structure depending on implementation
    if isinstance(combined_df.columns, pd.MultiIndex):
        assert ('PPG Signal', 'value') in combined_df.columns
        assert ('Accelerometer Signal', 'x') in combined_df.columns
        assert ('Accelerometer Signal', 'y') in combined_df.columns
        assert ('Accelerometer Signal', 'z') in combined_df.columns
        temp_value_cols = [col for col in combined_df.columns if "temp_0" in str(col)]
    else:
        assert "ppg_0" in combined_df.columns  # From PPG signal
        assert "accelerometer_0_x" in combined_df.columns  # From accelerometer signal
        assert "accelerometer_0_y" in combined_df.columns  # From accelerometer signal
        assert "accelerometer_0_z" in combined_df.columns  # From accelerometer signal
        temp_value_cols = [col for col in combined_df.columns if "temp_0" in col]
    # Check for the correct number of rows (should match grid length)
    assert len(combined_df) == 5  # Should have 5 rows matching the grid
    assert len(temp_value_cols) == 0 # Ensure temporary signal columns are not present

def test_export_csv(sample_signal_collection, temp_output_dir):
    """Test CSV export functionality."""
    # Generate and store the combined dataframe within the collection first
    sample_signal_collection.generate_alignment_grid()
    sample_signal_collection.apply_grid_alignment()
    sample_signal_collection.combine_aligned_signals()

    exporter = ExportModule(sample_signal_collection)
    exporter.export(formats=["csv"], output_dir=temp_output_dir, include_combined=True)

    # Check files were created
    assert os.path.isdir(os.path.join(temp_output_dir, "signals"))
    assert os.path.exists(os.path.join(temp_output_dir, "signals", "ppg_0.csv"))
    assert os.path.exists(os.path.join(temp_output_dir, "signals", "accelerometer_0.csv"))
    assert os.path.exists(os.path.join(temp_output_dir, "metadata.json"))
    assert os.path.exists(os.path.join(temp_output_dir, "combined.csv"))
    
    # Verify CSV content
    ppg_df = pd.read_csv(os.path.join(temp_output_dir, "signals", "ppg_0.csv"))
    assert "value" in ppg_df.columns
    assert len(ppg_df) == 5  # 5 data points
    
    accel_df = pd.read_csv(os.path.join(temp_output_dir, "signals", "accelerometer_0.csv"))
    assert "x" in accel_df.columns
    assert "y" in accel_df.columns
    assert "z" in accel_df.columns
    assert len(accel_df) == 5  # 5 data points

    # Verify combined CSV content
    combined_path = os.path.join(temp_output_dir, "combined.csv")
    # The export process resets the index, so the first column in the CSV
    # is the default 0-based index. Read it using index_col=0.
    combined_df = pd.read_csv(combined_path, index_col=0)

    # Check content length - should match the input signals
    assert len(combined_df) == 5  # 5 data points

    # Verify some column data is present, regardless of structure
    # Check for columns related to ppg_0 and accelerometer_0
    # The exact column names depend on whether multi-index was used,
    # so check for substrings.
    cols_str = str(combined_df.columns)
    assert 'ppg_0' in cols_str or 'PPG Signal' in cols_str # Check for PPG data
    assert 'accelerometer_0' in cols_str or 'Accelerometer Signal' in cols_str # Check for Accel data
    assert 'temp_0' not in cols_str # Ensure temporary signal is excluded

    # Verify metadata
    with open(os.path.join(temp_output_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    assert metadata["collection"]["collection_id"] == "test_collection"
    assert metadata["signals"]["ppg_0"]["signal_id"] == "ppg_test_id"

def test_export_pickle(sample_signal_collection, temp_output_dir):
    """Test Pickle export functionality."""
    # Generate and store the combined dataframe within the collection first
    sample_signal_collection.generate_alignment_grid()
    sample_signal_collection.apply_grid_alignment()
    sample_signal_collection.combine_aligned_signals()

    exporter = ExportModule(sample_signal_collection)
    exporter.export(formats=["pickle"], output_dir=temp_output_dir, include_combined=True)

    # Check file was created
    pickle_path = os.path.join(temp_output_dir, "signals.pkl")
    assert os.path.exists(pickle_path)
    
    # Verify pickle content
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    assert "metadata" in data
    assert "signals" in data
    assert "combined" in data
    
    assert data["metadata"]["collection"]["collection_id"] == "test_collection"
    assert "ppg_0" in data["signals"]
    assert "accelerometer_0" in data["signals"]
    assert "temp_0" in data["signals"]
    
    # Check the combined dataframe structure - we're more interested in the presence
    # of the right data and absence of temporary signals than the exact MultiIndex structure
    assert data["combined"] is not None # Ensure combined data is not None
    cols_str = str(data["combined"].columns)
    assert 'ppg_0' in cols_str or 'PPG Signal' in cols_str # Check for PPG data (handles MultiIndex)
    assert 'accelerometer_0' in cols_str or 'Accelerometer Signal' in cols_str # Check for Accel data
    assert 'temp_0' not in cols_str  # Temporary signal should not be included

def test_export_hdf5(sample_signal_collection, temp_output_dir):
    """Test HDF5 export functionality."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed, skipping HDF5 test")

    # Generate and store the combined dataframe within the collection first
    sample_signal_collection.generate_alignment_grid()
    sample_signal_collection.apply_grid_alignment()
    sample_signal_collection.combine_aligned_signals()

    exporter = ExportModule(sample_signal_collection)
    exporter.export(formats=["hdf5"], output_dir=temp_output_dir, include_combined=True)

    # Check file was created
    h5_path = os.path.join(temp_output_dir, "signals.h5")
    assert os.path.exists(h5_path)
    
    # Verify HDF5 content using pandas HDFStore
    with pd.HDFStore(h5_path, mode='r') as store:
        # Check signals data
        assert "/signals/ppg_0" in store
        assert "/signals/accelerometer_0" in store
        assert "/combined" in store
        
        # Check data integrity
        ppg_df = store["/signals/ppg_0"]
        assert "value" in ppg_df.columns
        assert len(ppg_df) == 5
        
        accel_df = store["/signals/accelerometer_0"]
        assert "x" in accel_df.columns
        assert "y" in accel_df.columns
        assert "z" in accel_df.columns
        
        # Check combined dataframe
        combined_df = store["/combined"]
        # Check the presence of expected columns and absence of temporary signal
        cols_str = str(combined_df.columns)
        assert 'ppg_0' in cols_str or 'PPG Signal' in cols_str # Check for PPG data (handles MultiIndex)
        assert 'accelerometer_0' in cols_str or 'Accelerometer Signal' in cols_str # Check for Accel data
        assert 'temp_0' not in cols_str # Ensure temporary signal is excluded

    # Check metadata using h5py
    with h5py.File(h5_path, 'r') as f:
        assert "metadata" in f
        metadata_json = f["metadata"][()]
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode('utf-8')
        metadata = json.loads(metadata_json)
        assert metadata["collection"]["collection_id"] == "test_collection"
        assert metadata["signals"]["ppg_0"]["signal_id"] == "ppg_test_id"

def test_unsupported_format(sample_signal_collection, temp_output_dir):
    """Test that using an unsupported format raises ValueError."""
    exporter = ExportModule(sample_signal_collection)
    with pytest.raises(ValueError, match="Unsupported format"):
        exporter.export(formats=["invalid_format"], output_dir=temp_output_dir)

def test_multiple_formats(sample_signal_collection, temp_output_dir):
    """Test exporting to multiple formats at once."""
    # Generate and store the combined dataframe within the collection first
    sample_signal_collection.generate_alignment_grid()
    sample_signal_collection.apply_grid_alignment()
    sample_signal_collection.combine_aligned_signals()

    exporter = ExportModule(sample_signal_collection)
    exporter.export(formats=["excel", "csv"], output_dir=temp_output_dir, include_combined=True)

    # Check that both formats were exported
    assert os.path.exists(os.path.join(temp_output_dir, "signals.xlsx"))
    assert os.path.isdir(os.path.join(temp_output_dir, "signals"))
    assert os.path.exists(os.path.join(temp_output_dir, "metadata.json"))
    assert os.path.exists(os.path.join(temp_output_dir, "combined.xlsx"))
    assert os.path.exists(os.path.join(temp_output_dir, "combined.csv"))

def test_framework_version_in_metadata(sample_signal_collection, temp_output_dir):
    """Test that framework version is included in the exported metadata."""
    exporter = ExportModule(sample_signal_collection)
    exporter.export(formats=["csv"], output_dir=temp_output_dir)
    
    # Check framework version in metadata
    with open(os.path.join(temp_output_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    assert metadata["collection"]["framework_version"] == __version__
    for signal_key in metadata["signals"]:
        assert metadata["signals"][signal_key]["framework_version"] == __version__
        
def test_export_with_custom_multiindex(sample_signal_collection, temp_output_dir):
    """Test exporting with a custom multi-index configuration."""
    # Configure the collection with custom indices
    sample_signal_collection.set_index_config(["signal_type", "body_position"])

    # Generate and store the combined dataframe within the collection first
    sample_signal_collection.generate_alignment_grid()
    sample_signal_collection.apply_grid_alignment()
    sample_signal_collection.combine_aligned_signals()

    # Export with the configured indices
    exporter = ExportModule(sample_signal_collection)
    exporter.export(formats=["csv"], output_dir=temp_output_dir, include_combined=True)

    # Read the combined CSV file
    combined_path = os.path.join(temp_output_dir, "combined.csv")
    assert os.path.exists(combined_path)

    # Read the combined CSV using pandas to check MultiIndex columns
    combined_df = pd.read_csv(combined_path, header=[0, 1], index_col=0) # Read with MultiIndex header

    # Check that we have data
    assert not combined_df.empty

    # Check MultiIndex structure based on index_config
    assert combined_df.columns.names == ["signal_type", "body_position"] # No 'column' level name

    # Check presence of expected data columns under the correct MultiIndex levels
    # The MultiIndex now only has two levels based on the index_config
    assert ('PPG', 'LEFT_WRIST') in combined_df.columns
    assert ('ACCELEROMETER', 'CHEST') in combined_df.columns # Check for the base tuple

    # Verify no temporary data is included by checking the first level of the MultiIndex
    assert 'TEMPORARY' not in combined_df.columns.get_level_values(0) # Assuming temporary signal has a type

# Duplicate test removed - this test was already defined earlier in the file
