"""Integration tests for export functionality within a complete workflow."""

import pytest
import os
import pandas as pd
import json
import tempfile
import shutil
from datetime import datetime

from sleep_analysis.core import SignalCollection
from sleep_analysis.signals import PPGSignal
from sleep_analysis.importers import PolarCSVImporter
from sleep_analysis.export import ExportModule
from sleep_analysis.signal_types import SignalType

@pytest.fixture
def temp_csv_file():
    """Fixture to create a temporary CSV file with PPG data."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create CSV file with sample data - Use 'value' as source column name
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
        "ppg_value": list(range(100))  # Use 'ppg_value' to match importer config {"value": "ppg_value"}
    })
    csv_path = os.path.join(temp_dir, "polar_sample_01.csv")
    data.to_csv(csv_path, index=False)
    
    yield csv_path
    
    # Cleanup after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_output_dir():
    """Fixture for a temporary directory to store export outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)  # Cleanup after test

def test_full_workflow(temp_csv_file, temp_output_dir):
    """Test a complete workflow from import to export."""
    # Step 1: Create a signal collection
    collection = SignalCollection({
        "collection_id": "test_workflow",
        "subject_id": "workflow_subject",
        "start_datetime": datetime(2023, 1, 1),
        "end_datetime": datetime(2023, 1, 2),
    })
    
    # Step 2: Import signal
    # Add target_timezone to the importer config for direct use
    importer = PolarCSVImporter({
        "column_mapping": {"timestamp": "timestamp", "value": "ppg_value"},
        "filename_pattern": r".*\.csv",
        "preserve_timestamp_column": True,
        "target_timezone": "UTC" # Added target timezone
    })
    ppg_signal = importer.import_signal(temp_csv_file, "PPG")
    collection.add_time_series_signal("ppg_raw", ppg_signal) # Use correct method

    # Step 3: Apply filtering operation
    filtered_signal = ppg_signal.apply_operation("filter_lowpass", cutoff=5.0)
    collection.add_time_series_signal("ppg_filtered", filtered_signal) # Use correct method

    # Step 4: Apply another operation to create a temporary signal
    temp_signal = filtered_signal.apply_operation("filter_lowpass", cutoff=2.0)
    temp_signal.metadata.temporary = True
    collection.add_time_series_signal("ppg_temp", temp_signal) # Use correct method

    # Step 5: Generate the combined dataframe BEFORE exporting
    # These steps populate collection._aligned_dataframe
    collection.generate_alignment_grid() # Calculate alignment parameters
    collection.align_and_combine_signals() # Align using merge_asof and combine

    # Step 6: Export all signals (now including the generated combined data)
    exporter = ExportModule(collection)
    exporter.export(
        formats=["csv", "excel"],
        output_dir=temp_output_dir,
        # Use the 'content' argument to specify exporting all time series and combined data
        content=["all_ts", "combined_ts"]
    )

    # Verify exports:
    
    # 1. Check CSV files
    assert os.path.isdir(os.path.join(temp_output_dir, "signals"))
    assert os.path.exists(os.path.join(temp_output_dir, "signals", "ppg_raw.csv"))
    assert os.path.exists(os.path.join(temp_output_dir, "signals", "ppg_filtered.csv"))
    # Temporary signals are NOT exported individually when using "all_ts" content keyword
    # assert os.path.exists(os.path.join(temp_output_dir, "signals", "ppg_temp.csv")) # REMOVED THIS ASSERTION
    assert os.path.exists(os.path.join(temp_output_dir, "metadata.json"))
    # Check for the correct combined time-series filename
    assert os.path.exists(os.path.join(temp_output_dir, "combined_ts.csv"))

    # 2. Check Excel files
    assert os.path.exists(os.path.join(temp_output_dir, "signals.xlsx"))
    assert os.path.exists(os.path.join(temp_output_dir, "combined.xlsx"))

    # 3. Check combined dataframe excludes temporary signals
    # Read the correct combined time-series CSV file
    combined_df = pd.read_csv(os.path.join(temp_output_dir, "combined_ts.csv"), index_col=0)
    # Timestamp column and one column each for ppg_raw and ppg_filtered
    # The temporary signal should be excluded
    temp_cols = [col for col in combined_df.columns if "ppg_temp" in col]
    assert len(temp_cols) == 0
    
    # 4. Check metadata
    with open(os.path.join(temp_output_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Verify collection metadata
    assert metadata["collection"]["collection_id"] == "test_workflow"
    assert metadata["collection"]["subject_id"] == "workflow_subject"
    
    # Verify signal metadata
    assert "ppg_raw" in metadata["signals"]
    assert "ppg_filtered" in metadata["signals"]
    assert "ppg_temp" in metadata["signals"]
    
    # Verify signal relationships
    assert metadata["signals"]["ppg_filtered"]["derived_from"][0][0] == ppg_signal.metadata.signal_id
    assert metadata["signals"]["ppg_filtered"]["operations"][0]["operation_name"] == "filter_lowpass"
    assert metadata["signals"]["ppg_filtered"]["operations"][0]["parameters"]["cutoff"] == 5.0
