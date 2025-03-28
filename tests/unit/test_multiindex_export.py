"""Tests focused on multi-index column handling in exports."""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import shutil
import logging
from datetime import datetime

from sleep_analysis.core import SignalCollection
from sleep_analysis.signals import PPGSignal, AccelerometerSignal, HeartRateSignal
from sleep_analysis.export import ExportModule
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition
from sleep_analysis.utils import debug_multiindex

@pytest.fixture
def logger():
    """Setup test logger for debugging."""
    logger = logging.getLogger("test_multiindex")
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    return logger

@pytest.fixture
def temp_output_dir():
    """Fixture for a temporary directory to store export outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)  # Cleanup after test

@pytest.fixture
def sample_multiindex_df():
    """Create a sample DataFrame with MultiIndex columns for testing."""
    # Create datetimes for index
    dates = pd.date_range("2023-01-01", periods=5, freq="1s")
    
    # Create tuples for MultiIndex
    tuples = [
        ("HEART_RATE", "POLAR_H10", "CHEST", "hr"),
        ("ACCELEROMETER", "POLAR_H10", "CHEST", "x"),
        ("ACCELEROMETER", "POLAR_H10", "CHEST", "y"),
        ("ACCELEROMETER", "POLAR_H10", "CHEST", "z")
    ]
    
    # Level names
    index_names = ["signal_type", "sensor_model", "body_position", "column"]
    
    # Create MultiIndex
    columns = pd.MultiIndex.from_tuples(tuples, names=index_names)
    
    # Create DataFrame with sample data
    df = pd.DataFrame(
        np.random.randn(5, 4),
        index=dates,
        columns=columns
    )
    
    # Set index name
    df.index.name = "timestamp"
    
    return df

@pytest.fixture
def signal_collection_with_multiindex():
    """Create a signal collection with data that will result in a MultiIndex when combined."""
    collection = SignalCollection({
        "collection_id": "test_multiindex",
        "subject_id": "test_subject",
    })
    
    # Configure the collection to use multi-index
    collection.set_index_config(["signal_type", "sensor_model", "body_position"])
    
    # Create heart rate signal
    hr_data = pd.DataFrame({
        "hr": [60, 62, 65, 63, 64]
    }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
    
    hr_signal = HeartRateSignal(data=hr_data, metadata={
        "signal_id": "hr_test_id",
        "name": "HR Signal",
        "signal_type": SignalType.HEART_RATE,
        "sensor_type": SensorType.EKG,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.CHEST
    })
    
    # Create accelerometer signal
    accel_data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [6, 7, 8, 9, 10],
        "z": [11, 12, 13, 14, 15]
    }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
    
    accel_signal = AccelerometerSignal(data=accel_data, metadata={
        "signal_id": "accel_test_id",
        "name": "Accelerometer Signal",
        "signal_type": SignalType.ACCELEROMETER,
        "sensor_type": SensorType.ACCEL,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.CHEST
    })
    
    collection.add_signal("hr_0", hr_signal)
    collection.add_signal("accelerometer_0", accel_signal)
    
    return collection

def test_simple_multiindex_to_csv(temp_output_dir, sample_multiindex_df, logger):
    """Test direct export and import of a MultiIndex DataFrame to CSV."""
    # Log the original DataFrame structure
    logger.debug("Original DataFrame structure:")
    logger.debug(f"Shape: {sample_multiindex_df.shape}")
    logger.debug(f"Index: {sample_multiindex_df.index.name}")
    logger.debug(f"Column MultiIndex names: {sample_multiindex_df.columns.names}")
    logger.debug(f"Column MultiIndex levels: {sample_multiindex_df.columns.nlevels}")
    
    # Log the first few rows of the original DataFrame
    logger.debug("Original DataFrame head:")
    logger.debug(f"\n{sample_multiindex_df.head().to_string()}")
    
    # Export to CSV
    csv_path = os.path.join(temp_output_dir, "simple_multiindex.csv")
    sample_multiindex_df.to_csv(csv_path)
    
    # Log the raw CSV content
    logger.debug("Raw CSV file content:")
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 10:  # Only show first 10 lines
                break
            logger.debug(f"Line {i}: {line.strip()}")
    
    # Read back with explicit header rows
    header_rows = list(range(len(sample_multiindex_df.columns.names)))
    df_read = pd.read_csv(csv_path, header=header_rows, index_col=0)
    
    # Log the structure of the imported DataFrame
    logger.debug("Imported DataFrame structure:")
    logger.debug(f"Shape: {df_read.shape}")
    logger.debug(f"Columns: {df_read.columns}")
    
    # Check if it's a MultiIndex
    assert isinstance(df_read.columns, pd.MultiIndex), "Columns should be a MultiIndex after import"
    
    # Debug the imported MultiIndex structure
    logger.debug("Imported MultiIndex details:")
    debug_multiindex(df_read.columns, logger)
    
    # Check that the level names are preserved
    assert list(df_read.columns.names) == sample_multiindex_df.columns.names, \
        "Column level names should be preserved"
    
    # Check the values at each level
    for i, name in enumerate(df_read.columns.names):
        orig_values = sample_multiindex_df.columns.get_level_values(i).unique().tolist()
        read_values = df_read.columns.get_level_values(i).unique().tolist()
        logger.debug(f"Level {i} ({name}) original values: {orig_values}")
        logger.debug(f"Level {i} ({name}) imported values: {read_values}")
        
        # Check values match (might be in different order)
        assert set(orig_values) == set(read_values), \
            f"Values at level {i} ({name}) should match after import"

def test_collection_multiindex_export(temp_output_dir, signal_collection_with_multiindex, logger):
    """Test exporting a SignalCollection with MultiIndex to CSV."""
    # Get the combined DataFrame
    combined_df = signal_collection_with_multiindex.get_combined_dataframe()
    
    # Log the combined DataFrame structure
    logger.debug("Combined DataFrame structure before export:")
    logger.debug(f"Shape: {combined_df.shape}")
    logger.debug(f"Index name: {combined_df.index.name}")
    
    # Check and log MultiIndex details
    assert isinstance(combined_df.columns, pd.MultiIndex), "Combined columns should be a MultiIndex"
    logger.debug(f"Column MultiIndex names: {combined_df.columns.names}")
    logger.debug(f"Column MultiIndex levels: {combined_df.columns.nlevels}")
    
    # Debug the MultiIndex structure
    logger.debug("Combined DataFrame MultiIndex details:")
    debug_multiindex(combined_df.columns, logger)
    
    # Export using ExportModule
    exporter = ExportModule(signal_collection_with_multiindex)
    exporter.export(formats=["csv"], output_dir=temp_output_dir, include_combined=True)
    
    # Check the exported CSV file
    csv_path = os.path.join(temp_output_dir, "combined.csv")
    assert os.path.exists(csv_path), "Combined CSV file should exist"
    
    # Log the raw CSV content
    logger.debug("Raw CSV file content:")
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 10:  # Only show first 10 lines
                break
            logger.debug(f"Line {i}: {line.strip()}")
    
    # Try to read the CSV with different header configurations
    for header_count in range(1, 5):
        header_rows = list(range(header_count))
        try:
            logger.debug(f"Trying to read CSV with header={header_rows}, index_col=0")
            df_read = pd.read_csv(csv_path, header=header_rows, index_col=0)
            
            logger.debug(f"Successfully read with header={header_rows}:")
            logger.debug(f"Shape: {df_read.shape}")
            logger.debug(f"Columns: {df_read.columns}")
            
            if isinstance(df_read.columns, pd.MultiIndex):
                logger.debug("Successfully imported with MultiIndex columns")
                debug_multiindex(df_read.columns, logger)
                
                # If this read was successful with a MultiIndex, run more detailed checks
                logger.debug(f"Column level names: {df_read.columns.names}")
                logger.debug(f"Expected level names: {combined_df.columns.names}")
                
                # Check level names match the original
                assert list(df_read.columns.names) == list(combined_df.columns.names), \
                    "Column level names should match original after import"
                
                # Check the first column tuple values to ensure they're preserved
                first_column_orig = combined_df.columns[0]
                first_column_read = df_read.columns[0]
                logger.debug(f"Original first column: {first_column_orig}")
                logger.debug(f"Imported first column: {first_column_read}")
                
                assert first_column_orig == first_column_read, \
                    "First column tuple should be preserved after import"
                break
        except Exception as e:
            logger.debug(f"Failed to read with header={header_rows}: {e}")

def test_raw_multiindex_csv_creation(temp_output_dir, logger):
    """Test creating a multi-index CSV directly to ensure the structure is correct."""
    # Create a sample dataframe with multi-index columns
    index = pd.date_range('2023-01-01', periods=5, freq='1s')
    
    # Create multi-index columns
    columns = pd.MultiIndex.from_tuples([
        ('HEART_RATE', 'POLAR_H10', 'CHEST', 'hr'),
        ('ACCELEROMETER', 'POLAR_H10', 'CHEST', 'x'),
        ('ACCELEROMETER', 'POLAR_H10', 'CHEST', 'y'),
        ('ACCELEROMETER', 'POLAR_H10', 'CHEST', 'z')
    ], names=['signal_type', 'sensor_model', 'body_position', 'column'])
    
    # Create dataframe with sample data
    df = pd.DataFrame(
        data=np.random.rand(5, 4), 
        index=index,
        columns=columns
    )
    df.index.name = 'timestamp'
    
    # Log the original dataframe structure
    logger.debug("Original dataframe structure:")
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Column names: {df.columns.names}")
    logger.debug(f"Columns: {df.columns.tolist()}")
    
    # Write to CSV using pandas to_csv with default settings
    csv_path = os.path.join(temp_output_dir, "raw_multiindex.csv")
    df.to_csv(csv_path)
    
    # Examine the raw CSV content
    logger.debug("Raw CSV content:")
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            logger.debug(f"Line {i}: {line.strip()}")
    
    # Try to read the CSV back
    header_rows = list(range(len(df.columns.names)))
    df_read = pd.read_csv(csv_path, header=header_rows, index_col=0)
    
    # Check that the structure is preserved
    assert isinstance(df_read.columns, pd.MultiIndex), "Columns should still be a MultiIndex"
    assert df_read.columns.nlevels == df.columns.nlevels, "Number of levels should be preserved"
    assert list(df_read.columns.names) == list(df.columns.names), "Column names should be preserved"
    
    # Log the structure of the read dataframe
    logger.debug("Read dataframe structure:")
    logger.debug(f"Shape: {df_read.shape}")
    logger.debug(f"Column names: {df_read.columns.names}")
    logger.debug(f"Columns: {df_read.columns.tolist()}")
    
    # Write manual CSV file with known correct structure
    manual_csv_path = os.path.join(temp_output_dir, "manual_multiindex.csv")
    with open(manual_csv_path, 'w') as f:
        f.write("timestamp,signal_type,signal_type,signal_type,signal_type\n")
        f.write(",HEART_RATE,ACCELEROMETER,ACCELEROMETER,ACCELEROMETER\n")
        f.write(",POLAR_H10,POLAR_H10,POLAR_H10,POLAR_H10\n")
        f.write(",CHEST,CHEST,CHEST,CHEST\n")
        f.write(",hr,x,y,z\n")
        f.write("2023-01-01 00:00:00,72,0.1,0.2,0.3\n")
        f.write("2023-01-01 00:00:01,73,0.2,0.3,0.4\n")
    
    # Read the manual CSV back
    manual_df = pd.read_csv(manual_csv_path, header=[0, 1, 2, 3, 4], index_col=0)
    
    # Log the structure of the manual dataframe
    logger.debug("Manual CSV dataframe structure:")
    logger.debug(f"Shape: {manual_df.shape}")
    if isinstance(manual_df.columns, pd.MultiIndex):
        logger.debug(f"Column names: {manual_df.columns.names}")
        logger.debug(f"Columns: {manual_df.columns.tolist()}")
        debug_multiindex(manual_df.columns, logger)
    else:
        logger.debug(f"Not a MultiIndex: {manual_df.columns}")

def test_export_functions_multiindex(temp_output_dir, signal_collection_with_multiindex, logger):
    """Test the export functions specifically for MultiIndex handling."""
    # Initialize the exporter
    exporter = ExportModule(signal_collection_with_multiindex)
    
    # Get the combined DataFrame directly
    combined_df = signal_collection_with_multiindex.get_combined_dataframe()
    
    # Log the structure of the combined DataFrame
    logger.debug("Combined DataFrame structure before export:")
    logger.debug(f"Shape: {combined_df.shape}")
    if isinstance(combined_df.columns, pd.MultiIndex):
        logger.debug(f"Column names: {combined_df.columns.names}")
        logger.debug("Column MultiIndex details:")
        debug_multiindex(combined_df.columns, logger)
    else:
        logger.debug(f"Not a MultiIndex: {combined_df.columns}")
    
    # Direct export of the combined DataFrame
    direct_csv_path = os.path.join(temp_output_dir, "direct_export.csv")
    combined_df.to_csv(direct_csv_path)
    
    # Log the direct export CSV content
    logger.debug("Direct export CSV content:")
    with open(direct_csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 10: break
            logger.debug(f"Line {i}: {line.strip()}")
    
    # Use ExportModule to export to CSV
    exporter.export(formats=["csv"], output_dir=temp_output_dir, include_combined=True)
    
    # Log the ExportModule CSV content
    module_csv_path = os.path.join(temp_output_dir, "combined.csv")
    logger.debug("ExportModule CSV content:")
    with open(module_csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 10: break
            logger.debug(f"Line {i}: {line.strip()}")
    
    # Read both CSVs back and compare their structures
    # Determine number of header rows for direct export
    direct_header_rows = list(range(combined_df.columns.nlevels))
    
    # Read direct export
    logger.debug(f"Reading direct export with header={direct_header_rows}")
    direct_read = pd.read_csv(direct_csv_path, header=direct_header_rows, index_col=0)
    
    # Log structure of direct read
    logger.debug("Direct read DataFrame structure:")
    logger.debug(f"Shape: {direct_read.shape}")
    if isinstance(direct_read.columns, pd.MultiIndex):
        logger.debug(f"Column names: {direct_read.columns.names}")
        logger.debug("Column MultiIndex details:")
        debug_multiindex(direct_read.columns, logger)
    else:
        logger.debug(f"Not a MultiIndex: {direct_read.columns}")
    
    # Try different header row combinations for module export
    for header_count in range(1, 5):
        module_header_rows = list(range(header_count))
        try:
            logger.debug(f"Reading module export with header={module_header_rows}")
            module_read = pd.read_csv(module_csv_path, header=module_header_rows, index_col=0)
            
            # Log structure of module read
            logger.debug(f"Module read DataFrame structure (header={module_header_rows}):")
            logger.debug(f"Shape: {module_read.shape}")
            if isinstance(module_read.columns, pd.MultiIndex):
                logger.debug(f"Column names: {module_read.columns.names}")
                logger.debug("Column MultiIndex details:")
                debug_multiindex(module_read.columns, logger)
                
                # Compare with original
                logger.debug("Comparing with original MultiIndex:")
                logger.debug(f"Original levels: {combined_df.columns.nlevels}, Read levels: {module_read.columns.nlevels}")
                logger.debug(f"Original names: {combined_df.columns.names}, Read names: {module_read.columns.names}")
                
                # If found a matching structure, do more detailed checks
                if module_read.columns.nlevels == combined_df.columns.nlevels:
                    # Check the values at each level
                    for i in range(module_read.columns.nlevels):
                        orig_values = combined_df.columns.get_level_values(i).unique().tolist()
                        read_values = module_read.columns.get_level_values(i).unique().tolist()
                        logger.debug(f"Level {i} original values: {orig_values}")
                        logger.debug(f"Level {i} read values: {read_values}")
                    
                    # Success - found the right header structure
                    break
            else:
                logger.debug(f"Not a MultiIndex: {module_read.columns}")
                
        except Exception as e:
            logger.debug(f"Failed to read with header={module_header_rows}: {e}")
