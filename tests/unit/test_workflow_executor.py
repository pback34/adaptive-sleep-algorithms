"""Tests for the WorkflowExecutor class."""

import pytest
import pandas as pd
import os
import yaml
import tempfile
import shutil
from datetime import datetime

from sleep_analysis.workflows.workflow_executor import WorkflowExecutor
from sleep_analysis.core.signal_collection import SignalCollection
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition
from sleep_analysis.signals import PPGSignal, AccelerometerSignal
from sleep_analysis.importers import PolarCSVImporter

# ===== Fixtures =====

@pytest.fixture
def empty_workflow_executor():
    """Return a fresh WorkflowExecutor instance with an empty collection."""
    return WorkflowExecutor()

@pytest.fixture
def signal_collection_with_data():
    """Return a SignalCollection with sample signals."""
    collection = SignalCollection()
    
    # Add a PPG signal
    ppg_data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
    ppg_signal = PPGSignal(data=ppg_data, metadata={
        "signal_id": "ppg_test_id",
        "name": "PPG Signal",
        "sensor_type": SensorType.PPG,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.LEFT_WRIST
    })
    collection.add_time_series_signal("ppg_0", ppg_signal) # Use add_time_series_signal

    # Add another PPG signal
    ppg_data2 = pd.DataFrame({
        "value": [10, 20, 30, 40, 50]
    }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
    ppg_signal2 = PPGSignal(data=ppg_data2, metadata={
        "signal_id": "ppg_test_id_2",
        "name": "PPG Signal 2",
        "sensor_type": SensorType.PPG,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.RIGHT_WRIST
    })
    collection.add_time_series_signal("ppg_1", ppg_signal2) # Use add_time_series_signal

    # Add an accelerometer signal
    accel_data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [6, 7, 8, 9, 10],
        "z": [11, 12, 13, 14, 15]
    }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
    accel_signal = AccelerometerSignal(data=accel_data, metadata={
        "signal_id": "accel_test_id",
        "name": "Accelerometer Signal",
        "sensor_type": SensorType.ACCEL,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.CHEST
    })
    collection.add_time_series_signal("accelerometer_0", accel_signal) # Use add_time_series_signal

    return collection

@pytest.fixture
def workflow_executor_with_data(signal_collection_with_data):
    """Return a WorkflowExecutor with sample signals."""
    return WorkflowExecutor(container=signal_collection_with_data)

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing imports."""
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
        "ppg_value": list(range(100))  # Changed to match PolarCSVImporter's expected column
    })
    csv_path = tmp_path / "polar_sample_01.csv"  # Changed to match PolarCSVImporter's expected pattern
    data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def polar_csv(tmp_path):
    """Create a sample Polar CSV file for testing imports."""
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
        "ppg_value": list(range(100))
    })
    csv_path = tmp_path / "polar_subject123_01.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def sample_workflow_config():
    """Return a sample workflow configuration dictionary."""
    return {
        "steps": [
            {
                "operation": "filter_lowpass",
                "input": "ppg_0",
                "output": "filtered_ppg",
                "parameters": {"cutoff": 5.0}
            },
            {
                "operation": "filter_lowpass",
                "input": "ppg_1",
                "output": "filtered_ppg_1",
                "parameters": {"cutoff": 10.0}
            }
        ]
    }

@pytest.fixture
def import_workflow_config(polar_csv):
    """Return a workflow configuration with import section."""
    return {
        "import": [
            {
                "signal_type": "ppg",
                "importer": "PolarCSVImporter",
                "source": polar_csv,
                "config": {
                    "column_mapping": {"value": "ppg_value", "timestamp": "timestamp"},
                    "filename_pattern": r"polar_(?P<subject_id>\w+)_(?P<session>\d+)\.csv",
                    "preserve_timestamp_column": True,  # Ensure timestamp stays as column
                    "origin_timezone": "UTC" # Added origin timezone for naive data
                },
                "sensor_type": "PPG",
                "sensor_model": "POLAR_H10",
                "body_position": "LEFT_WRIST",
                "base_name": "ppg_imported"
            }
        ],
        "steps": [
            {
                "operation": "filter_lowpass",
                "input": "ppg_imported_0",
                "output": "filtered_ppg",
                "parameters": {"cutoff": 5.0}
            }
        ]
    }

@pytest.fixture
def temp_output_dir():
    """Temporary directory for export outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)  # Cleanup after test

@pytest.fixture
def export_workflow_config(temp_output_dir):
    """Return a workflow configuration with export section."""
    return {
        "steps": [
            {
                "operation": "filter_lowpass",
                "input": "ppg_0",
                "output": "filtered_ppg",
                "parameters": {"cutoff": 5.0}
            }
        ],
        # Export section expects a LIST of configurations
        "export": [
            {
                "formats": ["csv"],
                "output_dir": temp_output_dir,
                # Add the required 'content' key
                "content": {
                    "time_series": ["all"], # Export all individual time series signals
                    "features": ["all"],    # Export all individual features (if any)
                    "combined_time_series": True, # Export the combined time series dataframe
                    "combined_features": True, # Export the combined feature matrix (if any)
                    "metadata": True        # Export the metadata.json file
                }
                # Removed include_combined as it's replaced by content structure
            }
        ]
    }

# ===== Tests =====

class TestWorkflowExecutor:
    """Test suite for the WorkflowExecutor class."""
    
    def test_initialization(self, empty_workflow_executor):
        """Test initializing WorkflowExecutor."""
        assert empty_workflow_executor.container is not None
        assert empty_workflow_executor.strict_validation is True
        
        # Test initializing with custom container
        collection = SignalCollection()
        executor = WorkflowExecutor(container=collection, strict_validation=False)
        assert executor.container is collection
        assert executor.strict_validation is False
    
    def test_str_to_enum(self):
        """Test converting strings to enum values using the utils function."""
        from sleep_analysis.utils import str_to_enum
        
        # Test case-insensitive matching
        assert str_to_enum("ppg", SignalType) == SignalType.PPG
        assert str_to_enum("PPG", SignalType) == SignalType.PPG
        assert str_to_enum("pPg", SignalType) == SignalType.PPG
        
        # Test for different enum classes
        assert str_to_enum("LEFT_WRIST", BodyPosition) == BodyPosition.LEFT_WRIST
        assert str_to_enum("polar_h10", SensorModel) == SensorModel.POLAR_H10
        
        # Test invalid value
        with pytest.raises(ValueError):
            str_to_enum("invalid_value", SignalType)
    
    def test_get_signals_by_input_specifier(self, workflow_executor_with_data):
        """Test retrieving signals by input specifier (base name or indexed name)."""
        # Get specific indexed signal
        signals = workflow_executor_with_data.container.get_signals_from_input_spec("ppg_0")
        assert len(signals) == 1
        assert signals[0].metadata.name == "ppg_0" # Name is set to the key by add_time_series_signal

        # Get all signals with base name
        signals = workflow_executor_with_data.container.get_signals_from_input_spec("ppg")
        assert len(signals) == 2
        # Assert that the names are now the keys assigned by the collection
        signal_names = sorted([s.metadata.name for s in signals])
        assert signal_names == ["ppg_0", "ppg_1"]

        # Test non-existent signal
        signals = workflow_executor_with_data.container.get_signals_from_input_spec("nonexistent")
        assert len(signals) == 0
    
    def test_get_signals_from_input_spec(self, workflow_executor_with_data):
        """Test retrieving signals by metadata criteria using get_signals_from_input_spec."""
        # Get signals by criteria without base name
        input_spec = {
            "criteria": {
                "body_position": "LEFT_WRIST"
            }
        }
        signals = workflow_executor_with_data.container.get_signals_from_input_spec(input_spec)
        assert len(signals) == 1
        assert signals[0].metadata.body_position == BodyPosition.LEFT_WRIST
        
        # Get signals by base name and criteria
        input_spec = {
            "base_name": "ppg",
            "criteria": {
                "body_position": "RIGHT_WRIST"
            }
        }
        signals = workflow_executor_with_data.container.get_signals_from_input_spec(input_spec)
        assert len(signals) == 1
        assert signals[0].metadata.body_position == BodyPosition.RIGHT_WRIST
        
        # Test with no matching signals
        input_spec = {
            "criteria": {
                "body_position": "HEAD"  # No signals with HEAD position
            }
        }
        signals = workflow_executor_with_data.container.get_signals_from_input_spec(input_spec)
        assert len(signals) == 0
    
    def test_execute_step_single_signal(self, workflow_executor_with_data):
        """Test executing a processing step on a single signal."""
        step = {
            "operation": "filter_lowpass",
            "input": "ppg_0",
            "output": "filtered_ppg",
            "parameters": {"cutoff": 5.0}
        }

        # Before execution
        assert "filtered_ppg" not in workflow_executor_with_data.container.time_series_signals # Check specific dict

        # Execute step
        workflow_executor_with_data.execute_step(step)

        # After execution
        assert "filtered_ppg_0" in workflow_executor_with_data.container.time_series_signals # Check specific dict for indexed key
        filtered_signal = workflow_executor_with_data.container.get_signal("filtered_ppg_0") # Use indexed key
        assert filtered_signal.metadata.operations[0].operation_name == "filter_lowpass"
        assert filtered_signal.metadata.operations[0].parameters["cutoff"] == 5.0
    
    def test_execute_step_multiple_signals(self, workflow_executor_with_data):
        """Test executing a step on multiple signals using base name."""
        step = {
            "operation": "filter_lowpass",
            "input": "ppg",  # Base name referring to ppg_0 and ppg_1
            "output": "filtered_ppg",
            "parameters": {"cutoff": 5.0}
        }
        
        # Execute step
        workflow_executor_with_data.execute_step(step)

        # Should create filtered_ppg_0 and filtered_ppg_1
        assert "filtered_ppg_0" in workflow_executor_with_data.container.time_series_signals # Check specific dict
        assert "filtered_ppg_1" in workflow_executor_with_data.container.time_series_signals # Check specific dict

    def test_execute_step_criteria_filter(self, workflow_executor_with_data):
        """Test executing a step using metadata criteria filter."""
        step = {
            "operation": "filter_lowpass",
            "input": {
                "criteria": {
                    "body_position": "LEFT_WRIST"
                }
            },
            "output": "filtered_left_wrist",
            "parameters": {"cutoff": 5.0}
        }
        
        # Execute step
        workflow_executor_with_data.execute_step(step)

        # Should create filtered_left_wrist_0 (indexed from ppg_0)
        assert "filtered_left_wrist_0" in workflow_executor_with_data.container.time_series_signals # Check specific dict for indexed key
        filtered_signal = workflow_executor_with_data.container.get_signal("filtered_left_wrist_0") # Use indexed key
        assert filtered_signal.metadata.operations[0].operation_name == "filter_lowpass"
    def test_execute_step_in_place(self, workflow_executor_with_data):
        """Test executing a step with in-place operation."""
        # Get reference to original signal
        original_signal = workflow_executor_with_data.container.get_signal("ppg_0")
        original_data = original_signal.get_data().copy()
        
        step = {
            "operation": "filter_lowpass",
            "input": "ppg_0",
            "inplace": True,
            "parameters": {"cutoff": 5.0}
        }
        
        # Execute step
        workflow_executor_with_data.execute_step(step)
        
        # Signal should be modified in-place
        modified_signal = workflow_executor_with_data.container.get_signal("ppg_0")
        assert len(modified_signal.metadata.operations) > 0
        assert modified_signal.metadata.operations[-1].operation_name == "filter_lowpass"
        # Data should be different after filtering
        modified_data = modified_signal.get_data()
        assert not pd.DataFrame.equals(modified_data, original_data)
    
    def test_execute_step_list_inputs(self, workflow_executor_with_data):
        """Test executing a step with list of inputs and outputs."""
        step = {
            "operation": "filter_lowpass",
            "input": ["ppg_0", "ppg_1"],
            "output": ["filtered_a", "filtered_b"],
            "parameters": {"cutoff": 5.0}
        }
        
        # Execute step
        workflow_executor_with_data.execute_step(step)

        # Should create filtered_a and filtered_b
        assert "filtered_a" in workflow_executor_with_data.container.time_series_signals # Check specific dict
        assert "filtered_b" in workflow_executor_with_data.container.time_series_signals # Check specific dict

        # Test invalid list inputs (different lengths)
        invalid_step = {
            "operation": "filter_lowpass",
            "input": ["ppg_0", "ppg_1"],
            "output": ["filtered_single"],  # Only one output for two inputs
            "parameters": {"cutoff": 5.0}
        }
        
        with pytest.raises(ValueError):
            workflow_executor_with_data.execute_step(invalid_step)
    
    def test_execute_workflow(self, workflow_executor_with_data, sample_workflow_config):
        """Test executing a complete workflow configuration."""
        # Before execution
        assert "filtered_ppg" not in workflow_executor_with_data.container.time_series_signals # Check specific dict
        assert "filtered_ppg_1" not in workflow_executor_with_data.container.time_series_signals # Check specific dict

        # Execute workflow
        workflow_executor_with_data.execute_workflow(sample_workflow_config)

        # After execution
        assert "filtered_ppg_0" in workflow_executor_with_data.container.time_series_signals # Check specific dict for indexed key
        assert "filtered_ppg_1_1" in workflow_executor_with_data.container.time_series_signals # Check specific dict for indexed key (output=filtered_ppg_1, input=ppg_1)

        filtered_signal = workflow_executor_with_data.container.get_signal("filtered_ppg_0") # Use indexed key
        assert filtered_signal.metadata.operations[0].operation_name == "filter_lowpass"
        assert filtered_signal.metadata.operations[0].parameters["cutoff"] == 5.0

        filtered_signal_1 = workflow_executor_with_data.container.get_signal("filtered_ppg_1_1") # Use indexed key
        assert filtered_signal_1.metadata.operations[0].parameters["cutoff"] == 10.0
    
    def test_import_section(self, empty_workflow_executor, import_workflow_config, polar_csv):
        """Test workflow with import section."""
        # Before execution
        assert "ppg_imported_0" not in empty_workflow_executor.container.time_series_signals # Check specific dict

        # Execute workflow
        empty_workflow_executor.execute_workflow(import_workflow_config)

        # After execution
        assert "ppg_imported_0" in empty_workflow_executor.container.time_series_signals # Check specific dict
        assert "filtered_ppg_0" in empty_workflow_executor.container.time_series_signals # Check specific dict for indexed key

        # Check that metadata was set correctly from import spec
        imported_signal = empty_workflow_executor.container.get_signal("ppg_imported_0")
        assert imported_signal.metadata.sensor_type == SensorType.PPG
        assert imported_signal.metadata.sensor_model == SensorModel.POLAR_H10
        assert imported_signal.metadata.body_position == BodyPosition.LEFT_WRIST
        
        # Check that extracted metadata from filename exists
        assert "subject_id" in imported_signal.metadata.sensor_info
        assert imported_signal.metadata.sensor_info["subject_id"] == "subject123"
        
        # Enhanced column standardization checks
        data = imported_signal.get_data()
        assert "value" in data.columns
        assert isinstance(data.index, pd.DatetimeIndex) or "timestamp" in data.columns
        if "timestamp" in data.columns:
            assert pd.api.types.is_datetime64_any_dtype(data["timestamp"])
    
    def test_export_section(self, workflow_executor_with_data, export_workflow_config, temp_output_dir):
        """Test workflow with export section."""
        # Mock the filter_lowpass operation to return proper DataFrame data
        def mock_filter_lowpass(data_list, parameters):
            """Mock implementation of lowpass filtering."""
            import pandas as pd
            
            # Ensure we have data we can work with
            if not data_list or not isinstance(data_list[0], pd.DataFrame):
                # Create a minimal default DataFrame for testing
                result = pd.DataFrame({
                    'value': [1, 2, 3, 4, 5]
                }, index=pd.date_range("2023-01-01", periods=5, freq="1s"))
                return result
                
            # If we have a DataFrame, apply the filtering
            data = data_list[0]
            result = data.copy()
            if 'value' in result.columns:
                result['value'] = result['value'] * 0.9  # Simulate filtering effect
            return result
        
        # Register the mock operation
        from sleep_analysis.signals import PPGSignal
        original_op = PPGSignal.registry.get("filter_lowpass", None)
        PPGSignal.registry["filter_lowpass"] = (mock_filter_lowpass, None)
    
        try:
            # --- Manually create and set a mock combined dataframe ---
            # This simulates the state after a combine operation would have run
            ppg0_data = workflow_executor_with_data.container.get_signal("ppg_0").get_data()
            # Simulate the output of the mock filter_lowpass
            filtered_data = mock_filter_lowpass([ppg0_data], {})
            
            # Combine the relevant dataframes (adjust columns as needed for your combine logic)
            # For this test, let's assume a simple combination
            mock_combined_df = pd.concat(
                [ppg0_data.add_prefix("ppg_0_"), filtered_data.add_prefix("filtered_ppg_")],
                axis=1
            )
            # Ensure it has the correct index type
            mock_combined_df.index = pd.to_datetime(mock_combined_df.index)

            # Set the mock combined dataframe on the collection
            workflow_executor_with_data.container._aligned_dataframe = mock_combined_df
            # Optionally set params if needed by export logic (not currently used by CSV export)
            # workflow_executor_with_data.container._aligned_dataframe_params = {"method_used": "mock"}
            # --- End setting mock combined dataframe ---

            # Execute workflow with our mock filter AND the pre-set combined data
            workflow_executor_with_data.execute_workflow(export_workflow_config)

            # Check export directory was created with expected files
            assert os.path.isdir(os.path.join(temp_output_dir, "signals"))
            assert os.path.exists(os.path.join(temp_output_dir, "signals", "ppg_0.csv")) # Original input signal
            assert os.path.exists(os.path.join(temp_output_dir, "signals", "filtered_ppg_0.csv")) # Check for indexed output key
            assert os.path.exists(os.path.join(temp_output_dir, "metadata.json"))
            assert os.path.exists(os.path.join(temp_output_dir, "combined.csv")) # Combined TS export
        finally:
            # Restore original operation or clean up
            if original_op:
                PPGSignal.registry["filter_lowpass"] = original_op
            else:
                del PPGSignal.registry["filter_lowpass"]
    
    def test_collection_get_signals_advanced(self, workflow_executor_with_data):
        """Test advanced signal retrieval methods in SignalCollection."""
        # Test with a list of input specifiers
        input_spec = ["ppg_0", "accelerometer_0"]
        signals = workflow_executor_with_data.container.get_signals_from_input_spec(input_spec)
        assert len(signals) == 2
        # Check that we got the correct signals, regardless of their exact name
        assert signals[0].metadata.signal_id == "ppg_test_id"
        assert signals[1].metadata.signal_id == "accel_test_id"
        
        # Test with mixed string and dict specifiers
        mixed_spec = [
            "ppg_0",
            {"criteria": {"body_position": "RIGHT_WRIST"}}
        ]
        signals = workflow_executor_with_data.container.get_signals_from_input_spec(mixed_spec)
        assert len(signals) == 2
        
        # Test with string enum values
        string_enum_spec = {
            "criteria": {
                "signal_type": "PPG"
            }
        }
        signals = workflow_executor_with_data.container.get_signals_from_input_spec(string_enum_spec)
        assert len(signals) == 2
        for signal in signals:
            assert signal.metadata.signal_type == SignalType.PPG
    
    def test_multi_file_import_workflow(self, empty_workflow_executor, tmp_path):
        """Test importing and merging multiple files in a workflow."""
        # Create sample CSV files for testing
        data1 = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=50, freq="1s"),
            "ppg_value": list(range(50))
        })
        data2 = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01 00:00:50", periods=50, freq="1s"),
            "ppg_value": list(range(50, 100))
        })
        
        # Write files to temporary directory
        csv1 = tmp_path / "polar_multi_01.csv"
        csv2 = tmp_path / "polar_multi_02.csv"
        data1.to_csv(csv1, index=False)
        data2.to_csv(csv2, index=False)
        
        # Create workflow configuration for multi-file import
        multi_file_config = {
            "import": [
                {
                    "signal_type": "ppg",
                    "importer": "MergingImporter",
                    "source": str(tmp_path),
                    "config": {
                        "file_pattern": "polar_multi_*.csv",
                        "time_column": "timestamp",
                        "column_mapping": {"value": "ppg_value", "timestamp": "timestamp"},
                        "merge": True,
                        "origin_timezone": "UTC" # Added origin timezone for naive data
                    },
                    "base_name": "ppg_merged"
                }
            ]
        }
        
        # Execute the workflow
        empty_workflow_executor.execute_workflow(multi_file_config)

        # Verify merged signal
        assert "ppg_merged_0" in empty_workflow_executor.container.time_series_signals # Check specific dict
        merged_signal = empty_workflow_executor.container.get_signal("ppg_merged_0")

        # Check merged data properties
        merged_data = merged_signal.get_data()
        assert len(merged_data) == 100  # Total rows from both files
        assert merged_signal.metadata.merged is True
        assert "source_files" in merged_signal.metadata.__dict__
        
    def test_feature_extraction_workflow(self, workflow_executor_with_data):
        """Test executing a workflow with feature extraction."""
        feature_workflow_config = {
            "steps": [
                {
                    "operation": "feature_mean",
                    "input": "ppg_0",
                    "output": "ppg_features",
                    "parameters": {"window_length": 2, "step_size": 1}
                }
            ]
        }
        
        # Mock the feature_mean operation on PPGSignal
        def mock_feature_mean(data_list, parameters):
            """Mock implementation of feature extraction."""
            import pandas as pd
            data = data_list[0]
            window = parameters.get("window_length", 2)
            step = parameters.get("step_size", 1)
            
            result = pd.DataFrame(index=data.index[::step][:-(window-1)])
            # Add both columns - "mean" for the test assertion and "value" for PPGSignal requirements
            result["mean"] = [data["value"][i:i+window].mean() for i in range(0, len(data)-window+1, step)]
            result["value"] = result["mean"]  # Include required column for PPGSignal
            return result
            
        # Register the mock operation
        from sleep_analysis.signals import PPGSignal
        from sleep_analysis.signal_types import SignalType
        PPGSignal.registry["feature_mean"] = (mock_feature_mean, None)
        
        # Execute the workflow
        workflow_executor_with_data.execute_workflow(feature_workflow_config)

        # Verify feature extraction results
        # Note: Feature extraction results might be stored in container.features
        # However, the mock operation currently returns a PPGSignal (due to registry setup)
        # If the operation were correctly registered to return a Feature, we'd check container.features
        assert "ppg_features_0" in workflow_executor_with_data.container.time_series_signals # Check time_series_signals for indexed key
        feature_signal = workflow_executor_with_data.container.get_signal("ppg_features_0") # Use indexed key
        assert "mean" in feature_signal.get_data().columns
        assert len(feature_signal.get_data()) == 4  # For 5 data points with window=2, step=1
        
        # Clean up the mock
        del PPGSignal.registry["feature_mean"]
