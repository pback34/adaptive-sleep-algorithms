"""Tests for the signal class hierarchy and registry."""

import pytest
import pandas as pd
from sleep_analysis.signals import TimeSeriesSignal, PPGSignal, AccelerometerSignal
from sleep_analysis.core.metadata import SignalMetadata
from sleep_analysis.signal_types import SignalType

def test_time_series_signal_abstract():
    """Test that TimeSeriesSignal cannot be instantiated directly."""
    with pytest.raises(TypeError):
        signal = TimeSeriesSignal(data=[1, 2, 3], metadata={})

def test_ppg_signal_instantiation(sample_metadata, sample_dataframe):
    """Test instantiation of PPGSignal."""
    metadata = sample_metadata.copy()
    metadata["signal_type"] = SignalType.PPG
    signal = PPGSignal(data=sample_dataframe, metadata=metadata)
    assert signal.signal_type == SignalType.PPG
    assert signal.get_data().equals(sample_dataframe)

def test_accelerometer_signal_instantiation(sample_metadata):
    """Test instantiation of AccelerometerSignal."""
    # Create appropriate dataframe for accelerometer (needs x, y, z columns)
    import pandas as pd
    import numpy as np
    
    # Create sample data with required columns for accelerometer
    dates = pd.date_range('2023-01-01', periods=100, freq='s')
    data = pd.DataFrame({
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'z': np.random.rand(100)
    }, index=dates)
    data.index.name = 'timestamp'
    
    metadata = sample_metadata.copy()
    metadata["signal_type"] = SignalType.ACCELEROMETER
    signal = AccelerometerSignal(data=data, metadata=metadata)
    assert signal.signal_type == SignalType.ACCELEROMETER

def test_registry_inheritance():
    """Test that operations are inherited correctly in the registry."""
    # Check that PPGSignal has its specific operations in registry
    ppg_registry = PPGSignal.get_registry()
    assert "normalize" in ppg_registry
    assert ppg_registry["normalize"][1] == PPGSignal
    
    # Check that filter_lowpass is an instance method, not in registry
    # Create DataFrame with required columns for PPG signal
    dates = pd.date_range('2023-01-01', periods=10, freq='s')
    data = pd.DataFrame({'value': range(10)}, index=dates)
    data.index.name = 'timestamp'
    ppg_signal = PPGSignal(data=data, metadata={"signal_id": "test"})
    assert hasattr(ppg_signal, "filter_lowpass")
    assert callable(getattr(ppg_signal, "filter_lowpass"))
    
    # Check that AccelerometerSignal inherits from TimeSeriesSignal but not from PPGSignal
    acc_registry = AccelerometerSignal.get_registry()
    assert "normalize" not in acc_registry  # PPGSignal-specific operation not inherited
    
    # Check that AccelerometerSignal has filter_lowpass as an instance method
    # Create DataFrame with required columns for AccelerometerSignal
    acc_dates = pd.date_range('2023-01-01', periods=10, freq='s')
    acc_data = pd.DataFrame({
        'x': [0] * 10,
        'y': [0] * 10,
        'z': [0] * 10
    }, index=acc_dates)
    acc_data.index.name = 'timestamp'
    acc_signal = AccelerometerSignal(data=acc_data, metadata={"signal_id": "test"})
    assert hasattr(acc_signal, "filter_lowpass")
    assert callable(getattr(acc_signal, "filter_lowpass"))

def test_apply_operation_inplace(sample_metadata, sample_dataframe):
    """Test applying an operation in-place."""
    metadata = sample_metadata.copy()
    metadata["signal_type"] = SignalType.PPG
    signal = PPGSignal(data=sample_dataframe, metadata=metadata)
    original_data = signal.get_data().copy()
    result = signal.apply_operation("filter_lowpass", inplace=True)
    assert result is signal
    # Data should be modified by the filter operation
    import pandas as pd
    assert not pd.DataFrame.equals(signal.get_data(), original_data)
    # Check that operation was recorded in metadata
    assert len(signal.metadata.operations) == 1
    assert signal.metadata.operations[0].operation_name == "filter_lowpass"

def test_apply_operation_non_inplace(sample_metadata, sample_dataframe):
    """Test applying an operation not in-place."""
    metadata = sample_metadata.copy()
    metadata["signal_type"] = SignalType.PPG
    signal = PPGSignal(data=sample_dataframe, metadata=metadata)
    original_data = signal.get_data().copy()
    new_signal = signal.apply_operation("filter_lowpass")
    assert isinstance(new_signal, PPGSignal)
    # The original signal's data should be unchanged
    import pandas as pd
    assert pd.DataFrame.equals(signal.get_data(), original_data)
    # The new signal's data should be different (filtered)
    assert not pd.DataFrame.equals(new_signal.get_data(), original_data)
    # Check metadata was properly updated
    assert len(new_signal.metadata.operations) == 1
    assert new_signal.metadata.operations[0].operation_name == "filter_lowpass"
    assert new_signal.metadata.derived_from == [(signal.metadata.signal_id, -1)]

def test_apply_operation_type_safety(sample_metadata):
    """Test that operations are type-safe."""
    # Create appropriate dataframe for accelerometer (needs x, y, z columns)
    import pandas as pd
    import numpy as np
    
    # Create sample data with required columns for accelerometer
    dates = pd.date_range('2023-01-01', periods=100, freq='s')
    data = pd.DataFrame({
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'z': np.random.rand(100)
    }, index=dates)
    data.index.name = 'timestamp'
    
    metadata = sample_metadata.copy()
    metadata["signal_type"] = SignalType.ACCELEROMETER
    signal = AccelerometerSignal(data=data, metadata=metadata)
    with pytest.raises(ValueError, match="Operation 'normalize' not found for AccelerometerSignal"):
        signal.apply_operation("normalize")
        
def test_apply_operation_traceability(sample_metadata, sample_dataframe):
    """Test that operation history and derived_from are properly maintained."""
    metadata = sample_metadata.copy()
    metadata["signal_type"] = SignalType.PPG
    signal = PPGSignal(data=sample_dataframe, metadata=metadata)
    
    # Apply first operation
    new_signal = signal.apply_operation("filter_lowpass", cutoff=5.0)
    
    # Check derived_from attribute
    assert new_signal.metadata.derived_from[0][0] == signal.metadata.signal_id
    assert new_signal.metadata.derived_from[0][1] == -1  # No prior operations
    
    # Apply another operation to the new signal
    newer_signal = new_signal.apply_operation("normalize")
    assert newer_signal.metadata.derived_from[0][0] == new_signal.metadata.signal_id
    assert newer_signal.metadata.derived_from[0][1] == 0  # Points to first operation
