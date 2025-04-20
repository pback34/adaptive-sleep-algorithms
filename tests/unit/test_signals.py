"""Tests for the signal class hierarchy and registry."""

import pytest
import pandas as pd
from sleep_analysis.signals import TimeSeriesSignal, PPGSignal, AccelerometerSignal
from sleep_analysis.core.metadata import TimeSeriesMetadata # Changed SignalMetadata to TimeSeriesMetadata
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

    # Check that 'filter_lowpass' is registered for PPGSignal (overriding TimeSeriesSignal)
    # Check that 'filter_lowpass' is NOT registered for PPGSignal (it's handled as a method)
    assert "filter_lowpass" not in ppg_registry

    # Check that AccelerometerSignal inherits from TimeSeriesSignal but not from PPGSignal
    acc_registry = AccelerometerSignal.get_registry()
    assert "normalize" not in acc_registry  # PPGSignal-specific operation not inherited

    # Check that AccelerometerSignal does NOT have 'filter_lowpass' in its registry (it's handled as a method)
    assert "filter_lowpass" not in acc_registry

    # --- Test that apply_operation still works for methods ---
    # Create instances to test apply_operation
    ppg_signal = create_test_signal(freq_hz=100) # Helper creates PPGSignal
    acc_signal = AccelerometerSignal(data=pd.DataFrame({'x':[1,2],'y':[1,2],'z':[1,2]}, index=pd.date_range('2023-01-01', periods=2, freq='s')), metadata={"signal_id": "acc_test"})

    # Check that apply_operation finds the method on PPGSignal
    try:
        ppg_signal.apply_operation("filter_lowpass", cutoff=5)
    except ValueError as e:
        pytest.fail(f"PPGSignal.apply_operation('filter_lowpass') failed: {e}")

    # Check that apply_operation finds the method on AccelerometerSignal (inherited from TimeSeriesSignal)
    try:
        acc_signal.apply_operation("filter_lowpass", cutoff=5)
    except ValueError as e:
        pytest.fail(f"AccelerometerSignal.apply_operation('filter_lowpass') failed: {e}")

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


# ===== Sample Rate Tests =====

def create_test_signal(freq_hz=None, num_points=100, irregular=False, constant_time=False, start_time="2023-01-01"):
    """Helper to create a TimeSeriesSignal for testing sample rate."""
    import numpy as np
    if constant_time:
        index = pd.DatetimeIndex([pd.Timestamp(start_time)] * num_points)
    elif irregular:
        # Create irregular timestamps
        base_index = pd.date_range(start=start_time, periods=num_points, freq="1s")
        noise = np.random.uniform(-0.4, 0.4, num_points) # Add noise up to +/- 400ms
        irregular_seconds = base_index.astype(np.int64) / 1e9 + noise
        index = pd.to_datetime(irregular_seconds, unit='s')
    elif freq_hz is not None and freq_hz > 0:
        freq_str = f"{1000/freq_hz:.6f}ms"
        index = pd.date_range(start=start_time, periods=num_points, freq=freq_str)
    else: # Default to 1Hz if no freq specified
        index = pd.date_range(start=start_time, periods=num_points, freq="1s")

    data = pd.DataFrame({"value": range(num_points)}, index=index)
    data.index.name = 'timestamp'
    # Use PPGSignal as a concrete TimeSeriesSignal subclass
    return PPGSignal(data=data, metadata={"signal_id": f"test_signal_{freq_hz or 'irr'}"})

def test_sample_rate_metadata_on_init_regular():
    """Test sample_rate metadata is set correctly on init for regular data."""
    signal_100hz = create_test_signal(freq_hz=100)
    assert signal_100hz.metadata.sample_rate == "100.0000Hz"

    signal_50hz = create_test_signal(freq_hz=50)
    assert signal_50hz.metadata.sample_rate == "50.0000Hz"

    signal_1hz = create_test_signal(freq_hz=1)
    assert signal_1hz.metadata.sample_rate == "1.0000Hz"

def test_sample_rate_metadata_on_init_irregular():
    """Test sample_rate metadata is 'Variable' on init for irregular data."""
    signal_irregular = create_test_signal(irregular=True)
    # get_sampling_rate should return None for irregular data
    assert signal_irregular.get_sampling_rate() is None
    # _update_sample_rate_metadata should set metadata to "Variable"
    assert signal_irregular.metadata.sample_rate == "Variable"

def test_sample_rate_metadata_on_init_insufficient_data():
    """Test sample_rate metadata is 'Unknown' for insufficient data."""
    signal_1pt = create_test_signal(num_points=1)
    # get_sampling_rate should return None for insufficient data
    assert signal_1pt.get_sampling_rate() is None
    # _update_sample_rate_metadata should set metadata to "Unknown"
    assert signal_1pt.metadata.sample_rate == "Unknown"

    signal_0pt = create_test_signal(num_points=0)
    # get_sampling_rate should return None for insufficient data
    assert signal_0pt.get_sampling_rate() is None
    # _update_sample_rate_metadata should set metadata to "Unknown"
    assert signal_0pt.metadata.sample_rate == "Unknown"

def test_sample_rate_metadata_on_init_constant_time():
    """Test sample_rate metadata is 'Unknown' for constant timestamps."""
    signal_const = create_test_signal(constant_time=True)
    # get_sampling_rate should return None for constant time (zero median diff)
    assert signal_const.get_sampling_rate() is None
    # _update_sample_rate_metadata should set metadata to "Unknown"
    assert signal_const.metadata.sample_rate == "Unknown"

def test_sample_rate_metadata_update_after_operation_inplace():
    """Test sample_rate metadata update after inplace operation."""
    signal = create_test_signal(freq_hz=100)
    assert signal.metadata.sample_rate == "100.0000Hz"

    # Apply filter_lowpass inplace (doesn't change rate, but triggers update)
    signal.apply_operation("filter_lowpass", inplace=True, cutoff=5)
    # Rate should remain the same
    assert signal.metadata.sample_rate == "100.0000Hz"

def test_sample_rate_metadata_update_after_operation_new_signal():
    """Test sample_rate metadata update after non-inplace operation."""
    signal = create_test_signal(freq_hz=100)
    assert signal.metadata.sample_rate == "100.0000Hz"

    # Apply filter_lowpass (doesn't change rate, but creates new signal)
    new_signal = signal.apply_operation("filter_lowpass", inplace=False, cutoff=5)
    # New signal's metadata should have the correct rate calculated by its __init__
    assert new_signal.metadata.sample_rate == "100.0000Hz"
    # Original signal's metadata should be unchanged
    assert signal.metadata.sample_rate == "100.0000Hz"

# Mock resampling operation for testing rate changes
@PPGSignal.register("mock_resample", output_class=PPGSignal)
def mock_resample_op(data_list, parameters):
    """Mock resampling operation that changes the frequency."""
    import numpy as np
    data = data_list[0]
    new_freq_hz = parameters.get("new_freq", 50)
    num_points = int(len(data) * new_freq_hz / 100) # Assume original 100Hz for simplicity
    new_index = pd.date_range(start=data.index[0], periods=num_points, freq=f"{1000/new_freq_hz}ms")
    # Just create dummy data with the new index
    new_data = pd.DataFrame({"value": np.linspace(0, 1, num_points)}, index=new_index)
    return new_data

def test_sample_rate_metadata_update_after_resample_inplace():
    """Test sample_rate update after inplace operation that changes rate."""
    signal = create_test_signal(freq_hz=100)
    assert signal.metadata.sample_rate == "100.0000Hz"

    # Apply mock resampling inplace
    signal.apply_operation("mock_resample", inplace=True, new_freq=50)
    # Metadata should be updated to the new rate
    assert signal.metadata.sample_rate == "50.0000Hz"

def test_sample_rate_metadata_update_after_resample_new_signal():
    """Test sample_rate update after non-inplace operation that changes rate."""
    signal = create_test_signal(freq_hz=100)
    assert signal.metadata.sample_rate == "100.0000Hz"

    # Apply mock resampling to create a new signal
    new_signal = signal.apply_operation("mock_resample", inplace=False, new_freq=50)
    # New signal's metadata should reflect the new rate
    assert new_signal.metadata.sample_rate == "50.0000Hz"
    # Original signal's metadata remains unchanged
    assert signal.metadata.sample_rate == "100.0000Hz"
