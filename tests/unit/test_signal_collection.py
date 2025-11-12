"""Tests for the SignalCollection class."""

import pytest
import uuid
import pandas as pd
import logging # Added for caplog tests
from dataclasses import replace, asdict # Import replace and asdict for copying dataclasses
from sleep_analysis.core.signal_collection import SignalCollection, STANDARD_RATES
from sleep_analysis.signals.ppg_signal import PPGSignal
from sleep_analysis.signals.accelerometer_signal import AccelerometerSignal
from sleep_analysis.signals.heart_rate_signal import HeartRateSignal
# Import Feature and related types
from sleep_analysis.features.feature import Feature
from sleep_analysis.core.metadata import FeatureType # Assuming FeatureType is here
# Import base TimeSeriesSignal class
from sleep_analysis.signals.time_series_signal import TimeSeriesSignal
from sleep_analysis.signal_types import SignalType, SensorModel, BodyPosition

@pytest.fixture
def signal_collection():
    """Return a new empty SignalCollection."""
    return SignalCollection()

@pytest.fixture
def ppg_signal():
    """Return a sample PPG signal."""
    # Create datetime index instead of numeric index
    index = pd.date_range(start="2023-01-01", periods=3, freq="1s")
    data = pd.DataFrame({"value": [1, 2, 3]}, index=index)
    data.index.name = 'timestamp'  # Name the index 'timestamp'
    return PPGSignal(data=data, metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG})

@pytest.fixture
def sample_feature(ppg_signal): # Depends on ppg_signal for source info
    """Return a sample Feature object."""
    # Create epoch index aligned with ppg_signal for simplicity in later tests
    epoch_index = pd.date_range(start="2023-01-01", periods=2, freq="2s", name='epoch_start_time')
    feature_data = pd.DataFrame({
        ('ppg_0', 'mean'): [1.5, 2.5],
        ('ppg_0', 'std'): [0.5, 0.5]
    }, index=epoch_index)
    # Set column multi-index names
    feature_data.columns = pd.MultiIndex.from_tuples(feature_data.columns, names=['signal_key', 'feature'])
    # Extract feature names from the MultiIndex
    feature_names = feature_data.columns.get_level_values('feature').unique().tolist()

    # Basic metadata
    metadata = {
        "feature_id": str(uuid.uuid4()),
        "feature_type": FeatureType.STATISTICAL, # Example type
        "feature_names": feature_names, # Add the required feature names
        "source_signal_ids": [ppg_signal.metadata.signal_id],
        "source_signal_keys": ["ppg_0"], # Assume ppg_signal was added with this key
        "epoch_window_length": pd.Timedelta("2s"),
        "epoch_step_size": pd.Timedelta("2s"),
        "operations": [ # Example operation record
            {
                "operation_name": "feature_statistics",
                "parameters": {"stats": ["mean", "std"]},
                "timestamp": pd.Timestamp.now(tz='UTC')
            }
        ]
    }
    return Feature(data=feature_data, metadata=metadata)


def test_add_signal_with_base_name(signal_collection, ppg_signal):
    """Test adding signals with a base name and automatic indexing."""
    key1 = signal_collection.add_signal_with_base_name("ppg", ppg_signal)
    
    # Create a second signal with different data
    index = pd.date_range(start="2023-01-01", periods=3, freq="1s")
    data2 = pd.DataFrame({"value": [4, 5, 6]}, index=index)
    data2.index.name = 'timestamp'
    ppg_signal2 = PPGSignal(data=data2, 
                          metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG})
    key2 = signal_collection.add_signal_with_base_name("ppg", ppg_signal2)
    
    assert key1 == "ppg_0"
    assert key2 == "ppg_1"
    # Check the specific dictionary for time series signals
    assert len(signal_collection.time_series_signals) == 2
    
    # Test error handling
    with pytest.raises(ValueError):
        signal_collection.add_signal_with_base_name("", ppg_signal)

def test_get_signals_by_base_name(signal_collection, ppg_signal):
    """Test retrieving signals by base name."""
    signal_collection.add_signal_with_base_name("ppg", ppg_signal)
    
    # Add another PPG signal with same base name
    index = pd.date_range(start="2023-01-01", periods=3, freq="1s")
    data2 = pd.DataFrame({"value": [4, 5, 6]}, index=index)
    data2.index.name = 'timestamp'
    ppg_signal2 = PPGSignal(data=data2, 
                          metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG})
    signal_collection.add_signal_with_base_name("ppg", ppg_signal2)
    
    # Add a signal with different base name
    index = pd.date_range(start="2023-01-01", periods=3, freq="1s")
    data3 = pd.DataFrame({"value": [7, 8, 9]}, index=index)
    data3.index.name = 'timestamp'
    ppg_signal3 = PPGSignal(data=data3, 
                          metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG})
    signal_collection.add_signal_with_base_name("other", ppg_signal3)
    
    # Test retrieval by base name
    ppg_signals = signal_collection.get_signals(base_name="ppg")
    assert len(ppg_signals) == 2
    assert all(s.metadata.signal_type == SignalType.PPG for s in ppg_signals)
    
    # Test retrieval of other base name
    other_signals = signal_collection.get_signals(base_name="other")
    assert len(other_signals) == 1
    
    # Test non-existent base name
    nonexistent = signal_collection.get_signals(base_name="nonexistent")
    assert len(nonexistent) == 0

def test_get_signals_by_criteria(signal_collection, ppg_signal):
    """Test retrieving signals by metadata criteria."""
    signal_collection.add_time_series_signal("ppg_0", ppg_signal)
    
    # Get by signal type
    signals = signal_collection.get_signals(criteria={"signal_type": SignalType.PPG})
    assert len(signals) == 1
    assert signals[0].metadata.signal_type == SignalType.PPG
    
    # Get by non-matching criteria
    signals = signal_collection.get_signals(criteria={"signal_type": SignalType.ACCELEROMETER})
    assert len(signals) == 0
    
    # Get by multiple criteria
    index = pd.date_range(start="2023-01-01", periods=3, freq="1s")
    custom_data = pd.DataFrame({"value": [1, 2, 3]}, index=index)
    custom_data.index.name = 'timestamp'
    custom_signal = PPGSignal(data=custom_data, 
                           metadata={"signal_id": str(uuid.uuid4()), 
                                    "signal_type": SignalType.PPG,
                                    "sensor_info": {"custom_field": "test_value"}})
    signal_collection.add_time_series_signal("custom", custom_signal)
    
    signals = signal_collection.get_signals(criteria={
        "signal_type": SignalType.PPG,
        "sensor_info.custom_field": "test_value"
    })
    assert len(signals) == 1
    assert signals[0].metadata.signal_type == SignalType.PPG
    assert signals[0].metadata.sensor_info["custom_field"] == "test_value"

def test_add_signal_unique_signal_id(signal_collection, ppg_signal):
    """Test that add_time_series_signal ensures unique signal_id values."""
    signal_collection.add_time_series_signal("ppg_0", ppg_signal)
    
    # Create a second signal with the same signal_id
    index = pd.date_range(start="2023-01-01", periods=3, freq="1s")
    data2 = pd.DataFrame({"value": [4, 5, 6]}, index=index)
    data2.index.name = 'timestamp'
    ppg_signal2 = PPGSignal(data=data2, 
                          metadata={"signal_id": ppg_signal.metadata.signal_id})
    
    # Add to collection - should get a new signal_id
    signal_collection.add_time_series_signal("ppg_1", ppg_signal2)
    
    # Verify the signal_ids are different
    assert ppg_signal.metadata.signal_id != ppg_signal2.metadata.signal_id
    
    # Verify both signals are in the collection (check the correct dict)
    assert "ppg_0" in signal_collection.time_series_signals
    assert "ppg_1" in signal_collection.time_series_signals

def test_combined_functionality(signal_collection, ppg_signal):
    """Test the combination of base name indexing and criteria filtering."""
    # Add signals with base names
    signal_collection.add_signal_with_base_name("ppg", ppg_signal)
    
    index = pd.date_range(start="2023-01-01", periods=3, freq="1s")
    data2 = pd.DataFrame({"value": [4, 5, 6]}, index=index)
    data2.index.name = 'timestamp'
    ppg_signal2 = PPGSignal(data=data2, 
                          metadata={"signal_id": str(uuid.uuid4()), 
                                   "signal_type": SignalType.PPG,
                                   "sensor_info": {"custom_field": "special"}})
    signal_collection.add_signal_with_base_name("ppg", ppg_signal2)
    
    # Filter by both base name and criteria
    signals = signal_collection.get_signals(
        criteria={"sensor_info.custom_field": "special"},
        base_name="ppg"
    )
    
    assert len(signals) == 1
    assert signals[0].metadata.sensor_info["custom_field"] == "special"

# --- Tests for Feature Handling ---

def test_add_feature_success(signal_collection, sample_feature):
    """Test adding a valid Feature object."""
    key = "stats_0"
    signal_collection.add_feature(key, sample_feature)
    assert key in signal_collection.features
    assert signal_collection.features[key] is sample_feature
    assert signal_collection.features[key].metadata.name == key # Check name is set
    assert not signal_collection.time_series_signals # Ensure it didn't go to the wrong dict

def test_add_feature_type_error(signal_collection, ppg_signal):
    """Test that adding a non-Feature object raises TypeError."""
    with pytest.raises(TypeError, match="Object provided for key 'ts_signal' is not a Feature"):
        signal_collection.add_feature("ts_signal", ppg_signal) # Try adding a TimeSeriesSignal

def test_add_feature_duplicate_key(signal_collection, sample_feature):
    """Test that adding a feature with an existing key raises ValueError."""
    key = "stats_0"
    signal_collection.add_feature(key, sample_feature)
    # Create another feature instance using replace and asdict for metadata copy
    feature2_metadata_obj = replace(sample_feature.metadata)
    feature2_metadata_obj.feature_id = str(uuid.uuid4()) # Ensure different ID
    feature2 = Feature(data=sample_feature.get_data().copy(), metadata=asdict(feature2_metadata_obj))

    with pytest.raises(ValueError, match=f"Feature with key '{key}' already exists"):
        signal_collection.add_feature(key, feature2)

def test_add_feature_unique_id(signal_collection, sample_feature, caplog):
    """Test adding a feature with a conflicting ID assigns a new one."""
    caplog.set_level(logging.WARNING)
    key1 = "stats_0"
    signal_collection.add_feature(key1, sample_feature)
    original_id = sample_feature.metadata.feature_id

    # Create another feature instance with the *same* ID initially
    feature2_data = sample_feature.get_data().copy() + 1 # Different data
    # Use replace and asdict to create a new metadata dict with the same initial values (including ID)
    feature2_meta_obj = replace(sample_feature.metadata)
    feature2 = Feature(data=feature2_data, metadata=asdict(feature2_meta_obj))

    key2 = "stats_1"
    signal_collection.add_feature(key2, feature2)

    assert key2 in signal_collection.features
    new_id = signal_collection.features[key2].metadata.feature_id
    assert new_id != original_id
    assert f"Feature ID '{original_id}' conflicts" in caplog.text
    assert f"Assigning new ID: {new_id}" in caplog.text

def test_get_feature_success(signal_collection, sample_feature):
    """Test retrieving a feature successfully."""
    key = "stats_0"
    signal_collection.add_feature(key, sample_feature)
    retrieved_feature = signal_collection.get_feature(key)
    assert retrieved_feature is sample_feature

def test_get_feature_not_found(signal_collection):
    """Test retrieving a non-existent feature raises KeyError."""
    with pytest.raises(KeyError, match="No Feature with key 'not_a_feature' found"):
        signal_collection.get_feature("not_a_feature")

def test_get_signal_retrieves_feature(signal_collection, sample_feature):
    """Test that get_signal can retrieve a Feature object."""
    key = "stats_0"
    signal_collection.add_feature(key, sample_feature)
    retrieved_item = signal_collection.get_signal(key)
    assert isinstance(retrieved_item, Feature)
    assert retrieved_item is sample_feature

def test_get_signal_retrieves_timeseries(signal_collection, ppg_signal):
    """Test that get_signal can retrieve a TimeSeriesSignal object."""
    key = "ppg_0"
    signal_collection.add_time_series_signal(key, ppg_signal)
    retrieved_item = signal_collection.get_signal(key)
    assert isinstance(retrieved_item, TimeSeriesSignal)
    assert retrieved_item is ppg_signal

def test_get_signal_not_found(signal_collection):
    """Test get_signal raises KeyError if key matches neither type."""
    with pytest.raises(KeyError, match="No TimeSeriesSignal or Feature with key 'missing_key' found"):
        signal_collection.get_signal("missing_key")

def test_add_signal_with_base_name_feature(signal_collection, sample_feature):
    """Test add_signal_with_base_name correctly handles Feature objects."""
    key1 = signal_collection.add_signal_with_base_name("stats", sample_feature)
    assert key1 == "stats_0"
    assert "stats_0" in signal_collection.features
    assert signal_collection.features["stats_0"] is sample_feature

    # Add another feature with the same base name
    feature2_metadata_obj = replace(sample_feature.metadata) # Use replace for metadata copy
    feature2_metadata_obj.feature_id = str(uuid.uuid4()) # Ensure different ID
    feature2 = Feature(data=sample_feature.get_data().copy() + 1, metadata=asdict(feature2_metadata_obj))
    key2 = signal_collection.add_signal_with_base_name("stats", feature2)
    assert key2 == "stats_1"
    assert "stats_1" in signal_collection.features
    assert len(signal_collection.features) == 2
    assert not signal_collection.time_series_signals # Ensure none went to TS dict

def test_get_signals_filter_features(signal_collection, ppg_signal, sample_feature):
    """Test filtering Feature objects using get_signals."""
    # Add a time series signal for context
    signal_collection.add_time_series_signal("ppg_0", ppg_signal)

    # Add the sample feature (STATISTICAL type, source ppg_0)
    signal_collection.add_feature("stats_0", sample_feature)

    # Add another feature (different type, different source key)
    epoch_index2 = pd.date_range(start="2023-01-01", periods=2, freq="2s", name='epoch_start_time')
    feature_data2 = pd.DataFrame({'sleep_stage': [1, 2]}, index=epoch_index2)
    metadata2 = {
        "feature_id": str(uuid.uuid4()),
        "feature_type": FeatureType.SLEEP_STAGE, # Different type
        "source_signal_ids": ["some_other_id"],
        "source_signal_keys": ["hypnogram_raw"], # Different source key
        "epoch_window_length": pd.Timedelta("30s"), # Different epoch length
        "epoch_step_size": pd.Timedelta("30s"),
        "feature_names": feature_data2.columns.tolist(), # Add feature_names here too
        "name": "stages_0" # Explicitly set name
    }
    feature2 = Feature(data=feature_data2, metadata=metadata2)
    signal_collection.add_feature("stages_0", feature2)

    # Add a third feature with the same base name as the first
    # Use replace and asdict for consistent metadata copying/modification
    feature3_metadata_obj = replace(sample_feature.metadata)
    feature3_metadata_obj.feature_id = str(uuid.uuid4())
    feature3_metadata_obj.source_signal_keys = ["ppg_1"] # Different source key
    feature3 = Feature(data=sample_feature.get_data() + 10, metadata=asdict(feature3_metadata_obj))
    signal_collection.add_signal_with_base_name("stats", feature3) # Adds as "stats_1"

    # --- Test Filtering ---

    # 1. Filter by feature_type (enum)
    stat_features = signal_collection.get_signals(feature_type=FeatureType.STATISTICAL)
    assert len(stat_features) == 2
    assert all(isinstance(f, Feature) for f in stat_features)
    assert set(f.metadata.feature_type for f in stat_features) == {FeatureType.STATISTICAL}
    assert set(f.metadata.name for f in stat_features) == {"stats_0", "stats_1"}

    # 2. Filter by feature_type (string)
    stage_features = signal_collection.get_signals(feature_type="SLEEP_STAGE")
    assert len(stage_features) == 1
    assert stage_features[0].metadata.feature_type == FeatureType.SLEEP_STAGE
    assert stage_features[0].metadata.name == "stages_0"

    # 3. Filter by criteria (targeting feature metadata)
    ppg0_source_features = signal_collection.get_signals(criteria={"source_signal_keys": ["ppg_0"]})
    assert len(ppg0_source_features) == 1
    assert ppg0_source_features[0].metadata.name == "stats_0"

    # 4. Filter by base_name (targeting features)
    stats_base_features = signal_collection.get_signals(base_name="stats")
    assert len(stats_base_features) == 2
    assert set(f.metadata.name for f in stats_base_features) == {"stats_0", "stats_1"}

    # 5. Combine feature_type and criteria
    stats_from_ppg1 = signal_collection.get_signals(
        feature_type=FeatureType.STATISTICAL,
        criteria={"source_signal_keys": ["ppg_1"]}
    )
    assert len(stats_from_ppg1) == 1
    assert stats_from_ppg1[0].metadata.name == "stats_1"

    # 6. Criteria that only matches features
    epoch_30s_features = signal_collection.get_signals(criteria={"epoch_window_length": pd.Timedelta("30s")})
    assert len(epoch_30s_features) == 1
    assert epoch_30s_features[0].metadata.name == "stages_0"

    # 7. Criteria that only matches time series signals
    ppg_type_signals = signal_collection.get_signals(criteria={"signal_type": SignalType.PPG})
    assert len(ppg_type_signals) == 1
    assert isinstance(ppg_type_signals[0], TimeSeriesSignal)
    assert ppg_type_signals[0].metadata.name == "ppg_0"

    # 8. No filters - should return all items (1 TS, 3 Features)
    all_items = signal_collection.get_signals()
    assert len(all_items) == 4
    assert sum(1 for item in all_items if isinstance(item, TimeSeriesSignal)) == 1
    assert sum(1 for item in all_items if isinstance(item, Feature)) == 3

# --- End Feature Handling Tests ---


def test_new_get_signals_functionality(signal_collection, ppg_signal):
    """Test the new consolidated get_signals method with various input types."""
    # Set up test signals
    signal_collection.add_time_series_signal("ppg_0", ppg_signal)
    
    # Create an accelerometer signal
    accel_dates = pd.date_range('2023-01-01', periods=3, freq='1s')
    accel_data = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6],
        'z': [7, 8, 9]
    }, index=accel_dates)
    accel_data.index.name = 'timestamp'
    
    accel_signal = AccelerometerSignal(data=accel_data, metadata={
        "signal_id": str(uuid.uuid4()),
        "signal_type": SignalType.ACCELEROMETER,
        "body_position": BodyPosition.CHEST
    })
    signal_collection.add_time_series_signal("accel_0", accel_signal)
    
    # Test retrieving by signal type (enum)
    ppg_signals = signal_collection.get_signals(signal_type=SignalType.PPG)
    assert len(ppg_signals) == 1
    assert ppg_signals[0].metadata.signal_type == SignalType.PPG
    
    # Test retrieving by signal type (string)
    accel_signals = signal_collection.get_signals(signal_type="ACCELEROMETER")
    assert len(accel_signals) == 1
    assert accel_signals[0].metadata.signal_type == SignalType.ACCELEROMETER
    
    # Test input_spec as string (key)
    result = signal_collection.get_signals(input_spec="ppg_0")
    assert len(result) == 1
    assert result[0] is ppg_signal
    
    # Test input_spec as list of keys
    result = signal_collection.get_signals(input_spec=["ppg_0", "accel_0"])
    assert len(result) == 2
    assert set(s.metadata.signal_type for s in result) == {SignalType.PPG, SignalType.ACCELEROMETER}
    
    # Test input_spec as dict with criteria
    result = signal_collection.get_signals(input_spec={
        "criteria": {"body_position": "CHEST"}
    })
    assert len(result) == 1
    assert result[0].metadata.body_position == BodyPosition.CHEST
    
    # Test combined filtering
    # Add another PPG signal with different properties
    ppg2_data = pd.DataFrame({"value": [7, 8, 9]}, 
                           index=pd.date_range('2023-01-01', periods=3, freq='1s'))
    ppg2_data.index.name = 'timestamp'
    ppg_signal2 = PPGSignal(data=ppg2_data, metadata={
        "signal_id": str(uuid.uuid4()),
        "signal_type": SignalType.PPG,
        "body_position": BodyPosition.CHEST
    })
    signal_collection.add_time_series_signal("ppg_1", ppg_signal2)
    
    # Test filtering by signal_type and other criteria
    result = signal_collection.get_signals(
        signal_type=SignalType.PPG,
        criteria={"body_position": BodyPosition.CHEST}
    )
    assert len(result) == 1
    assert result[0].metadata.signal_type == SignalType.PPG
    assert result[0].metadata.body_position == BodyPosition.CHEST

def test_get_combined_dataframe_multiindex(signal_collection):
    """Test the get_combined_dataframe method with multi-index columns."""
    # Configure the collection to use multi-index
    signal_collection.set_index_config(["signal_type", "sensor_model", "body_position"])
    
    # Add an accelerometer signal
    accel_dates = pd.date_range('2023-01-01', periods=5, freq='1s')
    accel_data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [6, 7, 8, 9, 10],
        'z': [11, 12, 13, 14, 15]
    }, index=accel_dates)
    accel_data.index.name = 'timestamp'
    
    accel_signal = AccelerometerSignal(data=accel_data, metadata={
        "signal_id": str(uuid.uuid4()),
        "signal_type": SignalType.ACCELEROMETER,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.CHEST
    })
    signal_collection.add_time_series_signal("accelerometer_0", accel_signal)
    
    # Add a heart rate signal
    hr_dates = pd.date_range('2023-01-01', periods=5, freq='1s')
    hr_data = pd.DataFrame({
        'hr': [72, 73, 74, 75, 76]
    }, index=hr_dates)
    hr_data.index.name = 'timestamp'
    
    hr_signal = HeartRateSignal(data=hr_data, metadata={
        "signal_id": str(uuid.uuid4()),
        "signal_type": SignalType.HEART_RATE,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.CHEST
    })
    signal_collection.add_time_series_signal("hr_0", hr_signal)

    # --- Generate and retrieve the combined dataframe using the new workflow ---
    # 1. Calculate alignment grid parameters (using default highest rate)
    signal_collection.generate_alignment_grid()
    # 2. Align and combine using merge_asof (simpler for testing structure)
    signal_collection.align_and_combine_signals()
    # 3. Get the stored combined dataframe
    combined_df = signal_collection.get_stored_combined_dataframe()
    # --- End new workflow ---

    # Verify it's a dataframe with the correct structure
    assert combined_df is not None, "Combined dataframe should not be None"
    assert isinstance(combined_df, pd.DataFrame)
    assert not combined_df.empty
    
    # Verify we have a multi-index
    assert isinstance(combined_df.columns, pd.MultiIndex)
    
    # Verify the index config was applied
    assert combined_df.columns.names == ["signal_type", "sensor_model", "body_position", "column"]
    
    # Verify we have the expected levels
    assert combined_df.columns.nlevels == 4
    
    # Verify some expected values at each level
    assert "ACCELEROMETER" in combined_df.columns.get_level_values(0)
    assert "HEART_RATE" in combined_df.columns.get_level_values(0)
    assert "POLAR_H10" in combined_df.columns.get_level_values(1)
    assert "CHEST" in combined_df.columns.get_level_values(2)
    assert "x" in combined_df.columns.get_level_values(3)
    assert "hr" in combined_df.columns.get_level_values(3)
    
    # Verify the first column tuple has the correct structure
    first_tuple = combined_df.columns[0]
    assert len(first_tuple) == 4  # Should have 4 levels
    assert isinstance(first_tuple, tuple)  # Should be a proper tuple
    
    # Verify we can access data via the multi-index
    # Get x column from accelerometer
    x_data = combined_df[("ACCELEROMETER", "POLAR_H10", "CHEST", "x")]
    assert len(x_data) == 5
    assert x_data.iloc[0] == 1
    
    # Get hr column from heart rate
    hr_data = combined_df[("HEART_RATE", "POLAR_H10", "CHEST", "hr")]
    assert len(hr_data) == 5
    assert hr_data.iloc[0] == 72


def test_alignment_workflow(signal_collection):
    """Test the alignment workflow: generate_grid, apply_alignment, combine."""
    import pandas as pd
    from sleep_analysis.signals.time_series_signal import TimeSeriesSignal
    from sleep_analysis.signal_types import SignalType

    # Mock signal class for testing
    class MockSignal(TimeSeriesSignal):
        signal_type = SignalType.PPG
        required_columns = ["value"]
        def get_sampling_rate(self):
            # Estimate sample rate based on index frequency
            freq = self._data.index.freqstr or pd.infer_freq(self._data.index)
            return 1e9 / pd.Timedelta(freq).value if freq else 100.0

    # Create signals with different sample rates
    high_freq = pd.date_range("2023-01-01", periods=100, freq="10ms")  # 100 Hz
    low_freq = pd.date_range("2023-01-01", periods=50, freq="20ms")    # 50 Hz
    high_df = pd.DataFrame({"value": range(100)}, index=high_freq)
    low_df = pd.DataFrame({"value": range(50)}, index=low_freq)

    # Add signals to the collection
    signal_collection.add_time_series_signal("high_rate", MockSignal(high_df))
    signal_collection.add_time_series_signal("low_rate", MockSignal(low_df))
    
    # Set the index configuration to include the signal name
    signal_collection.metadata.index_config = ["name"]
    
    # Update the signals' metadata with names matching their keys (access via correct dict)
    signal_collection.time_series_signals["high_rate"].metadata.name = "high_rate"
    signal_collection.time_series_signals["low_rate"].metadata.name = "low_rate"

    # --- Test 1: Align to highest sample rate (100 Hz) using apply_grid_alignment + combine ---
    # Step 1: Generate grid (should default to 100 Hz)
    signal_collection.generate_alignment_grid()
    assert signal_collection.target_rate == 100.0
    assert signal_collection.grid_index is not None
    assert signal_collection.grid_index.freqstr in ["10ms", "10L"]

    # Step 2: Apply alignment in place using 'nearest'
    signal_collection.apply_grid_alignment(method='nearest')

    # Step 3: Combine the aligned signals
    signal_collection.combine_aligned_signals()
    combined_df_100hz = signal_collection.get_stored_combined_dataframe()

    assert combined_df_100hz is not None
    assert len(combined_df_100hz) == 100 # Matches high-rate signal length
    assert combined_df_100hz.index.is_unique
    assert combined_df_100hz.index.freqstr in ["10ms", "10L"]
    # Check columns (using multi-index with 'name')
    assert combined_df_100hz.columns.names == ["name", "column"]
    assert ('high_rate', 'value') in combined_df_100hz.columns
    assert ('low_rate', 'value') in combined_df_100hz.columns
    # Check that 'nearest' produced NaNs for the low_rate signal where no original point mapped
    assert combined_df_100hz[('low_rate', 'value')].isna().sum() > 0

    # --- Test 2: Downsample to 25 Hz using align_and_combine_signals (merge_asof) ---
    # Reset alignment parameters (or use a new collection instance)
    signal_collection._aligned_dataframe = None
    signal_collection._aligned_dataframe_params = None
    signal_collection._alignment_params_calculated = False

    # Step 1: Generate grid for 25 Hz
    signal_collection.generate_alignment_grid(target_sample_rate=25.0)
    assert signal_collection.target_rate == 25.0
    expected_freq = "40ms"
    assert signal_collection.grid_index.freqstr in [expected_freq, "40L"]

    # Step 2: Align and combine using merge_asof
    signal_collection.align_and_combine_signals()
    combined_df_25hz = signal_collection.get_stored_combined_dataframe()

    assert combined_df_25hz is not None
    assert combined_df_25hz.index.freqstr in [expected_freq, "40L"]
    assert ('high_rate', 'value') in combined_df_25hz.columns
    assert ('low_rate', 'value') in combined_df_25hz.columns
    # merge_asof (nearest) should fill most values.
    # Allow for potential NaN at the very last point due to merge_asof boundary behavior.
    assert combined_df_25hz[('high_rate', 'value')].notna().all()
    # Check that all but possibly the last value are not NaN for low_rate
    low_rate_values = combined_df_25hz[('low_rate', 'value')]
    assert low_rate_values.iloc[:-1].notna().all(), "All values except potentially the last should be non-NaN for low_rate"
    # Optionally, assert that *at most* one NaN exists
    assert low_rate_values.isna().sum() <= 1, "There should be at most one NaN value for low_rate"


    # --- Test 3: Handle signals with None sampling rates ---
    # Reset alignment parameters
    signal_collection._aligned_dataframe = None
    signal_collection._aligned_dataframe_params = None
    signal_collection._alignment_params_calculated = False

    class NoneRateSignal(TimeSeriesSignal):
        signal_type = SignalType.PPG
        required_columns = ["value"]
        def get_sampling_rate(self):
            return None
            
    # Create and add a signal with None sampling rate
    none_df = pd.DataFrame({"value": range(20)}, index=pd.date_range("2023-01-01", periods=20, freq="50ms"))
    # Add the signal only once
    none_rate_signal_instance = NoneRateSignal(none_df)
    signal_collection.add_time_series_signal("none_rate", none_rate_signal_instance)
    # Update metadata directly on the instance stored in the collection (access via correct dict)
    signal_collection.time_series_signals["none_rate"].metadata.name = "none_rate"

    # Align should still work, using the valid rates (100 Hz)
    # Step 1: Generate grid (should ignore None rate signal and use 100 Hz)
    signal_collection.generate_alignment_grid()
    assert signal_collection.target_rate == 100.0

    # Step 2: Align and combine using merge_asof
    signal_collection.align_and_combine_signals()
    combined_df_with_none = signal_collection.get_stored_combined_dataframe()

    assert combined_df_with_none is not None
    assert ('none_rate', 'value') in combined_df_with_none.columns
    assert combined_df_with_none[('none_rate', 'value')].notna().sum() > 0 # Ensure some data was merged

    # --- Test 4: Test helper methods related to alignment ---
    # (These seem okay, just ensure they are still relevant)
    # Test get_target_sample_rate
    assert signal_collection.get_target_sample_rate() in STANDARD_RATES
    assert signal_collection.get_target_sample_rate(200) == 200
    
    # Test get_nearest_standard_rate
    assert signal_collection.get_nearest_standard_rate(95) == 100
    assert signal_collection.get_nearest_standard_rate(112) == 100 # 100 is closer than 125
    assert signal_collection.get_nearest_standard_rate(137) == 125 # 125 is closer than 100 or 200
    # Check the default rate returned for None or invalid input
    assert signal_collection.get_nearest_standard_rate(None) == 1.0
    assert signal_collection.get_nearest_standard_rate(0) == 1.0
    assert signal_collection.get_nearest_standard_rate(-10) == 1.0
    
    # Test get_reference_time
    target_period = pd.Timedelta(milliseconds=10)  # 100 Hz
    ref_time = signal_collection.get_reference_time(target_period)
    assert isinstance(ref_time, pd.Timestamp)


def test_apply_collection_operation(signal_collection):
    """Test the apply_operation method for dispatching collection operations."""
    # Add a dummy signal for context
    dates = pd.date_range('2023-01-01', periods=5, freq='1s')
    data = pd.DataFrame({'value': range(5)}, index=dates)
    signal = PPGSignal(data=data, metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG})
    signal_collection.add_time_series_signal("dummy_0", signal)

    # --- Test Success Case ---
    # Use generate_alignment_grid as an example registered operation
    result = signal_collection.apply_operation("generate_alignment_grid", target_sample_rate=10.0)
    assert result is signal_collection # Should return self
    assert signal_collection._alignment_params_calculated is True
    assert signal_collection.target_rate == 10.0
    assert signal_collection.grid_index is not None

    # --- Test Operation Not Found ---
    with pytest.raises(ValueError, match="Collection operation 'non_existent_operation' not found"):
        signal_collection.apply_operation("non_existent_operation")

    # --- Test Underlying Operation Error ---
    # Mock apply_grid_alignment to raise an error *after* generate_alignment_grid has run
    # (apply_grid_alignment itself raises RuntimeError if grid isn't calculated,
    # so we need generate_alignment_grid to succeed first)
    signal_collection.generate_alignment_grid() # Ensure grid exists

    # Phase 2c: Mock the method directly on the instance instead of registry
    original_method = signal_collection.apply_grid_alignment
    def mock_apply_grid_alignment_error(*args, **kwargs):
        raise ConnectionError("Simulated underlying error")
    signal_collection.apply_grid_alignment = mock_apply_grid_alignment_error

    with pytest.raises(ConnectionError, match="Simulated underlying error"):
        signal_collection.apply_operation("apply_grid_alignment", method='nearest')

    # Restore original method
    signal_collection.apply_grid_alignment = original_method


def test_add_signal_timezone_warning(caplog):
    """Test warning when adding naive signal to timezone-aware collection."""
    import logging
    caplog.set_level(logging.WARNING)

    # Create collection with a specific timezone
    collection = SignalCollection(metadata={"timezone": "America/New_York"})
    assert collection.metadata.timezone == "America/New_York"

    # Create a signal with a naive timestamp index
    naive_dates = pd.date_range('2023-01-01', periods=5, freq='1s') # Naive
    naive_data = pd.DataFrame({'value': range(5)}, index=naive_dates)
    naive_signal = PPGSignal(data=naive_data, metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG})

    # Add the naive signal - should trigger a warning
    collection.add_time_series_signal("naive_signal", naive_signal)

    # Check captured logs for the expected warning
    assert "has a naive timestamp index (timezone: None)" in caplog.text
    assert "collection timezone is 'America/New_York'" in caplog.text

    # --- Test adding aware signal to naive collection ---
    caplog.clear()
    naive_collection = SignalCollection(metadata={"timezone": None}) # Naive collection
    aware_dates = pd.date_range('2023-01-01', periods=5, freq='1s', tz="UTC") # Aware
    aware_data = pd.DataFrame({'value': range(5)}, index=aware_dates)
    aware_signal = PPGSignal(data=aware_data, metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG})

    naive_collection.add_time_series_signal("aware_signal", aware_signal)
    # Check the specific warning message for aware signal -> naive collection
    assert "timezone string ('UTC') does not match collection timezone string ('None')" in caplog.text

    # --- Test adding aware signal with mismatching timezone ---
    caplog.clear()
    collection_aware = SignalCollection(metadata={"timezone": "America/New_York"})
    # aware_signal is UTC
    collection_aware.add_time_series_signal("aware_signal_mismatch", aware_signal)
    # Check the specific warning message for mismatching aware timezones
    assert "timezone string ('UTC') does not match collection timezone string ('America/New_York')" in caplog.text
