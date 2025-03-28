"""Tests for the signal_types module."""

import pytest
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit

def test_signal_type_enum():
    """Test that the SignalType enum is correctly defined."""
    assert SignalType.PPG.value == "PPG"
    assert SignalType.ACCELEROMETER.value == "Accelerometer"
    assert SignalType.HEART_RATE.value == "Heart Rate"
    assert SignalType.FEATURES.value == "Features"
    
    # Test accessing by string key
    assert SignalType["PPG"] == SignalType.PPG
    
def test_sensor_type_enum():
    """Test that the SensorType enum is correctly defined."""
    assert SensorType.PPG.value == "PPG"
    assert SensorType.ACCEL.value == "Accel"
    
def test_sensor_model_enum():
    """Test that the SensorModel enum is correctly defined."""
    assert SensorModel.POLAR_H10.value == "PolarH10"
    assert SensorModel.POLAR_SENSE.value == "PolarSense"
    
def test_body_position_enum():
    """Test that the BodyPosition enum is correctly defined."""
    assert BodyPosition.CHEST.value == "chest"
    assert BodyPosition.LEFT_WRIST.value == "left_wrist"
    
def test_unit_enum():
    """Test that the Unit enum is correctly defined."""
    assert Unit.G.value == "g"
    assert Unit.BPM.value == "bpm"
    assert Unit.HZ.value == "Hz"
