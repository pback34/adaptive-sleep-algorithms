"""Test fixtures and configuration for the sleep analysis framework."""

import pytest
import pandas as pd
from datetime import datetime
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit


@pytest.fixture
def sample_metadata():
    """Fixture for sample metadata dictionary."""
    return {
        "signal_id": "test_signal_001",
        "name": "Test Signal",
        # "sample_rate": "100Hz", # Removed - now calculated automatically
        "units": Unit.BPM,
        "start_time": datetime(2023, 1, 1, 0, 0, 0),
        "end_time": datetime(2023, 1, 1, 1, 0, 0),
        "sensor_type": SensorType.PPG,
        "sensor_model": SensorModel.POLAR_H10,
        "body_position": BodyPosition.LEFT_WRIST
    }

@pytest.fixture
def sample_dataframe():
    """Fixture for a sample pandas DataFrame."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
        "value": list(range(100))
    }).set_index("timestamp")
