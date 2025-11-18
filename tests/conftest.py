"""Test fixtures and configuration for the sleep analysis framework."""

import pytest
import pandas as pd
from datetime import datetime
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit


@pytest.fixture(scope="session", autouse=True)
def disable_parallel_processing():
    """
    Automatically disable parallel processing for all tests.

    Parallel processing with ProcessPoolExecutor can cause issues in pytest:
    - Worker processes can fork the entire test suite
    - Causes deadlocks and hanging tests
    - Interferes with test collection

    This fixture disables parallel processing globally for the test session.
    """
    from sleep_analysis.utils.parallel import disable_parallel_processing
    disable_parallel_processing()
    yield
    # Re-enable after tests if needed
    from sleep_analysis.utils.parallel import enable_parallel_processing
    enable_parallel_processing()


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
