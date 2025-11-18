"""
Unit tests for AlignmentGridService.

Tests cover:
- Service initialization
- Target sample rate determination
- Reference time calculation
- Grid index generation
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from src.sleep_analysis.core.services.alignment_grid_service import (
    AlignmentGridService,
    STANDARD_RATES
)
from src.sleep_analysis.core.models.alignment_state import AlignmentGridState
from src.sleep_analysis.core.repositories.signal_repository import SignalRepository
from src.sleep_analysis.core.metadata import TimeSeriesMetadata
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.signal_types import SignalType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_metadata_handler():
    """Create a mock metadata handler."""
    handler = Mock()
    handler.create_time_series_metadata.return_value = Mock(spec=TimeSeriesMetadata)
    return handler


@pytest.fixture
def signal_repository(mock_metadata_handler):
    """Create a SignalRepository instance."""
    return SignalRepository(mock_metadata_handler, collection_timezone="UTC")


@pytest.fixture
def alignment_grid_service(signal_repository):
    """Create an AlignmentGridService instance."""
    return AlignmentGridService(signal_repository)


@pytest.fixture
def sample_signal_100hz():
    """Create a sample signal at 100 Hz."""
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=1000, freq="10ms", tz="UTC")
    data = pd.DataFrame(
        {"hr": 70.0 + np.random.randn(1000) * 5},
        index=timestamps
    )

    metadata = {
        "signal_id": "test_signal_100hz",
        "signal_type": SignalType.HEART_RATE,
        "sampling_rate": 100.0
    }

    signal = HeartRateSignal(data, metadata)
    return signal


@pytest.fixture
def sample_signal_50hz():
    """Create a sample signal at 50 Hz."""
    timestamps = pd.date_range("2024-01-01 00:00:05", periods=500, freq="20ms", tz="UTC")
    data = pd.DataFrame(
        {"hr": 68.0 + np.random.randn(500) * 5},
        index=timestamps
    )

    metadata = {
        "signal_id": "test_signal_50hz",
        "signal_type": SignalType.HEART_RATE,
        "sampling_rate": 50.0
    }

    signal = HeartRateSignal(data, metadata)
    return signal


@pytest.fixture
def sample_signal_25hz():
    """Create a sample signal at 25 Hz."""
    timestamps = pd.date_range("2024-01-01 00:00:02", periods=250, freq="40ms", tz="UTC")
    data = pd.DataFrame(
        {"hr": 72.0 + np.random.randn(250) * 5},
        index=timestamps
    )

    metadata = {
        "signal_id": "test_signal_25hz",
        "signal_type": SignalType.HEART_RATE,
        "sampling_rate": 25.0
    }

    signal = HeartRateSignal(data, metadata)
    return signal


# ============================================================================
# Test AlignmentGridService Initialization
# ============================================================================

class TestAlignmentGridServiceBasics:
    """Test basic initialization and properties."""

    def test_initialization(self, alignment_grid_service, signal_repository):
        """Test service initializes correctly."""
        assert alignment_grid_service.repository is signal_repository
        assert isinstance(alignment_grid_service.state, AlignmentGridState)
        assert alignment_grid_service.state.is_calculated is False

    def test_initial_state_not_valid(self, alignment_grid_service):
        """Test initial state is not valid."""
        assert alignment_grid_service.state.is_valid() is False

    def test_state_property_returns_state(self, alignment_grid_service):
        """Test state property returns AlignmentGridState."""
        state = alignment_grid_service.state
        assert isinstance(state, AlignmentGridState)
        assert state.target_rate is None
        assert state.reference_time is None
        assert state.grid_index is None


# ============================================================================
# Test Target Sample Rate Determination
# ============================================================================

class TestTargetSampleRate:
    """Test target sample rate determination logic."""

    def test_user_specified_rate(self, alignment_grid_service):
        """Test user-specified rate is returned."""
        rate = alignment_grid_service._get_target_sample_rate(user_specified=123.45)
        assert rate == 123.45

    def test_auto_calculate_from_single_signal(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test auto-calculation with single signal."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        rate = alignment_grid_service._get_target_sample_rate()
        # Should return largest standard rate <= 100.0, which is 100.0
        assert rate == 100.0

    def test_auto_calculate_from_multiple_signals(self, alignment_grid_service, signal_repository,
                                                   sample_signal_100hz, sample_signal_50hz):
        """Test auto-calculation with multiple signals."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)
        signal_repository.add_time_series_signal("sig2", sample_signal_50hz)

        rate = alignment_grid_service._get_target_sample_rate()
        # Should return largest standard rate <= max(100, 50) = 100, which is 100.0
        assert rate == 100.0

    def test_auto_calculate_no_signals_defaults_to_100(self, alignment_grid_service):
        """Test default rate when no signals present."""
        rate = alignment_grid_service._get_target_sample_rate()
        assert rate == 100.0

    def test_auto_calculate_with_non_standard_rate(self, alignment_grid_service, signal_repository):
        """Test auto-calculation with non-standard rate."""
        # Create signal with non-standard rate (e.g., 75 Hz)
        timestamps = pd.date_range("2024-01-01 00:00:00", periods=750, freq="13.333333ms", tz="UTC")
        data = pd.DataFrame(
            {"hr": 70.0 + np.random.randn(750) * 5},
            index=timestamps
        )
        metadata = {
            "signal_id": "test_signal_75hz",
            "signal_type": SignalType.HEART_RATE,
            "sampling_rate": 75.0
        }
        signal = HeartRateSignal(data, metadata)
        signal_repository.add_time_series_signal("sig1", signal)

        rate = alignment_grid_service._get_target_sample_rate()
        # Should return largest standard rate <= 75, which is 50.0
        assert rate == 50.0


# ============================================================================
# Test Nearest Standard Rate
# ============================================================================

class TestNearestStandardRate:
    """Test nearest standard rate finding logic."""

    def test_exact_match(self, alignment_grid_service):
        """Test exact match returns same rate."""
        assert alignment_grid_service.get_nearest_standard_rate(100.0) == 100.0
        assert alignment_grid_service.get_nearest_standard_rate(50.0) == 50.0

    def test_closest_match(self, alignment_grid_service):
        """Test finding closest standard rate."""
        assert alignment_grid_service.get_nearest_standard_rate(99.5) == 100.0
        assert alignment_grid_service.get_nearest_standard_rate(51.0) == 50.0
        assert alignment_grid_service.get_nearest_standard_rate(26.0) == 25.0

    def test_invalid_rate_returns_default(self, alignment_grid_service):
        """Test invalid rates return default 1 Hz."""
        assert alignment_grid_service.get_nearest_standard_rate(None) == 1.0
        assert alignment_grid_service.get_nearest_standard_rate(0) == 1.0
        assert alignment_grid_service.get_nearest_standard_rate(-10) == 1.0


# ============================================================================
# Test Reference Time Calculation
# ============================================================================

class TestReferenceTime:
    """Test reference time calculation logic."""

    def test_reference_time_with_single_signal(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test reference time calculation with single signal."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        target_period = pd.Timedelta(seconds=0.01)  # 100 Hz
        ref_time = alignment_grid_service._get_reference_time(target_period)

        assert isinstance(ref_time, pd.Timestamp)
        assert ref_time.tz is not None  # Should be timezone-aware
        assert ref_time <= sample_signal_100hz.get_data().index.min()

    def test_reference_time_with_multiple_signals(self, alignment_grid_service, signal_repository,
                                                   sample_signal_100hz, sample_signal_50hz):
        """Test reference time with multiple signals."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)
        signal_repository.add_time_series_signal("sig2", sample_signal_50hz)

        target_period = pd.Timedelta(seconds=0.01)  # 100 Hz
        ref_time = alignment_grid_service._get_reference_time(target_period)

        # Reference time should be <= earliest signal start
        earliest = min(sample_signal_100hz.get_data().index.min(),
                      sample_signal_50hz.get_data().index.min())
        assert ref_time <= earliest

    def test_reference_time_no_signals_returns_epoch(self, alignment_grid_service):
        """Test reference time with no signals returns epoch."""
        target_period = pd.Timedelta(seconds=0.01)
        ref_time = alignment_grid_service._get_reference_time(target_period)

        assert ref_time == pd.Timestamp("1970-01-01", tz='UTC')

    def test_reference_time_zero_period_raises_error(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test zero period raises ValueError."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        with pytest.raises(ValueError, match="Target period cannot be zero"):
            alignment_grid_service._get_reference_time(pd.Timedelta(seconds=0))


# ============================================================================
# Test Grid Index Calculation
# ============================================================================

class TestGridIndexCalculation:
    """Test grid index calculation logic."""

    def test_grid_index_with_single_signal(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test grid index calculation with single signal."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        target_rate = 100.0
        target_period = pd.Timedelta(seconds=1 / target_rate)
        ref_time = alignment_grid_service._get_reference_time(target_period)

        grid_index = alignment_grid_service._calculate_grid_index(target_rate, ref_time)

        assert grid_index is not None
        assert isinstance(grid_index, pd.DatetimeIndex)
        assert len(grid_index) > 0
        assert grid_index.tz is not None

    def test_grid_index_spans_all_signals(self, alignment_grid_service, signal_repository,
                                          sample_signal_100hz, sample_signal_50hz, sample_signal_25hz):
        """Test grid index spans all signals."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)
        signal_repository.add_time_series_signal("sig2", sample_signal_50hz)
        signal_repository.add_time_series_signal("sig3", sample_signal_25hz)

        target_rate = 100.0
        target_period = pd.Timedelta(seconds=1 / target_rate)
        ref_time = alignment_grid_service._get_reference_time(target_period)

        grid_index = alignment_grid_service._calculate_grid_index(target_rate, ref_time)

        # Grid should span from earliest to latest signal
        earliest = min(sample_signal_100hz.get_data().index.min(),
                      sample_signal_50hz.get_data().index.min(),
                      sample_signal_25hz.get_data().index.min())
        latest = max(sample_signal_100hz.get_data().index.max(),
                    sample_signal_50hz.get_data().index.max(),
                    sample_signal_25hz.get_data().index.max())

        assert grid_index.min() <= earliest
        assert grid_index.max() >= latest

    def test_grid_index_invalid_rate_returns_none(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test invalid rate returns None."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        ref_time = pd.Timestamp("2024-01-01", tz="UTC")
        grid_index = alignment_grid_service._calculate_grid_index(-10.0, ref_time)

        assert grid_index is None

    def test_grid_index_no_signals_returns_none(self, alignment_grid_service):
        """Test no signals returns None."""
        target_rate = 100.0
        ref_time = pd.Timestamp("2024-01-01", tz="UTC")

        grid_index = alignment_grid_service._calculate_grid_index(target_rate, ref_time)

        assert grid_index is None


# ============================================================================
# Test generate_alignment_grid (Integration)
# ============================================================================

class TestGenerateAlignmentGrid:
    """Test full alignment grid generation."""

    def test_generate_grid_with_signals(self, alignment_grid_service, signal_repository,
                                        sample_signal_100hz, sample_signal_50hz):
        """Test successful grid generation."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)
        signal_repository.add_time_series_signal("sig2", sample_signal_50hz)

        state = alignment_grid_service.generate_alignment_grid()

        assert state is not None
        assert state.is_calculated is True
        assert state.is_valid() is True
        assert state.target_rate == 100.0
        assert state.reference_time is not None
        assert state.grid_index is not None
        assert len(state.grid_index) > 0

    def test_generate_grid_with_specified_rate(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test grid generation with user-specified rate."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        state = alignment_grid_service.generate_alignment_grid(target_sample_rate=50.0)

        assert state.target_rate == 50.0
        assert state.is_valid() is True

    def test_generate_grid_no_signals_raises_error(self, alignment_grid_service):
        """Test grid generation with no signals raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No time-series signals found"):
            alignment_grid_service.generate_alignment_grid()

    def test_generate_grid_updates_state(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test grid generation updates internal state."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        # Check initial state
        assert alignment_grid_service.state.is_calculated is False

        # Generate grid
        state = alignment_grid_service.generate_alignment_grid()

        # Check state was updated
        assert alignment_grid_service.state is state
        assert alignment_grid_service.state.is_calculated is True

    def test_state_immutability(self, alignment_grid_service, signal_repository, sample_signal_100hz):
        """Test that returned state is independent."""
        signal_repository.add_time_series_signal("sig1", sample_signal_100hz)

        state1 = alignment_grid_service.generate_alignment_grid(target_sample_rate=100.0)
        state2 = alignment_grid_service.generate_alignment_grid(target_sample_rate=50.0)

        # Second call should create a new state
        assert state2.target_rate == 50.0
        assert state2 is not state1
        # Service should hold the new state
        assert alignment_grid_service.state is state2
