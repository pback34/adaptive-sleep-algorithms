"""
Unit tests for EpochGridService.

Tests cover:
- Service initialization
- Epoch grid generation with default time range
- Epoch grid generation with custom time range
- Configuration validation
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.sleep_analysis.core.services.epoch_grid_service import EpochGridService
from src.sleep_analysis.core.models.epoch_state import EpochGridState
from src.sleep_analysis.core.repositories.signal_repository import SignalRepository
from src.sleep_analysis.core.metadata import CollectionMetadata
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.signal_types import SignalType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_metadata_handler():
    """Create a mock metadata handler."""
    handler = Mock()
    return handler


@pytest.fixture
def signal_repository(mock_metadata_handler):
    """Create a SignalRepository instance."""
    return SignalRepository(mock_metadata_handler, collection_timezone="UTC")


@pytest.fixture
def collection_metadata():
    """Create a CollectionMetadata instance with epoch grid config."""
    return CollectionMetadata(
        collection_id="test_collection",
        subject_id="test_subject",
        timezone="UTC",
        epoch_grid_config={
            "window_length": "30s",
            "step_size": "30s"
        }
    )


@pytest.fixture
def epoch_grid_service(signal_repository, collection_metadata):
    """Create an EpochGridService instance."""
    return EpochGridService(signal_repository, collection_metadata)


@pytest.fixture
def sample_signal():
    """Create a sample signal spanning 5 minutes."""
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=300, freq="1s", tz="UTC")
    data = pd.DataFrame(
        {"hr": 70.0 + np.random.randn(300) * 5},
        index=timestamps
    )

    metadata = {
        "signal_id": "test_signal",
        "signal_type": SignalType.HEART_RATE,
        "sampling_rate": 1.0
    }

    return HeartRateSignal(data, metadata)


@pytest.fixture
def sample_signal_long():
    """Create a longer sample signal spanning 10 minutes."""
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=600, freq="1s", tz="UTC")
    data = pd.DataFrame(
        {"hr": 68.0 + np.random.randn(600) * 5},
        index=timestamps
    )

    metadata = {
        "signal_id": "test_signal_long",
        "signal_type": SignalType.HEART_RATE,
        "sampling_rate": 1.0
    }

    return HeartRateSignal(data, metadata)


# ============================================================================
# Test EpochGridService Initialization
# ============================================================================

class TestEpochGridServiceBasics:
    """Test basic initialization and properties."""

    def test_initialization(self, epoch_grid_service, signal_repository, collection_metadata):
        """Test service initializes correctly."""
        assert epoch_grid_service.repository is signal_repository
        assert epoch_grid_service.collection_metadata is collection_metadata
        assert isinstance(epoch_grid_service.state, EpochGridState)
        assert epoch_grid_service.state.is_calculated is False

    def test_initial_state_not_valid(self, epoch_grid_service):
        """Test initial state is not valid."""
        assert epoch_grid_service.state.is_valid() is False

    def test_state_property_returns_state(self, epoch_grid_service):
        """Test state property returns EpochGridState."""
        state = epoch_grid_service.state
        assert isinstance(state, EpochGridState)
        assert state.epoch_grid_index is None
        assert state.window_length is None
        assert state.step_size is None


# ============================================================================
# Test Epoch Grid Generation
# ============================================================================

class TestGenerateEpochGrid:
    """Test epoch grid generation logic."""

    def test_generate_grid_with_default_range(self, epoch_grid_service, signal_repository, sample_signal):
        """Test successful grid generation with default time range from signal."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        state = epoch_grid_service.generate_epoch_grid()

        assert state is not None
        assert state.is_calculated is True
        assert state.is_valid() is True
        assert state.window_length == pd.Timedelta("30s")
        assert state.step_size == pd.Timedelta("30s")
        assert state.epoch_grid_index is not None
        assert len(state.epoch_grid_index) > 0

        # Should have approximately 5 minutes / 30 seconds = 10 epochs
        assert len(state.epoch_grid_index) >= 9  # Allow some tolerance

    def test_generate_grid_with_custom_time_range(self, epoch_grid_service, signal_repository, sample_signal):
        """Test grid generation with custom time range."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        # Generate grid for only first 2 minutes
        state = epoch_grid_service.generate_epoch_grid(
            start_time="2024-01-01 00:00:00",
            end_time="2024-01-01 00:02:00"
        )

        assert state.is_valid() is True
        # Should have 2 minutes / 30 seconds = 4 epochs
        assert len(state.epoch_grid_index) == 4

    def test_generate_grid_with_timestamp_objects(self, epoch_grid_service, signal_repository, sample_signal):
        """Test grid generation with pd.Timestamp objects."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        start = pd.Timestamp("2024-01-01 00:01:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 00:03:00", tz="UTC")

        state = epoch_grid_service.generate_epoch_grid(start_time=start, end_time=end)

        assert state.is_valid() is True
        # Should have 2 minutes / 30 seconds = 4 epochs
        assert len(state.epoch_grid_index) == 4

    def test_generate_grid_with_multiple_signals(self, epoch_grid_service, signal_repository,
                                                 sample_signal, sample_signal_long):
        """Test grid generation spans all signals."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)
        signal_repository.add_time_series_signal("hr_1", sample_signal_long)

        state = epoch_grid_service.generate_epoch_grid()

        # Grid should span the longer signal (10 minutes)
        # 10 minutes / 30 seconds = 20 epochs
        assert len(state.epoch_grid_index) >= 19  # Allow some tolerance

    def test_generate_grid_updates_state(self, epoch_grid_service, signal_repository, sample_signal):
        """Test grid generation updates internal state."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        # Check initial state
        assert epoch_grid_service.state.is_calculated is False

        # Generate grid
        state = epoch_grid_service.generate_epoch_grid()

        # Check state was updated
        assert epoch_grid_service.state is state
        assert epoch_grid_service.state.is_calculated is True

    def test_generate_grid_preserves_timezone(self, epoch_grid_service, signal_repository, sample_signal):
        """Test grid generation preserves collection timezone."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        state = epoch_grid_service.generate_epoch_grid()

        assert state.epoch_grid_index.tz is not None
        assert str(state.epoch_grid_index.tz) == "UTC"


# ============================================================================
# Test Configuration Validation
# ============================================================================

class TestConfigurationValidation:
    """Test epoch grid configuration validation."""

    def test_missing_config_raises_error(self, signal_repository, sample_signal):
        """Test missing epoch_grid_config raises RuntimeError."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        metadata = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config=None
        )
        service = EpochGridService(signal_repository, metadata)

        with pytest.raises(RuntimeError, match="Missing or incomplete 'epoch_grid_config'"):
            service.generate_epoch_grid()

    def test_incomplete_config_raises_error(self, signal_repository, sample_signal):
        """Test incomplete epoch_grid_config raises RuntimeError."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        metadata = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config={"window_length": "30s"}  # Missing step_size
        )
        service = EpochGridService(signal_repository, metadata)

        with pytest.raises(RuntimeError, match="Missing or incomplete 'epoch_grid_config'"):
            service.generate_epoch_grid()

    def test_invalid_window_length_raises_error(self, signal_repository, sample_signal):
        """Test invalid window_length raises RuntimeError."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        metadata = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config={"window_length": "invalid", "step_size": "30s"}
        )
        service = EpochGridService(signal_repository, metadata)

        with pytest.raises(RuntimeError, match="Invalid epoch_grid_config parameters"):
            service.generate_epoch_grid()

    def test_zero_window_length_raises_error(self, signal_repository, sample_signal):
        """Test zero window_length raises ValueError."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        metadata = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config={"window_length": "0s", "step_size": "30s"}
        )
        service = EpochGridService(signal_repository, metadata)

        with pytest.raises(RuntimeError, match="window_length and step_size must be positive"):
            service.generate_epoch_grid()

    def test_negative_step_size_raises_error(self, signal_repository, sample_signal):
        """Test negative step_size raises ValueError."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        metadata = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config={"window_length": "30s", "step_size": "-10s"}
        )
        service = EpochGridService(signal_repository, metadata)

        with pytest.raises(RuntimeError, match="window_length and step_size must be positive"):
            service.generate_epoch_grid()


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_signals_raises_error(self, epoch_grid_service):
        """Test grid generation with no signals raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No time-series signals found"):
            epoch_grid_service.generate_epoch_grid()

    def test_invalid_start_time_raises_error(self, epoch_grid_service, signal_repository, sample_signal):
        """Test invalid start_time raises ValueError."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        with pytest.raises(ValueError, match="Invalid start_time or end_time"):
            epoch_grid_service.generate_epoch_grid(start_time="not-a-date")

    def test_start_after_end_raises_error(self, epoch_grid_service, signal_repository, sample_signal):
        """Test start_time after end_time raises ValueError."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        with pytest.raises(ValueError, match="must be before end time"):
            epoch_grid_service.generate_epoch_grid(
                start_time="2024-01-01 00:05:00",
                end_time="2024-01-01 00:01:00"
            )

    def test_state_immutability(self, epoch_grid_service, signal_repository, sample_signal):
        """Test that returned state is independent."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        state1 = epoch_grid_service.generate_epoch_grid()
        state2 = epoch_grid_service.generate_epoch_grid(
            start_time="2024-01-01 00:00:00",
            end_time="2024-01-01 00:01:00"
        )

        # Second call should create a new state
        assert len(state2.epoch_grid_index) < len(state1.epoch_grid_index)
        assert state2 is not state1
        # Service should hold the new state
        assert epoch_grid_service.state is state2

    def test_different_window_step_configurations(self, signal_repository, sample_signal):
        """Test different window/step configurations work correctly."""
        signal_repository.add_time_series_signal("hr_0", sample_signal)

        # Test overlapping windows (step < window)
        metadata1 = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config={"window_length": "60s", "step_size": "30s"}
        )
        service1 = EpochGridService(signal_repository, metadata1)
        state1 = service1.generate_epoch_grid()

        assert state1.is_valid()
        assert state1.window_length == pd.Timedelta("60s")
        assert state1.step_size == pd.Timedelta("30s")

        # Test non-overlapping windows (step == window)
        metadata2 = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config={"window_length": "30s", "step_size": "30s"}
        )
        service2 = EpochGridService(signal_repository, metadata2)
        state2 = service2.generate_epoch_grid()

        assert state2.is_valid()
        assert state2.window_length == pd.Timedelta("30s")
        assert state2.step_size == pd.Timedelta("30s")

        # Test sparse windows (step > window)
        metadata3 = CollectionMetadata(
            collection_id="test",
            subject_id="test",
            timezone="UTC",
            epoch_grid_config={"window_length": "30s", "step_size": "60s"}
        )
        service3 = EpochGridService(signal_repository, metadata3)
        state3 = service3.generate_epoch_grid()

        assert state3.is_valid()
        assert state3.window_length == pd.Timedelta("30s")
        assert state3.step_size == pd.Timedelta("60s")
