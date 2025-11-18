"""
Unit tests for AlignmentExecutor.

Tests cover:
- Service initialization
- Grid alignment application
- Method validation
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.sleep_analysis.core.services.alignment_executor import (
    AlignmentExecutor,
    ALLOWED_METHODS
)
from src.sleep_analysis.core.services.alignment_grid_service import AlignmentGridService
from src.sleep_analysis.core.repositories.signal_repository import SignalRepository
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
def alignment_grid_service(signal_repository):
    """Create an AlignmentGridService instance."""
    return AlignmentGridService(signal_repository)


@pytest.fixture
def alignment_executor(signal_repository, alignment_grid_service):
    """Create an AlignmentExecutor instance."""
    return AlignmentExecutor(signal_repository, alignment_grid_service)


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

    return HeartRateSignal(data, metadata)


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

    return HeartRateSignal(data, metadata)


# ============================================================================
# Test AlignmentExecutor Initialization
# ============================================================================

class TestAlignmentExecutorBasics:
    """Test basic initialization and properties."""

    def test_initialization(self, alignment_executor, signal_repository, alignment_grid_service):
        """Test executor initializes correctly."""
        assert alignment_executor.repository is signal_repository
        assert alignment_executor.alignment_grid_service is alignment_grid_service

    def test_allowed_methods_defined(self):
        """Test allowed methods constant is properly defined."""
        assert 'nearest' in ALLOWED_METHODS
        assert 'pad' in ALLOWED_METHODS
        assert 'ffill' in ALLOWED_METHODS
        assert 'backfill' in ALLOWED_METHODS
        assert 'bfill' in ALLOWED_METHODS


# ============================================================================
# Test Grid Alignment Application
# ============================================================================

class TestApplyGridAlignment:
    """Test grid alignment application logic."""

    def test_apply_grid_alignment_no_grid_raises_error(self, alignment_executor):
        """Test applying alignment without grid raises RuntimeError."""
        with pytest.raises(RuntimeError, match="generate_alignment_grid must be run successfully"):
            alignment_executor.apply_grid_alignment()

    def test_apply_grid_alignment_with_valid_grid(
        self, alignment_executor, signal_repository, alignment_grid_service, sample_signal_100hz
    ):
        """Test successful grid alignment application."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)

        # Generate grid first
        alignment_grid_service.generate_alignment_grid(target_sample_rate=100.0)

        # Apply alignment
        count = alignment_executor.apply_grid_alignment(method='nearest')

        assert count == 1  # Should have aligned 1 signal

    def test_apply_grid_alignment_to_specific_signals(
        self, alignment_executor, signal_repository, alignment_grid_service,
        sample_signal_100hz, sample_signal_50hz
    ):
        """Test aligning specific signals."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        signal_repository.add_time_series_signal("hr_1", sample_signal_50hz)

        # Generate grid
        alignment_grid_service.generate_alignment_grid()

        # Align only hr_0
        count = alignment_executor.apply_grid_alignment(signals_to_align=["hr_0"])

        assert count == 1

    def test_apply_grid_alignment_invalid_method_uses_nearest(
        self, alignment_executor, signal_repository, alignment_grid_service, sample_signal_100hz, caplog
    ):
        """Test invalid method defaults to 'nearest' with warning."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        alignment_grid_service.generate_alignment_grid()

        count = alignment_executor.apply_grid_alignment(method='invalid_method')

        assert count == 1
        assert "not in allowed list" in caplog.text

    def test_apply_grid_alignment_skips_missing_signal(
        self, alignment_executor, signal_repository, alignment_grid_service, sample_signal_100hz, caplog
    ):
        """Test alignment skips non-existent signals."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        alignment_grid_service.generate_alignment_grid()

        count = alignment_executor.apply_grid_alignment(signals_to_align=["hr_0", "hr_missing"])

        assert count == 1  # Only hr_0 was aligned
        assert "not found" in caplog.text

    def test_apply_grid_alignment_skips_empty_signal(
        self, alignment_executor, signal_repository, alignment_grid_service, caplog
    ):
        """Test alignment skips signals with empty data."""
        # Create signal with empty data
        empty_data = pd.DataFrame(
            {"hr": []},
            index=pd.DatetimeIndex([], tz="UTC")
        )
        signal = HeartRateSignal(
            empty_data,
            metadata={"signal_id": "empty", "signal_type": SignalType.HEART_RATE}
        )

        # Add another valid signal to generate grid
        timestamps = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1s", tz="UTC")
        valid_data = pd.DataFrame({"hr": [70.0] * 100}, index=timestamps)
        valid_signal = HeartRateSignal(
            valid_data,
            metadata={"signal_id": "valid", "signal_type": SignalType.HEART_RATE, "sampling_rate": 1.0}
        )

        signal_repository.add_time_series_signal("empty", signal)
        signal_repository.add_time_series_signal("valid", valid_signal)

        alignment_grid_service.generate_alignment_grid()

        count = alignment_executor.apply_grid_alignment()

        # Should process valid but skip empty
        assert count == 1
        assert "Skipping alignment" in caplog.text


# ============================================================================
# Test Different Alignment Methods
# ============================================================================

class TestAlignmentMethods:
    """Test different alignment methods work correctly."""

    def test_nearest_method(
        self, alignment_executor, signal_repository, alignment_grid_service, sample_signal_100hz
    ):
        """Test nearest neighbor alignment."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        alignment_grid_service.generate_alignment_grid()

        count = alignment_executor.apply_grid_alignment(method='nearest')
        assert count == 1

    def test_ffill_method(
        self, alignment_executor, signal_repository, alignment_grid_service, sample_signal_100hz
    ):
        """Test forward fill alignment."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        alignment_grid_service.generate_alignment_grid()

        count = alignment_executor.apply_grid_alignment(method='ffill')
        assert count == 1

    def test_bfill_method(
        self, alignment_executor, signal_repository, alignment_grid_service, sample_signal_100hz
    ):
        """Test backward fill alignment."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        alignment_grid_service.generate_alignment_grid()

        count = alignment_executor.apply_grid_alignment(method='bfill')
        assert count == 1


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_apply_alignment_with_operation_failure(
        self, alignment_executor, signal_repository, alignment_grid_service, sample_signal_100hz
    ):
        """Test handling of signal operation failures."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        alignment_grid_service.generate_alignment_grid()

        # Mock apply_operation to raise an exception
        with patch.object(sample_signal_100hz, 'apply_operation', side_effect=Exception("Test error")):
            with pytest.raises(RuntimeError, match="Failed to apply grid alignment"):
                alignment_executor.apply_grid_alignment()

    def test_alignment_all_signals_by_default(
        self, alignment_executor, signal_repository, alignment_grid_service,
        sample_signal_100hz, sample_signal_50hz
    ):
        """Test that all signals are aligned when no specific list provided."""
        signal_repository.add_time_series_signal("hr_0", sample_signal_100hz)
        signal_repository.add_time_series_signal("hr_1", sample_signal_50hz)

        alignment_grid_service.generate_alignment_grid()

        count = alignment_executor.apply_grid_alignment()

        assert count == 2  # Both signals should be aligned
