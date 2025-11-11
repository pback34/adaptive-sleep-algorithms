"""Tests for AlignmentService class."""

import pytest
import pandas as pd
import numpy as np
import uuid
from unittest.mock import Mock, patch

from sleep_analysis.services.alignment_service import AlignmentService, STANDARD_RATES
from sleep_analysis.core.signal_collection import SignalCollection
from sleep_analysis.signals.ppg_signal import PPGSignal
from sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from sleep_analysis.signal_types import SignalType, Unit


@pytest.fixture
def alignment_service():
    """Create an AlignmentService instance for testing."""
    return AlignmentService()


@pytest.fixture
def signal_collection_with_signals():
    """Create a SignalCollection with sample signals at different rates."""
    collection = SignalCollection(metadata={"timezone": "UTC"})

    # Add PPG signal at 100 Hz
    ppg_index = pd.date_range(start="2023-01-01", periods=1000, freq="10ms", tz="UTC")
    ppg_data = pd.DataFrame({"value": np.sin(np.linspace(0, 10, 1000))}, index=ppg_index)
    ppg_data.index.name = 'timestamp'
    ppg_signal = PPGSignal(
        data=ppg_data,
        metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
    )
    collection.add_time_series_signal("ppg_0", ppg_signal)

    # Add heart rate signal at 1 Hz
    hr_index = pd.date_range(start="2023-01-01", periods=10, freq="1s", tz="UTC")
    hr_data = pd.DataFrame({"hr": [70, 72, 71, 73, 72, 74, 73, 75, 74, 76]}, index=hr_index)
    hr_data.index.name = 'timestamp'
    hr_signal = HeartRateSignal(
        data=hr_data,
        metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.HEART_RATE}
    )
    collection.add_time_series_signal("hr_0", hr_signal)

    return collection


class TestAlignmentServiceInit:
    """Tests for AlignmentService initialization."""

    def test_init(self, alignment_service):
        """Test service initialization."""
        assert alignment_service is not None


class TestGetTargetSampleRate:
    """Tests for get_target_sample_rate method."""

    def test_user_specified_rate(self, alignment_service, signal_collection_with_signals):
        """Test with user-specified rate."""
        rate = alignment_service.get_target_sample_rate(signal_collection_with_signals, 50.0)
        assert rate == 50.0

    def test_auto_detect_rate(self, alignment_service, signal_collection_with_signals):
        """Test automatic rate detection from signals."""
        rate = alignment_service.get_target_sample_rate(signal_collection_with_signals, None)
        # Should select highest standard rate <= max signal rate (100 Hz)
        assert rate == 100.0
        assert rate in STANDARD_RATES

    def test_no_valid_rates(self, alignment_service):
        """Test with no valid signal rates."""
        collection = SignalCollection(metadata={"timezone": "UTC"})
        rate = alignment_service.get_target_sample_rate(collection, None)
        # Should default to 100 Hz
        assert rate == 100.0

    def test_rate_selection_from_standards(self, alignment_service):
        """Test that selected rate is from standard rates."""
        collection = SignalCollection(metadata={"timezone": "UTC"})

        # Add signal with non-standard rate (e.g., 75 Hz)
        index = pd.date_range(start="2023-01-01", periods=750, freq="13.333ms", tz="UTC")
        data = pd.DataFrame({"value": range(750)}, index=index)
        data.index.name = 'timestamp'
        signal = PPGSignal(
            data=data,
            metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
        )
        collection.add_time_series_signal("test_0", signal)

        rate = alignment_service.get_target_sample_rate(collection, None)
        # Should select largest standard rate <= actual rate
        assert rate in STANDARD_RATES
        assert rate <= signal.get_sampling_rate()


class TestGetReferenceTime:
    """Tests for get_reference_time method."""

    def test_reference_time_calculation(self, alignment_service, signal_collection_with_signals):
        """Test reference time calculation."""
        target_period = pd.Timedelta(seconds=1/100)  # 100 Hz
        ref_time = alignment_service.get_reference_time(signal_collection_with_signals, target_period)

        assert isinstance(ref_time, pd.Timestamp)
        assert ref_time.tz is not None  # Should be timezone-aware

    def test_reference_time_alignment(self, alignment_service, signal_collection_with_signals):
        """Test that reference time aligns to epoch on target period."""
        target_period = pd.Timedelta(seconds=1/100)
        ref_time = alignment_service.get_reference_time(signal_collection_with_signals, target_period)

        # Reference time should be aligned to 1970 epoch on the target period
        epoch = pd.Timestamp("1970-01-01", tz=ref_time.tz)
        delta_ns = (ref_time - epoch).total_seconds() * 1e9
        period_ns = target_period.total_seconds() * 1e9

        # Should be exact multiple of period
        assert delta_ns % period_ns < 1.0  # Within 1 nanosecond tolerance

    def test_zero_period_error(self, alignment_service, signal_collection_with_signals):
        """Test that zero period raises error."""
        with pytest.raises(ValueError, match="Target period cannot be zero"):
            alignment_service.get_reference_time(
                signal_collection_with_signals,
                pd.Timedelta(0)
            )


class TestCalculateGridIndex:
    """Tests for _calculate_grid_index method."""

    def test_grid_index_generation(self, alignment_service, signal_collection_with_signals):
        """Test grid index generation."""
        target_rate = 10.0  # 10 Hz for easier testing
        target_period = pd.Timedelta(seconds=1/target_rate)
        ref_time = pd.Timestamp("2023-01-01", tz="UTC")

        grid_index = alignment_service._calculate_grid_index(
            signal_collection_with_signals,
            target_rate,
            ref_time
        )

        assert grid_index is not None
        assert isinstance(grid_index, pd.DatetimeIndex)
        assert len(grid_index) > 0
        assert grid_index.freq == target_period or grid_index.inferred_freq is not None

    def test_grid_covers_signal_range(self, alignment_service, signal_collection_with_signals):
        """Test that grid covers entire signal time range."""
        target_rate = 10.0
        ref_time = pd.Timestamp("2023-01-01", tz="UTC")

        grid_index = alignment_service._calculate_grid_index(
            signal_collection_with_signals,
            target_rate,
            ref_time
        )

        # Get min/max from signals
        min_time = min(s.get_data().index.min() for s in signal_collection_with_signals.time_series_signals.values())
        max_time = max(s.get_data().index.max() for s in signal_collection_with_signals.time_series_signals.values())

        # Grid should cover the range
        assert grid_index.min() <= min_time
        assert grid_index.max() >= max_time

    def test_invalid_rate(self, alignment_service, signal_collection_with_signals):
        """Test that invalid rate returns None."""
        ref_time = pd.Timestamp("2023-01-01", tz="UTC")
        grid_index = alignment_service._calculate_grid_index(
            signal_collection_with_signals,
            -10.0,  # Invalid negative rate
            ref_time
        )
        assert grid_index is None


class TestGenerateAlignmentGrid:
    """Tests for generate_alignment_grid method."""

    def test_generate_grid(self, alignment_service, signal_collection_with_signals):
        """Test full grid generation."""
        alignment_service.generate_alignment_grid(signal_collection_with_signals, target_sample_rate=10.0)

        assert signal_collection_with_signals._alignment_params_calculated is True
        assert signal_collection_with_signals.target_rate == 10.0
        assert signal_collection_with_signals.ref_time is not None
        assert signal_collection_with_signals.grid_index is not None
        assert len(signal_collection_with_signals.grid_index) > 0

    def test_generate_grid_auto_rate(self, alignment_service, signal_collection_with_signals):
        """Test grid generation with automatic rate detection."""
        alignment_service.generate_alignment_grid(signal_collection_with_signals)

        assert signal_collection_with_signals._alignment_params_calculated is True
        assert signal_collection_with_signals.target_rate is not None
        assert signal_collection_with_signals.target_rate in STANDARD_RATES

    def test_generate_grid_empty_collection(self, alignment_service):
        """Test grid generation with empty collection raises error."""
        empty_collection = SignalCollection(metadata={"timezone": "UTC"})

        with pytest.raises(RuntimeError, match="No time-series signals"):
            alignment_service.generate_alignment_grid(empty_collection)


class TestApplyGridAlignment:
    """Tests for apply_grid_alignment method."""

    def test_apply_alignment(self, alignment_service, signal_collection_with_signals):
        """Test applying grid alignment to signals."""
        # First generate grid
        alignment_service.generate_alignment_grid(signal_collection_with_signals, target_sample_rate=10.0)

        # Then apply alignment
        alignment_service.apply_grid_alignment(signal_collection_with_signals, method='nearest')

        # Check that signals were modified
        for signal in signal_collection_with_signals.time_series_signals.values():
            data = signal.get_data()
            # Signal index should now match grid
            assert isinstance(data.index, pd.DatetimeIndex)

    def test_apply_alignment_subset(self, alignment_service, signal_collection_with_signals):
        """Test applying alignment to subset of signals."""
        alignment_service.generate_alignment_grid(signal_collection_with_signals, target_sample_rate=10.0)

        # Apply only to PPG signal
        alignment_service.apply_grid_alignment(
            signal_collection_with_signals,
            method='nearest',
            signals_to_align=['ppg_0']
        )

        # PPG should be aligned, HR might not be (depending on implementation)
        ppg_data = signal_collection_with_signals.get_time_series_signal('ppg_0').get_data()
        assert isinstance(ppg_data.index, pd.DatetimeIndex)

    def test_apply_without_grid_fails(self, alignment_service, signal_collection_with_signals):
        """Test that applying alignment without grid generation fails."""
        with pytest.raises(RuntimeError, match="generate_alignment_grid must be run"):
            alignment_service.apply_grid_alignment(signal_collection_with_signals)

    def test_invalid_method(self, alignment_service, signal_collection_with_signals, caplog):
        """Test that invalid alignment method falls back to nearest."""
        alignment_service.generate_alignment_grid(signal_collection_with_signals, target_sample_rate=10.0)

        alignment_service.apply_grid_alignment(
            signal_collection_with_signals,
            method='invalid_method'
        )

        # Should log warning about invalid method
        assert "invalid_method" in caplog.text or "nearest" in caplog.text


class TestAlignAndCombineSignals:
    """Tests for align_and_combine_signals method."""

    def test_align_and_combine(self, alignment_service, signal_collection_with_signals):
        """Test aligning and combining signals."""
        alignment_service.generate_alignment_grid(signal_collection_with_signals, target_sample_rate=1.0)

        combined_df = alignment_service.align_and_combine_signals(signal_collection_with_signals)

        assert isinstance(combined_df, pd.DataFrame)
        assert not combined_df.empty
        assert isinstance(combined_df.index, pd.DatetimeIndex)
        # Should have columns from both signals
        assert len(combined_df.columns) >= 2

    def test_combine_without_grid_fails(self, alignment_service, signal_collection_with_signals):
        """Test that combining without grid generation fails."""
        with pytest.raises(RuntimeError, match="generate_alignment_grid must be run"):
            alignment_service.align_and_combine_signals(signal_collection_with_signals)

    def test_combined_df_has_correct_index(self, alignment_service, signal_collection_with_signals):
        """Test that combined dataframe has correct grid index."""
        alignment_service.generate_alignment_grid(signal_collection_with_signals, target_sample_rate=1.0)
        combined_df = alignment_service.align_and_combine_signals(signal_collection_with_signals)

        # Combined df index should match grid index
        assert len(combined_df) == len(signal_collection_with_signals.grid_index)


class TestIntegration:
    """Integration tests for AlignmentService."""

    def test_full_alignment_workflow(self, alignment_service):
        """Test complete alignment workflow."""
        # Create collection with misaligned signals
        collection = SignalCollection(metadata={"timezone": "UTC"})

        # Signal 1: 100 Hz starting at exact second
        index1 = pd.date_range(start="2023-01-01 00:00:00", periods=1000, freq="10ms", tz="UTC")
        data1 = pd.DataFrame({"value": range(1000)}, index=index1)
        data1.index.name = 'timestamp'
        signal1 = PPGSignal(
            data=data1,
            metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
        )
        collection.add_time_series_signal("signal1", signal1)

        # Signal 2: 50 Hz starting offset by 5ms
        index2 = pd.date_range(start="2023-01-01 00:00:00.005", periods=500, freq="20ms", tz="UTC")
        data2 = pd.DataFrame({"value": range(500)}, index=index2)
        data2.index.name = 'timestamp'
        signal2 = PPGSignal(
            data=data2,
            metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
        )
        collection.add_time_series_signal("signal2", signal2)

        # Generate grid
        alignment_service.generate_alignment_grid(collection, target_sample_rate=10.0)
        assert collection._alignment_params_calculated

        # Apply alignment
        alignment_service.apply_grid_alignment(collection)

        # Combine
        combined_df = alignment_service.align_and_combine_signals(collection)

        # Verify results
        assert not combined_df.empty
        assert isinstance(combined_df.index, pd.DatetimeIndex)
        # Signals should now be on common grid
        assert combined_df.index.equals(collection.grid_index)
