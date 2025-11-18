"""
Unit tests for SignalSummaryReporter.

Tests cover:
- Summary generation for signals and features
- Field inclusion and filtering
- Cell formatting for display
- Getters for stored results
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
import sys

from src.sleep_analysis.core.services import SignalSummaryReporter
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.signals.ppg_signal import PPGSignal
from src.sleep_analysis.features.feature import Feature
from src.sleep_analysis.signal_types import SignalType


class TestSignalSummaryReporterInitialization:
    """Tests for SignalSummaryReporter initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        reporter = SignalSummaryReporter()

        assert reporter._summary_dataframe is None
        assert reporter._summary_dataframe_params is None


class TestSummarizeSignals:
    """Tests for summarize_signals method."""

    @pytest.fixture
    def setup_signals(self):
        """Setup test signals and features."""
        # Create time-series signals
        index = pd.date_range('2024-01-01', periods=100, freq='1s', tz=timezone.utc)
        hr_data = pd.DataFrame({'hr': [70] * 100}, index=index)
        hr_signal = HeartRateSignal(
            hr_data,
            metadata={'name': 'hr_0', 'signal_type': SignalType.HR}
        )

        ppg_data = pd.DataFrame({'value': [98] * 100}, index=index)
        ppg_signal = PPGSignal(
            ppg_data,
            metadata={'name': 'ppg_0', 'signal_type': SignalType.PPG}
        )

        # Create feature
        feat_index = pd.date_range('2024-01-01', periods=10, freq='10s', tz=timezone.utc)
        feat_data = pd.DataFrame({'mean': [70.0] * 10}, index=feat_index)
        hr_feature = Feature(
            feat_data,
            metadata={
                'name': 'hr_features',
                'epoch_window_length': pd.Timedelta('10s'),
                'epoch_step_size': pd.Timedelta('10s'),
                'feature_names': ['mean'],
                'source_signal_keys': ['hr_0'],
                'source_signal_ids': ['hr_0_id']
            }
        )

        time_series = {'hr_0': hr_signal, 'ppg_0': ppg_signal}
        features = {'hr_features': hr_feature}

        return time_series, features

    def test_summarize_signals_basic(self, setup_signals):
        """Test basic summary generation."""
        time_series, features = setup_signals
        reporter = SignalSummaryReporter()

        summary = reporter.summarize_signals(
            time_series_signals=time_series,
            features=features,
            print_summary=False
        )

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3  # 2 signals + 1 feature
        assert 'hr_0' in summary.index
        assert 'ppg_0' in summary.index
        assert 'hr_features' in summary.index

    def test_summarize_signals_with_item_type_column(self, setup_signals):
        """Test that summary includes item_type column."""
        time_series, features = setup_signals
        reporter = SignalSummaryReporter()

        summary = reporter.summarize_signals(
            time_series,
            features,
            print_summary=False
        )

        assert 'item_type' in summary.columns
        assert summary.loc['hr_0', 'item_type'] == 'TimeSeries'
        assert summary.loc['hr_features', 'item_type'] == 'Feature'

    def test_summarize_signals_with_data_shape(self, setup_signals):
        """Test that summary includes data_shape column."""
        time_series, features = setup_signals
        reporter = SignalSummaryReporter()

        summary = reporter.summarize_signals(
            time_series,
            features,
            fields_to_include=['key', 'item_type', 'data_shape'],
            print_summary=False
        )

        assert 'data_shape' in summary.columns
        assert summary.loc['hr_0', 'data_shape'] == (100, 1)

    def test_summarize_empty_collection(self):
        """Test summary of empty collection."""
        reporter = SignalSummaryReporter()

        summary = reporter.summarize_signals(
            time_series_signals={},
            features={},
            print_summary=False
        )

        assert isinstance(summary, pd.DataFrame)
        assert summary.empty

    def test_summarize_with_custom_fields(self, setup_signals):
        """Test summary with custom field selection."""
        time_series, features = setup_signals
        reporter = SignalSummaryReporter()

        custom_fields = ['key', 'item_type', 'data_shape']
        summary = reporter.summarize_signals(
            time_series,
            features,
            fields_to_include=custom_fields,
            print_summary=False
        )

        # Should only have the requested columns (minus 'key' which becomes index)
        assert set(summary.columns) <= set(['item_type', 'data_shape'])

    def test_summarize_stores_summary(self, setup_signals):
        """Test that summary is stored internally."""
        time_series, features = setup_signals
        reporter = SignalSummaryReporter()

        reporter.summarize_signals(time_series, features, print_summary=False)

        assert reporter._summary_dataframe is not None
        assert reporter._summary_dataframe_params is not None

    def test_summarize_stores_params(self, setup_signals):
        """Test that summary parameters are stored."""
        time_series, features = setup_signals
        reporter = SignalSummaryReporter()

        fields = ['key', 'item_type']
        reporter.summarize_signals(
            time_series,
            features,
            fields_to_include=fields,
            print_summary=False
        )

        params = reporter._summary_dataframe_params
        assert params['fields_to_include'] == fields
        assert params['print_summary'] == False

    def test_summarize_with_print(self, setup_signals, capsys):
        """Test that summary prints to console when requested."""
        time_series, features = setup_signals
        reporter = SignalSummaryReporter()

        reporter.summarize_signals(
            time_series,
            features,
            fields_to_include=['key', 'item_type'],
            print_summary=True
        )

        captured = capsys.readouterr()
        assert "Signal Collection Summary" in captured.out
        assert "hr_0" in captured.out

    def test_summarize_only_time_series(self, setup_signals):
        """Test summary with only time-series signals."""
        time_series, _ = setup_signals
        reporter = SignalSummaryReporter()

        summary = reporter.summarize_signals(
            time_series,
            features={},
            print_summary=False
        )

        assert len(summary) == 2
        assert all(summary['item_type'] == 'TimeSeries')

    def test_summarize_only_features(self, setup_signals):
        """Test summary with only features."""
        _, features = setup_signals
        reporter = SignalSummaryReporter()

        summary = reporter.summarize_signals(
            time_series_signals={},
            features=features,
            print_summary=False
        )

        assert len(summary) == 1
        assert all(summary['item_type'] == 'Feature')


class TestGetSummaryMethods:
    """Tests for getter methods."""

    def test_get_summary_dataframe_before_generation(self):
        """Test getting summary before any summary is generated."""
        reporter = SignalSummaryReporter()

        summary = reporter.get_summary_dataframe()

        assert summary is None

    def test_get_summary_dataframe_after_generation(self):
        """Test getting summary after generation."""
        reporter = SignalSummaryReporter()

        index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        signal = HeartRateSignal(
            pd.DataFrame({'hr': [70] * 10}, index=index),
            metadata={'name': 'hr_0'}
        )

        reporter.summarize_signals({'hr_0': signal}, {}, print_summary=False)
        summary = reporter.get_summary_dataframe()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1

    def test_get_summary_params_before_generation(self):
        """Test getting params before any summary is generated."""
        reporter = SignalSummaryReporter()

        params = reporter.get_summary_params()

        assert params is None

    def test_get_summary_params_after_generation(self):
        """Test getting params after generation."""
        reporter = SignalSummaryReporter()

        index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        signal = HeartRateSignal(
            pd.DataFrame({'hr': [70] * 10}, index=index),
            metadata={'name': 'hr_0'}
        )

        reporter.summarize_signals({'hr_0': signal}, {}, print_summary=False)
        params = reporter.get_summary_params()

        assert isinstance(params, dict)
        assert 'fields_to_include' in params
        assert 'print_summary' in params


class TestFormatSummaryCell:
    """Tests for _format_summary_cell helper method."""

    def test_format_list(self):
        """Test formatting of list values."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell([1, 2, 3], 'test_col')

        assert result == "<list len=3>"

    def test_format_empty_list(self):
        """Test formatting of empty list."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell([], 'test_col')

        assert result == "<list len=0>"

    def test_format_tuple(self):
        """Test formatting of tuple values."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell((1, 2), 'test_col')

        assert result == "<tuple len=2>"

    def test_format_dict(self):
        """Test formatting of dict values."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell({'a': 1, 'b': 2}, 'test_col')

        assert result == "<dict len=2>"

    def test_format_enum(self):
        """Test formatting of Enum values."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell(SignalType.HR, 'signal_type')

        # HR is an alias to HEART_RATE in the enum, so .name returns 'HEART_RATE'
        assert result == 'HEART_RATE'

    def test_format_timestamp(self):
        """Test formatting of Timestamp values."""
        reporter = SignalSummaryReporter()

        ts = pd.Timestamp('2024-01-01 12:00:00', tz=timezone.utc)
        result = reporter._format_summary_cell(ts, 'time_col')

        assert '2024-01-01' in result
        assert '12:00:00' in result

    def test_format_nat_timestamp(self):
        """Test formatting of NaT timestamp."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell(pd.NaT, 'time_col')

        assert result == 'NaT'

    def test_format_timedelta(self):
        """Test formatting of Timedelta values."""
        reporter = SignalSummaryReporter()

        td = pd.Timedelta('30s')
        result = reporter._format_summary_cell(td, 'duration_col')

        assert '30' in result

    def test_format_nat_timedelta(self):
        """Test formatting of NaT timedelta."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell(pd.NaT, 'duration_col')

        assert result == 'NaT'

    def test_format_data_shape_tuple(self):
        """Test formatting of data_shape tuple."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell((100, 5), 'data_shape')

        assert result == '(100, 5)'

    def test_format_none(self):
        """Test formatting of None values."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell(None, 'test_col')

        assert result == 'N/A'

    def test_format_nan(self):
        """Test formatting of NaN values."""
        reporter = SignalSummaryReporter()

        result = reporter._format_summary_cell(np.nan, 'test_col')

        assert result == 'N/A'

    def test_format_scalar_values(self):
        """Test formatting of scalar values (no change)."""
        reporter = SignalSummaryReporter()

        # Numbers
        assert reporter._format_summary_cell(42, 'num_col') == 42
        assert reporter._format_summary_cell(3.14, 'float_col') == 3.14

        # Strings
        assert reporter._format_summary_cell("test", 'str_col') == "test"

        # Booleans
        assert reporter._format_summary_cell(True, 'bool_col') == True

    def test_format_series_error(self, capsys):
        """Test that Series values are handled with error logging."""
        reporter = SignalSummaryReporter()

        series = pd.Series([1, 2, 3])
        result = reporter._format_summary_cell(series, 'bad_col')

        assert "<ERROR:" in result
