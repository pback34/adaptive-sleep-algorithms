"""
Unit tests for OperationExecutor.

Tests cover:
- Collection-level operation execution
- Multi-signal operation execution
- Single-signal operations with storage
- Batch operations on multiple signals
- Metadata propagation for features
- Error handling and validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timezone
from unittest.mock import Mock, MagicMock

from src.sleep_analysis.core.services import OperationExecutor
from src.sleep_analysis.core.models import EpochGridState
from src.sleep_analysis.signals.time_series_signal import TimeSeriesSignal
from src.sleep_analysis.features.feature import Feature
from src.sleep_analysis.signal_types import SignalType


class TestOperationExecutorInitialization:
    """Tests for OperationExecutor initialization."""

    def test_initialization_with_all_params(self):
        """Test initialization with all parameters."""
        collection_registry = {'op1': lambda x: x}
        multi_signal_registry = {'op2': (lambda x: x, TimeSeriesSignal)}
        get_ts = Mock()
        get_feat = Mock()
        add_ts = Mock()
        add_feat = Mock()
        epoch_index = pd.date_range('2024-01-01', periods=10, freq='10s', tz=timezone.utc)
        epoch_state = EpochGridState(
            epoch_grid_index=epoch_index,
            window_length=pd.Timedelta('10s'),
            step_size=pd.Timedelta('10s'),
            is_calculated=True
        )

        executor = OperationExecutor(
            collection_op_registry=collection_registry,
            multi_signal_registry=multi_signal_registry,
            get_time_series_signal=get_ts,
            get_feature=get_feat,
            add_time_series_signal=add_ts,
            add_feature=add_feat,
            epoch_state=epoch_state,
            feature_index_config=['name', 'signal_type'],
            global_epoch_window_length=pd.Timedelta('10s'),
            global_epoch_step_size=pd.Timedelta('10s')
        )

        assert executor.collection_op_registry == collection_registry
        assert executor.multi_signal_registry == multi_signal_registry
        assert executor.epoch_state == epoch_state
        assert executor.feature_index_config == ['name', 'signal_type']

    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        assert executor.epoch_state is None
        assert executor.feature_index_config is None


class TestApplyCollectionOperation:
    """Tests for apply_collection_operation method."""

    def test_apply_collection_operation_success(self):
        """Test successful execution of collection operation."""
        # Create a mock operation
        def test_operation(collection, param1, param2):
            return param1 + param2

        registry = {'test_op': test_operation}
        executor = OperationExecutor(
            collection_op_registry=registry,
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        mock_collection = Mock()
        result = executor.apply_collection_operation(
            'test_op',
            mock_collection,
            param1=5,
            param2=3
        )

        assert result == 8

    def test_apply_collection_operation_not_found(self):
        """Test that missing operation raises ValueError."""
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        with pytest.raises(ValueError, match="not found"):
            executor.apply_collection_operation(
                'nonexistent_op',
                Mock()
            )

    def test_apply_collection_operation_with_exception(self):
        """Test that operation exceptions are propagated."""
        def failing_operation(collection):
            raise RuntimeError("Test error")

        registry = {'failing_op': failing_operation}
        executor = OperationExecutor(
            collection_op_registry=registry,
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        with pytest.raises(RuntimeError, match="Test error"):
            executor.apply_collection_operation('failing_op', Mock())


class TestApplyMultiSignalOperation:
    """Tests for apply_multi_signal_operation method."""

    @pytest.fixture
    def setup_multi_signal_operation(self):
        """Setup for multi-signal operation tests."""
        # Create epoch state
        epoch_index = pd.date_range('2024-01-01', periods=10, freq='10s', tz=timezone.utc)
        epoch_state = EpochGridState(
            epoch_grid_index=epoch_index,
            window_length=pd.Timedelta('10s'),
            step_size=pd.Timedelta('10s'),
            is_calculated=True
        )

        # Create mock signals
        grid_index = pd.date_range('2024-01-01', periods=100, freq='1s', tz=timezone.utc)
        signal1 = TimeSeriesSignal(
            pd.DataFrame({'hr': [70] * 100}, index=grid_index),
            metadata={'name': 'hr_0', 'signal_type': SignalType.HR}
        )
        signal2 = TimeSeriesSignal(
            pd.DataFrame({'hr': [75] * 100}, index=grid_index),
            metadata={'name': 'hr_1', 'signal_type': SignalType.HR}
        )

        signals = {'hr_0': signal1, 'hr_1': signal2}

        # Mock get functions
        def get_ts_signal(key):
            if key not in signals:
                raise KeyError(f"Signal {key} not found")
            return signals[key]

        get_feat = Mock()
        add_ts = Mock()
        add_feat = Mock()

        return epoch_state, signals, get_ts_signal, get_feat, add_ts, add_feat, epoch_index

    def test_apply_multi_signal_operation_time_series(self, setup_multi_signal_operation):
        """Test multi-signal operation producing TimeSeriesSignal."""
        epoch_state, signals, get_ts, get_feat, add_ts, add_feat, _ = setup_multi_signal_operation

        # Create a mock operation that returns TimeSeriesSignal
        def mock_operation(signals, **params):
            grid_index = signals[0].get_data().index
            result_data = pd.DataFrame({'combined': [1] * len(grid_index)}, index=grid_index)
            return TimeSeriesSignal(result_data, metadata={'name': 'combined'})

        registry = {'combine_signals': (mock_operation, TimeSeriesSignal)}

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry=registry,
            get_time_series_signal=get_ts,
            get_feature=get_feat,
            add_time_series_signal=add_ts,
            add_feature=add_feat,
            epoch_state=epoch_state
        )

        result = executor.apply_multi_signal_operation(
            'combine_signals',
            ['hr_0', 'hr_1'],
            {}
        )

        assert isinstance(result, TimeSeriesSignal)

    def test_apply_multi_signal_operation_feature(self, setup_multi_signal_operation):
        """Test multi-signal operation producing Feature."""
        epoch_state, signals, get_ts, get_feat, add_ts, add_feat, epoch_index = setup_multi_signal_operation

        # Create a mock operation that returns Feature
        def mock_feature_op(signals, epoch_grid_index, parameters, global_window_length, global_step_size):
            result_data = pd.DataFrame({'mean': [70.0] * len(epoch_grid_index)}, index=epoch_grid_index)
            return Feature(result_data, metadata={'name': 'hr_features'})

        registry = {'feature_stats': (mock_feature_op, Feature)}

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry=registry,
            get_time_series_signal=get_ts,
            get_feature=get_feat,
            add_time_series_signal=add_ts,
            add_feature=add_feat,
            epoch_state=epoch_state,
            global_epoch_window_length=pd.Timedelta('10s'),
            global_epoch_step_size=pd.Timedelta('10s')
        )

        result = executor.apply_multi_signal_operation(
            'feature_stats',
            ['hr_0'],
            {}
        )

        assert isinstance(result, Feature)

    def test_apply_multi_signal_operation_not_found(self):
        """Test that missing operation raises ValueError."""
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        with pytest.raises(ValueError, match="not found in registry"):
            executor.apply_multi_signal_operation('nonexistent', ['hr_0'], {})

    def test_apply_multi_signal_operation_signal_not_found(self, setup_multi_signal_operation):
        """Test that missing signal raises ValueError."""
        epoch_state, signals, get_ts, get_feat, add_ts, add_feat, _ = setup_multi_signal_operation

        registry = {'op': (Mock(), TimeSeriesSignal)}
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry=registry,
            get_time_series_signal=get_ts,
            get_feature=get_feat,
            add_time_series_signal=add_ts,
            add_feature=add_feat
        )

        with pytest.raises(ValueError, match="not found"):
            executor.apply_multi_signal_operation('op', ['nonexistent_signal'], {})

    def test_apply_multi_signal_operation_no_epoch_state_for_feature(self, setup_multi_signal_operation):
        """Test that feature operation without epoch state raises RuntimeError."""
        _, signals, get_ts, get_feat, add_ts, add_feat, _ = setup_multi_signal_operation

        registry = {'feature_op': (Mock(), Feature)}
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry=registry,
            get_time_series_signal=get_ts,
            get_feature=get_feat,
            add_time_series_signal=add_ts,
            add_feature=add_feat,
            epoch_state=None  # No epoch state
        )

        with pytest.raises(RuntimeError, match="generate_epoch_grid must be run"):
            executor.apply_multi_signal_operation('feature_op', ['hr_0'], {})

    def test_apply_multi_signal_operation_wrong_return_type(self, setup_multi_signal_operation):
        """Test that wrong return type raises TypeError."""
        epoch_state, signals, get_ts, get_feat, add_ts, add_feat, _ = setup_multi_signal_operation

        # Operation returns wrong type
        def wrong_type_op(signals, **params):
            return "wrong_type"

        registry = {'wrong_op': (wrong_type_op, TimeSeriesSignal)}
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry=registry,
            get_time_series_signal=get_ts,
            get_feature=get_feat,
            add_time_series_signal=add_ts,
            add_feature=add_feat
        )

        with pytest.raises(TypeError, match="returned unexpected type"):
            executor.apply_multi_signal_operation('wrong_op', ['hr_0'], {})


class TestApplyAndStoreOperation:
    """Tests for apply_and_store_operation method."""

    def test_apply_and_store_operation_success(self):
        """Test successful operation application and storage."""
        grid_index = pd.date_range('2024-01-01', periods=50, freq='1s', tz=timezone.utc)
        signal = TimeSeriesSignal(
            pd.DataFrame({'hr': [70] * 50}, index=grid_index),
            metadata={'name': 'hr_0'}
        )

        # Mock the apply_operation method to return a new signal
        def mock_apply_op(op_name, **params):
            result_data = pd.DataFrame({'hr_filtered': [70] * 50}, index=grid_index)
            return TimeSeriesSignal(result_data, metadata={'name': 'hr_filtered'})

        signal.apply_operation = mock_apply_op

        get_ts = Mock(return_value=signal)
        add_ts = Mock()

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=get_ts,
            get_feature=Mock(),
            add_time_series_signal=add_ts,
            add_feature=Mock()
        )

        result = executor.apply_and_store_operation(
            'hr_0',
            'filter',
            {'low': 0.5, 'high': 4.0},
            'hr_filtered'
        )

        assert isinstance(result, TimeSeriesSignal)
        add_ts.assert_called_once()

    def test_apply_and_store_operation_signal_not_found(self):
        """Test that missing signal raises KeyError."""
        def get_ts_raises(key):
            raise KeyError(f"Signal {key} not found")

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=get_ts_raises,
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        with pytest.raises(KeyError):
            executor.apply_and_store_operation('nonexistent', 'op', {}, 'output')

    def test_apply_and_store_operation_wrong_return_type(self):
        """Test that wrong return type raises TypeError."""
        signal = Mock()
        signal.apply_operation = Mock(return_value="wrong_type")

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(return_value=signal),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        with pytest.raises(TypeError, match="returned unexpected type"):
            executor.apply_and_store_operation('sig', 'op', {}, 'output')


class TestApplyOperationToSignals:
    """Tests for apply_operation_to_signals method."""

    @pytest.fixture
    def setup_batch_operations(self):
        """Setup for batch operation tests."""
        grid_index = pd.date_range('2024-01-01', periods=50, freq='1s', tz=timezone.utc)
        signal1 = TimeSeriesSignal(
            pd.DataFrame({'hr': [70] * 50}, index=grid_index),
            metadata={'name': 'hr_0'}
        )
        signal2 = TimeSeriesSignal(
            pd.DataFrame({'hr': [75] * 50}, index=grid_index),
            metadata={'name': 'hr_1'}
        )

        signals = {'hr_0': signal1, 'hr_1': signal2}

        def get_ts(key):
            if key not in signals:
                raise KeyError(f"Signal {key} not found")
            return signals[key]

        add_ts = Mock()

        return signals, get_ts, add_ts, grid_index

    def test_apply_operation_to_signals_inplace(self, setup_batch_operations):
        """Test in-place operation on multiple signals."""
        signals, get_ts, add_ts, _ = setup_batch_operations

        # Mock apply_operation for in-place modification
        for sig in signals.values():
            sig.apply_operation = Mock()

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=get_ts,
            get_feature=Mock(),
            add_time_series_signal=add_ts,
            add_feature=Mock()
        )

        results = executor.apply_operation_to_signals(
            ['hr_0', 'hr_1'],
            'normalize',
            {},
            inplace=True
        )

        assert len(results) == 2
        # Should not call add_time_series_signal for in-place
        add_ts.assert_not_called()

    def test_apply_operation_to_signals_not_inplace(self, setup_batch_operations):
        """Test non-inplace operation on multiple signals."""
        signals, get_ts, add_ts, grid_index = setup_batch_operations

        # Mock apply_operation to return new signals
        def make_mock_apply(name):
            def mock_apply(op_name, **params):
                result_data = pd.DataFrame({f'{name}_norm': [1.0] * 50}, index=grid_index)
                return TimeSeriesSignal(result_data, metadata={'name': f'{name}_norm'})
            return mock_apply

        signals['hr_0'].apply_operation = make_mock_apply('hr_0')
        signals['hr_1'].apply_operation = make_mock_apply('hr_1')

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=get_ts,
            get_feature=Mock(),
            add_time_series_signal=add_ts,
            add_feature=Mock()
        )

        results = executor.apply_operation_to_signals(
            ['hr_0', 'hr_1'],
            'normalize',
            {},
            inplace=False,
            output_keys=['hr_0_norm', 'hr_1_norm']
        )

        assert len(results) == 2
        assert add_ts.call_count == 2

    def test_apply_operation_to_signals_missing_output_keys(self):
        """Test that missing output_keys raises ValueError when inplace=False."""
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        with pytest.raises(ValueError, match="Must provide matching output_keys"):
            executor.apply_operation_to_signals(
                ['hr_0', 'hr_1'],
                'op',
                {},
                inplace=False,
                output_keys=None
            )

    def test_apply_operation_to_signals_mismatched_output_keys(self):
        """Test that mismatched output_keys raises ValueError."""
        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock()
        )

        with pytest.raises(ValueError, match="Must provide matching output_keys"):
            executor.apply_operation_to_signals(
                ['hr_0', 'hr_1'],
                'op',
                {},
                inplace=False,
                output_keys=['output_0']  # Only one key for two signals
            )


class TestPropagateFeatureMetadata:
    """Tests for _propagate_feature_metadata helper method."""

    def test_propagate_metadata_single_source(self):
        """Test metadata propagation from single source signal."""
        grid_index = pd.date_range('2024-01-01', periods=50, freq='1s', tz=timezone.utc)
        signal = TimeSeriesSignal(
            pd.DataFrame({'hr': [70] * 50}, index=grid_index),
            metadata={'name': 'hr_0', 'signal_type': SignalType.HR}
        )

        epoch_index = pd.date_range('2024-01-01', periods=5, freq='10s', tz=timezone.utc)
        feature = Feature(
            pd.DataFrame({'mean': [70.0] * 5}, index=epoch_index),
            metadata={'name': 'hr_features'}
        )

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock(),
            feature_index_config=['signal_type']
        )

        executor._propagate_feature_metadata(feature, [signal], 'test_op')

        # signal_type should be propagated
        assert hasattr(feature.metadata, 'signal_type')

    def test_propagate_metadata_multiple_sources_common_value(self):
        """Test metadata propagation with multiple sources having common value."""
        grid_index = pd.date_range('2024-01-01', periods=50, freq='1s', tz=timezone.utc)
        signal1 = TimeSeriesSignal(
            pd.DataFrame({'hr': [70] * 50}, index=grid_index),
            metadata={'name': 'hr_0', 'signal_type': SignalType.HR}
        )
        signal2 = TimeSeriesSignal(
            pd.DataFrame({'hr': [75] * 50}, index=grid_index),
            metadata={'name': 'hr_1', 'signal_type': SignalType.HR}
        )

        epoch_index = pd.date_range('2024-01-01', periods=5, freq='10s', tz=timezone.utc)
        feature = Feature(
            pd.DataFrame({'mean': [72.5] * 5}, index=epoch_index),
            metadata={'name': 'hr_features'}
        )

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock(),
            feature_index_config=['signal_type']
        )

        executor._propagate_feature_metadata(feature, [signal1, signal2], 'test_op')

        # signal_type should be propagated as common value
        assert feature.metadata.signal_type == SignalType.HR

    def test_propagate_metadata_no_config(self):
        """Test that no propagation occurs when feature_index_config is None."""
        signal = Mock()
        feature = Mock()
        feature.metadata = Mock()
        feature.metadata.source_signal_ids = []
        feature.metadata.source_signal_keys = []

        executor = OperationExecutor(
            collection_op_registry={},
            multi_signal_registry={},
            get_time_series_signal=Mock(),
            get_feature=Mock(),
            add_time_series_signal=Mock(),
            add_feature=Mock(),
            feature_index_config=None
        )

        # Should not raise, just skip propagation
        executor._propagate_feature_metadata(feature, [signal], 'test_op')
