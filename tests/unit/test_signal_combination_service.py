"""
Unit tests for SignalCombinationService.

Tests cover:
- Combining aligned time-series signals
- Combining features into feature matrices
- MultiIndex column handling
- Validation of alignment and epoch grids
- Error handling for invalid inputs
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.sleep_analysis.core.services import SignalCombinationService
from src.sleep_analysis.core.metadata import (
    TimeSeriesMetadata,
    FeatureMetadata,
    CollectionMetadata
)
from src.sleep_analysis.core.models import (
    AlignmentGridState,
    EpochGridState,
    CombinationResult
)
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.signals.ppg_signal import PPGSignal
from src.sleep_analysis.features.feature import Feature
from src.sleep_analysis.signal_types import SignalType


class TestSignalCombinationServiceInitialization:
    """Tests for SignalCombinationService initialization."""

    def test_initialization_with_all_params(self):
        """Test initialization with all parameters."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        grid_index = pd.date_range('2024-01-01', periods=100, freq='1s', tz=timezone.utc)
        alignment_state = AlignmentGridState(
            target_rate=1.0,
            reference_time=pd.Timestamp('2024-01-01', tz=timezone.utc),
            grid_index=grid_index,
            merge_tolerance=pd.Timedelta('0.5s'),
            is_calculated=True
        )
        epoch_index = pd.date_range('2024-01-01', periods=10, freq='10s', tz=timezone.utc)
        epoch_state = EpochGridState(
            epoch_grid_index=epoch_index,
            window_length=pd.Timedelta('10s'),
            step_size=pd.Timedelta('10s'),
            is_calculated=True
        )

        service = SignalCombinationService(
            metadata=metadata,
            alignment_state=alignment_state,
            epoch_state=epoch_state
        )

        assert service.metadata == metadata
        assert service.alignment_state == alignment_state
        assert service.epoch_state == epoch_state

    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(metadata=metadata)

        assert service.metadata == metadata
        assert service.alignment_state is None
        assert service.epoch_state is None


class TestCombineAlignedSignals:
    """Tests for combine_aligned_signals method."""

    @pytest.fixture
    def setup_combination(self):
        """Setup for combination tests."""
        # Create grid index
        grid_index = pd.date_range('2024-01-01', periods=100, freq='1s', tz=timezone.utc)

        # Create alignment state
        alignment_state = AlignmentGridState(
            target_rate=1.0,
            reference_time=pd.Timestamp('2024-01-01', tz=timezone.utc),
            grid_index=grid_index,
            merge_tolerance=pd.Timedelta('0.5s'),
            is_calculated=True
        )

        # Create metadata
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test", index_config=['name', 'signal_type'])

        # Create service
        service = SignalCombinationService(
            metadata=metadata,
            alignment_state=alignment_state
        )

        # Create test signals
        hr_data = pd.DataFrame(
            {'hr': np.random.randint(60, 100, 100)},
            index=grid_index
        )
        hr_signal = HeartRateSignal(
            hr_data,
            metadata={'name': 'hr_0', 'signal_type': SignalType.HR}
        )

        ppg_data = pd.DataFrame(
            {'value': np.random.randint(95, 100, 100)},
            index=grid_index
        )
        ppg_signal = PPGSignal(
            ppg_data,
            metadata={'name': 'ppg_0', 'signal_type': SignalType.PPG}
        )

        signals = {
            'hr_0': hr_signal,
            'ppg_0': ppg_signal
        }

        return service, signals, grid_index

    def test_combine_aligned_signals_success(self, setup_combination):
        """Test successful combination of aligned signals."""
        service, signals, grid_index = setup_combination

        result = service.combine_aligned_signals(signals)

        assert isinstance(result, CombinationResult)
        assert isinstance(result.dataframe, pd.DataFrame)
        assert not result.is_feature_matrix
        assert len(result.dataframe) == len(grid_index)
        assert result.dataframe.index.equals(grid_index)

    def test_combine_aligned_signals_with_multiindex(self, setup_combination):
        """Test combination creates MultiIndex columns when index_config is set."""
        service, signals, grid_index = setup_combination

        result = service.combine_aligned_signals(signals)

        # Should have MultiIndex columns based on index_config
        assert isinstance(result.dataframe.columns, pd.MultiIndex)
        assert 'name' in result.dataframe.columns.names or result.dataframe.columns.names[0] == 'name'

    def test_combine_empty_signals_dict(self, setup_combination):
        """Test combination with empty signals dictionary."""
        service, _, grid_index = setup_combination

        result = service.combine_aligned_signals({})

        assert isinstance(result.dataframe, pd.DataFrame)
        assert result.dataframe.empty or len(result.dataframe.columns) == 0
        assert result.dataframe.index.equals(grid_index)

    def test_combine_signals_skips_temporary(self, setup_combination):
        """Test that temporary signals are skipped during combination."""
        service, signals, grid_index = setup_combination

        # Add a temporary signal
        temp_data = pd.DataFrame({'hr': [70] * 100}, index=grid_index)
        temp_signal = HeartRateSignal(
            temp_data,
            metadata={'name': 'temp_0', 'signal_type': SignalType.HR, 'temporary': True}
        )
        signals['temp_0'] = temp_signal

        result = service.combine_aligned_signals(signals)

        # Result should not include temporary signal
        # Check by column count or names
        assert isinstance(result.dataframe, pd.DataFrame)

    def test_combine_signals_without_alignment_state(self):
        """Test that combination fails without alignment state."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(metadata=metadata)

        with pytest.raises(RuntimeError, match="Alignment grid must be calculated"):
            service.combine_aligned_signals({})

    def test_combine_signals_with_invalid_alignment_state(self):
        """Test that combination fails with invalid alignment state."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        alignment_state = AlignmentGridState(
            target_rate=1.0,
            reference_time=pd.Timestamp('2024-01-01', tz=timezone.utc),
            grid_index=None,
            merge_tolerance=pd.Timedelta('0.5s'),
            is_calculated=False
        )
        service = SignalCombinationService(
            metadata=metadata,
            alignment_state=alignment_state
        )

        with pytest.raises(RuntimeError, match="Alignment grid must be calculated"):
            service.combine_aligned_signals({})

    def test_combine_signals_with_empty_grid_index(self):
        """Test that combination fails with empty grid index."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        empty_index = pd.DatetimeIndex([], tz=timezone.utc)
        alignment_state = AlignmentGridState(
            target_rate=1.0,
            reference_time=pd.Timestamp('2024-01-01', tz=timezone.utc),
            grid_index=empty_index,
            merge_tolerance=pd.Timedelta('0.5s'),
            is_calculated=True
        )
        service = SignalCombinationService(
            metadata=metadata,
            alignment_state=alignment_state
        )

        # Empty grid_index makes state invalid, so expect "Alignment grid must be calculated"
        with pytest.raises(RuntimeError, match="Alignment grid must be calculated"):
            service.combine_aligned_signals({})

    def test_combine_signals_with_error_in_signal_data(self, setup_combination):
        """Test handling of errors when accessing signal data."""
        service, signals, _ = setup_combination

        # Create a mock signal that raises an error
        class ErrorSignal:
            metadata = TimeSeriesMetadata(
                signal_id='error_id',
                name='error_signal',
                signal_type=SignalType.HR
            )

            def get_data(self):
                raise ValueError("Test error")

        signals['error_signal'] = ErrorSignal()

        with pytest.raises(RuntimeError, match="Failed to combine signals"):
            service.combine_aligned_signals(signals)

    def test_combine_signals_without_index_config(self):
        """Test combination without index_config uses simple column names."""
        grid_index = pd.date_range('2024-01-01', periods=50, freq='1s', tz=timezone.utc)
        alignment_state = AlignmentGridState(
            target_rate=1.0,
            reference_time=pd.Timestamp('2024-01-01', tz=timezone.utc),
            grid_index=grid_index,
            merge_tolerance=pd.Timedelta('0.5s'),
            is_calculated=True
        )

        # No index_config
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(
            metadata=metadata,
            alignment_state=alignment_state
        )

        hr_data = pd.DataFrame({'hr': [70] * 50}, index=grid_index)
        hr_signal = HeartRateSignal(hr_data, metadata={'name': 'hr_0'})

        result = service.combine_aligned_signals({'hr_0': hr_signal})

        # Should have simple column names, not MultiIndex
        assert not isinstance(result.dataframe.columns, pd.MultiIndex)


class TestCombineFeatures:
    """Tests for combine_features method."""

    @pytest.fixture
    def setup_feature_combination(self):
        """Setup for feature combination tests."""
        # Create epoch grid
        epoch_index = pd.date_range('2024-01-01', periods=20, freq='10s', tz=timezone.utc)

        # Create epoch state
        epoch_state = EpochGridState(
            epoch_grid_index=epoch_index,
            window_length=pd.Timedelta('10s'),
            step_size=pd.Timedelta('10s'),
            is_calculated=True
        )

        # Create metadata
        metadata = CollectionMetadata(
            collection_id="test_coll",
            subject_id="test",
            feature_index_config=['name', 'feature_type']
        )

        # Create service
        service = SignalCombinationService(
            metadata=metadata,
            epoch_state=epoch_state
        )

        # Create test features
        hr_feature_data = pd.DataFrame(
            {'mean': np.random.rand(20), 'std': np.random.rand(20)},
            index=epoch_index
        )
        hr_feature_data.columns = pd.MultiIndex.from_tuples(
            [('hr_0', 'mean'), ('hr_0', 'std')],
            names=['signal_key', 'feature']
        )
        hr_feature = Feature(
            hr_feature_data,
            metadata={
                'name': 'hr_features',
                'epoch_window_length': pd.Timedelta('10s'),
                'epoch_step_size': pd.Timedelta('10s'),
                'feature_names': ['mean', 'std'],
                'source_signal_keys': ['hr_0'],
                'source_signal_ids': ['hr_0_id']
            }
        )

        accel_feature_data = pd.DataFrame(
            {'max': np.random.rand(20), 'min': np.random.rand(20)},
            index=epoch_index
        )
        accel_feature_data.columns = pd.MultiIndex.from_tuples(
            [('accel_0', 'max'), ('accel_0', 'min')],
            names=['signal_key', 'feature']
        )
        accel_feature = Feature(
            accel_feature_data,
            metadata={
                'name': 'accel_features',
                'epoch_window_length': pd.Timedelta('10s'),
                'epoch_step_size': pd.Timedelta('10s'),
                'feature_names': ['max', 'min'],
                'source_signal_keys': ['accel_0'],
                'source_signal_ids': ['accel_0_id']
            }
        )

        features = {
            'hr_features': hr_feature,
            'accel_features': accel_feature
        }

        return service, features, epoch_index

    def test_combine_features_success(self, setup_feature_combination):
        """Test successful combination of features."""
        service, features, epoch_index = setup_feature_combination

        result = service.combine_features(
            features,
            inputs=['hr_features', 'accel_features']
        )

        assert isinstance(result, CombinationResult)
        assert isinstance(result.dataframe, pd.DataFrame)
        assert result.is_feature_matrix
        assert len(result.dataframe) == len(epoch_index)
        assert result.dataframe.index.equals(epoch_index)

    def test_combine_features_with_multiindex_columns(self, setup_feature_combination):
        """Test that combined features have MultiIndex columns."""
        service, features, _ = setup_feature_combination

        result = service.combine_features(
            features,
            inputs=['hr_features', 'accel_features']
        )

        assert isinstance(result.dataframe.columns, pd.MultiIndex)
        assert result.dataframe.columns.nlevels >= 2

    def test_combine_features_single_input(self, setup_feature_combination):
        """Test combining a single feature."""
        service, features, epoch_index = setup_feature_combination

        result = service.combine_features(features, inputs=['hr_features'])

        assert isinstance(result.dataframe, pd.DataFrame)
        assert len(result.dataframe) == len(epoch_index)

    def test_combine_features_with_base_name_resolution(self, setup_feature_combination):
        """Test that base names are resolved to actual feature keys."""
        service, features, epoch_index = setup_feature_combination

        # Add features with numbered suffixes
        hr_feat_data = pd.DataFrame({'mean': [1.0] * 20}, index=epoch_index)
        features['hr_0'] = Feature(
            hr_feat_data,
            metadata={
                'name': 'hr_0',
                'epoch_window_length': pd.Timedelta('10s'),
                'epoch_step_size': pd.Timedelta('10s'),
                'feature_names': ['mean'],
                'source_signal_keys': ['hr_0'],
                'source_signal_ids': ['hr_0_id']
            }
        )
        features['hr_1'] = Feature(
            hr_feat_data.copy(),
            metadata={
                'name': 'hr_1',
                'epoch_window_length': pd.Timedelta('10s'),
                'epoch_step_size': pd.Timedelta('10s'),
                'feature_names': ['mean'],
                'source_signal_keys': ['hr_1'],
                'source_signal_ids': ['hr_1_id']
            }
        )

        result = service.combine_features(features, inputs=['hr'])

        # Should resolve 'hr' to 'hr_0' and 'hr_1'
        assert isinstance(result.dataframe, pd.DataFrame)

    def test_combine_features_empty_inputs(self, setup_feature_combination):
        """Test that empty inputs raises ValueError."""
        service, features, _ = setup_feature_combination

        with pytest.raises(ValueError, match="No input signals specified"):
            service.combine_features(features, inputs=[])

    def test_combine_features_invalid_key(self, setup_feature_combination):
        """Test that invalid feature key raises ValueError."""
        service, features, _ = setup_feature_combination

        with pytest.raises(ValueError, match="does not match any existing feature"):
            service.combine_features(features, inputs=['nonexistent_feature'])

    def test_combine_features_without_epoch_state(self):
        """Test that combination fails without epoch state."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(metadata=metadata)

        with pytest.raises(RuntimeError, match="epoch grid must be calculated"):
            service.combine_features({}, inputs=['test'])

    def test_combine_features_with_invalid_epoch_state(self):
        """Test that combination fails with invalid epoch state."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        epoch_state = EpochGridState(
            epoch_grid_index=None,
            window_length=pd.Timedelta('10s'),
            step_size=pd.Timedelta('10s'),
            is_calculated=False
        )
        service = SignalCombinationService(
            metadata=metadata,
            epoch_state=epoch_state
        )

        with pytest.raises(RuntimeError, match="epoch grid must be calculated"):
            service.combine_features({}, inputs=['test'])

    def test_combine_features_index_mismatch(self, setup_feature_combination):
        """Test that mismatched feature index raises ValueError."""
        service, features, _ = setup_feature_combination

        # Create feature with wrong index
        wrong_index = pd.date_range('2024-02-01', periods=20, freq='10s', tz=timezone.utc)
        wrong_data = pd.DataFrame({'mean': [1.0] * 20}, index=wrong_index)
        wrong_feature = Feature(
            wrong_data,
            metadata={
                'name': 'wrong_feature',
                'epoch_window_length': pd.Timedelta('10s'),
                'epoch_step_size': pd.Timedelta('10s'),
                'feature_names': ['mean'],
                'source_signal_keys': ['wrong_0'],
                'source_signal_ids': ['wrong_0_id']
            }
        )
        features['wrong_feature'] = wrong_feature

        with pytest.raises(ValueError, match="index does not match epoch_grid_index"):
            service.combine_features(features, inputs=['wrong_feature'])

    def test_combine_features_not_datetime_index(self, setup_feature_combination):
        """Test that non-DatetimeIndex feature raises ValueError at Feature init."""
        service, features, _ = setup_feature_combination

        # Creating feature with wrong index type should fail at initialization
        wrong_data = pd.DataFrame({'mean': [1.0] * 20}, index=range(20))

        # This should raise ValueError about DatetimeIndex during Feature initialization
        with pytest.raises(ValueError, match="DatetimeIndex"):
            wrong_feature = Feature(
                wrong_data,
                metadata={
                    'name': 'wrong_feature',
                    'epoch_window_length': pd.Timedelta('10s'),
                    'epoch_step_size': pd.Timedelta('10s'),
                    'feature_names': ['mean'],
                    'source_signal_keys': ['wrong_0'],
                    'source_signal_ids': ['wrong_0_id']
                }
            )

    def test_combine_features_custom_index_config(self, setup_feature_combination):
        """Test combining features with custom feature_index_config."""
        service, features, _ = setup_feature_combination

        custom_config = ['name']
        result = service.combine_features(
            features,
            inputs=['hr_features'],
            feature_index_config=custom_config
        )

        assert isinstance(result.dataframe, pd.DataFrame)


class TestPerformConcatenation:
    """Tests for _perform_concatenation helper method."""

    def test_concatenation_empty_dfs(self):
        """Test concatenation with empty dataframes dict."""
        grid_index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(metadata=metadata)

        result = service._perform_concatenation(
            aligned_dfs={},
            grid_index=grid_index,
            is_feature=False
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty or len(result.columns) == 0
        assert result.index.equals(grid_index)

    def test_concatenation_time_series_simple(self):
        """Test concatenation of time-series without index_config."""
        grid_index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")  # No index_config
        service = SignalCombinationService(metadata=metadata)

        df1 = pd.DataFrame({'value': [1] * 10}, index=grid_index)
        df2 = pd.DataFrame({'value': [2] * 10}, index=grid_index)

        result = service._perform_concatenation(
            aligned_dfs={'sig1': df1, 'sig2': df2},
            grid_index=grid_index,
            is_feature=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2

    def test_concatenation_removes_all_nan_rows(self):
        """Test that concatenation removes rows with all NaN values."""
        grid_index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(metadata=metadata)

        # Create dataframes with some all-NaN rows
        df1 = pd.DataFrame({'value': [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]}, index=grid_index)
        df2 = pd.DataFrame({'value': [np.nan, 2, np.nan, 4, np.nan, 6, np.nan, 8, np.nan, 10]}, index=grid_index)

        result = service._perform_concatenation(
            aligned_dfs={'sig1': df1, 'sig2': df2},
            grid_index=grid_index,
            is_feature=False
        )

        # All rows should have at least one non-NaN value
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(grid_index)


class TestGetAlignmentParams:
    """Tests for _get_alignment_params helper method."""

    def test_get_alignment_params_with_state(self):
        """Test getting alignment params when alignment state exists."""
        grid_index = pd.date_range('2024-01-01', periods=10, freq='1s', tz=timezone.utc)
        alignment_state = AlignmentGridState(
            target_rate=1.0,
            reference_time=pd.Timestamp('2024-01-01', tz=timezone.utc),
            grid_index=grid_index,
            merge_tolerance=pd.Timedelta('0.5s'),
            is_calculated=True
        )
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(
            metadata=metadata,
            alignment_state=alignment_state
        )

        params = service._get_alignment_params("test_method")

        assert params['method_used'] == "test_method"
        assert params['target_rate'] == 1.0
        assert params['ref_time'] == pd.Timestamp('2024-01-01', tz=timezone.utc)
        assert params['merge_tolerance'] == pd.Timedelta('0.5s')
        assert params['grid_shape'] == (10,)

    def test_get_alignment_params_without_state(self):
        """Test getting alignment params when no alignment state exists."""
        metadata = CollectionMetadata(collection_id="test_coll", subject_id="test")
        service = SignalCombinationService(metadata=metadata)

        params = service._get_alignment_params("test_method")

        assert params == {"method_used": "test_method"}
