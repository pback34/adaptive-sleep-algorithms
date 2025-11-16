"""
Unit tests for feature extraction operations and Feature class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.sleep_analysis.features.feature import Feature
from src.sleep_analysis.operations import feature_extraction
from src.sleep_analysis.core.metadata import FeatureMetadata, FeatureType, OperationInfo


class TestFeatureClass:
    """Tests for the Feature class including lazy evaluation."""

    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data for testing."""
        index = pd.date_range('2024-01-01', periods=10, freq='30s')
        data = pd.DataFrame({
            ('signal_a', 'mean'): np.random.randn(10),
            ('signal_a', 'std'): np.random.rand(10),
            ('signal_b', 'mean'): np.random.randn(10),
            ('signal_b', 'std'): np.random.rand(10),
        }, index=index)
        data.columns = pd.MultiIndex.from_tuples(
            data.columns, names=['signal_key', 'feature']
        )
        return data

    @pytest.fixture
    def sample_metadata(self):
        """Create sample feature metadata."""
        return {
            'name': 'test_feature',
            'epoch_window_length': pd.Timedelta('30s'),
            'epoch_step_size': pd.Timedelta('30s'),
            'feature_names': ['mean', 'std'],
            'feature_type': FeatureType.STATISTICAL,
            'source_signal_keys': ['signal_a', 'signal_b'],
            'source_signal_ids': ['id_a', 'id_b'],
            'operations': [OperationInfo('compute_feature_statistics', {})]
        }

    def test_feature_eager_initialization(self, sample_feature_data, sample_metadata):
        """Test creating a Feature with eager evaluation (traditional mode)."""
        feature = Feature(data=sample_feature_data, metadata=sample_metadata)

        assert feature.is_lazy() is False
        assert feature.is_computed() is True
        assert feature.metadata.name == 'test_feature'
        assert len(feature.get_data()) == 10
        assert isinstance(feature.get_data().index, pd.DatetimeIndex)

    def test_feature_lazy_initialization(self, sample_feature_data, sample_metadata):
        """Test creating a Feature with lazy evaluation."""
        def compute_func():
            return sample_feature_data

        feature = Feature(
            metadata=sample_metadata,
            lazy=True,
            computation_function=compute_func
        )

        assert feature.is_lazy() is True
        assert feature.is_computed() is False

        # Data should be computed on first access
        data = feature.get_data()
        assert feature.is_computed() is True
        assert len(data) == 10
        pd.testing.assert_frame_equal(data, sample_feature_data)

    def test_feature_lazy_with_args(self, sample_metadata):
        """Test lazy evaluation with computation arguments."""
        def compute_with_args(value, multiplier):
            index = pd.date_range('2024-01-01', periods=5, freq='30s')
            data = pd.DataFrame({
                ('test', 'result'): [value * multiplier] * 5
            }, index=index)
            data.columns = pd.MultiIndex.from_tuples(
                data.columns, names=['signal_key', 'feature']
            )
            return data

        metadata = sample_metadata.copy()
        metadata['feature_names'] = ['result']
        metadata['source_signal_keys'] = ['test']
        metadata['source_signal_ids'] = ['test_id']

        feature = Feature(
            metadata=metadata,
            lazy=True,
            computation_function=compute_with_args,
            computation_args={'value': 10, 'multiplier': 5}
        )

        data = feature.get_data()
        assert all(data[('test', 'result')] == 50)

    def test_feature_lazy_validation(self, sample_metadata):
        """Test validation for lazy feature initialization."""
        # Should fail without computation_function
        with pytest.raises(ValueError, match="lazy=True requires computation_function"):
            Feature(metadata=sample_metadata, lazy=True)

        # Should fail with non-callable computation_function
        with pytest.raises(TypeError, match="computation_function must be callable"):
            Feature(
                metadata=sample_metadata,
                lazy=True,
                computation_function="not_callable"
            )

    def test_feature_clear_data_eager(self, sample_feature_data, sample_metadata):
        """Test clearing data from an eager feature."""
        feature = Feature(data=sample_feature_data, metadata=sample_metadata)

        feature.clear_data()
        assert feature._data is None

        # Should raise error when accessing cleared eager feature
        with pytest.raises(RuntimeError, match="data for .* has been cleared"):
            feature.get_data()

    def test_feature_clear_data_lazy(self, sample_metadata):
        """Test clearing data from a lazy feature (can be recomputed)."""
        call_count = 0

        def compute_func():
            nonlocal call_count
            call_count += 1
            index = pd.date_range('2024-01-01', periods=3, freq='30s')
            data = pd.DataFrame({
                ('test', 'value'): [call_count] * 3
            }, index=index)
            data.columns = pd.MultiIndex.from_tuples(
                data.columns, names=['signal_key', 'feature']
            )
            return data

        metadata = sample_metadata.copy()
        metadata['feature_names'] = ['value']
        metadata['source_signal_keys'] = ['test']
        metadata['source_signal_ids'] = ['test_id']

        feature = Feature(
            metadata=metadata,
            lazy=True,
            computation_function=compute_func
        )

        # First access - computes data
        data1 = feature.get_data()
        assert call_count == 1
        assert all(data1[('test', 'value')] == 1)

        # Clear and access again - recomputes data
        feature.clear_data()
        assert feature.is_computed() is False

        data2 = feature.get_data()
        assert call_count == 2
        assert all(data2[('test', 'value')] == 2)

    def test_feature_repr(self, sample_feature_data, sample_metadata):
        """Test Feature string representation."""
        # Eager feature
        feature_eager = Feature(data=sample_feature_data, metadata=sample_metadata)
        repr_str = repr(feature_eager)
        assert 'test_feature' in repr_str
        assert 'shape=(10, 4)' in repr_str

        # Lazy feature (not computed)
        feature_lazy = Feature(
            metadata=sample_metadata,
            lazy=True,
            computation_function=lambda: sample_feature_data
        )
        repr_str = repr(feature_lazy)
        assert 'Lazy(not computed)' in repr_str


class TestFeatureCaching:
    """Tests for the feature caching decorator."""

    def setup_method(self):
        """Clear cache before each test."""
        feature_extraction.clear_feature_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        feature_extraction.clear_feature_cache()

    def test_cache_enable_disable(self):
        """Test enabling and disabling the cache."""
        assert feature_extraction._CACHE_ENABLED is True

        feature_extraction.enable_feature_cache(False)
        assert feature_extraction._CACHE_ENABLED is False

        feature_extraction.enable_feature_cache(True)
        assert feature_extraction._CACHE_ENABLED is True

    def test_cache_clear(self):
        """Test clearing the cache."""
        # Add something to cache manually
        feature_extraction._FEATURE_CACHE['test_key'] = 'test_value'
        assert len(feature_extraction._FEATURE_CACHE) == 1

        feature_extraction.clear_feature_cache()
        assert len(feature_extraction._FEATURE_CACHE) == 0

    def test_cache_stats(self):
        """Test getting cache statistics."""
        feature_extraction.clear_feature_cache()
        stats = feature_extraction.get_cache_stats()

        assert stats['enabled'] is True
        assert stats['size'] == 0
        assert stats['keys'] == []

        # Add items
        feature_extraction._FEATURE_CACHE['key1'] = 'value1'
        feature_extraction._FEATURE_CACHE['key2'] = 'value2'

        stats = feature_extraction.get_cache_stats()
        assert stats['size'] == 2
        assert set(stats['keys']) == {'key1', 'key2'}

    def test_compute_cache_key(self):
        """Test cache key computation."""
        key1 = feature_extraction._compute_cache_key(
            signal_ids=['sig1', 'sig2'],
            operation_name='test_op',
            parameters={'param1': 'value1'},
            epoch_grid_hash='abc123'
        )

        # Same inputs should produce same key
        key2 = feature_extraction._compute_cache_key(
            signal_ids=['sig1', 'sig2'],
            operation_name='test_op',
            parameters={'param1': 'value1'},
            epoch_grid_hash='abc123'
        )
        assert key1 == key2

        # Different inputs should produce different keys
        key3 = feature_extraction._compute_cache_key(
            signal_ids=['sig1', 'sig3'],  # Different signal
            operation_name='test_op',
            parameters={'param1': 'value1'},
            epoch_grid_hash='abc123'
        )
        assert key1 != key3

        # Order of signal_ids shouldn't matter (they're sorted)
        key4 = feature_extraction._compute_cache_key(
            signal_ids=['sig2', 'sig1'],  # Different order
            operation_name='test_op',
            parameters={'param1': 'value1'},
            epoch_grid_hash='abc123'
        )
        assert key1 == key4


class TestFeatureMetadata:
    """Tests for FeatureMetadata class."""

    def test_feature_metadata_creation(self):
        """Test creating FeatureMetadata with required fields."""
        metadata = FeatureMetadata(
            feature_id='test_id',
            name='test_feature',
            epoch_window_length=pd.Timedelta('30s'),
            epoch_step_size=pd.Timedelta('15s'),
            feature_names=['mean', 'std', 'max'],
            feature_type=FeatureType.STATISTICAL,
            source_signal_keys=['signal_a'],
            source_signal_ids=['id_a']
        )

        assert metadata.feature_id == 'test_id'
        assert metadata.name == 'test_feature'
        assert metadata.epoch_window_length == pd.Timedelta('30s')
        assert metadata.feature_type == FeatureType.STATISTICAL
        assert metadata.feature_names == ['mean', 'std', 'max']

    def test_feature_metadata_with_operations(self):
        """Test FeatureMetadata with operation history."""
        op1 = OperationInfo('filter_lowpass', {'cutoff': 10.0})
        op2 = OperationInfo('compute_stats', {'aggs': ['mean', 'std']})

        metadata = FeatureMetadata(
            feature_id='test_id',
            name='test_feature',
            epoch_window_length=pd.Timedelta('30s'),
            epoch_step_size=pd.Timedelta('15s'),
            feature_names=['mean', 'std'],
            feature_type=FeatureType.STATISTICAL,
            source_signal_keys=['signal_a'],
            source_signal_ids=['id_a'],
            operations=[op1, op2]
        )

        assert len(metadata.operations) == 2
        assert metadata.operations[0].operation_name == 'filter_lowpass'
        assert metadata.operations[1].parameters == {'aggs': ['mean', 'std']}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
