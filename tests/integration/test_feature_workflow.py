"""
Integration tests for complete feature extraction workflows.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.sleep_analysis.core.signal_collection import SignalCollection
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.signals.magnitude_signal import MagnitudeSignal
from src.sleep_analysis.workflows.workflow_executor import WorkflowExecutor
from src.sleep_analysis.operations.feature_extraction import (
    clear_feature_cache,
    get_cache_stats
)


class TestFeatureExtractionWorkflow:
    """Integration tests for complete feature extraction workflows."""

    @pytest.fixture
    def sample_signals(self):
        """Create sample time-series signals for testing."""
        # Create timestamp index (timezone-aware to match epoch grid)
        index = pd.date_range('2024-01-01 00:00:00', periods=120, freq='1s', tz='UTC')

        # Signal A: Simulated heart rate
        hr_data = pd.DataFrame({
            'hr': 60 + 10 * np.sin(np.arange(120) * 0.1) + np.random.randn(120)
        }, index=index)

        signal_a = HeartRateSignal(
            data=hr_data,
            metadata={
                'name': 'heart_rate',
                'signal_id': 'hr_001',
                'sampling_rate': 1.0,
                'units': 'bpm'
            }
        )

        # Signal B: Simulated accelerometer magnitude
        accel_data = pd.DataFrame({
            'magnitude': 1.0 + 0.2 * np.random.randn(120)
        }, index=index)

        signal_b = MagnitudeSignal(
            data=accel_data,
            metadata={
                'name': 'accel_mag',
                'signal_id': 'acc_001',
                'sampling_rate': 1.0,
                'units': 'g'
            }
        )

        return {'heart_rate': signal_a, 'accel_mag': signal_b}

    @pytest.fixture
    def collection_with_signals(self, sample_signals):
        """Create a SignalCollection with sample signals."""
        collection = SignalCollection()

        # Set collection metadata with epoch grid config
        collection.metadata.epoch_grid_config = {
            'window_length': pd.Timedelta('30s'),
            'step_size': pd.Timedelta('15s')
        }

        # Add signals
        for key, signal in sample_signals.items():
            collection.add_time_series_signal(key, signal)

        return collection

    def setup_method(self):
        """Clear feature cache before each test."""
        clear_feature_cache()

    def test_generate_epoch_grid(self, collection_with_signals):
        """Test epoch grid generation."""
        collection = collection_with_signals

        # Generate epoch grid
        collection.generate_epoch_grid()

        assert collection._epoch_grid_calculated is True
        assert collection.epoch_grid_index is not None
        assert len(collection.epoch_grid_index) > 0
        assert isinstance(collection.epoch_grid_index, pd.DatetimeIndex)

        # Check step size
        expected_steps = (120 - 30) // 15 + 1  # Approximate
        assert len(collection.epoch_grid_index) >= expected_steps - 1

    def test_feature_extraction_basic_stats(self, collection_with_signals):
        """Test basic feature extraction with statistics."""
        collection = collection_with_signals

        # Generate epoch grid
        collection.generate_epoch_grid()

        # Extract features from heart rate signal
        feature = collection.apply_multi_signal_operation(
            operation_name='feature_statistics',
            input_signal_keys=['heart_rate'],
            parameters={
                'aggregations': ['mean', 'std', 'min', 'max']
            }
        )

        # Store the feature
        collection.add_feature('hr_stats', feature)

        assert feature is not None
        assert 'hr_stats' in collection.features

        # Check feature properties
        feature_data = feature.get_data()
        assert isinstance(feature_data, pd.DataFrame)
        assert isinstance(feature_data.index, pd.DatetimeIndex)
        assert len(feature_data) == len(collection.epoch_grid_index)

        # Check columns (should have MultiIndex: signal_key, feature)
        assert isinstance(feature_data.columns, pd.MultiIndex)
        assert feature_data.columns.names == ['signal_key', 'feature']

        # Check feature names
        feature_names = set(feature_data.columns.get_level_values('feature'))
        expected_features = {'hr_mean', 'hr_std', 'hr_min', 'hr_max'}
        assert feature_names == expected_features

    def test_feature_extraction_multiple_signals(self, collection_with_signals):
        """Test feature extraction from multiple signals."""
        collection = collection_with_signals

        # Generate epoch grid
        collection.generate_epoch_grid()

        # Extract features from both signals
        feature = collection.apply_multi_signal_operation(
            operation_name='feature_statistics',
            input_signal_keys=['heart_rate', 'accel_mag'],
            parameters={
                'aggregations': ['mean', 'std']
            }
        )
        collection.add_feature('combined_stats', feature)

        feature_data = feature.get_data()

        # Check that features from both signals are present
        signal_keys = set(feature_data.columns.get_level_values('signal_key'))
        assert signal_keys == {'heart_rate', 'accel_mag'}

        # Each signal should have mean and std features
        hr_features = [col for col in feature_data.columns
                      if col[0] == 'heart_rate']
        assert len(hr_features) == 2  # mean and std

        accel_features = [col for col in feature_data.columns
                         if col[0] == 'accel_mag']
        assert len(accel_features) == 2  # mean and std

    def test_combine_features(self, collection_with_signals):
        """Test combining multiple features into a matrix."""
        collection = collection_with_signals

        # Generate epoch grid
        collection.generate_epoch_grid()

        # Extract features separately
        hr_feature = collection.apply_multi_signal_operation(
            operation_name='feature_statistics',
            input_signal_keys=['heart_rate'],
            parameters={'aggregations': ['mean', 'std']}
        )
        collection.add_feature('hr_features', hr_feature)

        accel_feature = collection.apply_multi_signal_operation(
            operation_name='feature_statistics',
            input_signal_keys=['accel_mag'],
            parameters={'aggregations': ['mean', 'std']}
        )
        collection.add_feature('accel_features', accel_feature)

        # Combine features
        collection.combine_features(inputs=['hr_features', 'accel_features'])

        # Check combined feature matrix
        combined_df = collection.get_combined_feature_matrix()
        assert combined_df is not None
        assert isinstance(combined_df, pd.DataFrame)
        assert isinstance(combined_df.columns, pd.MultiIndex)

        # Should have features from both sources
        assert len(combined_df.columns) > 0

    def test_feature_caching(self, collection_with_signals):
        """Test that feature caching works correctly."""
        collection = collection_with_signals

        # Generate epoch grid
        collection.generate_epoch_grid()

        # Clear cache
        clear_feature_cache()
        initial_stats = get_cache_stats()
        assert initial_stats['size'] == 0

        # Extract features - should be cached
        feature1 = collection.apply_multi_signal_operation(
            operation_name='feature_statistics',
            input_signal_keys=['heart_rate'],
            parameters={'aggregations': ['mean']}
        )
        collection.add_feature('hr_mean_1', feature1)

        # Cache should now have one entry
        cache_stats = get_cache_stats()
        assert cache_stats['size'] == 1

        # Extract same features again - should use cache
        feature2 = collection.apply_multi_signal_operation(
            operation_name='feature_statistics',
            input_signal_keys=['heart_rate'],
            parameters={'aggregations': ['mean']}
        )
        collection.add_feature('hr_mean_2', feature2)

        # Cache size should still be 1 (same computation)
        cache_stats = get_cache_stats()
        assert cache_stats['size'] == 1

        # Results should be identical
        pd.testing.assert_frame_equal(
            feature1.get_data(),
            feature2.get_data()
        )

    def test_workflow_with_validation(self, collection_with_signals):
        """Test complete workflow with validation enabled."""
        executor = WorkflowExecutor(container=collection_with_signals, strict_validation=True)

        # Valid workflow configuration
        workflow_config = {
            'collection_settings': {
                'epoch_grid_config': {
                    'window_length': '30s',
                    'step_size': '15s'
                }
            },
            'steps': [
                {
                    'operation': 'generate_epoch_grid',
                    'type': 'collection'
                },
                {
                    'operation': 'feature_statistics',
                    'inputs': ['heart_rate'],
                    'output': 'hr_features',
                    'parameters': {
                        'aggregations': ['mean', 'std', 'min', 'max']
                    }
                }
            ]
        }

        # Execute workflow
        executor.execute_workflow(workflow_config)

        # Check that features were created
        assert 'hr_features' in executor.container.features

    def test_workflow_validation_errors(self, collection_with_signals):
        """Test that workflow validation catches errors."""
        executor = WorkflowExecutor(container=collection_with_signals, strict_validation=True)

        # Invalid workflow: feature extraction before epoch grid
        invalid_workflow = {
            'collection_settings': {},
            'steps': [
                {
                    'operation': 'feature_statistics',  # Requires epoch grid
                    'inputs': ['heart_rate'],
                    'output': 'hr_features',
                    'parameters': {
                        'aggregations': ['mean']
                    }
                }
            ]
        }

        # Should raise validation error
        with pytest.raises(ValueError, match="requires epoch grid"):
            executor.execute_workflow(invalid_workflow)

    def test_lazy_feature_in_workflow(self, sample_signals):
        """Test lazy feature evaluation within a workflow context."""
        # Note: This is a conceptual test - actual lazy feature integration
        # would require modifications to the workflow executor

        from src.sleep_analysis.features.feature import Feature

        # Create a lazy feature
        index = pd.date_range('2024-01-01', periods=10, freq='30s')

        def compute_feature():
            return pd.DataFrame({
                ('test', 'value'): np.random.randn(10)
            }, index=index)

        lazy_feature = Feature(
            metadata={
                'name': 'lazy_test',
                'epoch_window_length': pd.Timedelta('30s'),
                'epoch_step_size': pd.Timedelta('15s'),
                'feature_names': ['value'],
                'source_signal_keys': ['test'],
                'source_signal_ids': ['test_id']
            },
            lazy=True,
            computation_function=compute_feature
        )

        assert lazy_feature.is_lazy() is True
        assert lazy_feature.is_computed() is False

        # Access data - triggers computation
        data = lazy_feature.get_data()
        assert lazy_feature.is_computed() is True
        assert len(data) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
