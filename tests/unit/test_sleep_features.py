"""
Unit tests for sleep-specific feature extraction operations:
- HRV (Heart Rate Variability) features
- Movement/Activity features from accelerometer
- Multi-signal correlation features
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.sleep_analysis.operations.feature_extraction import (
    compute_hrv_features,
    compute_movement_features,
    compute_correlation_features,
    _compute_hrv_time_domain,
    _compute_hrv_from_heart_rate,
    _compute_movement_features,
    _compute_correlation_features,
)
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.signals.accelerometer_signal import AccelerometerSignal
from src.sleep_analysis.core.metadata import FeatureType
from src.sleep_analysis.features.feature import Feature


class TestHRVFeatures:
    """Tests for HRV feature extraction functions."""

    @pytest.fixture
    def rr_interval_data(self):
        """Create sample RR interval data (in milliseconds)."""
        # Typical RR intervals for resting heart rate ~60 bpm
        # RR interval = 60000 / HR, so for 60 bpm = 1000ms
        rr_intervals = pd.Series([
            950, 980, 1000, 1020, 1050, 980, 960, 1010, 1030, 990,
            1000, 970, 1040, 1020, 990, 1000, 980, 1010, 1020, 1000
        ])
        return rr_intervals

    @pytest.fixture
    def hr_dataframe(self):
        """Create sample heart rate DataFrame."""
        index = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1s', tz='UTC')
        # Simulate heart rate varying between 55-75 bpm
        hr_values = 65 + 10 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 2
        return pd.DataFrame({'hr': hr_values}, index=index)

    @pytest.fixture
    def hr_signal(self, hr_dataframe):
        """Create a HeartRateSignal for testing."""
        return HeartRateSignal(
            data=hr_dataframe,
            metadata={
                'name': 'test_hr',
                'signal_type': 'heart_rate',
                'sample_rate': '1Hz'
            }
        )

    def test_compute_hrv_time_domain_valid_data(self, rr_interval_data):
        """Test HRV time-domain metrics with valid RR interval data."""
        result = _compute_hrv_time_domain(rr_interval_data)

        # Check all expected keys are present
        assert 'rr_mean' in result
        assert 'rr_std' in result
        assert 'sdnn' in result
        assert 'rmssd' in result
        assert 'pnn50' in result
        assert 'sdsd' in result

        # Check values are reasonable
        assert 900 < result['rr_mean'] < 1100  # Mean RR should be ~1000ms
        assert result['rr_std'] > 0  # Should have some variability
        assert result['sdnn'] > 0
        assert result['rmssd'] > 0
        assert 0 <= result['pnn50'] <= 100  # Percentage
        assert result['sdsd'] > 0

        # SDNN should equal rr_std for this implementation
        assert result['sdnn'] == result['rr_std']

    def test_compute_hrv_time_domain_insufficient_data(self):
        """Test HRV with insufficient data points."""
        # Empty series
        result_empty = _compute_hrv_time_domain(pd.Series([]))
        assert all(np.isnan(v) for v in result_empty.values())

        # Single data point
        result_single = _compute_hrv_time_domain(pd.Series([1000]))
        assert all(np.isnan(v) for v in result_single.values())

    def test_compute_hrv_from_heart_rate_valid_data(self, hr_dataframe):
        """Test HRV approximation from heart rate data."""
        result = _compute_hrv_from_heart_rate(hr_dataframe)

        # Check all expected keys are present
        assert 'hr_mean' in result
        assert 'hr_std' in result
        assert 'hr_cv' in result
        assert 'hr_range' in result

        # Check values are reasonable
        assert 50 < result['hr_mean'] < 80  # Should be around 65 bpm
        assert result['hr_std'] > 0  # Should have variability
        assert result['hr_cv'] > 0  # Coefficient of variation > 0
        assert result['hr_range'] > 0  # Range should be positive

    def test_compute_hrv_from_heart_rate_empty_data(self):
        """Test HRV approximation with empty/invalid data."""
        # Empty DataFrame
        result_empty = _compute_hrv_from_heart_rate(pd.DataFrame())
        assert all(np.isnan(v) for v in result_empty.values())

        # DataFrame without 'hr' column
        result_no_hr = _compute_hrv_from_heart_rate(pd.DataFrame({'other': [1, 2, 3]}))
        assert all(np.isnan(v) for v in result_no_hr.values())

    def test_compute_hrv_features_wrapper(self, hr_signal):
        """Test the compute_hrv_features wrapper function."""
        # Create epoch grid
        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=3,
            freq='30s',
            tz='UTC'
        )

        parameters = {
            'hrv_metrics': ['hr_mean', 'hr_std', 'hr_cv'],
            'use_rr_intervals': False
        }

        result = compute_hrv_features(
            signals=[hr_signal],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('30s'),
            global_step_size=pd.Timedelta('30s')
        )

        # Check result is a Feature object
        assert isinstance(result, Feature)
        assert result.metadata.feature_type == FeatureType.HRV

        # Check data structure
        data = result.get_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3  # Should have 3 epochs
        assert isinstance(data.columns, pd.MultiIndex)

        # Check feature names
        assert 'hr_mean' in result.metadata.feature_names
        assert 'hr_std' in result.metadata.feature_names
        assert 'hr_cv' in result.metadata.feature_names

    def test_compute_hrv_features_all_metrics(self, hr_signal):
        """Test HRV features with 'all' metrics option."""
        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=2,
            freq='30s',
            tz='UTC'
        )

        parameters = {
            'hrv_metrics': 'all',
            'use_rr_intervals': False
        }

        result = compute_hrv_features(
            signals=[hr_signal],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('30s'),
            global_step_size=pd.Timedelta('30s')
        )

        # Should get all HR-based metrics
        expected_metrics = ['hr_mean', 'hr_std', 'hr_cv', 'hr_range']
        for metric in expected_metrics:
            assert metric in result.metadata.feature_names


class TestMovementFeatures:
    """Tests for movement/activity feature extraction from accelerometer."""

    @pytest.fixture
    def accel_data(self):
        """Create sample accelerometer data."""
        index = pd.date_range('2024-01-01 00:00:00', periods=100, freq='0.1s', tz='UTC')

        # Simulate periods of activity and stillness
        t = np.linspace(0, 10, 100)
        x = np.sin(t) * 100 + np.random.randn(100) * 10  # mg units
        y = np.cos(t) * 100 + np.random.randn(100) * 10
        z = 1000 + np.random.randn(100) * 20  # Mostly gravity on z-axis

        return pd.DataFrame({'x': x, 'y': y, 'z': z}, index=index)

    @pytest.fixture
    def accel_signal(self, accel_data):
        """Create an AccelerometerSignal for testing."""
        return AccelerometerSignal(
            data=accel_data,
            metadata={
                'name': 'test_accel',
                'signal_type': 'accelerometer',
                'sample_rate': '10Hz'
            }
        )

    def test_compute_movement_features_valid_data(self, accel_data):
        """Test movement feature computation with valid accelerometer data."""
        result = _compute_movement_features(accel_data)

        # Check all expected keys are present
        expected_keys = [
            'magnitude_mean', 'magnitude_std', 'magnitude_max',
            'activity_count', 'stillness_ratio',
            'x_std', 'y_std', 'z_std'
        ]
        for key in expected_keys:
            assert key in result

        # Check values are reasonable
        assert result['magnitude_mean'] > 0
        assert result['magnitude_std'] > 0
        assert result['magnitude_max'] > result['magnitude_mean']
        assert result['activity_count'] >= 0
        assert 0 <= result['stillness_ratio'] <= 100
        assert result['x_std'] > 0
        assert result['y_std'] > 0
        assert result['z_std'] > 0

        # Sanity check: activity_count + stillness should account for all samples
        total_samples = len(accel_data)
        still_samples = (result['stillness_ratio'] / 100) * total_samples
        active_samples = result['activity_count']
        assert abs((active_samples + still_samples) - total_samples) < 1  # Allow rounding error

    def test_compute_movement_features_missing_columns(self):
        """Test movement features with missing required columns."""
        # Missing z column
        incomplete_data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = _compute_movement_features(incomplete_data)
        assert all(np.isnan(v) for v in result.values())

        # Empty DataFrame
        result_empty = _compute_movement_features(pd.DataFrame())
        assert all(np.isnan(v) for v in result_empty.values())

    def test_compute_movement_features_insufficient_data(self):
        """Test movement features with too few data points."""
        small_data = pd.DataFrame({
            'x': [1], 'y': [2], 'z': [3]
        })
        result = _compute_movement_features(small_data)
        assert all(np.isnan(v) for v in result.values())

    def test_compute_movement_features_wrapper(self, accel_signal):
        """Test the compute_movement_features wrapper function."""
        # Create epoch grid
        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=3,
            freq='3s',
            tz='UTC'
        )

        parameters = {
            'movement_metrics': 'all'
        }

        result = compute_movement_features(
            signals=[accel_signal],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('3s'),
            global_step_size=pd.Timedelta('3s')
        )

        # Check result is a Feature object
        assert isinstance(result, Feature)
        assert result.metadata.feature_type == FeatureType.MOVEMENT

        # Check data structure
        data = result.get_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3  # Should have 3 epochs
        assert isinstance(data.columns, pd.MultiIndex)

        # Check all expected features are present
        expected_features = [
            'magnitude_mean', 'magnitude_std', 'magnitude_max',
            'activity_count', 'stillness_ratio',
            'x_std', 'y_std', 'z_std'
        ]
        for feature in expected_features:
            assert feature in result.metadata.feature_names

    def test_compute_movement_features_selective_metrics(self, accel_signal):
        """Test movement features with selective metric choice."""
        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=2,
            freq='3s',
            tz='UTC'
        )

        parameters = {
            'movement_metrics': ['magnitude_mean', 'stillness_ratio']
        }

        result = compute_movement_features(
            signals=[accel_signal],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('3s'),
            global_step_size=pd.Timedelta('3s')
        )

        # Should only have selected metrics
        assert 'magnitude_mean' in result.metadata.feature_names
        assert 'stillness_ratio' in result.metadata.feature_names
        # Should not have other metrics
        assert 'activity_count' not in result.metadata.feature_names or \
               not any('activity_count' in str(col) for col in result.get_data().columns)


class TestCorrelationFeatures:
    """Tests for multi-signal correlation feature extraction."""

    @pytest.fixture
    def correlated_signals(self):
        """Create two signals with known correlation."""
        index = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1s', tz='UTC')

        # Create two positively correlated signals
        x = np.linspace(0, 10, 100)
        signal1_data = pd.DataFrame({
            'value': x + np.random.randn(100) * 0.5
        }, index=index)
        signal2_data = pd.DataFrame({
            'value': 2 * x + 3 + np.random.randn(100) * 0.5
        }, index=index)

        signal1 = HeartRateSignal(
            data=signal1_data.rename(columns={'value': 'hr'}),
            metadata={'name': 'signal1', 'signal_type': 'heart_rate'}
        )
        signal2 = HeartRateSignal(
            data=signal2_data.rename(columns={'value': 'hr'}),
            metadata={'name': 'signal2', 'signal_type': 'heart_rate'}
        )

        return signal1, signal2

    def test_compute_correlation_features_valid_data(self):
        """Test correlation computation with valid aligned data."""
        index = pd.date_range('2024-01-01', periods=50, freq='1s')

        # Perfect positive correlation
        signal1_data = pd.DataFrame({'col1': np.arange(50)}, index=index)
        signal2_data = pd.DataFrame({'col2': np.arange(50) * 2}, index=index)

        result = _compute_correlation_features(
            signal1_data, signal2_data,
            'col1', 'col2',
            method='pearson'
        )

        assert 'pearson_corr' in result
        assert result['pearson_corr'] > 0.99  # Should be near perfect correlation

    def test_compute_correlation_features_negative_correlation(self):
        """Test correlation with negatively correlated data."""
        index = pd.date_range('2024-01-01', periods=50, freq='1s')

        # Negative correlation
        signal1_data = pd.DataFrame({'col1': np.arange(50)}, index=index)
        signal2_data = pd.DataFrame({'col2': -np.arange(50)}, index=index)

        result = _compute_correlation_features(
            signal1_data, signal2_data,
            'col1', 'col2',
            method='pearson'
        )

        assert 'pearson_corr' in result
        assert result['pearson_corr'] < -0.99  # Should be near perfect negative correlation

    def test_compute_correlation_features_missing_columns(self):
        """Test correlation with missing columns."""
        signal1_data = pd.DataFrame({'col1': [1, 2, 3]})
        signal2_data = pd.DataFrame({'col2': [4, 5, 6]})

        # Request non-existent column
        result = _compute_correlation_features(
            signal1_data, signal2_data,
            'col1', 'missing_col',
            method='pearson'
        )

        assert np.isnan(result['pearson_corr'])

    def test_compute_correlation_features_insufficient_overlap(self):
        """Test correlation with insufficient overlapping data points."""
        index1 = pd.date_range('2024-01-01', periods=2, freq='1s')
        index2 = pd.date_range('2024-01-01 00:00:10', periods=2, freq='1s')

        # No overlap in timestamps
        signal1_data = pd.DataFrame({'col1': [1, 2]}, index=index1)
        signal2_data = pd.DataFrame({'col2': [3, 4]}, index=index2)

        result = _compute_correlation_features(
            signal1_data, signal2_data,
            'col1', 'col2',
            method='pearson'
        )

        assert np.isnan(result['pearson_corr'])

    def test_compute_correlation_features_wrapper(self, correlated_signals):
        """Test the compute_correlation_features wrapper function."""
        signal1, signal2 = correlated_signals

        # Create epoch grid
        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=3,
            freq='30s',
            tz='UTC'
        )

        parameters = {
            'signal1_column': 'hr',
            'signal2_column': 'hr',
            'method': 'pearson'
        }

        result = compute_correlation_features(
            signals=[signal1, signal2],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('30s'),
            global_step_size=pd.Timedelta('30s')
        )

        # Check result is a Feature object
        assert isinstance(result, Feature)
        assert result.metadata.feature_type == FeatureType.CORRELATION

        # Check data structure
        data = result.get_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3  # Should have 3 epochs
        assert isinstance(data.columns, pd.MultiIndex)

        # Check correlation values are valid
        assert 'pearson_corr' in result.metadata.feature_names

    def test_compute_correlation_features_different_methods(self, correlated_signals):
        """Test correlation with different correlation methods."""
        signal1, signal2 = correlated_signals

        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=2,
            freq='30s',
            tz='UTC'
        )

        # Test Pearson (doesn't require scipy)
        parameters = {
            'signal1_column': 'hr',
            'signal2_column': 'hr',
            'method': 'pearson'
        }

        result = compute_correlation_features(
            signals=[signal1, signal2],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('30s'),
            global_step_size=pd.Timedelta('30s')
        )

        assert 'pearson_corr' in result.metadata.feature_names
        data = result.get_data()
        # Should have correlation values (not all NaN) for pearson
        assert not data.isna().all().all()

        # Test other methods only if scipy is available
        try:
            import scipy
            methods_to_test = ['spearman', 'kendall']
        except ImportError:
            pytest.skip("scipy not installed, skipping spearman/kendall correlation tests")
            return

        for method in methods_to_test:
            parameters = {
                'signal1_column': 'hr',
                'signal2_column': 'hr',
                'method': method
            }

            result = compute_correlation_features(
                signals=[signal1, signal2],
                epoch_grid_index=epoch_grid,
                parameters=parameters,
                global_window_length=pd.Timedelta('30s'),
                global_step_size=pd.Timedelta('30s')
            )

            assert f'{method}_corr' in result.metadata.feature_names
            data = result.get_data()
            # Should have correlation values (not all NaN)
            assert not data.isna().all().all()

    def test_compute_correlation_features_wrong_signal_count(self, correlated_signals):
        """Test that correlation requires exactly 2 signals."""
        signal1, _ = correlated_signals

        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=2,
            freq='30s',
            tz='UTC'
        )

        parameters = {
            'signal1_column': 'hr',
            'signal2_column': 'hr',
            'method': 'pearson'
        }

        # Test with 1 signal - should raise ValueError
        with pytest.raises(ValueError, match="exactly 2 signals"):
            compute_correlation_features(
                signals=[signal1],
                epoch_grid_index=epoch_grid,
                parameters=parameters,
                global_window_length=pd.Timedelta('30s'),
                global_step_size=pd.Timedelta('30s')
            )

        # Test with 3 signals - should raise ValueError
        # Create signal3 with proper DatetimeIndex
        index3 = pd.date_range('2024-01-01 00:00:00', periods=3, freq='1s', tz='UTC')
        signal3 = HeartRateSignal(
            data=pd.DataFrame({'hr': [1, 2, 3]}, index=index3),
            metadata={'name': 'signal3', 'signal_type': 'heart_rate'}
        )
        with pytest.raises(ValueError, match="exactly 2 signals"):
            compute_correlation_features(
                signals=[signal1, signal1, signal3],
                epoch_grid_index=epoch_grid,
                parameters=parameters,
                global_window_length=pd.Timedelta('30s'),
                global_step_size=pd.Timedelta('30s')
            )


class TestFeatureCaching:
    """Tests for feature caching decorator with new sleep features."""

    @pytest.fixture
    def hr_signal(self):
        """Create a simple heart rate signal for caching tests."""
        index = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1s', tz='UTC')
        hr_values = 65 + np.random.randn(100) * 5
        data = pd.DataFrame({'hr': hr_values}, index=index)
        return HeartRateSignal(
            data=data,
            metadata={'name': 'test_hr', 'signal_type': 'heart_rate'}
        )

    def test_hrv_feature_caching(self, hr_signal):
        """Test that HRV features are properly cached."""
        from src.sleep_analysis.operations.feature_extraction import (
            clear_feature_cache,
            get_cache_stats,
            enable_feature_cache
        )

        # Clear cache and enable caching
        clear_feature_cache()
        enable_feature_cache(True)

        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=3,
            freq='30s',
            tz='UTC'
        )

        parameters = {
            'hrv_metrics': ['hr_mean', 'hr_std'],
            'use_rr_intervals': False
        }

        # First call - should compute and cache
        initial_cache_size = get_cache_stats()['size']
        result1 = compute_hrv_features(
            signals=[hr_signal],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('30s'),
            global_step_size=pd.Timedelta('30s')
        )

        # Cache should have one more entry
        assert get_cache_stats()['size'] == initial_cache_size + 1

        # Second call with same parameters - should use cache
        result2 = compute_hrv_features(
            signals=[hr_signal],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('30s'),
            global_step_size=pd.Timedelta('30s')
        )

        # Should be the exact same object (from cache)
        assert result1 is result2

        # Cache size should not increase
        assert get_cache_stats()['size'] == initial_cache_size + 1

    def test_movement_feature_cache_invalidation(self):
        """Test that cache is properly invalidated for movement features."""
        from src.sleep_analysis.operations.feature_extraction import (
            clear_feature_cache,
            get_cache_stats
        )

        clear_feature_cache()

        # Create two different accelerometer signals
        index = pd.date_range('2024-01-01 00:00:00', periods=100, freq='0.1s', tz='UTC')
        data1 = pd.DataFrame({
            'x': np.random.randn(100) * 100,
            'y': np.random.randn(100) * 100,
            'z': 1000 + np.random.randn(100) * 20
        }, index=index)

        signal1 = AccelerometerSignal(
            data=data1,
            metadata={'name': 'accel1', 'signal_type': 'accelerometer'}
        )

        data2 = pd.DataFrame({
            'x': np.random.randn(100) * 50,  # Different data
            'y': np.random.randn(100) * 50,
            'z': 1000 + np.random.randn(100) * 10
        }, index=index)

        signal2 = AccelerometerSignal(
            data=data2,
            metadata={'name': 'accel2', 'signal_type': 'accelerometer'}
        )

        epoch_grid = pd.date_range(
            '2024-01-01 00:00:00',
            periods=3,
            freq='3s',
            tz='UTC'
        )

        parameters = {'movement_metrics': 'all'}

        # Compute features for signal1
        result1 = compute_movement_features(
            signals=[signal1],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('3s'),
            global_step_size=pd.Timedelta('3s')
        )

        # Compute features for signal2 - should create new cache entry
        result2 = compute_movement_features(
            signals=[signal2],
            epoch_grid_index=epoch_grid,
            parameters=parameters,
            global_window_length=pd.Timedelta('3s'),
            global_step_size=pd.Timedelta('3s')
        )

        # Results should be different (different signals)
        assert result1 is not result2

        # Both should be in cache
        assert get_cache_stats()['size'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
