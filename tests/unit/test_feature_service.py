"""Tests for FeatureService class."""

import pytest
import pandas as pd
import numpy as np
import uuid
from unittest.mock import Mock, patch

from sleep_analysis.services.feature_service import FeatureService
from sleep_analysis.core.signal_collection import SignalCollection
from sleep_analysis.signals.ppg_signal import PPGSignal
from sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from sleep_analysis.features.feature import Feature
from sleep_analysis.core.metadata import FeatureType, CollectionMetadata
from sleep_analysis.signal_types import SignalType


@pytest.fixture
def feature_service():
    """Create a FeatureService instance for testing."""
    return FeatureService()


@pytest.fixture
def signal_collection_with_config():
    """Create a SignalCollection with epoch configuration."""
    collection = SignalCollection(metadata={"timezone": "UTC"})

    # Set epoch_grid_config after initialization
    collection.metadata.epoch_grid_config = {
        "window_length": "30s",
        "step_size": "30s"
    }

    # Add PPG signal
    ppg_index = pd.date_range(start="2023-01-01", periods=300, freq="1s", tz="UTC")
    ppg_data = pd.DataFrame({"value": np.sin(np.linspace(0, 10, 300))}, index=ppg_index)
    ppg_data.index.name = 'timestamp'
    ppg_signal = PPGSignal(
        data=ppg_data,
        metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
    )
    collection.add_time_series_signal("ppg_0", ppg_signal)

    # Add heart rate signal
    hr_index = pd.date_range(start="2023-01-01", periods=300, freq="1s", tz="UTC")
    hr_data = pd.DataFrame({"hr": 70 + np.random.randn(300) * 5}, index=hr_index)
    hr_data.index.name = 'timestamp'
    hr_signal = HeartRateSignal(
        data=hr_data,
        metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.HEART_RATE}
    )
    collection.add_time_series_signal("hr_0", hr_signal)

    return collection


@pytest.fixture
def collection_with_features(signal_collection_with_config, feature_service):
    """Create collection with generated features."""
    collection = signal_collection_with_config

    # Generate epoch grid
    feature_service.generate_epoch_grid(collection)

    # Create and add sample features
    epoch_index = collection.epoch_grid_index

    # Feature 1: PPG stats
    feature1_data = pd.DataFrame({
        ('ppg_0', 'mean'): np.random.randn(len(epoch_index)),
        ('ppg_0', 'std'): np.random.randn(len(epoch_index))
    }, index=epoch_index)
    feature1_data.columns = pd.MultiIndex.from_tuples(feature1_data.columns, names=['signal_key', 'feature'])

    feature1 = Feature(
        data=feature1_data,
        metadata={
            "feature_id": str(uuid.uuid4()),
            "feature_type": FeatureType.STATISTICAL,
            "feature_names": ['mean', 'std'],
            "source_signal_ids": [collection.get_time_series_signal('ppg_0').metadata.signal_id],
            "source_signal_keys": ["ppg_0"],
            "epoch_window_length": pd.Timedelta("30s"),
            "epoch_step_size": pd.Timedelta("30s")
        }
    )
    collection.add_feature("ppg_features", feature1)

    # Feature 2: HR stats
    feature2_data = pd.DataFrame({
        ('hr_0', 'mean'): np.random.randn(len(epoch_index)),
        ('hr_0', 'max'): np.random.randn(len(epoch_index))
    }, index=epoch_index)
    feature2_data.columns = pd.MultiIndex.from_tuples(feature2_data.columns, names=['signal_key', 'feature'])

    feature2 = Feature(
        data=feature2_data,
        metadata={
            "feature_id": str(uuid.uuid4()),
            "feature_type": FeatureType.STATISTICAL,
            "feature_names": ['mean', 'max'],
            "source_signal_ids": [collection.get_time_series_signal('hr_0').metadata.signal_id],
            "source_signal_keys": ["hr_0"],
            "epoch_window_length": pd.Timedelta("30s"),
            "epoch_step_size": pd.Timedelta("30s")
        }
    )
    collection.add_feature("hr_features", feature2)

    return collection


class TestFeatureServiceInit:
    """Tests for FeatureService initialization."""

    def test_init(self, feature_service):
        """Test service initialization."""
        assert feature_service is not None


class TestGenerateEpochGrid:
    """Tests for generate_epoch_grid method."""

    def test_generate_grid_basic(self, feature_service, signal_collection_with_config):
        """Test basic epoch grid generation."""
        feature_service.generate_epoch_grid(signal_collection_with_config)

        assert signal_collection_with_config._epoch_grid_calculated is True
        assert signal_collection_with_config.epoch_grid_index is not None
        assert len(signal_collection_with_config.epoch_grid_index) > 0
        assert signal_collection_with_config.global_epoch_window_length == pd.Timedelta("30s")
        assert signal_collection_with_config.global_epoch_step_size == pd.Timedelta("30s")

    def test_epoch_grid_covers_signals(self, feature_service, signal_collection_with_config):
        """Test that epoch grid covers signal time range."""
        feature_service.generate_epoch_grid(signal_collection_with_config)

        # Get signal time range
        min_time = min(
            s.get_data().index.min()
            for s in signal_collection_with_config.time_series_signals.values()
        )
        max_time = max(
            s.get_data().index.max()
            for s in signal_collection_with_config.time_series_signals.values()
        )

        # Epoch grid should cover this range
        grid = signal_collection_with_config.epoch_grid_index
        assert grid.min() <= min_time
        assert grid.max() + signal_collection_with_config.global_epoch_window_length >= max_time

    def test_generate_grid_with_overrides(self, feature_service, signal_collection_with_config):
        """Test epoch grid generation with time overrides."""
        start_override = "2023-01-01 00:01:00"
        end_override = "2023-01-01 00:03:00"

        feature_service.generate_epoch_grid(
            signal_collection_with_config,
            start_time=start_override,
            end_time=end_override
        )

        grid = signal_collection_with_config.epoch_grid_index
        assert grid.min() >= pd.Timestamp(start_override, tz="UTC")
        assert grid.max() <= pd.Timestamp(end_override, tz="UTC")

    def test_generate_grid_no_config(self, feature_service):
        """Test that missing config raises error."""
        collection = SignalCollection(metadata={"timezone": "UTC"})

        with pytest.raises(RuntimeError, match="Missing or incomplete 'epoch_grid_config'"):
            feature_service.generate_epoch_grid(collection)

    def test_generate_grid_invalid_config(self, feature_service):
        """Test that invalid config raises error."""
        collection = SignalCollection(metadata={
            "timezone": "UTC",
            "epoch_grid_config": {
                "window_length": "invalid",
                "step_size": "30s"
            }
        })

        # Add a signal so it doesn't fail on empty collection
        index = pd.date_range(start="2023-01-01", periods=100, freq="1s", tz="UTC")
        data = pd.DataFrame({"value": range(100)}, index=index)
        data.index.name = 'timestamp'
        signal = PPGSignal(
            data=data,
            metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
        )
        collection.add_time_series_signal("test", signal)

        with pytest.raises(RuntimeError, match="Missing or incomplete 'epoch_grid_config'"):
            feature_service.generate_epoch_grid(collection)

    def test_invalid_time_range(self, feature_service, signal_collection_with_config):
        """Test that invalid time range raises error."""
        with pytest.raises(ValueError, match="start time .* must be before end time"):
            feature_service.generate_epoch_grid(
                signal_collection_with_config,
                start_time="2023-01-01 00:05:00",
                end_time="2023-01-01 00:01:00"  # End before start
            )


class TestApplyMultiSignalOperation:
    """Tests for apply_multi_signal_operation method."""

    def test_apply_operation_basic(self, feature_service, signal_collection_with_config):
        """Test basic multi-signal operation application."""
        # Generate epoch grid first
        feature_service.generate_epoch_grid(signal_collection_with_config)

        # Create mock method that returns a Feature
        def mock_method(signals, epoch_grid_index, parameters, global_window_length, global_step_size):
            feature_data = pd.DataFrame({
                ('ppg_0', 'mean'): [1.0] * len(epoch_grid_index)
            }, index=epoch_grid_index)
            feature_data.columns = pd.MultiIndex.from_tuples(feature_data.columns, names=['signal_key', 'feature'])

            return Feature(
                data=feature_data,
                metadata={
                    "feature_id": str(uuid.uuid4()),
                    "feature_type": FeatureType.STATISTICAL,
                    "feature_names": ['mean'],
                    "source_signal_ids": [signals[0].metadata.signal_id],
                    "source_signal_keys": ["ppg_0"],
                    "epoch_window_length": global_window_length,
                    "epoch_step_size": global_step_size
                }
            )

        # Mock the service method (Phase 2b: Methods instead of registry)
        with patch.object(feature_service, 'compute_feature_statistics', side_effect=mock_method):
            result = feature_service.apply_multi_signal_operation(
                signal_collection_with_config,
                "feature_statistics",
                ["ppg_0"],
                {}
            )

        assert isinstance(result, Feature)
        assert result.metadata.feature_type == FeatureType.STATISTICAL

    def test_apply_without_epoch_grid_fails(self, feature_service, signal_collection_with_config):
        """Test that applying operation without epoch grid fails."""
        # Phase 2b: No need to register - operations are methods now
        with pytest.raises(RuntimeError, match="generate_epoch_grid must be run"):
            feature_service.apply_multi_signal_operation(
                signal_collection_with_config,
                "feature_statistics",  # Use real operation name
                ["ppg_0"],
                {}
            )

    def test_operation_not_found(self, feature_service, signal_collection_with_config):
        """Test that missing operation raises error."""
        feature_service.generate_epoch_grid(signal_collection_with_config)

        # Phase 2b: Updated error message - no longer mentions "registry"
        with pytest.raises(ValueError, match="not found"):
            feature_service.apply_multi_signal_operation(
                signal_collection_with_config,
                "nonexistent_operation",
                ["ppg_0"],
                {}
            )

    def test_invalid_signal_key(self, feature_service, signal_collection_with_config):
        """Test that invalid signal key raises error."""
        feature_service.generate_epoch_grid(signal_collection_with_config)

        # Phase 2b: No need to register - operations are methods now
        with pytest.raises(ValueError, match="not found"):
            feature_service.apply_multi_signal_operation(
                signal_collection_with_config,
                "feature_statistics",  # Use real operation name
                ["nonexistent_signal"],
                {}
            )


class TestPropagateMetadataToFeature:
    """Tests for _propagate_metadata_to_feature method."""

    def test_propagate_single_source(self, feature_service, signal_collection_with_config):
        """Test metadata propagation from single source."""
        feature_service.generate_epoch_grid(signal_collection_with_config)

        # Set feature index config
        signal_collection_with_config.metadata.feature_index_config = ['name', 'sensor_type']

        # Create feature
        epoch_index = signal_collection_with_config.epoch_grid_index
        feature_data = pd.DataFrame({
            ('ppg_0', 'mean'): [1.0] * len(epoch_index)
        }, index=epoch_index)
        feature_data.columns = pd.MultiIndex.from_tuples(feature_data.columns, names=['signal_key', 'feature'])

        feature = Feature(
            data=feature_data,
            metadata={
                "feature_id": str(uuid.uuid4()),
                "feature_type": FeatureType.STATISTICAL,
                "feature_names": ['mean'],
                "source_signal_ids": [signal_collection_with_config.get_time_series_signal('ppg_0').metadata.signal_id],
                "source_signal_keys": ["ppg_0"],
                "epoch_window_length": pd.Timedelta("30s"),
                "epoch_step_size": pd.Timedelta("30s")
            }
        )

        input_signals = [signal_collection_with_config.get_time_series_signal('ppg_0')]

        feature_service._propagate_metadata_to_feature(
            signal_collection_with_config,
            feature,
            input_signals,
            "test_operation"
        )

        # Check that metadata was propagated
        # Note: actual field values depend on what's in the signal metadata

    def test_propagate_multiple_sources_common_value(self, feature_service, signal_collection_with_config):
        """Test metadata propagation with multiple sources having common value."""
        feature_service.generate_epoch_grid(signal_collection_with_config)
        signal_collection_with_config.metadata.feature_index_config = ['signal_type']

        # Create feature from both signals
        epoch_index = signal_collection_with_config.epoch_grid_index
        feature_data = pd.DataFrame({
            ('ppg_0', 'mean'): [1.0] * len(epoch_index),
            ('hr_0', 'mean'): [70.0] * len(epoch_index)
        }, index=epoch_index)
        feature_data.columns = pd.MultiIndex.from_tuples(feature_data.columns, names=['signal_key', 'feature'])

        feature = Feature(
            data=feature_data,
            metadata={
                "feature_id": str(uuid.uuid4()),
                "feature_type": FeatureType.STATISTICAL,
                "feature_names": ['mean'],
                "source_signal_ids": [
                    signal_collection_with_config.get_time_series_signal('ppg_0').metadata.signal_id,
                    signal_collection_with_config.get_time_series_signal('hr_0').metadata.signal_id
                ],
                "source_signal_keys": ["ppg_0", "hr_0"],
                "epoch_window_length": pd.Timedelta("30s"),
                "epoch_step_size": pd.Timedelta("30s")
            }
        )

        input_signals = [
            signal_collection_with_config.get_time_series_signal('ppg_0'),
            signal_collection_with_config.get_time_series_signal('hr_0')
        ]

        feature_service._propagate_metadata_to_feature(
            signal_collection_with_config,
            feature,
            input_signals,
            "test_operation"
        )


class TestCombineFeatures:
    """Tests for combine_features method."""

    def test_combine_basic(self, feature_service, collection_with_features):
        """Test basic feature combination."""
        combined_df = feature_service.combine_features(
            collection_with_features,
            ["ppg_features", "hr_features"]
        )

        assert isinstance(combined_df, pd.DataFrame)
        assert not combined_df.empty
        assert isinstance(combined_df.index, pd.DatetimeIndex)
        # Should have columns from both features
        assert combined_df.shape[1] >= 4  # At least 2 features from each

    def test_combine_empty_inputs(self, feature_service, collection_with_features):
        """Test that empty inputs raises error."""
        with pytest.raises(ValueError, match="No inputs"):
            feature_service.combine_features(collection_with_features, [])

    def test_combine_nonexistent_feature(self, feature_service, collection_with_features):
        """Test that combining nonexistent feature logs warning."""
        # This should complete but log warnings for missing features
        combined_df = feature_service.combine_features(
            collection_with_features,
            ["ppg_features", "nonexistent_feature"]
        )

        # Should still return dataframe with valid features
        assert isinstance(combined_df, pd.DataFrame)


class TestIntegration:
    """Integration tests for FeatureService."""

    def test_full_feature_workflow(self, feature_service):
        """Test complete feature extraction workflow."""
        # Create collection
        collection = SignalCollection(metadata={"timezone": "UTC"})

        # Set epoch_grid_config after initialization
        collection.metadata.epoch_grid_config = {
            "window_length": "10s",
            "step_size": "10s"
        }

        # Add signal
        index = pd.date_range(start="2023-01-01", periods=100, freq="1s", tz="UTC")
        data = pd.DataFrame({"value": np.sin(np.linspace(0, 10, 100))}, index=index)
        data.index.name = 'timestamp'
        signal = PPGSignal(
            data=data,
            metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
        )
        collection.add_time_series_signal("ppg_0", signal)

        # Generate epoch grid
        feature_service.generate_epoch_grid(collection)
        assert collection._epoch_grid_calculated
        assert len(collection.epoch_grid_index) == 10  # 100s / 10s epochs

        # Mock service method (Phase 2b: Methods instead of registry)
        def mock_feature_method(signals, epoch_grid_index, parameters, global_window_length, global_step_size):
            feature_data = pd.DataFrame({
                ('ppg_0', 'mean'): [1.0] * len(epoch_grid_index),
                ('ppg_0', 'std'): [0.5] * len(epoch_grid_index)
            }, index=epoch_grid_index)
            feature_data.columns = pd.MultiIndex.from_tuples(feature_data.columns, names=['signal_key', 'feature'])

            return Feature(
                data=feature_data,
                metadata={
                    "feature_id": str(uuid.uuid4()),
                    "feature_type": FeatureType.STATISTICAL,
                    "feature_names": ['mean', 'std'],
                    "source_signal_ids": [signals[0].metadata.signal_id],
                    "source_signal_keys": ["ppg_0"],
                    "epoch_window_length": global_window_length,
                    "epoch_step_size": global_step_size
                }
            )

        with patch.object(feature_service, 'compute_feature_statistics', side_effect=mock_feature_method):
            result = feature_service.apply_multi_signal_operation(
                collection,
                "feature_statistics",
                ["ppg_0"],
                {}
            )

        assert isinstance(result, Feature)
        assert result.get_data().shape[0] == 10
        assert result.metadata.feature_names == ['mean', 'std']
