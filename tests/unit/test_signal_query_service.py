"""Tests for SignalQueryService class."""

import pytest
import pandas as pd
from unittest.mock import Mock

from src.sleep_analysis.core.services.signal_query_service import SignalQueryService
from src.sleep_analysis.core.repositories.signal_repository import SignalRepository
from src.sleep_analysis.core.metadata_handler import MetadataHandler
from src.sleep_analysis.signals.heart_rate_signal import HeartRateSignal
from src.sleep_analysis.features.feature import Feature
from src.sleep_analysis.signal_types import SignalType, SensorType, SensorModel
from src.sleep_analysis.core.metadata import FeatureType


@pytest.fixture
def metadata_handler():
    """Create a metadata handler for testing."""
    return MetadataHandler()


@pytest.fixture
def signal_repository(metadata_handler):
    """Create a signal repository with some test data."""
    repo = SignalRepository(metadata_handler, collection_timezone="UTC")

    # Add some test signals
    for i in range(3):
        data = pd.DataFrame(
            {"hr": [float(70 + i), float(71 + i), float(72 + i)]},
            index=pd.date_range("2025-01-01", periods=3, freq="1s", tz="UTC")
        )
        signal = HeartRateSignal(
            data=data,
            metadata={
                "signal_id": f"hr_signal_{i:03d}",
                "signal_type": SignalType.HEART_RATE,
                "sensor_type": SensorType.PPG,
                "sensor_model": SensorModel.POLAR_H10 if i == 0 else SensorModel.POLAR_SENSE
            }
        )
        repo.add_time_series_signal(f"hr_{i}", signal)

    # Add some PPG signals
    for i in range(2):
        data = pd.DataFrame(
            {"value": [float(100 + i), float(101 + i), float(102 + i)]},
            index=pd.date_range("2025-01-01", periods=3, freq="1s", tz="UTC")
        )
        from src.sleep_analysis.signals.ppg_signal import PPGSignal
        signal = PPGSignal(
            data=data,
            metadata={
                "signal_id": f"ppg_signal_{i:03d}",
                "signal_type": SignalType.PPG,
                "sensor_type": SensorType.PPG
            }
        )
        repo.add_time_series_signal(f"ppg_{i}", signal)

    # Add some features
    for i in range(2):
        data = pd.DataFrame(
            {"mean_hr": [float(70 + i), float(71 + i)]},
            index=pd.date_range("2025-01-01", periods=2, freq="30s", tz="UTC")
        )
        feature = Feature(
            data=data,
            metadata={
                "feature_id": f"hrv_feature_{i:03d}",
                "sensor_type": SensorType.PPG,
                "feature_type": FeatureType.HRV,
                "epoch_window_length": pd.Timedelta(seconds=30),
                "epoch_step_size": pd.Timedelta(seconds=30),
                "feature_names": ["mean_hr"],
                "source_signal_keys": [f"hr_{i}"],
                "source_signal_ids": [f"hr_signal_{i:03d}"]
            }
        )
        repo.add_feature(f"hrv_{i}", feature)

    return repo


@pytest.fixture
def query_service(signal_repository):
    """Create a query service for testing."""
    return SignalQueryService(signal_repository)


class TestSignalQueryServiceBasics:
    """Test basic query service initialization."""

    def test_init_sets_repository(self, signal_repository):
        """Test that initialization sets the repository."""
        service = SignalQueryService(signal_repository)
        assert service.repository is signal_repository


class TestGetSignalsByExactKey:
    """Test retrieving signals by exact key."""

    def test_get_signal_by_exact_key(self, query_service):
        """Test retrieving a signal by exact key."""
        signals = query_service.get_signals("hr_0")
        assert len(signals) == 1
        assert signals[0].metadata.signal_id == "hr_signal_000"

    def test_get_feature_by_exact_key(self, query_service):
        """Test retrieving a feature by exact key."""
        features = query_service.get_signals("hrv_0")
        assert len(features) == 1
        assert features[0].metadata.feature_id == "hrv_feature_000"

    def test_get_signal_by_nonexistent_key(self, query_service):
        """Test that nonexistent key returns empty list."""
        signals = query_service.get_signals("nonexistent")
        assert len(signals) == 0

    def test_get_signal_by_exact_key_with_criteria_match(self, query_service):
        """Test exact key with matching criteria returns signal."""
        signals = query_service.get_signals(
            "hr_0",
            criteria={"signal_type": SignalType.HEART_RATE}
        )
        assert len(signals) == 1

    def test_get_signal_by_exact_key_with_criteria_mismatch(self, query_service):
        """Test exact key with non-matching criteria returns empty."""
        signals = query_service.get_signals(
            "hr_0",
            criteria={"signal_type": SignalType.PPG}
        )
        assert len(signals) == 0


class TestGetSignalsByBaseName:
    """Test retrieving signals by base name pattern."""

    def test_get_signals_by_base_name(self, query_service):
        """Test retrieving all signals with a base name."""
        signals = query_service.get_signals(base_name="hr")
        assert len(signals) == 3
        assert all(s.metadata.signal_type == SignalType.HEART_RATE for s in signals)

    def test_get_signals_by_ppg_base_name(self, query_service):
        """Test retrieving PPG signals by base name."""
        signals = query_service.get_signals(base_name="ppg")
        assert len(signals) == 2
        assert all(s.metadata.signal_type == SignalType.PPG for s in signals)

    def test_get_features_by_base_name(self, query_service):
        """Test retrieving features by base name."""
        features = query_service.get_signals(base_name="hrv")
        assert len(features) == 2
        assert all(f.metadata.feature_type == FeatureType.HRV for f in features)

    def test_get_signals_by_nonexistent_base_name(self, query_service):
        """Test that nonexistent base name returns empty list."""
        signals = query_service.get_signals(base_name="nonexistent")
        assert len(signals) == 0


class TestGetSignalsBySignalType:
    """Test filtering by signal type."""

    def test_get_signals_by_signal_type_enum(self, query_service):
        """Test filtering by SignalType enum."""
        signals = query_service.get_signals(signal_type=SignalType.HEART_RATE)
        assert len(signals) == 3
        assert all(s.metadata.signal_type == SignalType.HEART_RATE for s in signals)

    def test_get_signals_by_signal_type_string(self, query_service):
        """Test filtering by SignalType string."""
        signals = query_service.get_signals(signal_type="PPG")
        assert len(signals) == 2
        assert all(s.metadata.signal_type == SignalType.PPG for s in signals)

    def test_get_signals_by_signal_type_with_base_name(self, query_service):
        """Test combining signal type with base name."""
        # Should return only HR signals with base name "hr"
        signals = query_service.get_signals(
            base_name="hr",
            signal_type=SignalType.HEART_RATE
        )
        assert len(signals) == 3


class TestGetSignalsByFeatureType:
    """Test filtering by feature type."""

    def test_get_signals_by_feature_type_enum(self, query_service):
        """Test filtering by FeatureType enum."""
        features = query_service.get_signals(feature_type=FeatureType.HRV)
        assert len(features) == 2
        assert all(f.metadata.feature_type == FeatureType.HRV for f in features)

    def test_get_signals_by_feature_type_string(self, query_service):
        """Test filtering by FeatureType string."""
        features = query_service.get_signals(feature_type="HRV")
        assert len(features) == 2


class TestGetSignalsByCriteria:
    """Test filtering by metadata criteria."""

    def test_get_signals_by_sensor_model_criteria(self, query_service):
        """Test filtering by sensor model criteria."""
        signals = query_service.get_signals(
            criteria={"sensor_model": SensorModel.POLAR_H10}
        )
        assert len(signals) == 1
        assert signals[0].metadata.sensor_model == SensorModel.POLAR_H10

    def test_get_signals_by_multiple_criteria(self, query_service):
        """Test filtering by multiple criteria."""
        signals = query_service.get_signals(
            criteria={
                "signal_type": SignalType.HEART_RATE,
                "sensor_type": SensorType.PPG
            }
        )
        assert len(signals) == 3
        assert all(s.metadata.signal_type == SignalType.HEART_RATE for s in signals)
        assert all(s.metadata.sensor_type == SensorType.PPG for s in signals)

    def test_get_signals_by_criteria_no_match(self, query_service):
        """Test criteria with no matching signals."""
        signals = query_service.get_signals(
            criteria={"sensor_model": SensorModel.ENCHANTED_WAVE}
        )
        assert len(signals) == 0


class TestGetSignalsByInputSpecDict:
    """Test using dictionary input spec."""

    def test_get_signals_with_dict_base_name(self, query_service):
        """Test dictionary input spec with base_name."""
        signals = query_service.get_signals({"base_name": "hr"})
        assert len(signals) == 3

    def test_get_signals_with_dict_criteria(self, query_service):
        """Test dictionary input spec with criteria."""
        signals = query_service.get_signals({
            "criteria": {"signal_type": SignalType.PPG}
        })
        assert len(signals) == 2

    def test_get_signals_with_dict_base_name_and_criteria(self, query_service):
        """Test dictionary input spec with both base_name and criteria."""
        signals = query_service.get_signals({
            "base_name": "hr",
            "criteria": {"sensor_model": SensorModel.POLAR_H10}
        })
        assert len(signals) == 1


class TestGetSignalsByInputSpecList:
    """Test using list input spec."""

    def test_get_signals_with_list_of_keys(self, query_service):
        """Test list input spec with exact keys."""
        signals = query_service.get_signals(["hr_0", "hr_1", "ppg_0"])
        assert len(signals) == 3

    def test_get_signals_with_list_of_base_names(self, query_service):
        """Test list input spec with base names."""
        signals = query_service.get_signals(["hr", "ppg"])
        assert len(signals) == 5  # 3 hr + 2 ppg

    def test_get_signals_with_list_deduplication(self, query_service):
        """Test that list results are deduplicated."""
        signals = query_service.get_signals(["hr_0", "hr_0"])
        assert len(signals) == 1


class TestProcessEnumCriteria:
    """Test enum criteria processing."""

    def test_process_enum_criteria_signal_type_string(self, query_service):
        """Test converting signal_type string to enum."""
        criteria = {"signal_type": "HEART_RATE"}
        processed = query_service._process_enum_criteria(criteria)
        assert processed["signal_type"] == SignalType.HEART_RATE

    def test_process_enum_criteria_sensor_type_string(self, query_service):
        """Test converting sensor_type string to enum."""
        criteria = {"sensor_type": "PPG"}
        processed = query_service._process_enum_criteria(criteria)
        assert processed["sensor_type"] == SensorType.PPG

    def test_process_enum_criteria_keeps_non_string_values(self, query_service):
        """Test that non-string values are kept as-is."""
        criteria = {"signal_type": SignalType.HEART_RATE}
        processed = query_service._process_enum_criteria(criteria)
        assert processed["signal_type"] == SignalType.HEART_RATE

    def test_process_enum_criteria_invalid_enum_value(self, query_service, caplog):
        """Test that invalid enum values are kept as strings with warning."""
        criteria = {"signal_type": "INVALID_TYPE"}
        processed = query_service._process_enum_criteria(criteria)
        assert processed["signal_type"] == "INVALID_TYPE"
        assert "Invalid enum value" in caplog.text

    def test_process_enum_criteria_non_enum_field(self, query_service):
        """Test that non-enum fields are passed through."""
        criteria = {"custom_field": "custom_value"}
        processed = query_service._process_enum_criteria(criteria)
        assert processed["custom_field"] == "custom_value"


class TestMatchesCriteria:
    """Test criteria matching logic."""

    def test_matches_criteria_empty_criteria(self, query_service, signal_repository):
        """Test that empty criteria matches any signal."""
        signal = signal_repository.get_time_series_signal("hr_0")
        assert query_service._matches_criteria(signal, {})

    def test_matches_criteria_single_field_match(self, query_service, signal_repository):
        """Test matching a single field."""
        signal = signal_repository.get_time_series_signal("hr_0")
        criteria = {"signal_type": SignalType.HEART_RATE}
        assert query_service._matches_criteria(signal, criteria)

    def test_matches_criteria_single_field_mismatch(self, query_service, signal_repository):
        """Test non-matching single field."""
        signal = signal_repository.get_time_series_signal("hr_0")
        criteria = {"signal_type": SignalType.PPG}
        assert not query_service._matches_criteria(signal, criteria)

    def test_matches_criteria_multiple_fields_all_match(self, query_service, signal_repository):
        """Test matching multiple fields."""
        signal = signal_repository.get_time_series_signal("hr_0")
        criteria = {
            "signal_type": SignalType.HEART_RATE,
            "sensor_type": SensorType.PPG
        }
        assert query_service._matches_criteria(signal, criteria)

    def test_matches_criteria_multiple_fields_one_mismatch(self, query_service, signal_repository):
        """Test that one mismatch fails the match."""
        signal = signal_repository.get_time_series_signal("hr_0")
        criteria = {
            "signal_type": SignalType.HEART_RATE,
            "sensor_type": SensorType.ACCEL  # Mismatch (signal has PPG)
        }
        assert not query_service._matches_criteria(signal, criteria)

    def test_matches_criteria_nonexistent_field(self, query_service, signal_repository):
        """Test that nonexistent field fails the match."""
        signal = signal_repository.get_time_series_signal("hr_0")
        criteria = {"nonexistent_field": "value"}
        assert not query_service._matches_criteria(signal, criteria)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_signals_with_none_input_spec(self, query_service):
        """Test that None input_spec returns all signals."""
        signals = query_service.get_signals(None)
        # Should return all 7 signals (3 hr + 2 ppg + 2 hrv features)
        assert len(signals) == 7

    def test_get_signals_with_empty_list(self, query_service):
        """Test that empty list returns empty result."""
        signals = query_service.get_signals([])
        assert len(signals) == 0

    def test_get_signals_combining_all_filters(self, query_service):
        """Test combining all filter types together."""
        signals = query_service.get_signals(
            base_name="hr",
            signal_type=SignalType.HEART_RATE,
            criteria={"sensor_model": SensorModel.POLAR_H10}
        )
        assert len(signals) == 1
        assert signals[0].metadata.sensor_model == SensorModel.POLAR_H10
