"""Signal repository for CRUD operations on signals and features."""

import uuid
import logging
from typing import Dict, List, Union, Optional
import pandas as pd

from ..metadata_handler import MetadataHandler
from ...signals.time_series_signal import TimeSeriesSignal
from ...features.feature import Feature

logger = logging.getLogger(__name__)


class SignalRepository:
    """Manages storage and basic access to signals and features.

    This class encapsulates all CRUD (Create, Read, Update, Delete) operations
    for TimeSeriesSignals and Features, extracted from the original SignalCollection
    God object.

    Responsibilities:
        - Add and retrieve TimeSeriesSignals
        - Add and retrieve Features
        - Auto-increment signal names with base names
        - Validate ID uniqueness
        - Batch import operations

    Example:
        >>> from ..metadata_handler import MetadataHandler
        >>> handler = MetadataHandler()
        >>> repo = SignalRepository(handler, collection_timezone="UTC")
        >>> # Add signals
        >>> repo.add_time_series_signal("hr_0", hr_signal)
        >>> # Retrieve signals
        >>> signal = repo.get_time_series_signal("hr_0")
    """

    def __init__(
        self,
        metadata_handler: MetadataHandler,
        collection_timezone: str = "UTC"
    ):
        """Initialize the signal repository.

        Args:
            metadata_handler: Handler for signal metadata operations
            collection_timezone: Timezone for the collection (default: "UTC")
        """
        self.time_series_signals: Dict[str, TimeSeriesSignal] = {}
        self.features: Dict[str, Feature] = {}
        self.metadata_handler = metadata_handler
        self.collection_timezone = collection_timezone

    def add_time_series_signal(self, key: str, signal: TimeSeriesSignal) -> None:
        """Add a TimeSeriesSignal to the repository.

        Args:
            key: Unique identifier for the signal in this collection
            signal: The TimeSeriesSignal instance to add

        Raises:
            TypeError: If signal is not a TimeSeriesSignal instance
            ValueError: If a signal with the given key already exists

        Example:
            >>> repo.add_time_series_signal("hr_sensor_1", hr_signal)
        """
        if not isinstance(signal, TimeSeriesSignal):
            raise TypeError(
                f"Signal provided for key '{key}' is not a TimeSeriesSignal "
                f"(type: {type(signal).__name__})."
            )
        if key in self.time_series_signals:
            raise ValueError(
                f"TimeSeriesSignal with key '{key}' already exists in the collection."
            )

        # Check for signal_id uniqueness across *all* signals (time series and features)
        existing_ids = {s.metadata.signal_id for s in self.time_series_signals.values()} | \
                       {f.metadata.feature_id for f in self.features.values()}
        if signal.metadata.signal_id in existing_ids:
            new_id = str(uuid.uuid4())
            logger.warning(
                f"TimeSeriesSignal ID '{signal.metadata.signal_id}' conflicts with an "
                f"existing signal/feature ID. Assigning new ID: {new_id}"
            )
            signal.metadata.signal_id = new_id

        # Validate timestamp index and timezone
        self._validate_timestamp_index(signal)
        self._validate_timezone(key, signal)

        # Set the signal's name to the key if not already set
        if signal.handler:
            signal.handler.set_name(signal.metadata, key=key)
        else:
            signal.handler = self.metadata_handler
            signal.handler.set_name(signal.metadata, key=key)

        self.time_series_signals[key] = signal

    def add_feature(self, key: str, feature: Feature) -> None:
        """Add a Feature object to the repository.

        Args:
            key: Unique identifier for the feature set in this collection
            feature: The Feature instance to add

        Raises:
            TypeError: If feature is not a Feature instance
            ValueError: If a feature with the given key already exists

        Example:
            >>> repo.add_feature("hrv_features_0", hrv_feature)
        """
        if not isinstance(feature, Feature):
            raise TypeError(
                f"Object provided for key '{key}' is not a Feature "
                f"(type: {type(feature).__name__})."
            )
        if key in self.features:
            raise ValueError(
                f"Feature with key '{key}' already exists in the collection."
            )

        # Check for feature_id uniqueness across *all* signals/features
        existing_ids = {s.metadata.signal_id for s in self.time_series_signals.values()} | \
                       {f.metadata.feature_id for f in self.features.values()}
        if feature.metadata.feature_id in existing_ids:
            new_id = str(uuid.uuid4())
            logger.warning(
                f"Feature ID '{feature.metadata.feature_id}' conflicts with an "
                f"existing signal/feature ID. Assigning new ID: {new_id}"
            )
            feature.metadata.feature_id = new_id

        # Set the feature's name to the key if not already set
        if feature.handler:
            feature.handler.set_name(feature.metadata, key=key)
        else:
            feature.handler = self.metadata_handler
            feature.handler.set_name(feature.metadata, key=key)

        self.features[key] = feature

    def add_signal_with_base_name(
        self,
        base_name: str,
        signal: Union[TimeSeriesSignal, Feature]
    ) -> str:
        """Add a TimeSeriesSignal or Feature with a base name, appending an index if needed.

        Args:
            base_name: Base name for the signal/feature (e.g., "ppg", "hr_stats")
            signal: The TimeSeriesSignal or Feature instance to add

        Returns:
            The key assigned to the signal/feature (e.g., "ppg_0", "hr_stats_1")

        Raises:
            ValueError: If the base name is empty
            TypeError: If signal is not a TimeSeriesSignal or Feature

        Example:
            >>> key = repo.add_signal_with_base_name("accel", accel_signal)
            >>> print(key)  # "accel_0"
            >>> key2 = repo.add_signal_with_base_name("accel", accel_signal2)
            >>> print(key2)  # "accel_1"
        """
        if not base_name:
            raise ValueError("Base name cannot be empty")

        target_dict = None
        if isinstance(signal, TimeSeriesSignal):
            target_dict = self.time_series_signals
        elif isinstance(signal, Feature):
            target_dict = self.features
        else:
            raise TypeError(
                f"Input must be a TimeSeriesSignal or Feature, "
                f"got {type(signal).__name__}"
            )

        index = 0
        while True:
            key = f"{base_name}_{index}"
            if key not in target_dict:
                # Use the appropriate add method
                if isinstance(signal, TimeSeriesSignal):
                    self.add_time_series_signal(key, signal)
                else:  # Must be Feature
                    self.add_feature(key, signal)
                return key
            index += 1

    def add_imported_signals(
        self,
        signals: List[TimeSeriesSignal],
        base_name: str,
        start_index: int = 0
    ) -> List[str]:
        """Add imported TimeSeriesSignals to the repository with sequential indexing.

        Args:
            signals: List of TimeSeriesSignal instances to add
            base_name: Base name for all signals (e.g., "imported")
            start_index: Starting index for sequential naming (default: 0)

        Returns:
            List of keys assigned to the added signals

        Example:
            >>> signals = [signal1, signal2, signal3]
            >>> keys = repo.add_imported_signals(signals, "polar_h10")
            >>> print(keys)  # ["polar_h10_0", "polar_h10_1", "polar_h10_2"]
        """
        keys = []
        current_index = start_index

        for signal in signals:
            if not isinstance(signal, TimeSeriesSignal):
                logger.warning(
                    f"Skipping object of type {type(signal).__name__} during "
                    f"add_imported_signals (expected TimeSeriesSignal)."
                )
                continue

            key = f"{base_name}_{current_index}"
            try:
                self.add_time_series_signal(key, signal)
                keys.append(key)
                current_index += 1
            except ValueError as e:
                # Handle case where key might already exist unexpectedly
                logger.error(
                    f"Failed to add imported signal with key '{key}': {e}. "
                    f"Trying next index."
                )
                # Try incrementing index again to find next available slot
                current_index += 1
                key = f"{base_name}_{current_index}"
                try:
                    self.add_time_series_signal(key, signal)
                    keys.append(key)
                    current_index += 1
                except ValueError as e2:
                    logger.error(
                        f"Failed again to add imported signal with key '{key}': {e2}. "
                        f"Skipping this signal."
                    )

        return keys

    def get_time_series_signal(self, key: str) -> TimeSeriesSignal:
        """Retrieve a TimeSeriesSignal by its key.

        Args:
            key: The key of the signal to retrieve

        Returns:
            The TimeSeriesSignal instance

        Raises:
            KeyError: If no signal with the given key exists

        Example:
            >>> signal = repo.get_time_series_signal("hr_0")
        """
        if key not in self.time_series_signals:
            raise KeyError(
                f"No TimeSeriesSignal with key '{key}' found in the collection."
            )
        return self.time_series_signals[key]

    def get_feature(self, key: str) -> Feature:
        """Retrieve a Feature object by its key.

        Args:
            key: The key of the feature to retrieve

        Returns:
            The Feature instance

        Raises:
            KeyError: If no feature with the given key exists

        Example:
            >>> feature = repo.get_feature("hrv_features_0")
        """
        if key not in self.features:
            raise KeyError(
                f"No Feature with key '{key}' found in the collection."
            )
        return self.features[key]

    def get_by_key(self, key: str) -> Union[TimeSeriesSignal, Feature]:
        """Retrieve a TimeSeriesSignal or Feature by its key.

        Checks both time_series_signals and features dictionaries.

        Args:
            key: The key of the signal or feature to retrieve

        Returns:
            The TimeSeriesSignal or Feature instance

        Raises:
            KeyError: If no signal or feature with the given key exists

        Example:
            >>> signal_or_feature = repo.get_by_key("hr_0")
        """
        if key in self.time_series_signals:
            return self.time_series_signals[key]
        elif key in self.features:
            return self.features[key]
        else:
            raise KeyError(
                f"No TimeSeriesSignal or Feature with key '{key}' found in the collection."
            )

    def get_all_time_series(self) -> Dict[str, TimeSeriesSignal]:
        """Get all TimeSeriesSignals in the repository.

        Returns:
            Dictionary mapping keys to TimeSeriesSignal instances

        Example:
            >>> all_signals = repo.get_all_time_series()
            >>> print(f"Repository contains {len(all_signals)} time-series signals")
        """
        return self.time_series_signals

    def get_all_features(self) -> Dict[str, Feature]:
        """Get all Features in the repository.

        Returns:
            Dictionary mapping keys to Feature instances

        Example:
            >>> all_features = repo.get_all_features()
            >>> print(f"Repository contains {len(all_features)} features")
        """
        return self.features

    def _validate_timestamp_index(self, signal: TimeSeriesSignal) -> None:
        """Validate that a signal has a proper DatetimeIndex.

        Args:
            signal: The signal to validate

        Raises:
            ValueError: If the signal doesn't have a DatetimeIndex
        """
        data = signal.get_data()
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                f"TimeSeriesSignal must have a DatetimeIndex, "
                f"got {type(data.index).__name__}"
            )

    def _validate_timezone(self, key: str, signal: TimeSeriesSignal) -> None:
        """Validate timezone consistency between signal and collection.

        Args:
            key: The key of the signal being added
            signal: The signal to validate

        Note:
            Logs warnings for timezone mismatches but does not raise exceptions
        """
        try:
            signal_tz = signal.get_data().index.tz
            collection_tz_str = self.collection_timezone

            # Convert collection timezone string to tzinfo object for robust comparison
            collection_tz = None
            if collection_tz_str:
                try:
                    # Use pandas to interpret the timezone string robustly
                    collection_tz = pd.Timestamp('now', tz=collection_tz_str).tz
                except Exception as tz_parse_err:
                    logger.warning(
                        f"Could not parse collection timezone string '{collection_tz_str}' "
                        f"for validation: {tz_parse_err}"
                    )
                    collection_tz = None

            # Perform comparison using string representations for robustness
            signal_tz_str = str(signal_tz) if signal_tz is not None else "None"

            if signal_tz is None and collection_tz_str != "None":
                logger.warning(
                    f"Signal '{key}' has a naive timestamp index (timezone: {signal_tz_str}), "
                    f"while collection timezone is '{collection_tz_str}'. "
                    f"Potential inconsistency."
                )
            elif signal_tz is not None and collection_tz_str == "None":
                logger.warning(
                    f"Signal '{key}' has timezone '{signal_tz_str}', "
                    f"while collection timezone is not set ('{collection_tz_str}'). "
                    f"Potential inconsistency."
                )
            elif (signal_tz is not None and collection_tz_str != "None" and
                  signal_tz_str != collection_tz_str):
                logger.warning(
                    f"Signal '{key}' timezone string ('{signal_tz_str}') does not match "
                    f"collection timezone string ('{collection_tz_str}'). "
                    f"Potential inconsistency."
                )

        except AttributeError:
            logger.warning(
                f"Could not access index or timezone for signal '{key}' during validation."
            )
        except Exception as val_err:
            logger.error(
                f"Error during timezone validation for signal '{key}': {val_err}",
                exc_info=True
            )
