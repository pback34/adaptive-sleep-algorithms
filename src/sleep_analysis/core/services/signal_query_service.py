"""
SignalQueryService for querying and filtering signals.

This service extracts the signal querying and filtering logic from SignalCollection,
providing a focused service for finding signals based on various criteria.
"""

import logging
from typing import Dict, List, Any, Union, Optional

from ...signal_types import SignalType, SensorType, SensorModel, BodyPosition
from ..metadata import TimeSeriesMetadata, FeatureMetadata, FeatureType
from ...signals.time_series_signal import TimeSeriesSignal
from ...features.feature import Feature
from ...utils import str_to_enum
from ..repositories.signal_repository import SignalRepository

# Initialize logger
logger = logging.getLogger(__name__)


class SignalQueryService:
    """
    Service for querying and filtering signals based on various criteria.

    This service provides flexible signal retrieval with support for:
    - Exact key matching
    - Base name pattern matching (e.g., "ppg_0", "ppg_1" from base "ppg")
    - Metadata criteria filtering
    - Signal type and feature type filtering
    - Enum field processing

    Examples:
        >>> repo = SignalRepository(metadata_handler)
        >>> query_service = SignalQueryService(repo)
        >>>
        >>> # Get signal by exact key
        >>> signals = query_service.get_signals("hr_0")
        >>>
        >>> # Get all signals with base name "ppg"
        >>> signals = query_service.get_signals(base_name="ppg")
        >>>
        >>> # Get signals by criteria
        >>> signals = query_service.get_signals(
        ...     criteria={"signal_type": SignalType.HEART_RATE}
        ... )
        >>>
        >>> # Get signals with multiple criteria
        >>> signals = query_service.get_signals(
        ...     signal_type=SignalType.PPG,
        ...     criteria={"sensor_model": SensorModel.POLAR_H10}
        ... )
    """

    def __init__(self, repository: SignalRepository):
        """
        Initialize SignalQueryService.

        Args:
            repository: SignalRepository instance to query from
        """
        self.repository = repository

    def get_signals(
        self,
        input_spec: Union[str, Dict[str, Any], List[str], None] = None,
        signal_type: Union[SignalType, str, None] = None,
        feature_type: Union[FeatureType, str, None] = None,
        criteria: Dict[str, Any] = None,
        base_name: str = None
    ) -> List[Union[TimeSeriesSignal, Feature]]:
        """
        Retrieve TimeSeriesSignals and/or Features based on flexible criteria.

        Searches both time-series signals and features from the repository.

        Args:
            input_spec: Can be:
                - String ID or base name ("ppg", "ppg_0", "hr_stats_0")
                - Dictionary with criteria/base_name
                - List of string IDs or base names
            signal_type: A SignalType enum/string to filter TimeSeriesSignals.
            feature_type: A FeatureType enum/string to filter Features.
            criteria: Dictionary of metadata field/value pairs to match (searches
                both TimeSeriesMetadata and FeatureMetadata).
            base_name: Base name to filter signals/features (e.g., "ppg", "hr_stats").

        Returns:
            List of matching TimeSeriesSignal and/or Feature instances.

        Examples:
            >>> # Get by exact key
            >>> signals = query_service.get_signals("hr_0")
            >>>
            >>> # Get by base name
            >>> signals = query_service.get_signals(base_name="ppg")
            >>>
            >>> # Get by signal type
            >>> signals = query_service.get_signals(signal_type=SignalType.HEART_RATE)
            >>>
            >>> # Get with criteria
            >>> signals = query_service.get_signals(
            ...     criteria={"sensor_model": SensorModel.POLAR_H10}
            ... )
            >>>
            >>> # Get with list input
            >>> signals = query_service.get_signals(["hr_0", "ppg_0", "ppg_1"])
        """
        results = []

        # Combine both dictionaries for searching
        search_space = {
            **self.repository.time_series_signals,
            **self.repository.features
        }

        # --- Prepare Criteria ---
        processed_criteria = criteria.copy() if criteria else {}

        # Add signal_type to criteria if provided
        if signal_type is not None:
            st = str_to_enum(signal_type, SignalType) if isinstance(signal_type, str) else signal_type
            processed_criteria["signal_type"] = st

        # Add feature_type to criteria if provided
        if feature_type is not None:
            ft = str_to_enum(feature_type, FeatureType) if isinstance(feature_type, str) else feature_type
            processed_criteria["feature_type"] = ft

        # --- Process input_spec ---
        if input_spec is not None:
            if isinstance(input_spec, dict):
                # Dictionary spec: extract base_name and merge criteria
                if "base_name" in input_spec:
                    base_name = input_spec["base_name"]
                if "criteria" in input_spec:
                    spec_criteria = self._process_enum_criteria(input_spec["criteria"])
                    processed_criteria.update(spec_criteria)

            elif isinstance(input_spec, list):
                # List spec: recursively call get_signals for each item
                for spec_item in input_spec:
                    # Pass down existing filters
                    results.extend(
                        self.get_signals(
                            input_spec=spec_item,
                            signal_type=signal_type,
                            feature_type=feature_type,
                            criteria=criteria,  # Pass original criteria dict
                            base_name=base_name  # Pass original base_name
                        )
                    )
                # Deduplicate results based on object ID
                return list({id(s): s for s in results}.values())

            else:  # String spec
                spec_str = str(input_spec)
                # Check if it's an exact key
                if spec_str in search_space:
                    signal = search_space[spec_str]
                    if self._matches_criteria(signal, processed_criteria):
                        return [signal]
                    else:
                        return []  # Key found but doesn't match criteria
                # If not an exact key, treat it as a base name
                else:
                    base_name = spec_str

        # --- Apply Filtering (Base Name and Criteria) ---
        if base_name:
            # Filter by base name first
            filtered_signals = []
            for key, signal in search_space.items():
                # Check if key matches the base name pattern (e.g., "basename_0")
                if key.startswith(f"{base_name}_") and key[len(base_name)+1:].isdigit():
                    filtered_signals.append(signal)
        else:
            # If no base name, start with all signals/features
            filtered_signals = list(search_space.values())

        # Apply criteria filtering
        final_results = [s for s in filtered_signals if self._matches_criteria(s, processed_criteria)]

        # Deduplicate final results
        return list({id(s): s for s in final_results}.values())

    def _process_enum_criteria(self, criteria_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string enum values in criteria to Enum objects.

        Args:
            criteria_dict: Dictionary of criteria with potential string enum values

        Returns:
            Dictionary with string enum values converted to Enum objects

        Examples:
            >>> criteria = {"signal_type": "HEART_RATE", "sensor_type": "PPG"}
            >>> processed = query_service._process_enum_criteria(criteria)
            >>> # Returns: {"signal_type": SignalType.HEART_RATE, "sensor_type": SensorType.PPG}
        """
        processed = {}
        enum_map = {
            "signal_type": SignalType,
            "sensor_type": SensorType,
            "sensor_model": SensorModel,
            "body_position": BodyPosition,
            "feature_type": FeatureType,
            # Add other enum fields here if needed
        }
        for key, value in criteria_dict.items():
            if key in enum_map and isinstance(value, str):
                try:
                    processed[key] = str_to_enum(value, enum_map[key])
                except ValueError:
                    logger.warning(
                        f"Invalid enum value '{value}' for criteria key '{key}'. "
                        f"Keeping as string."
                    )
                    processed[key] = value  # Keep original string if conversion fails
            else:
                processed[key] = value
        return processed

    def _matches_criteria(
        self,
        signal: Union[TimeSeriesSignal, Feature],
        criteria: Dict[str, Any]
    ) -> bool:
        """
        Check if a TimeSeriesSignal or Feature matches all criteria.

        Args:
            signal: The signal or feature to check
            criteria: Dictionary of metadata field/value pairs to match

        Returns:
            True if signal matches all criteria, False otherwise

        Examples:
            >>> signal = HeartRateSignal(...)
            >>> criteria = {"signal_type": SignalType.HEART_RATE}
            >>> matches = query_service._matches_criteria(signal, criteria)
            >>> # Returns: True if signal is HEART_RATE type
        """
        if not criteria:
            return True

        metadata_obj = signal.metadata  # Get the correct metadata object

        for key, value in criteria.items():
            # Handle nested fields (e.g., "sensor_info.device_id") - applies only to TimeSeriesMetadata
            if "." in key and isinstance(metadata_obj, TimeSeriesMetadata):
                parts = key.split(".", 1)
                container_name, field_name = parts
                if hasattr(metadata_obj, container_name):
                    container = getattr(metadata_obj, container_name)
                    if isinstance(container, dict) and field_name in container:
                        if container[field_name] != value:
                            return False
                    else:
                        return False  # Container not dict or field missing
                else:
                    return False  # Container attribute missing
            # Handle standard fields
            elif hasattr(metadata_obj, key):
                if getattr(metadata_obj, key) != value:
                    return False
            else:  # Field doesn't exist on this metadata type
                return False
        return True
