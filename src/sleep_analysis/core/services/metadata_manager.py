"""
MetadataManager for managing signal and feature metadata.

This service extracts metadata management logic from SignalCollection,
providing focused operations for updating and validating metadata.
"""

import logging
import pandas as pd
from dataclasses import fields
from typing import Dict, Any

from ...signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from ..metadata import TimeSeriesMetadata, FeatureMetadata, FeatureType
from ...signals.time_series_signal import TimeSeriesSignal
from ...features.feature import Feature
from ...utils import str_to_enum
from ..metadata_handler import MetadataHandler

# Initialize logger
logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Service for managing and updating signal and feature metadata.

    This service provides centralized metadata operations including:
    - Updating time-series signal metadata
    - Updating feature metadata
    - Enum field processing and validation
    - Timedelta field handling
    - Sensor info updates

    Examples:
        >>> metadata_handler = MetadataHandler()
        >>> manager = MetadataManager(metadata_handler)
        >>>
        >>> # Update time-series metadata
        >>> manager.update_time_series_metadata(
        ...     signal,
        ...     {"sensor_model": "POLAR_H10", "quality_score": 0.95}
        ... )
        >>>
        >>> # Update feature metadata
        >>> manager.update_feature_metadata(
        ...     feature,
        ...     {"feature_type": "HRV", "epoch_window_length": "30s"}
        ... )
    """

    def __init__(self, metadata_handler: MetadataHandler):
        """
        Initialize MetadataManager.

        Args:
            metadata_handler: MetadataHandler instance for metadata operations
        """
        self.metadata_handler = metadata_handler

    def update_time_series_metadata(
        self,
        signal: TimeSeriesSignal,
        metadata_spec: Dict[str, Any]
    ) -> None:
        """
        Update a TimeSeriesSignal's metadata.

        Handles enum field conversion, sensor info updates, and general metadata
        field updates using the metadata handler.

        Args:
            signal: The TimeSeriesSignal to update
            metadata_spec: Dictionary of metadata fields to update

        Raises:
            TypeError: If signal is not a TimeSeriesSignal

        Examples:
            >>> manager.update_time_series_metadata(
            ...     signal,
            ...     {
            ...         "sensor_model": "POLAR_H10",
            ...         "quality_score": 0.95,
            ...         "sensor_info": {"device_id": "ABC123"}
            ...     }
            ... )
        """
        if not isinstance(signal, TimeSeriesSignal):
            raise TypeError(f"Expected TimeSeriesSignal, got {type(signal).__name__}")

        # Process enum fields specifically for TimeSeriesMetadata
        processed_metadata = {}
        enum_map = {
            "signal_type": SignalType,
            "sensor_type": SensorType,
            "sensor_model": SensorModel,
            "body_position": BodyPosition,
            "units": Unit
        }

        for field, enum_cls in enum_map.items():
            if field in metadata_spec and isinstance(metadata_spec[field], str):
                try:
                    processed_metadata[field] = str_to_enum(metadata_spec[field], enum_cls)
                except ValueError:
                    logger.warning(
                        f"Invalid enum value '{metadata_spec[field]}' for field '{field}'. "
                        f"Skipping update."
                    )

        # Handle sensor_info separately
        if "sensor_info" in metadata_spec and isinstance(metadata_spec["sensor_info"], dict):
            if signal.metadata.sensor_info is None:
                signal.metadata.sensor_info = {}
            signal.metadata.sensor_info.update(metadata_spec["sensor_info"])

        # Add other valid TimeSeriesMetadata fields
        valid_fields = {f.name for f in fields(TimeSeriesMetadata)}
        for field in valid_fields:
            if field in metadata_spec and field not in processed_metadata and field != "sensor_info":
                processed_metadata[field] = metadata_spec[field]

        # Use the metadata handler to update
        handler = signal.handler or self.metadata_handler
        handler.update_metadata(signal.metadata, **processed_metadata)

    def update_feature_metadata(
        self,
        feature: Feature,
        metadata_spec: Dict[str, Any]
    ) -> None:
        """
        Update a Feature's metadata.

        Handles feature type enum conversion, timedelta field parsing, and general
        metadata field updates using the metadata handler.

        Args:
            feature: The Feature to update
            metadata_spec: Dictionary of metadata fields to update

        Raises:
            TypeError: If feature is not a Feature

        Examples:
            >>> manager.update_feature_metadata(
            ...     feature,
            ...     {
            ...         "feature_type": "HRV",
            ...         "epoch_window_length": "30s",
            ...         "custom_field": "value"
            ...     }
            ... )
        """
        if not isinstance(feature, Feature):
            raise TypeError(f"Expected Feature, got {type(feature).__name__}")

        processed_metadata = {}

        # Process FeatureType enum
        if "feature_type" in metadata_spec and isinstance(metadata_spec["feature_type"], str):
            try:
                processed_metadata["feature_type"] = str_to_enum(
                    metadata_spec["feature_type"],
                    FeatureType
                )
            except ValueError:
                logger.warning(
                    f"Invalid enum value '{metadata_spec['feature_type']}' for field "
                    f"'feature_type'. Skipping update."
                )

        # Add other valid FeatureMetadata fields
        valid_fields = {f.name for f in fields(FeatureMetadata)}
        for field in valid_fields:
            if field in metadata_spec and field not in processed_metadata:
                # Special handling for timedeltas if provided as strings
                if field in ['epoch_window_length', 'epoch_step_size'] and isinstance(
                    metadata_spec[field], str
                ):
                    try:
                        processed_metadata[field] = pd.Timedelta(metadata_spec[field])
                    except ValueError:
                        logger.warning(
                            f"Invalid timedelta format '{metadata_spec[field]}' for field "
                            f"'{field}'. Skipping update."
                        )
                else:
                    processed_metadata[field] = metadata_spec[field]

        # Use the metadata handler to update
        handler = feature.handler or self.metadata_handler
        handler.update_metadata(feature.metadata, **processed_metadata)

    def get_valid_time_series_fields(self) -> set:
        """
        Get the set of valid TimeSeriesMetadata field names.

        Returns:
            Set of valid field names for TimeSeriesMetadata

        Examples:
            >>> fields = manager.get_valid_time_series_fields()
            >>> 'signal_type' in fields
            True
        """
        return {f.name for f in fields(TimeSeriesMetadata)}

    def get_valid_feature_fields(self) -> set:
        """
        Get the set of valid FeatureMetadata field names.

        Returns:
            Set of valid field names for FeatureMetadata

        Examples:
            >>> fields = manager.get_valid_feature_fields()
            >>> 'feature_type' in fields
            True
        """
        return {f.name for f in fields(FeatureMetadata)}

    def validate_time_series_metadata_spec(self, metadata_spec: Dict[str, Any]) -> None:
        """
        Validate that metadata spec contains only valid TimeSeriesMetadata fields.

        Args:
            metadata_spec: Dictionary of metadata fields to validate

        Raises:
            ValueError: If metadata_spec contains invalid fields

        Examples:
            >>> manager.validate_time_series_metadata_spec(
            ...     {"signal_type": SignalType.HEART_RATE}
            ... )  # Valid
            >>> manager.validate_time_series_metadata_spec(
            ...     {"invalid_field": "value"}
            ... )  # Raises ValueError
        """
        valid_fields = self.get_valid_time_series_fields()
        invalid_fields = [field for field in metadata_spec if field not in valid_fields]
        if invalid_fields:
            raise ValueError(
                f"Invalid TimeSeriesMetadata fields: {invalid_fields}. "
                f"Valid fields: {sorted(valid_fields)}"
            )

    def validate_feature_metadata_spec(self, metadata_spec: Dict[str, Any]) -> None:
        """
        Validate that metadata spec contains only valid FeatureMetadata fields.

        Args:
            metadata_spec: Dictionary of metadata fields to validate

        Raises:
            ValueError: If metadata_spec contains invalid fields

        Examples:
            >>> manager.validate_feature_metadata_spec(
            ...     {"feature_type": FeatureType.HRV}
            ... )  # Valid
            >>> manager.validate_feature_metadata_spec(
            ...     {"invalid_field": "value"}
            ... )  # Raises ValueError
        """
        valid_fields = self.get_valid_feature_fields()
        invalid_fields = [field for field in metadata_spec if field not in valid_fields]
        if invalid_fields:
            raise ValueError(
                f"Invalid FeatureMetadata fields: {invalid_fields}. "
                f"Valid fields: {sorted(valid_fields)}"
            )
