"""
Import service for handling signal imports.

This service encapsulates all logic related to importing signals from various sources,
updating metadata, and validating imported signals.
"""

import os
import glob
import warnings
import logging
from typing import Dict, Any, List
from dataclasses import fields

from ..signals.time_series_signal import TimeSeriesSignal
from ..features.feature import Feature
from ..core.metadata import TimeSeriesMetadata, FeatureMetadata, FeatureType
from ..signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from ..utils import str_to_enum
from ..core.metadata_handler import MetadataHandler

logger = logging.getLogger(__name__)


class ImportService:
    """
    Service for importing signals from various sources.

    This service handles:
    - Importing signals using importer instances
    - Validating imported signals
    - Updating signal metadata from specifications
    - Managing file patterns and batch imports
    """

    def __init__(self, metadata_handler: MetadataHandler = None):
        """
        Initialize the import service.

        Args:
            metadata_handler: Optional metadata handler for managing metadata updates
        """
        self.metadata_handler = metadata_handler or MetadataHandler()

    def import_signals_from_source(
        self,
        importer_instance,
        source: str,
        spec: Dict[str, Any]
    ) -> List[TimeSeriesSignal]:
        """
        Import TimeSeriesSignals from a source using the specified importer.

        Args:
            importer_instance: The importer instance to use.
            source: Source path or identifier.
            spec: Import specification containing configuration.

        Returns:
            List of imported TimeSeriesSignals.

        Raises:
            ValueError: If the source doesn't exist or no signals can be imported.
            TypeError: If the imported object is not a TimeSeriesSignal.
        """
        signal_type_str = spec["signal_type"]
        strict_validation = spec.get("strict_validation", True)

        # Determine expected output type based on signal_type_str
        expected_type = TimeSeriesSignal
        imported_objects: List[Any] = []

        # --- File Pattern Handling ---
        if "file_pattern" in spec:
            if not os.path.isdir(source):
                if strict_validation:
                    raise ValueError(f"Source directory not found: {source}")
                else:
                    warnings.warn(f"Source directory not found: {source}, skipping")
                    return []

            # Delegate pattern handling to importer if supported
            if hasattr(importer_instance, 'import_signals'):
                try:
                    imported_objects = importer_instance.import_signals(source, signal_type_str)
                except FileNotFoundError as e:
                    if strict_validation:
                        raise e
                    else:
                        warnings.warn(f"No files found matching pattern in {source} for importer: {e}")
                        return []
                except Exception as e:
                    if strict_validation:
                        raise
                    else:
                        warnings.warn(f"Error importing from {source} with pattern: {e}, skipping")
                        return []
            else:
                # Manual globbing if importer doesn't handle patterns
                file_pattern = os.path.join(source, spec["file_pattern"])
                matching_files = glob.glob(file_pattern)
                if not matching_files:
                    if strict_validation:
                        raise ValueError(f"No files found matching pattern: {file_pattern}")
                    else:
                        warnings.warn(f"No files found matching pattern: {file_pattern}, skipping")
                        return []

                for file_path in matching_files:
                    try:
                        signal_obj = importer_instance.import_signal(file_path, signal_type_str)
                        imported_objects.append(signal_obj)
                    except Exception as e:
                        if strict_validation:
                            raise
                        else:
                            warnings.warn(f"Error importing {file_path}: {e}, skipping")
        else:
            # --- Regular File Import ---
            if not os.path.exists(source):
                if strict_validation:
                    raise ValueError(f"Source file not found: {source}")
                else:
                    warnings.warn(f"Source file not found: {source}, skipping")
                    return []
            try:
                signal_obj = importer_instance.import_signal(source, signal_type_str)
                imported_objects.append(signal_obj)
            except Exception as e:
                if strict_validation:
                    raise
                else:
                    warnings.warn(f"Error importing {source}: {e}, skipping")
                    return []

        # --- Validate and Filter Results ---
        validated_signals: List[TimeSeriesSignal] = []
        for obj in imported_objects:
            if isinstance(obj, expected_type):
                if isinstance(obj, TimeSeriesSignal):
                    validated_signals.append(obj)
                else:
                    logger.warning(
                        f"Importer returned object of type {type(obj).__name__} "
                        f"which is not TimeSeriesSignal. Skipping."
                    )
            else:
                logger.warning(
                    f"Importer returned unexpected type {type(obj).__name__} "
                    f"(expected {expected_type.__name__}). Skipping."
                )

        return validated_signals

    def update_time_series_metadata(
        self,
        signal: TimeSeriesSignal,
        metadata_spec: Dict[str, Any]
    ) -> None:
        """
        Update a TimeSeriesSignal's metadata from a specification.

        Args:
            signal: The TimeSeriesSignal to update
            metadata_spec: Dictionary containing metadata fields to update

        Raises:
            TypeError: If signal is not a TimeSeriesSignal
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
        Update a Feature's metadata from a specification.

        Args:
            feature: The Feature to update
            metadata_spec: Dictionary containing metadata fields to update

        Raises:
            TypeError: If feature is not a Feature
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
                    f"Invalid enum value '{metadata_spec['feature_type']}' for field 'feature_type'. "
                    f"Skipping update."
                )

        # Add other valid FeatureMetadata fields
        valid_fields = {f.name for f in fields(FeatureMetadata)}
        for field in valid_fields:
            if field in metadata_spec and field not in processed_metadata:
                processed_metadata[field] = metadata_spec[field]

        # Use the metadata handler to update
        handler = feature.handler or self.metadata_handler
        handler.update_metadata(feature.metadata, **processed_metadata)
