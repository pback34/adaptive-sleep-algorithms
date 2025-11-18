"""
Data import service for importing signals from external sources.

This module provides the DataImportService class, which handles:
- Importing time-series signals from files using importer instances
- File pattern matching and glob support
- Adding imported signals with sequential naming
- Validation and error handling
"""

# Standard library imports
import os
import glob
import warnings
import logging
from typing import List, Dict, Any, Callable

# Local application imports
from ...signals.time_series_signal import TimeSeriesSignal

# Initialize logger for the module
logger = logging.getLogger(__name__)


class DataImportService:
    """
    Service for importing signals from external data sources.

    This service handles:
    - Importing TimeSeriesSignals from files using importer instances
    - File pattern matching with glob support
    - Batch importing with automatic naming
    - Validation of imported signals
    - Error handling with strict/non-strict modes

    Example:
        >>> service = DataImportService(
        ...     add_time_series_signal=repo.add_time_series_signal
        ... )
        >>> signals = service.import_signals_from_source(
        ...     importer,
        ...     '/data/hr_files/',
        ...     {'signal_type': 'hr', 'file_pattern': '*.csv'}
        ... )
        >>> keys = service.add_imported_signals(signals, 'hr', start_index=0)
    """

    def __init__(self, add_time_series_signal: Callable[[str, TimeSeriesSignal], None]):
        """
        Initialize the DataImportService.

        Args:
            add_time_series_signal: Function to add time-series signal to repository
        """
        self.add_time_series_signal = add_time_series_signal

    def import_signals_from_source(
        self,
        importer_instance: Any,
        source: str,
        spec: Dict[str, Any]
    ) -> List[TimeSeriesSignal]:
        """
        Import TimeSeriesSignals from a source using the specified importer.

        Handles both single file imports and batch imports using file patterns.
        Supports strict and non-strict validation modes.

        Args:
            importer_instance: The importer instance to use (must have import_signal
                             or import_signals method)
            source: Source path (file or directory)
            spec: Import specification containing:
                - signal_type: Type of signal being imported
                - file_pattern: Optional glob pattern for batch imports
                - strict_validation: Whether to raise errors (default: True)

        Returns:
            List of imported TimeSeriesSignals

        Raises:
            ValueError: If source doesn't exist or no signals found (strict mode)
            TypeError: If imported object is not a TimeSeriesSignal (strict mode)

        Example:
            >>> signals = service.import_signals_from_source(
            ...     csv_importer,
            ...     '/data/hr/',
            ...     {'signal_type': 'hr', 'file_pattern': '*.csv', 'strict_validation': True}
            ... )
        """
        signal_type_str = spec["signal_type"]
        strict_validation = spec.get("strict_validation", True)
        expected_type = TimeSeriesSignal

        imported_objects: List[Any] = []

        # File pattern handling
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
                        warnings.warn(
                            f"No files found matching pattern in {source} for importer: {e}"
                        )
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
            # Regular file import
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

        # Validate and filter results
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

    def add_imported_signals(
        self,
        signals: List[TimeSeriesSignal],
        base_name: str,
        start_index: int = 0
    ) -> List[str]:
        """
        Add imported TimeSeriesSignals with sequential indexing.

        Creates keys using the pattern {base_name}_{index} and adds each signal
        to the repository. Handles key conflicts by incrementing the index.

        Args:
            signals: List of TimeSeriesSignals to add
            base_name: Base name for generating keys (e.g., 'hr' -> 'hr_0', 'hr_1')
            start_index: Starting index for sequential naming (default: 0)

        Returns:
            List of keys used to store the signals

        Example:
            >>> keys = service.add_imported_signals(
            ...     [signal1, signal2, signal3],
            ...     'hr',
            ...     start_index=0
            ... )
            >>> print(keys)
            ['hr_0', 'hr_1', 'hr_2']
        """
        keys = []
        current_index = start_index

        for signal in signals:
            if not isinstance(signal, TimeSeriesSignal):
                logger.warning(
                    f"Skipping object of type {type(signal).__name__} during add_imported_signals "
                    f"(expected TimeSeriesSignal)."
                )
                continue

            key = f"{base_name}_{current_index}"
            try:
                self.add_time_series_signal(key, signal)
                keys.append(key)
                current_index += 1
            except ValueError as e:
                # Handle case where key might already exist unexpectedly
                logger.error(f"Failed to add imported signal with key '{key}': {e}. Trying next index.")
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

        logger.info(f"Added {len(keys)} imported signals with base name '{base_name}'")
        return keys
