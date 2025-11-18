"""
Operation executor service for applying operations to signals and collections.

This module provides the OperationExecutor class, which handles:
- Collection-level operations (e.g., generate_alignment_grid)
- Multi-signal operations that produce new signals/features
- Single-signal operations
- Batch operations on multiple signals
"""

# Standard library imports
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type

# Local application imports
from ..models import EpochGridState
from ...signals.time_series_signal import TimeSeriesSignal
from ...features.feature import Feature
from ..signal_data import SignalData

# Initialize logger for the module
logger = logging.getLogger(__name__)


class OperationExecutor:
    """
    Service for executing operations on signals and collections.

    This service handles:
    - Collection-level operations registered via decorator
    - Multi-signal operations that create new signals or features
    - Single-signal operations with result storage
    - Batch operations on multiple signals

    The executor requires access to:
    - Operation registries (collection_operation_registry, multi_signal_registry)
    - Signal repository for retrieving and storing signals
    - Epoch grid state for feature operations

    Example:
        >>> executor = OperationExecutor(
        ...     collection_op_registry=collection_registry,
        ...     multi_signal_registry=multi_signal_registry,
        ...     get_time_series_signal=repo.get_time_series_signal,
        ...     add_time_series_signal=repo.add_time_series_signal,
        ...     add_feature=repo.add_feature,
        ...     epoch_state=epoch_state
        ... )
        >>> result = executor.apply_multi_signal_operation(
        ...     'feature_statistics',
        ...     ['hr_0', 'hr_1'],
        ...     {'stat_type': 'mean'}
        ... )
    """

    def __init__(
        self,
        collection_op_registry: Dict[str, Callable],
        multi_signal_registry: Dict[str, Tuple[Callable, Type[SignalData]]],
        get_time_series_signal: Callable[[str], TimeSeriesSignal],
        get_feature: Callable[[str], Feature],
        add_time_series_signal: Callable[[str, TimeSeriesSignal], None],
        add_feature: Callable[[str, Feature], None],
        epoch_state: Optional[EpochGridState] = None,
        feature_index_config: Optional[List[str]] = None,
        global_epoch_window_length: Optional[Any] = None,
        global_epoch_step_size: Optional[Any] = None
    ):
        """
        Initialize the OperationExecutor.

        Args:
            collection_op_registry: Registry of collection-level operations
            multi_signal_registry: Registry of multi-signal operations
            get_time_series_signal: Function to retrieve time-series signal by key
            get_feature: Function to retrieve feature by key
            add_time_series_signal: Function to add time-series signal
            add_feature: Function to add feature
            epoch_state: Optional epoch grid state for feature operations
            feature_index_config: Optional list of metadata fields for feature index
            global_epoch_window_length: Global window length for epoch operations
            global_epoch_step_size: Global step size for epoch operations
        """
        self.collection_op_registry = collection_op_registry
        self.multi_signal_registry = multi_signal_registry
        self.get_time_series_signal = get_time_series_signal
        self.get_feature = get_feature
        self.add_time_series_signal = add_time_series_signal
        self.add_feature = add_feature
        self.epoch_state = epoch_state
        self.feature_index_config = feature_index_config
        self.global_epoch_window_length = global_epoch_window_length
        self.global_epoch_step_size = global_epoch_step_size

    def apply_collection_operation(
        self,
        operation_name: str,
        collection_instance: Any,
        **parameters: Any
    ) -> Any:
        """
        Applies a registered collection-level operation by name.

        Looks up the operation in the collection_operation_registry and executes
        the corresponding method on the collection instance.

        Args:
            operation_name: The name of the operation to execute
            collection_instance: The SignalCollection instance to operate on
            **parameters: Keyword arguments to pass to the operation method

        Returns:
            The result returned by the executed operation method

        Raises:
            ValueError: If the operation_name is not found in the registry
            Exception: If the underlying operation method raises an exception

        Example:
            >>> result = executor.apply_collection_operation(
            ...     'combine_features',
            ...     collection,
            ...     inputs=['hr_features']
            ... )
        """
        logger.info(f"Applying collection operation '{operation_name}' with parameters: {parameters}")

        if operation_name not in self.collection_op_registry:
            logger.error(f"Collection operation '{operation_name}' not found in registry.")
            raise ValueError(f"Collection operation '{operation_name}' not found.")

        operation_method = self.collection_op_registry[operation_name]

        try:
            # Call the registered method, passing collection instance as first argument
            result = operation_method(collection_instance, **parameters)
            logger.info(f"Successfully applied collection operation '{operation_name}'.")
            return result
        except Exception as e:
            logger.error(f"Error executing collection operation '{operation_name}': {e}", exc_info=True)
            raise

    def apply_multi_signal_operation(
        self,
        operation_name: str,
        input_signal_keys: List[str],
        parameters: Dict[str, Any]
    ) -> Union[TimeSeriesSignal, Feature]:
        """
        Applies an operation that takes multiple signals as input and produces one output.

        Handles operations registered in multi_signal_registry. These operations
        typically create new signals or features from multiple input signals.

        Args:
            operation_name: Name of the operation (e.g., "feature_statistics")
            input_signal_keys: List of keys for the input TimeSeriesSignals
            parameters: Dictionary of parameters for the operation

        Returns:
            The resulting Feature or TimeSeriesSignal object

        Raises:
            ValueError: If operation is not found, inputs are invalid
            RuntimeError: If the operation execution fails or prerequisites not met

        Example:
            >>> feature = executor.apply_multi_signal_operation(
            ...     'feature_statistics',
            ...     ['hr_0', 'hr_1'],
            ...     {'stat_type': 'mean'}
            ... )
        """
        logger.info(f"Applying multi-signal operation '{operation_name}' to inputs: {input_signal_keys}")

        if operation_name not in self.multi_signal_registry:
            raise ValueError(f"Multi-signal operation '{operation_name}' not found in registry.")

        operation_func, output_class = self.multi_signal_registry[operation_name]

        # Input resolution and validation
        input_signals: List[TimeSeriesSignal] = []
        for key in input_signal_keys:
            try:
                signal = self.get_time_series_signal(key)
                input_signals.append(signal)
            except KeyError:
                raise ValueError(
                    f"Input TimeSeriesSignal key '{key}' not found for operation '{operation_name}'."
                )

        if not input_signals:
            raise ValueError(f"No valid input TimeSeriesSignals for operation '{operation_name}'.")

        # Prerequisite checks for feature operations
        is_feature_op = issubclass(output_class, Feature)
        if is_feature_op:
            if not self.epoch_state or not self.epoch_state.is_valid():
                raise RuntimeError(
                    f"Cannot execute feature operation '{operation_name}': "
                    f"generate_epoch_grid must be run first."
                )
            if self.epoch_state.epoch_grid_index is None or self.epoch_state.epoch_grid_index.empty:
                raise RuntimeError(
                    f"Cannot execute feature operation '{operation_name}': "
                    f"epoch_grid_index is None or empty."
                )

        # Function execution
        try:
            logger.debug(f"Executing operation function '{operation_func.__name__}'...")
            if is_feature_op:
                # Make a copy to avoid modifying the original dict
                params_copy = parameters.copy()
                # Remove epoch grid params if accidentally included
                params_copy.pop('epoch_grid_index', None)
                params_copy.pop('global_epoch_window_length', None)
                params_copy.pop('global_epoch_step_size', None)

                # Call feature function with explicit global args
                result_object = operation_func(
                    signals=input_signals,
                    epoch_grid_index=self.epoch_state.epoch_grid_index,
                    parameters=params_copy,
                    global_window_length=self.global_epoch_window_length,
                    global_step_size=self.global_epoch_step_size
                )
            else:
                # For non-feature ops, pass parameters as before
                result_object = operation_func(signals=input_signals, **parameters)

            logger.debug(f"Operation function '{operation_func.__name__}' completed.")
        except Exception as e:
            logger.error(
                f"Error executing multi-signal operation function '{operation_func.__name__}': {e}",
                exc_info=True
            )
            raise RuntimeError(f"Execution of operation '{operation_name}' failed.") from e

        # Result validation
        if not isinstance(result_object, output_class):
            raise TypeError(
                f"Operation '{operation_name}' returned unexpected type {type(result_object).__name__}. "
                f"Expected {output_class.__name__}."
            )

        # Metadata propagation for Feature outputs
        if isinstance(result_object, Feature):
            self._propagate_feature_metadata(result_object, input_signals, operation_name)

        logger.info(
            f"Successfully applied multi-signal operation '{operation_name}'. "
            f"Result type: {type(result_object).__name__}"
        )
        return result_object

    def apply_and_store_operation(
        self,
        signal_key: str,
        operation_name: str,
        parameters: Dict[str, Any],
        output_key: str
    ) -> Union[TimeSeriesSignal, Feature]:
        """
        Apply an operation to a TimeSeriesSignal and store the result.

        Args:
            signal_key: Key of the TimeSeriesSignal to operate on
            operation_name: Name of the operation to apply (must be registered
                          in the TimeSeriesSignal's registry)
            parameters: Parameters for the operation
            output_key: Key to use when storing the result (must be unique)

        Returns:
            The resulting TimeSeriesSignal that was stored

        Raises:
            KeyError: If the signal key doesn't exist or is not a TimeSeriesSignal
            ValueError: If the operation fails or output_key exists
            TypeError: If the operation returns an unexpected type

        Example:
            >>> result = executor.apply_and_store_operation(
            ...     'hr_0',
            ...     'bandpass_filter',
            ...     {'low_freq': 0.5, 'high_freq': 4.0},
            ...     'hr_filtered'
            ... )
        """
        signal = self.get_time_series_signal(signal_key)
        result = signal.apply_operation(operation_name, **parameters)

        # Check result type and add to appropriate storage
        if isinstance(result, TimeSeriesSignal):
            self.add_time_series_signal(output_key, result)
        else:
            raise TypeError(
                f"Operation '{operation_name}' on signal '{signal_key}' "
                f"returned unexpected type {type(result).__name__}"
            )

        return result

    def apply_operation_to_signals(
        self,
        signal_keys: List[str],
        operation_name: str,
        parameters: Dict[str, Any],
        inplace: bool = False,
        output_keys: Optional[List[str]] = None
    ) -> List[Union[TimeSeriesSignal, Feature]]:
        """
        Apply an operation to multiple TimeSeriesSignals.

        Args:
            signal_keys: List of keys for TimeSeriesSignals to operate on
            operation_name: Name of the operation to apply
            parameters: Parameters for the operation
            inplace: Whether to apply the operation in place
            output_keys: Keys for storing results (required if inplace=False)

        Returns:
            List of TimeSeriesSignals that were created or modified

        Raises:
            ValueError: If inplace=False and output_keys mismatch, or if a key
                       is not a valid TimeSeriesSignal key
            TypeError: If operation returns unexpected type

        Example:
            >>> results = executor.apply_operation_to_signals(
            ...     ['hr_0', 'hr_1'],
            ...     'normalize',
            ...     {},
            ...     inplace=True
            ... )
        """
        if not inplace and (not output_keys or len(output_keys) != len(signal_keys)):
            raise ValueError("Must provide matching output_keys when inplace=False")

        results = []
        for i, key in enumerate(signal_keys):
            signal = self.get_time_series_signal(key)

            if inplace:
                signal.apply_operation(operation_name, inplace=True, **parameters)
                results.append(signal)
            else:
                result = signal.apply_operation(operation_name, **parameters)
                output_key = output_keys[i]  # type: ignore

                # Check result type and add
                if isinstance(result, TimeSeriesSignal):
                    self.add_time_series_signal(output_key, result)
                else:
                    raise TypeError(
                        f"Operation '{operation_name}' on signal '{key}' "
                        f"returned unexpected type {type(result).__name__}"
                    )
                results.append(result)

        return results  # type: ignore

    def _propagate_feature_metadata(
        self,
        feature: Feature,
        input_signals: List[TimeSeriesSignal],
        operation_name: str
    ) -> None:
        """
        Propagate metadata from input signals to feature result.

        Args:
            feature: The Feature object to update
            input_signals: List of input TimeSeriesSignals
            operation_name: Name of the operation (for logging)
        """
        logger.debug(f"Propagating metadata for Feature result of '{operation_name}'...")
        feature_meta = feature.metadata
        fields_to_propagate = self.feature_index_config

        if fields_to_propagate:
            if len(input_signals) == 1:
                # Single input: Copy directly
                source_meta = input_signals[0].metadata
                for field in fields_to_propagate:
                    if hasattr(source_meta, field) and hasattr(feature_meta, field):
                        value = getattr(source_meta, field)
                        setattr(feature_meta, field, value)
                        logger.debug(f"  Propagated '{field}' = {value} (from single source)")
                    elif hasattr(feature_meta, field):
                        logger.debug(
                            f"  Field '{field}' exists in FeatureMetadata but not in source. "
                            f"Skipping propagation."
                        )

            elif len(input_signals) > 1:
                # Multiple inputs: Check for common values
                for field in fields_to_propagate:
                    if hasattr(feature_meta, field):
                        values = set()
                        all_sources_have_field = True
                        for source_signal in input_signals:
                            if hasattr(source_signal.metadata, field):
                                values.add(getattr(source_signal.metadata, field))
                            else:
                                all_sources_have_field = False
                                logger.debug(
                                    f"  Source signal '{source_signal.metadata.name}' "
                                    f"missing field '{field}' for propagation."
                                )
                                break

                        if not all_sources_have_field:
                            logger.debug(
                                f"  Field '{field}' not present in all sources. Setting to None."
                            )
                            setattr(feature_meta, field, None)
                        elif len(values) == 1:
                            common_value = values.pop()
                            setattr(feature_meta, field, common_value)
                            logger.debug(f"  Propagated '{field}' = {common_value} (common value)")
                        else:
                            # Different values found
                            setattr(feature_meta, field, "mixed")
                            logger.debug(f"  Propagated '{field}' = 'mixed' (values differ)")
        else:
            logger.debug("No feature_index_config set. Skipping metadata propagation.")

        # Ensure source signal IDs and keys are set
        if not feature_meta.source_signal_ids:
            feature_meta.source_signal_ids = [s.metadata.signal_id for s in input_signals]
        if not feature_meta.source_signal_keys:
            feature_meta.source_signal_keys = [s.metadata.name for s in input_signals]
