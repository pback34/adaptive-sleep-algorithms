"""
Workflow executor for processing signal data.

This module defines the WorkflowExecutor class for executing workflow definitions
specified in YAML/JSON format, including import, processing steps, and export.
"""

import os
import importlib
import warnings
import os # Added os import
from typing import Dict, Any, List, Optional, Type, Union, Callable

from ..core.signal_collection import SignalCollection
# Removed SignalData import as it's less directly used here
# Import TimeSeriesSignal and Feature for type checking
from ..signals.time_series_signal import TimeSeriesSignal
from ..features.feature import Feature
from ..signals.eeg_sleep_stage_signal import EEGSleepStageSignal
from ..signal_types import SignalType, SensorType, SensorModel, BodyPosition
from ..utils import str_to_enum

# Added imports for logging and timezone handling
import logging
try:
    import tzlocal
    import pytz # Optional: for validation
except ImportError:
    tzlocal = None
    # pytz = None # Keep commented if only used for optional validation

# Initialize logger for this module
logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Class for running workflow definitions.
    
    This class is a thin coordinator that delegates operations to a SignalCollection.
    It translates declarative YAML/JSON workflow definitions into method calls on
    a SignalCollection instance, which is where all the actual signal processing logic resides.
    """
    
    def __init__(self, container=None, strict_validation=True, data_dir=None):
        """
        Initialize a WorkflowExecutor.
        
        Args:
            container: SignalCollection to use for signal management. If None, a new one is created.
            strict_validation: If True, raises errors for missing signals or operations.
                               If False, issues warnings and skips invalid steps.
            data_dir: Base directory for resolving relative paths in the workflow's import section.
        """
        self.container = container or SignalCollection()
        self.strict_validation = strict_validation
        self.data_dir = data_dir
        self.default_input_timezone: Optional[str] = None
        self.target_timezone: str = "UTC" # Default target timezone

    def _resolve_target_timezone(self, target_tz_config: Optional[str]) -> str:
        """Resolves the target timezone based on config, system, or fallback."""
        if target_tz_config is None or target_tz_config.lower() in ["system", "local"]:
            if tzlocal:
                try:
                    system_tz = tzlocal.get_localzone_name()
                    # Optional: Validate system_tz before returning using pytz
                    # if pytz:
                    #     pytz.timezone(system_tz)
                    logger.info(f"Using system local timezone: {system_tz}")
                    return system_tz
                except Exception as e:
                    logger.warning(f"Failed to detect system timezone using tzlocal: {e}. Falling back to UTC.")
                    return "UTC"
            else:
                logger.warning("tzlocal library not found. Cannot detect system timezone. Falling back to UTC.")
                return "UTC"
        else:
            # Optional: Validate the provided timezone string using pytz
            # if pytz:
            #     try:
            #         pytz.timezone(target_tz_config)
            #     except pytz.UnknownTimeZoneError:
            #         logger.error(f"Invalid target_timezone specified: '{target_tz_config}'")
            #         raise ValueError(f"Invalid target_timezone specified: '{target_tz_config}'")
            logger.info(f"Using specified target timezone: {target_tz_config}")
            return target_tz_config

    def execute_workflow(self, workflow_config: Dict[str, Any]):
        """
        Execute a workflow defined in a configuration dictionary.
        
        Args:
            workflow_config: Dictionary containing workflow definition with sections:
                - import: List of signal import specifications
                - steps: List of processing step specifications
                - export: Export configuration (optional)
                - visualization: Visualization configuration (optional)

        Raises:
            ValueError: If the workflow configuration is invalid or execution fails.
        """
        # --- Timezone Configuration ---
        self.default_input_timezone = workflow_config.get("default_input_timezone") # Can be None
        if not self.default_input_timezone:
             logger.warning("No 'default_input_timezone' specified in workflow. Naive timestamps in source data without an 'origin_timezone' override may be handled ambiguously (often assumed UTC).")

        target_tz_config = workflow_config.get("target_timezone") # Can be None, 'system', or a specific zone
        self.target_timezone = self._resolve_target_timezone(target_tz_config)
        logger.info(f"Resolved target timezone for workflow: {self.target_timezone}")

        # Update collection metadata with the resolved target timezone
        if self.container:
            self.container.metadata.timezone = self.target_timezone
            logger.debug(f"Set SignalCollection timezone to: {self.target_timezone}")
        # --- End Timezone Configuration ---

        # --- Process Collection Settings (e.g., index_config) ---
        if "collection_settings" in workflow_config:
            settings = workflow_config["collection_settings"]
            if "index_config" in settings: # Check if index_config exists
                try: # Add try block
                    self.container.set_index_config(settings["index_config"]) # Correct indent
                    logger.info(f"Set collection index_config to: {settings['index_config']}") # Correct indent
                except ValueError as e: # Correct indent for except
                    logger.error(f"Invalid index_config in collection_settings: {e}") # Correct indent
                    raise # Re-raise error if config is invalid # Correct indent
            if "feature_index_config" in settings:
                try:
                    self.container.set_feature_index_config(settings["feature_index_config"]) # Correct indent
                    logger.info(f"Set collection feature_index_config to: {settings['feature_index_config']}")
                except ValueError as e:
                    logger.error(f"Invalid feature_index_config in collection_settings: {e}")
                    raise # Re-raise error if config is invalid
            if "epoch_grid_config" in settings:
                try:
                    # Store the config dict on the collection metadata
                    self.container.metadata.epoch_grid_config = settings["epoch_grid_config"]
                    logger.info(f"Set collection epoch_grid_config to: {settings['epoch_grid_config']}")
                except Exception as e: # Catch potential issues setting the attribute
                    logger.error(f"Failed to set epoch_grid_config on collection metadata: {e}")
                    raise # Re-raise error if setting fails
        # --- End Collection Settings ---

        # Handle import section if present
        if "import" in workflow_config:
            self._process_import_section(workflow_config["import"])

        # Execute processing steps
        if "steps" in workflow_config:
            for step in workflow_config["steps"]:
                self.execute_step(step)
        
        # Handle visualization section if present
        if "visualization" in workflow_config:
            self._process_visualization_section(workflow_config["visualization"])
                
        # Handle export section if present
        if "export" in workflow_config:
            self._process_export_section(workflow_config["export"])
    
    def execute_step(self, step: Dict[str, Any]):
        """
        Execute a single workflow step by delegating to SignalCollection.
        
        Args:
            step: Dictionary containing step specification.
                
        Raises:
            ValueError: If the step specification is invalid or execution fails.
        """
        # Validate required fields
        if "operation" not in step:
            raise ValueError("Step missing required 'operation' field")
            
        operation_name = step["operation"]
        parameters = step.get("parameters", {})
        inplace = step.get("inplace", False)
        output_key = step.get("output") # Use output_key for clarity

        # Validate output specification for non-inplace operations that produce output
        # Collection operations might not produce output to be stored via 'output' key
        produces_output = step.get("type") != "collection" and not inplace
        if produces_output and "output" not in step:
            raise ValueError(f"Step '{step.get('operation', 'unknown')}' requires 'output' key because it's non-inplace and not type 'collection'")

        try:
            step_type = step.get("type") # e.g., "collection", "time_series", "feature" (or inferred)

            # Handle collection-level operations using the new apply_operation method
            if step_type == "collection":
                # --- Handle Deprecated Operations First ---
                if operation_name in ['align_signals', 'generate_and_store_aligned_dataframe']:
                    error_msg = (f"Workflow operation '{operation_name}' is deprecated and removed. "
                                 f"Please update your workflow. Use 'generate_alignment_grid', "
                                 f"'apply_grid_alignment', 'combine_aligned_signals', or "
                                 f"'align_and_combine_signals' instead.")
                    logger.error(error_msg)
                    # Raise error immediately if strict, otherwise warn and skip
                    if self.strict_validation:
                        raise ValueError(error_msg)
                    else:
                        warnings.warn(error_msg + " Skipping step.")
                        return # Skip this step

                # --- Specific Handling for Operations with Top-Level Args ---
                # Updated combine_features call: output key is not used, config passed internally
                if operation_name == "combine_features":
                    inputs = step.get("inputs")
                    if inputs is None:
                        self._handle_error(ValueError(f"Collection operation '{operation_name}' requires 'inputs' defined in the workflow step."), operation_name=f"collection.{operation_name}")
                        return
                    try:
                        # Pass inputs from step, feature_index_config comes from collection metadata
                        self.container.combine_features(inputs=inputs)
                        logger.info(f"Successfully executed collection operation '{operation_name}'. Result stored internally.")
                    except Exception as e:
                        self._handle_error(e, operation_name=f"collection.{operation_name}")
                        return

                # --- Special handling for summarize_signals parameters ---
                elif operation_name == "summarize_signals":
                    try:
                        # Extract specific parameters for summarize_signals
                        fields = parameters.get("fields_to_include") # None if not present
                        print_flag = parameters.get("print_summary", True) # Default True
                        # Call directly with extracted params
                        self.container.summarize_signals(fields_to_include=fields, print_summary=print_flag)
                        logger.info(f"Successfully executed collection operation '{operation_name}' directly.")
                    except Exception as e:
                        self._handle_error(e, operation_name=f"collection.{operation_name}")
                        return # Stop processing this step

                # --- Handle generate_epoch_grid ---
                elif operation_name == "generate_epoch_grid":
                     try:
                          # Extract optional start/end time overrides from parameters
                          start_override = parameters.get("start_time")
                          end_override = parameters.get("end_time")
                          self.container.generate_epoch_grid(start_time=start_override, end_time=end_override)
                          logger.info(f"Successfully executed collection operation '{operation_name}'.")
                     except Exception as e:
                          self._handle_error(e, operation_name=f"collection.{operation_name}")
                          return

                # --- Fallback to Generic apply_operation for other collection ops ---
                else:
                    try:
                        # Pass parameters from the 'parameters' key in YAML
                        result = self.container.apply_operation(operation_name, **parameters)
                        # Logging is handled within apply_operation and the method itself
                    except Exception as e:
                        # Let _handle_error manage logging/raising based on strict_validation
                        self._handle_error(e, operation_name=f"collection.{operation_name}")
                        return # Stop processing this step

                # No need to handle 'output' or 'inplace' for most collection ops

            # Handle multi-signal operations (typically feature extraction)
            elif "inputs" in step:
                if inplace:
                    # While technically possible for some multi-signal ops, current design focuses on non-inplace feature generation
                    raise ValueError("Inplace operations currently not supported for multi-signal steps (feature extraction).")

                # Collect signal KEYS (metadata.name) from all input specifiers
                signal_keys = [] # Initialize signal_keys list here
                for input_spec in step["inputs"]:
                    signals = self.container.get_signals_from_input_spec(input_spec)
                    if not signals:
                         # If no signals found, handle based on strict_validation
                         if self.strict_validation:
                              raise ValueError(f"No signals found for input specifier '{input_spec}' in multi_signal step '{operation_name}'")
                         else:
                              warnings.warn(f"No signals found for input specifier '{input_spec}' in multi_signal step '{operation_name}'. Skipping this input.")
                              continue # Skip to the next input_spec in the list

                    # Ensure inputs are TimeSeriesSignals for feature extraction
                    if not all(isinstance(s, TimeSeriesSignal) for s in signals):
                         raise TypeError(f"Input specifier '{input_spec}' for multi_signal step '{operation_name}' resolved to non-TimeSeriesSignal objects.")
                    # Extend the list with the signal KEYS (stored in metadata.name)
                    signal_keys.extend([s.metadata.name for s in signals if s.metadata.name is not None]) # Use name (key)

                if not signal_keys:
                    # This check now happens after processing all input_specs
                    if self.strict_validation:
                        raise ValueError(f"No valid input TimeSeriesSignals resolved for operation '{operation_name}' after processing all input specifiers.")
                    else:
                        warnings.warn(f"No input TimeSeriesSignals found for operation '{operation_name}', skipping")
                        return

                # Remove step_size from parameters if present (it's global now)
                if 'step_size' in parameters:
                     del parameters['step_size']
                     logger.debug(f"Removed 'step_size' from parameters for feature operation '{operation_name}' (using global epoch grid).")

                # Apply the operation via the container
                result_object = self.container.apply_multi_signal_operation(
                    operation_name, signal_keys, parameters
                )

                # Add the result using the appropriate method based on type
                if isinstance(result_object, Feature):
                    self.container.add_feature(output_key, result_object)
                    logger.info(f"Stored Feature result with key '{output_key}'")
                elif isinstance(result_object, TimeSeriesSignal):
                    # This path might be used for future multi-signal ops producing TimeSeries
                    self.container.add_time_series_signal(output_key, result_object)
                    logger.info(f"Stored TimeSeriesSignal result with key '{output_key}'")
                else:
                    # Should not happen if registry is correct, but handle defensively
                    raise TypeError(f"Multi-signal operation '{operation_name}' returned unexpected type {type(result_object).__name__}")

            # Handle single signal operations (typically on TimeSeriesSignals)
            elif "input" in step:
                input_spec = step["input"]

                # Get the signals (can be TimeSeriesSignal or potentially Feature, though ops on Features are rare)
                signals = self.container.get_signals_from_input_spec(input_spec)
                if not signals:
                    if self.strict_validation:
                        raise ValueError(f"No signals found for input specifier '{input_spec}'")
                    else:
                        warnings.warn(f"No signals found for input specifier '{input_spec}', skipping")
                        return

                # --- Apply Operation to Resolved Signals ---
                processed_results = [] # Store results if not inplace

                for i, signal in enumerate(signals):
                    # Determine output key for non-inplace operations
                    current_output_key = None
                    if not inplace:
                        # Handle list vs single output key spec
                        if isinstance(output_key, list):
                            if i < len(output_key):
                                current_output_key = output_key[i]
                            else:
                                raise ValueError(f"Output key list length mismatch for input spec '{input_spec}'")
                        else: # Single output key string
                            current_output_key = f"{output_key}_{i}" if len(signals) > 1 else output_key

                        if current_output_key is None: # Should not happen if validation passed
                             raise ValueError("Could not determine output key.")

                    # Apply the operation
                    if inplace:
                        # Ensure signal has apply_operation (primarily TimeSeriesSignals)
                        if hasattr(signal, 'apply_operation'):
                            signal.apply_operation(operation_name, inplace=True, **parameters)
                            logger.debug(f"Applied inplace operation '{operation_name}' to '{signal.metadata.name}'")
                        else:
                             warnings.warn(f"Cannot apply inplace operation '{operation_name}' to object of type {type(signal).__name__} ('{signal.metadata.name}'). Skipping.")
                    else:
                        # Ensure signal has apply_operation
                        if hasattr(signal, 'apply_operation'):
                            result_object = signal.apply_operation(operation_name, **parameters) # inplace=False is default
                            processed_results.append((current_output_key, result_object))
                            logger.debug(f"Applied non-inplace operation '{operation_name}' to '{signal.metadata.name}'. Result key: '{current_output_key}'")
                        else:
                             warnings.warn(f"Cannot apply non-inplace operation '{operation_name}' to object of type {type(signal).__name__} ('{signal.metadata.name}'). Skipping.")


                # Store non-inplace results after processing all signals for this step
                if not inplace:
                    for key, result in processed_results:
                        # Add result using appropriate method
                        if isinstance(result, TimeSeriesSignal):
                            self.container.add_time_series_signal(key, result)
                        elif isinstance(result, Feature):
                            # This path is less common for single-signal ops but possible
                            self.container.add_feature(key, result)
                        else:
                            raise TypeError(f"Operation '{operation_name}' returned unexpected type {type(result).__name__}")
                        logger.info(f"Stored result with key '{key}' (type: {type(result).__name__})")

            # Neither collection-level, inputs, nor input was specified
            elif step_type != "collection": # Check step_type again
                raise ValueError("Step must specify 'input', 'inputs', or type 'collection'")
                
        except Exception as e:
            self._handle_error(e, operation_name, inplace)
    
    def _handle_error(self, error: Exception, operation_name: str = None, inplace: bool = False):
        """
        Handle an error based on validation settings.
        
        Args:
            error: The exception that occurred
            operation_name: Optional name of the operation that failed
            inplace: Whether the operation was in-place
        """
        if operation_name:
            operation_type = "in-place" if inplace else "non-in-place"
            error_msg = f"Error applying {operation_type} operation '{operation_name}': {error}"
        else:
            error_msg = f"Error: {error}"
        
        # Special handling for FileNotFoundError - always warn rather than fail
        if isinstance(error, FileNotFoundError):
            warnings.warn(error_msg)
            return
            
        if self.strict_validation:
            if isinstance(error, ValueError):
                raise ValueError(error_msg) from error
            else:
                raise RuntimeError(error_msg) from error
        else:
            warnings.warn(error_msg)
    
    def _get_importer_instance(self, importer_name: str, config: Dict[str, Any]):
        """
        Get an instance of the specified importer class.
        
        Args:
            importer_name: Name of the importer class
            config: Configuration for the importer
            
        Returns:
            Instance of the importer class
        """
        # Try different import paths
        import_paths = [
            f"..importers.{importer_name}",
            f"..importers.sensors.{importer_name.lower()}",
            f"..importers.formats.{importer_name.lower()}"
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path, package=__package__)
                if hasattr(module, importer_name):
                    importer_class = getattr(module, importer_name)
                    return importer_class(config)
            except (ImportError, ModuleNotFoundError):
                continue
                
        # Direct import from importers
        try:
            from ..importers import PolarCSVImporter, MergingImporter
            if importer_name == "PolarCSVImporter":
                return PolarCSVImporter(config)
            elif importer_name == "MergingImporter":
                return MergingImporter(config)
        except (ImportError, AttributeError):
            pass
        
        raise ImportError(f"Importer class {importer_name} not found")
    
    def _process_import_section(self, import_specs: List[Dict[str, Any]]):
        """
        Process the import section of a workflow.
        
        Args:
            import_specs: List of import specifications
        """
        # Dictionary to track counts for each base name
        base_name_counts = {}
        
        for spec in import_specs:
            # Add strict_validation to the spec for error handling
            # Add strict_validation to the spec for error handling within import_signals_from_source
            spec["strict_validation"] = self.strict_validation

            # Validate required fields
            required_fields = ["signal_type", "importer", "source"]
            missing_fields = [field for field in required_fields if field not in spec]
            if missing_fields:
                logger.error(f"Import specification missing required fields: {missing_fields}. Spec: {spec}")
                raise ValueError(f"Import specification missing required fields: {missing_fields}")

            try:
                # Prepare importer configuration, injecting timezones
                importer_config = spec.get("config", {}).copy()

                # Determine the origin timezone to use for this specific importer
                # Priority: importer's config -> workflow default -> None
                origin_timezone = importer_config.get("origin_timezone", self.default_input_timezone)
                logger.debug(f"Determined origin_timezone for importer '{spec['importer']}': {origin_timezone}")

                # Inject resolved target and determined origin timezones into the config passed to the importer
                importer_config["target_timezone"] = self.target_timezone
                importer_config["origin_timezone"] = origin_timezone # Pass the determined origin_timezone

                # Get importer instance with the updated config
                importer = self._get_importer_instance(spec["importer"], importer_config)

                # Inject timestamp format from collection metadata if importer has config
                if hasattr(importer, 'config') and importer.config is not None:
                     # Ensure config is a dictionary before accessing
                     if isinstance(importer.config, dict):
                          importer.config["timestamp_format"] = self.container.metadata.timestamp_format
                     else:
                          logger.warning(f"Importer '{spec['importer']}' config is not a dictionary, cannot set timestamp_format.")


                # Resolve source path relative to data_dir if provided
                source = spec["source"]
                if self.data_dir:
                    source = os.path.join(self.data_dir, source)
                
                try:
                    # Import signals using the collection's method
                    signals = self.container.import_signals_from_source(importer, source, spec)
                    
                    if not signals:
                        warnings.warn(f"No signals imported from {source}, skipping")
                        continue
                except FileNotFoundError as e:
                    # Handle file not found errors gracefully
                    warnings.warn(f"No files found for {spec['signal_type']} in {source}: {str(e)}")
                    continue
                except Exception as e:
                    # For strict validation, re-raise other errors
                    if self.strict_validation:
                        raise
                    # Otherwise, log a warning and continue
                    warnings.warn(f"Error importing {spec['signal_type']} from {source}: {str(e)}")
                    continue
                    
                # Get or create base name
                base_name = spec.get("base_name", spec["signal_type"].lower())
                
                # Initialize counter if this is a new base name
                if base_name not in base_name_counts:
                    base_name_counts[base_name] = 0
                
                # Add signals with incremental indices
                # Ensure we only process TimeSeriesSignals returned by the importer here
                time_series_signals_imported = [s for s in signals if isinstance(s, TimeSeriesSignal)]
                if len(time_series_signals_imported) != len(signals):
                     logger.warning(f"Importer for {spec['signal_type']} returned non-TimeSeriesSignal objects. Only TimeSeriesSignals will be added.")

                for signal in time_series_signals_imported:
                    # Update the signal's metadata from the spec using the correct method
                    self.container.update_time_series_metadata(signal, spec)

                    # Add to container with incremental index using the correct method
                    key = f"{base_name}_{base_name_counts[base_name]}"
                    try:
                         self.container.add_time_series_signal(key, signal)
                         base_name_counts[base_name] += 1
                         logger.info(f"Imported and added TimeSeriesSignal with key '{key}'")
                    except ValueError as add_err:
                         # Handle potential key collision if base_name_counts logic fails somehow
                         logger.error(f"Failed to add imported signal with key '{key}': {add_err}. Skipping this signal.")
                         # Don't increment count if add failed

            except Exception as e:
                self._handle_error(e)

    def _process_export_section(self, export_config: List[Dict[str, Any]]):
        """
        Process the export section of a workflow.

        Args:
            export_config: List of dictionaries, each containing an export configuration.
        """
        try:
            from ..export import ExportModule
        except ImportError:
            raise ImportError("ExportModule not found. Make sure the export module is available.")

        # Instantiate the exporter once, as it holds the collection reference
        exporter = ExportModule(self.container)

        # Iterate through each export configuration in the list
        for config_item in export_config:
            # Validate required fields for each item in the list
            required_fields = ["formats", "output_dir"]
            missing_fields = [f for f in required_fields if f not in config_item]
            if missing_fields:
                logger.error(f"Export configuration item missing required fields: {missing_fields}. Item: {config_item}")
                raise ValueError(f"Export configuration item missing required fields: {missing_fields}")

            # Set index configuration if provided for this specific export
            if "index_config" in config_item:
                try:
                    self.container.set_index_config(config_item["index_config"])
                    logger.debug(f"Temporarily set index_config for export to {config_item['output_dir']}")
                except ValueError as e:
                    logger.error(f"Invalid index_config in export item {config_item}: {e}")
                    raise  # Re-raise error

            # Call exporter's export method for the current configuration item
            logger.info(f"Executing export task to: {config_item['output_dir']}")
            try:
                # Extract parameters that match the ExportModule.export signature
                export_params = {
                    "formats": config_item["formats"],
                    "output_dir": config_item["output_dir"],
                    "include_combined": config_item.get("include_combined", False),
                    "include_summary": config_item.get("include_summary", False), # Add include_summary flag
                    "include_combined_features": config_item.get("include_combined_features", False) # Add missing flag
                }

                # Only add signals parameter if it exists in the config
                if "signals" in config_item:
                    export_params["signals"] = config_item["signals"]

                # Call export with the properly constructed parameters
                exporter.export(**export_params)
            except Exception as e:
                logger.error(f"Error during export task for {config_item['output_dir']}: {e}", exc_info=True)
                if self.strict_validation:
                    raise RuntimeError(f"Export task failed for {config_item['output_dir']}") from e
                else:
                    warnings.warn(f"Export task failed for {config_item['output_dir']}: {e}")


    def _process_visualization_section(self, vis_specs: List[Dict[str, Any]]):
        """
        Process the visualization section of a workflow.
        
        Args:
            vis_specs: List of visualization specifications
        """
        for spec in vis_specs:
            vis_type = spec.get('type', 'time_series') # Default to time_series if not specified
            backend = spec.get('backend', 'bokeh').lower()
            
            try:
                # Import the appropriate visualizer class
                if backend == 'bokeh':
                    from ..visualization import BokehVisualizer
                    visualizer = BokehVisualizer()
                elif backend == 'plotly':
                    from ..visualization import PlotlyVisualizer
                    visualizer = PlotlyVisualizer()
                else:
                    raise ValueError(f"Unsupported visualization backend: {backend}")

                # --- Process based on visualization type ---
                logger.info(f"Processing visualization type '{vis_type}' using {backend} backend: {spec.get('title', 'Untitled')}")

                if vis_type == 'hypnogram':
                    # Extract the sleep stage signal(s)
                    signals = self.container.get_signals(
                        input_spec=spec.get('signals'),
                        signal_type=SignalType.EEG_SLEEP_STAGE
                    )

                    if not signals:
                        warnings.warn(f"No EEG sleep stage signals found for hypnogram visualization spec: {spec.get('signals')}")
                        continue

                    # Process each signal and create visualizations
                    for i, signal in enumerate(signals):
                        # Ensure signal is of the correct type
                        if not isinstance(signal, EEGSleepStageSignal):
                             warnings.warn(f"Signal '{signal.metadata.signal_id}' is not an EEGSleepStageSignal, skipping hypnogram.")
                             continue

                        # Extract parameters
                        params = spec.get('parameters', {}).copy()
                        params['title'] = spec.get('title', f'Hypnogram - {signal.metadata.signal_id}') # Add title from spec

                        # Create the hypnogram
                        fig = visualizer.create_hypnogram_plot(signal, **params)

                        # Generate output filename
                        output = spec.get('output')
                        if output:
                            if len(signals) > 1:
                                # Add signal index for multiple signals
                                base, ext = os.path.splitext(output)
                                output_file = f"{base}_{i}{ext}"
                            else:
                                output_file = output

                            # Save the visualization
                            visualizer.save(fig, output_file, format=output_file.split('.')[-1], **params) # Pass params for save options
                        else:
                            # Show the visualization
                            visualizer.show(fig)

                elif vis_type == 'time_series':
                    # Existing logic for time_series plots
                    visualizer.process_visualization_config(spec, self.container)

                # Add other visualization types here (e.g., 'scatter', 'spectrogram') if needed
                else:
                     warnings.warn(f"Unsupported visualization type '{vis_type}' specified. Skipping.")

            except Exception as e:
                self._handle_error(e, operation_name=f"visualization ({vis_type}, {backend})")
