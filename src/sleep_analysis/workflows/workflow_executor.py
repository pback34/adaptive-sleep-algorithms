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
from ..core.signal_data import SignalData
from ..signals.eeg_sleep_stage_signal import EEGSleepStageSignal # Added import
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
            if "index_config" in settings:
                try:
                    self.container.set_index_config(settings["index_config"])
                    logger.info(f"Set collection index_config to: {settings['index_config']}")
                except ValueError as e:
                    logger.error(f"Invalid index_config in collection_settings: {e}")
                    raise # Re-raise error if config is invalid
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
        output_name = step.get("output")
        
        # Validate output specification for non-inplace operations
        if not inplace and "output" not in step and step.get("type") != "collection":
            raise ValueError("Output must be specified for non-inplace operations")
        
        try:
            # Handle collection-level operations using the new apply_operation method
            if "type" in step and step["type"] == "collection":
                # --- Handle Deprecated Operations First ---
                # It's cleaner to check for deprecated names here before calling apply_operation
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

                # --- Call the generic apply_operation method ---
                try:
                    # Pass the operation name and parameters directly
                    # The container (SignalCollection) now handles the dispatch
                    result = self.container.apply_operation(operation_name, **parameters)
                    # Logging is handled within apply_operation and the method itself
                except Exception as e:
                    # Let _handle_error manage logging/raising based on strict_validation
                    self._handle_error(e, operation_name=f"collection.{operation_name}")
                    return # Stop processing this step if an error occurred

                # No need to handle 'output' or 'inplace' for collection ops currently defined

            # Handle multi-signal operations
            elif "inputs" in step:
                if inplace:
                    raise ValueError("Inplace operations not supported for multi-signal steps")
                
                # Collect signal IDs from all input specifiers
                signal_ids = []
                for input_spec in step["inputs"]:
                    signals = self.container.get_signals_from_input_spec(input_spec)
                    if not signals and self.strict_validation:
                        raise ValueError(f"No signals found for input specifier '{input_spec}'")
                    signal_ids.extend([s.metadata.signal_id for s in signals])
                
                if not signal_ids:
                    if self.strict_validation:
                        raise ValueError(f"No input signals found for operation '{operation_name}'")
                    else:
                        warnings.warn(f"No input signals found for operation '{operation_name}', skipping")
                        return
                
                # Apply the operation via the container
                result = self.container.apply_multi_signal_operation(
                    operation_name, signal_ids, parameters
                )
                self.container.add_signal(output_name, result)
                
            # Handle single signal operations
            elif "input" in step:
                input_spec = step["input"]
                
                # Get the signals
                signals = self.container.get_signals_from_input_spec(input_spec)
                if not signals:
                    if self.strict_validation:
                        raise ValueError(f"No signals found for input specifier '{input_spec}'")
                    else:
                        warnings.warn(f"No signals found for input specifier '{input_spec}', skipping")
                        return
                
                # Handle list of inputs with list of outputs
                if isinstance(input_spec, list) and isinstance(output_name, list):
                    if len(input_spec) != len(output_name):
                        raise ValueError("'input' and 'output' lists must have the same length")
                    
                    # Create pairs of input signals and output keys
                    pairs = []
                    for i, in_spec in enumerate(input_spec):
                        in_signals = self.container.get_signals_from_input_spec(in_spec)
                        for j, signal in enumerate(in_signals):
                            out_key = f"{output_name[i]}_{j}" if len(in_signals) > 1 else output_name[i]
                            pairs.append((signal, out_key))
                    
                    # Apply operations using the pairs
                    for signal, out_key in pairs:
                        if inplace:
                            signal.apply_operation(operation_name, inplace=True, **parameters)
                        else:
                            result = signal.apply_operation(operation_name, **parameters)
                            self.container.add_signal(out_key, result)
                
                # Handle regular single signal operations
                else:
                    # For in-place operations, modify signals directly
                    if inplace:
                        for signal in signals:
                            signal.apply_operation(operation_name, inplace=True, **parameters)
                    # For non-in-place operations, create new signals
                    else:
                        for i, signal in enumerate(signals):
                            out_key = f"{output_name}_{i}" if len(signals) > 1 else output_name
                            result = signal.apply_operation(operation_name, **parameters)
                            self.container.add_signal(out_key, result)
            
            # Neither collection-level, inputs, nor input was specified
            elif step.get("type") != "collection":
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
                for signal in signals:
                    # Update the signal's metadata from the spec
                    self.container.update_signal_metadata(signal, spec)
                    
                    # Add to container with incremental index
                    key = f"{base_name}_{base_name_counts[base_name]}"
                    self.container.add_signal(key, signal)
                    base_name_counts[base_name] += 1
                    
            except Exception as e:
                self._handle_error(e)
    
    def _process_export_section(self, export_config: Dict[str, Any]):
        """
        Process the export section of a workflow.
        
        Args:
            export_config: Dictionary with export configuration
        """
        # Validate required fields
        required_fields = ["formats", "output_dir"]
        missing_fields = [f for f in required_fields if f not in export_config]
        if missing_fields:
            raise ValueError(f"Export configuration missing required fields: {missing_fields}")
            
        try:
            from ..export import ExportModule
        except ImportError:
            raise ImportError("ExportModule not found. Make sure the export module is available.")
            
        # Set index configuration if provided
        if "index_config" in export_config:
            self.container.set_index_config(export_config["index_config"])
            
        # Create and use exporter
        exporter = ExportModule(self.container)
        exporter.export(
            formats=export_config["formats"],
            output_dir=export_config["output_dir"],
            include_combined=export_config.get("include_combined", False)
        )
    
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
