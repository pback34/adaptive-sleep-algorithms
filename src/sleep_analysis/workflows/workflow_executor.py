"""
Workflow executor for processing signal data.

This module defines the WorkflowExecutor class for executing workflow definitions
specified in YAML/JSON format, including import, processing steps, and export.
"""

import os
import importlib
import warnings
from typing import Dict, Any, List, Optional, Type, Union, Callable

from ..core.signal_collection import SignalCollection
from ..core.signal_data import SignalData
from ..signal_types import SignalType, SensorType, SensorModel, BodyPosition
from ..utils import str_to_enum

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
    
    def execute_workflow(self, workflow_config: Dict[str, Any]):
        """
        Execute a workflow defined in a configuration dictionary.
        
        Args:
            workflow_config: Dictionary containing workflow definition with sections:
                - import: List of signal import specifications
                - steps: List of processing step specifications
                - export: Export configuration (optional)
                
        Raises:
            ValueError: If the workflow configuration is invalid or execution fails.
        """
        # Handle import section if present
        if "import" in workflow_config:
            self._process_import_section(workflow_config["import"])
            
        # Execute processing steps
        if "steps" in workflow_config:
            for step in workflow_config["steps"]:
                self.execute_step(step)
                
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
            # Handle collection-level operations
            if "type" in step and step["type"] == "collection":
                # Call the operation directly on the collection
                operation = getattr(self.container, operation_name, None)
                if not operation:
                    raise ValueError(f"Operation '{operation_name}' not found on SignalCollection")
                
                # Call the operation with parameters
                result = operation(**parameters)
                
                # Store the result with the output key if provided and not inplace
                if not inplace and "output" in step:
                    # Check if the result is a SignalData before adding it as a signal
                    from ..core.signal_data import SignalData
                    if isinstance(result, SignalData):
                        self.container.add_signal(output_name, result)
                    else:
                        # When a collection operation returns the collection itself,
                        # we don't need to add it as a signal
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"Collection operation '{operation_name}' returned {type(result).__name__}, not adding as signal")
                    
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
            spec["strict_validation"] = self.strict_validation
            
            # Validate required fields
            required_fields = ["signal_type", "importer", "source"]
            missing_fields = [field for field in required_fields if field not in spec]
            if missing_fields:
                raise ValueError(f"Import specification missing required fields: {missing_fields}")
            
            try:
                # Get importer instance
                importer = self._get_importer_instance(spec["importer"], spec.get("config", {}))
                if hasattr(importer, 'config'):
                    importer.config["timestamp_format"] = self.container.metadata.timestamp_format
                
                # Resolve source path
                source = spec["source"]
                if self.data_dir:
                    source = os.path.join(self.data_dir, source)
                
                # Import signals using the collection's method
                signals = self.container.import_signals_from_source(importer, source, spec)
                
                if not signals:
                    warnings.warn(f"No signals imported from {source}")
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
