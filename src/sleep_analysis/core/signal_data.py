"""
Abstract base class for signal data.

This module defines the SignalData abstract base class, which provides the foundation
for all signal types in the framework, including methods for accessing data and
applying operations with full traceability.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Callable, List, Tuple, TypeVar
import uuid
from datetime import datetime
import pandas as pd

from .metadata import SignalMetadata, OperationInfo
from ..signal_types import SignalType
from .. import __version__
from .metadata_handler import MetadataHandler

T = TypeVar('T', bound='SignalData')

class SignalData(ABC):
    """
    Abstract base class for all signal data.
    
    This class defines the common interface for all signal types, including
    methods for accessing data and applying operations.
    """
    _is_abstract = True
    # Class registry for operations
    registry: Dict[str, Tuple[Callable, Type['SignalData']]] = {}
    # Default empty list for required columns, to be overridden by subclasses
    required_columns = []
    
    def __init_subclass__(cls, **kwargs):
        """
        Initialize a subclass of SignalData.
        
        This method ensures subclasses inherit the registry properly without overwriting it.
        """
        super().__init_subclass__(**kwargs)
        # Ensure subclasses start with a copy of the parent registry
        if not hasattr(cls, 'registry'):
            cls.registry = {}
    
    # Signal type to be defined by subclasses
    signal_type: SignalType = None
    
    def __init__(self, data: Any, metadata: Optional[Dict[str, Any]] = None, handler: Optional[MetadataHandler] = None):
        """
        Initialize a SignalData instance.
        
        Args:
            data: The signal data (implementation-specific)
            metadata: Optional metadata dictionary
            handler: Optional metadata handler, will create one if not provided
        
        Raises:
            ValueError: If signal_type is not defined by the subclass
            ValueError: If required columns are missing from the data
        """
        # Only check signal_type for subclasses (not SignalData itself)
        if self.__class__ is not SignalData and self.signal_type is None:
            raise ValueError("Subclasses must define signal_type")
            
        # Validate that DataFrame has DatetimeIndex and contains required columns
        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Signal data must have a DatetimeIndex")
            
            if hasattr(self, 'required_columns') and self.required_columns:
                missing = [col for col in self.required_columns if col not in data.columns]
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")
        
        # Create or use the metadata handler
        self.handler = handler or MetadataHandler()
        
        # Initialize metadata using the handler
        metadata_kwargs = metadata or {}
        # Add signal_type to metadata if not explicitly provided
        if 'signal_type' not in metadata_kwargs:
            metadata_kwargs['signal_type'] = self.signal_type
            
        # Add framework version
        metadata_kwargs['framework_version'] = __version__
        
        # Initialize default empty collections if not provided
        for field in ['derived_from', 'operations', 'source_files']:
            if field not in metadata_kwargs:
                metadata_kwargs[field] = []
                
        # Initialize default dictionaries if not provided
        if 'sensor_info' not in metadata_kwargs:
            metadata_kwargs['sensor_info'] = {}
            
        # Initialize default booleans if not provided
        for field in ['temporary', 'merged']:
            if field not in metadata_kwargs:
                metadata_kwargs[field] = False

        # Initialize feature-specific defaults if not provided
        for field in ['feature_names', 'source_signal_keys']:
             if field not in metadata_kwargs:
                  metadata_kwargs[field] = []
        for field in ['epoch_window_length', 'epoch_step_size']:
             if field not in metadata_kwargs:
                  metadata_kwargs[field] = None

        # Use the handler to create the metadata
        self.metadata = self.handler.initialize_metadata(**metadata_kwargs)
        
        # Store the data
        self._data = data
    
    @abstractmethod
    def get_data(self) -> Any:
        """
        Get the signal data.
        
        Returns:
            The signal data in an implementation-specific format
        """
        pass
    
    @classmethod
    def get_registry(cls) -> Dict[str, Tuple[Callable, Type['SignalData']]]:
        """
        Get the combined registry from this class and its parent classes.
        
        This method properly handles inheritance by only including operations
        registered with this class and direct parent classes (not sibling classes).
        
        Returns:
            Dictionary mapping operation names to (function, output_class) tuples
        """
        all_registries = {}
        
        # First check if we have a class-specific registry dictionary
        for c in cls.__mro__:
            # Skip 'object' class
            if c is object:
                continue
                
            # Only get registries directly defined on the class
            # (not inherited from elsewhere)
            if 'registry' in c.__dict__:
                class_registry = c.__dict__['registry']
                
                # Add operations from this class's registry
                for op_name, (func, output_class) in class_registry.items():
                    # Only include if this operation wasn't already added from a more specific class
                    if op_name not in all_registries:
                        # Handle None output_class and ancestor class cases
                        if output_class is None:
                            # Use current class as output if None was specified
                            all_registries[op_name] = (func, cls)
                        elif issubclass(cls, output_class):
                            # If operation was registered to an ancestor class,
                            # but we're calling from a subclass, adjust the output class
                            all_registries[op_name] = (func, cls)
                        else:
                            all_registries[op_name] = (func, output_class)
                    
        return all_registries
    
    @classmethod
    def register(cls, operation_name: str, output_class: Type['SignalData'] = None):
        """
        Decorator for registering an operation with this signal class.
        
        Args:
            operation_name: Name of the operation
            output_class: Class of the output signal (defaults to the same class)
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            # Check if operation already exists and warn if overwriting
            if operation_name in cls.registry:
                import warnings
                warnings.warn(f"Operation '{operation_name}' is already registered. Overwriting.")
            
            # Ensure the class has its own registry dictionary
            if 'registry' not in cls.__dict__:
                cls.registry = {}
                
            # Default output class to this class if not specified
            output = output_class or cls
            cls.registry[operation_name] = (func, output)
            return func
        return decorator

    # Note: apply_operation is now concretely implemented in TimeSeriesSignal
    # to handle method/registry lookup and metadata.
    # Keep abstract here to enforce its presence, but subclasses like
    # TimeSeriesSignal will provide the actual working implementation.
    @abstractmethod
    def apply_operation(self, operation_name: str, inplace: bool = False, **parameters) -> 'SignalData':
        """
        Apply an operation to this signal by name.

        Concrete implementations (like in TimeSeriesSignal) handle method/registry
        lookup, execution, metadata updates, and inplace/new instance creation.

        Args:
            operation_name: String name of the operation.
            inplace: If True, attempts to modify this signal in place.
            **parameters: Keyword arguments passed to the operation's core logic.

        Returns:
            The resulting signal (self if inplace, or a new instance).

        Raises:
            NotImplementedError: If called on a class that doesn't implement it.
            ValueError: If operation not found, inplace fails, or core logic fails.
            AttributeError: If a non-callable attribute matches the operation name.
        """
        raise NotImplementedError("Subclasses like TimeSeriesSignal must implement apply_operation")

    def clear_data(self, skip_regeneration=False):
        """
        Clear the signal data to free memory.
        
        Args:
            skip_regeneration: If True, prevent automatic regeneration when get_data is called
        
        The data can be regenerated if needed using the operation history.
        """
        self._data = None
        # Store the flag to prevent regeneration if requested
        self._skip_regeneration = skip_regeneration

    def _regenerate_data(self):
        """
        Regenerate data based on operation history.

        This is called automatically by get_data() when data is None but operations exist.
        In this base implementation, regeneration is not fully supported without
        access to the original source signal (typically managed by a SignalCollection).
        Therefore, this method returns False to indicate regeneration failure,
        allowing tests to verify warning behavior in get_data().

        Returns:
            bool: False, indicating regeneration failed in this context.
        """
        # In a real scenario, this would need access to the source signal(s)
        # based on self.metadata.derived_from and potentially a SignalCollection.
        # Since that context isn't available here, we simulate failure.
        import warnings
        warnings.warn("SignalData._regenerate_data called, but full regeneration requires SignalCollection context. Simulating failure.", UserWarning)
        return False
