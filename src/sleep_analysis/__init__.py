"""Flexible Signal Processing Framework for Sleep Analysis."""

# Framework version
__version__ = "0.1.0" # Keep version consistent

# Import main components for easier access
# Updated imports for refactored metadata and feature classes
from .core import (
    SignalCollection, CollectionMetadata, TimeSeriesMetadata, FeatureMetadata,
    Feature, OperationInfo, FeatureType, MetadataHandler
)
from .signals import (
    SignalData, # Keep SignalData import here if it's still the base
    TimeSeriesSignal, PPGSignal, AccelerometerSignal, HeartRateSignal,
    MagnitudeSignal, AngleSignal, EEGSleepStageSignal
)
from .signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from .export import ExportModule
# Import visualization base if needed at top level
# from .visualization import VisualizerBase

# Define public API
__all__ = [
    # Core classes
    'SignalCollection', 'CollectionMetadata', 'TimeSeriesMetadata', 'FeatureMetadata',
    'Feature', 'OperationInfo', 'FeatureType', 'MetadataHandler',
    # Base Signal class
    'SignalData',
    # Concrete TimeSeriesSignal types
    'TimeSeriesSignal', 'PPGSignal', 'AccelerometerSignal', 'HeartRateSignal',
    'MagnitudeSignal', 'AngleSignal', 'EEGSleepStageSignal',
    # Enums
    'SignalType', 'SensorType', 'SensorModel', 'BodyPosition', 'Unit',
    # Modules
    'ExportModule',
    # Add other key components as needed, e.g., 'WorkflowExecutor', 'VisualizerBase'
]
