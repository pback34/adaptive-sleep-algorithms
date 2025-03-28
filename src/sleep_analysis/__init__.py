"""Flexible Signal Processing Framework for Sleep Analysis."""

# Framework version
__version__ = "0.1.0"

# Import main components for easier access
from .core import SignalData, SignalCollection, SignalMetadata, CollectionMetadata
from .signals import PPGSignal, AccelerometerSignal, TimeSeriesSignal, HeartRateSignal
from .signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit
from .export import ExportModule

__all__ = [
    'SignalData', 'SignalCollection', 'SignalMetadata', 'CollectionMetadata',
    'PPGSignal', 'AccelerometerSignal', 'TimeSeriesSignal', 'HeartRateSignal',
    'SignalType', 'SensorType', 'SensorModel', 'BodyPosition', 'Unit',
    'ExportModule'
]
