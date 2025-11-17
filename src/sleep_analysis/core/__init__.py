"""Core module for signal data, features, metadata, and collection classes."""

from .signal_data import SignalData
# Updated metadata imports
from .metadata import TimeSeriesMetadata, FeatureMetadata, CollectionMetadata, OperationInfo, FeatureType
from .signal_collection import SignalCollection
from .metadata_handler import MetadataHandler
# Import Feature class from its new location
from ..features.feature import Feature
# Import validation utilities
from . import validation

__all__ = [
    'SignalData',
    'TimeSeriesMetadata', 'FeatureMetadata', 'CollectionMetadata', 'OperationInfo', 'FeatureType', # Updated metadata
    'Feature', # Added Feature
    'SignalCollection',
    'MetadataHandler',
    'validation'
    ]
