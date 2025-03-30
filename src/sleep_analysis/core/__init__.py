"""Core module for signal data and metadata classes."""

from .signal_data import SignalData
from .metadata import SignalMetadata, CollectionMetadata, OperationInfo
from .signal_collection import SignalCollection
from .metadata_handler import MetadataHandler

__all__ = ['SignalData', 'SignalMetadata', 'CollectionMetadata', 'OperationInfo', 'SignalCollection', 'MetadataHandler']
