"""Core module for signal data and metadata classes."""

from .signal_data import SignalData
from .metadata import SignalMetadata, CollectionMetadata, OperationInfo
from .signal_collection import SignalCollection

__all__ = ['SignalData', 'SignalMetadata', 'CollectionMetadata', 'OperationInfo', 'SignalCollection']
