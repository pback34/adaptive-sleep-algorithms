"""
Abstract base class for signal importers.

This module defines the SignalImporter abstract base class that serves
as the foundation for the importer hierarchy.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

from ..core.signal_data import SignalData
from ..utils import get_logger

class SignalImporter(ABC):
    """
    Abstract base class for signal importers.

    Defines the interface for importing signals from various sources.
    """
    
    def __init__(self):
        """Initialize the importer with a logger."""
        self.logger = get_logger(__name__)

    @abstractmethod
    def import_signal(self, source: str, signal_type: str) -> SignalData:
        """
        Import a single signal from the specified source.

        Args:
            source: Path or identifier of the data source (e.g., file path).
            signal_type: Type of the signal to import (e.g., "PPG").

        Returns:
            An instance of a SignalData subclass corresponding to the signal_type.
        """
        pass

    @abstractmethod
    def import_signals(self, source: str, signal_type: str) -> List[SignalData]:
        """
        Import multiple signals from the specified source.

        Args:
            source: Path or identifier of the data source (e.g., file path).
            signal_type: Type of the signals to import (e.g., "PPG").

        Returns:
            A list of SignalData subclass instances.
        """
        pass
