"""
Signal importers for the sleep analysis framework.

This module defines the importer class hierarchy for converting raw data
from various formats and sensor types into standardized signals.
"""

from .base import SignalImporter
from .formats.csv import CSVImporterBase
from .sensors.polar import PolarCSVImporter
from .sensors.enchanted_wave import EnchantedWaveImporter
from .merging import MergingImporter

__all__ = [
    'SignalImporter',       # Abstract base class
    'CSVImporterBase',      # Format-specific abstract class
    'PolarCSVImporter',     # Concrete sensor-specific class
    'EnchantedWaveImporter',# Concrete sensor-specific class
    'MergingImporter',      # Specialized importer for merging multiple files
]
