"""
Services for signal processing operations.

This module contains service classes that handle complex operations
on signal collections, extracted from SignalCollection to follow
Single Responsibility Principle.
"""

from .import_service import ImportService
from .alignment_service import AlignmentService
from .feature_service import FeatureService

__all__ = [
    'ImportService',
    'AlignmentService',
    'FeatureService'
]
