"""Tests for the MetadataHandler class."""

import pytest
from datetime import datetime

from sleep_analysis.core.metadata_handler import MetadataHandler
from sleep_analysis.core.metadata import SignalMetadata
from sleep_analysis.signal_types import SignalType, SensorType, SensorModel, BodyPosition, Unit

def test_initialize_metadata():
    """Test initializing metadata with the handler."""
    handler = MetadataHandler()
    
    # Test with minimal fields
    metadata = handler.initialize_metadata(signal_id="test_signal")
    assert metadata.signal_id == "test_signal"
    assert metadata.name is None
    assert metadata.operations == []
    
    # Test with default values
    handler = MetadataHandler(default_values={"name": "Default Name", "units": Unit.BPM})
    metadata = handler.initialize_metadata(signal_id="test_signal_2")
    assert metadata.signal_id == "test_signal_2"
    assert metadata.name == "Default Name"
    assert metadata.units == Unit.BPM
    
    # Test with overrides
    metadata = handler.initialize_metadata(signal_id="test_signal_3", 
                                          name="Custom Name",
                                          units=Unit.G)
    assert metadata.signal_id == "test_signal_3"
    assert metadata.name == "Custom Name"
    assert metadata.units == Unit.G

def test_auto_generate_signal_id():
    """Test auto-generation of signal_id when not provided."""
    handler = MetadataHandler()
    metadata = handler.initialize_metadata()
    assert metadata.signal_id is not None
    assert isinstance(metadata.signal_id, str)
    assert len(metadata.signal_id) > 0

def test_filter_invalid_fields():
    """Test that invalid fields are filtered out."""
    handler = MetadataHandler()
    # Include a field not defined in SignalMetadata
    metadata = handler.initialize_metadata(signal_id="test_signal", 
                                          invalid_field="This should be ignored",
                                          source="This should also be ignored")
    
    # Verify the valid fields were set
    assert metadata.signal_id == "test_signal"
    
    # Verify invalid fields were filtered out
    assert not hasattr(metadata, "invalid_field")
    assert not hasattr(metadata, "source")

def test_update_metadata():
    """Test updating existing metadata."""
    handler = MetadataHandler()
    metadata = handler.initialize_metadata(signal_id="test_signal", name="Original Name")
    
    # Update with new values
    handler.update_metadata(metadata, name="Updated Name", units=Unit.BPM)
    
    # Verify updates were applied
    assert metadata.name == "Updated Name"
    assert metadata.units == Unit.BPM
    assert metadata.signal_id == "test_signal"  # Should not change
    
    # Try updating with invalid field (should be ignored)
    handler.update_metadata(metadata, invalid_field="Value")
    assert not hasattr(metadata, "invalid_field")

def test_set_name():
    """Test the name setting logic with different scenarios."""
    handler = MetadataHandler()
    
    # Test with explicit name
    metadata = handler.initialize_metadata(signal_id="test_id")
    handler.set_name(metadata, name="Explicit Name")
    assert metadata.name == "Explicit Name"
    
    # Test with key but no name
    metadata = handler.initialize_metadata(signal_id="test_id")
    handler.set_name(metadata, key="signal_key")
    assert metadata.name == "signal_key"
    
    # Test with neither name nor key
    metadata = handler.initialize_metadata(signal_id="abcdef1234567890")
    handler.set_name(metadata)
    assert metadata.name == "signal_abcdef12"  # Should use signal_id prefix
    
    # Test precedence (name over key)
    metadata = handler.initialize_metadata(signal_id="test_id")
    handler.set_name(metadata, name="Explicit Name", key="signal_key")
    assert metadata.name == "Explicit Name"
    
    # Test when name is already set (should not override)
    metadata = handler.initialize_metadata(signal_id="test_id", name="Original Name")
    handler.set_name(metadata, key="signal_key")
    assert metadata.name == "Original Name"

def test_record_operation():
    """Test recording operations in metadata."""
    handler = MetadataHandler()
    metadata = handler.initialize_metadata(signal_id="test_signal")
    
    # Record an operation
    handler.record_operation(metadata, "filter_lowpass", {"cutoff": 5.0})
    
    # Verify the operation was recorded
    assert len(metadata.operations) == 1
    assert metadata.operations[0].operation_name == "filter_lowpass"
    assert metadata.operations[0].parameters == {"cutoff": 5.0}
    
    # Record another operation
    handler.record_operation(metadata, "normalize", {})
    
    # Verify both operations are recorded in order
    assert len(metadata.operations) == 2
    assert metadata.operations[1].operation_name == "normalize"
