"""Tests for the SignalData abstract base class."""

import pytest
import pandas as pd
from sleep_analysis.core.signal_data import SignalData
from sleep_analysis.core.metadata import SignalMetadata
from sleep_analysis.signal_types import SignalType
from sleep_analysis import __version__
from sleep_analysis.signals import PPGSignal

# Create a minimal concrete implementation of SignalData for testing
class MockSignal(SignalData):
    """Concrete implementation of SignalData for testing."""
    signal_type = SignalType.PPG
    
    def get_data(self):
        """Return the signal data."""
        return self._data
    
    def apply_operation(self, operation_name, inplace=False, **parameters):
        """Apply an operation to this signal."""
        # For testing, just return self
        return self

def test_abstract_class():
    """Test that SignalData cannot be instantiated directly."""
    with pytest.raises(TypeError):
        # Cannot instantiate abstract class
        signal = SignalData(data=[1, 2, 3])

def test_concrete_subclass(sample_metadata, sample_dataframe):
    """Test that a concrete subclass can be instantiated."""
    signal = MockSignal(data=sample_dataframe, metadata=sample_metadata)
    assert signal.signal_type == SignalType.PPG
    assert signal.metadata.signal_id == sample_metadata["signal_id"]
    assert signal.get_data().equals(sample_dataframe)

def test_signal_type_required():
    """Test that subclasses must define signal_type."""
    # Create a subclass with undefined signal_type
    class InvalidSignal(SignalData):
        signal_type = None
        
        def get_data(self):
            return self._data
            
        def apply_operation(self, operation_name, inplace=False, **parameters):
            return self
    
    with pytest.raises(ValueError, match="Subclasses must define signal_type"):
        signal = InvalidSignal(data=[1, 2, 3])

def test_registry_inheritance():
    """Test that the registry is inherited correctly."""
    # Define parent class with an operation
    class ParentSignal(SignalData):
        signal_type = SignalType.PPG
        registry = {"operation1": (lambda x: x, None)}
        
        def get_data(self):
            return self._data
            
        def apply_operation(self, operation_name, inplace=False, **parameters):
            return self
    
    # Define child class with a different operation
    class ChildSignal(ParentSignal):
        registry = {"operation2": (lambda x: x, None)}
    
    # Get the registry for the child class
    registry = ChildSignal.get_registry()
    
    # Should contain both operations
    assert "operation1" in registry
    assert "operation2" in registry
    
    # Test __init_subclass__ works correctly
    class GrandchildSignal(ChildSignal):
        pass  # No explicit registry defined
        
        def get_data(self):
            return self._data
            
        def apply_operation(self, operation_name, inplace=False, **parameters):
            return self
    
    # Should inherit both operations via __init_subclass__
    assert hasattr(GrandchildSignal, 'registry')
    assert "operation1" in GrandchildSignal.get_registry()
    assert "operation2" in GrandchildSignal.get_registry()

def test_clear_data():
    """Test that clear_data removes data from the signal."""
    signal = MockSignal(data=[1, 2, 3], metadata={"signal_id": "test"})
    assert signal.get_data() == [1, 2, 3]
    
    signal.clear_data()
    assert signal.get_data() is None

def test_regenerate_data_after_clear(sample_metadata, sample_dataframe):
    """Test that a derived signal can regenerate its data after clearing."""
    metadata = sample_metadata.copy()
    metadata["signal_type"] = SignalType.PPG
    signal = PPGSignal(data=sample_dataframe, metadata=metadata)
    filtered_signal = signal.apply_operation("filter_lowpass", cutoff=5.0)
    original_data = filtered_signal.get_data().copy()
    
    # Clear data with skip_regeneration flag
    filtered_signal.clear_data(skip_regeneration=True)
    assert filtered_signal.get_data() is None
    
    # Now reset the skip_regeneration flag and regenerate the data by re-applying the operation
    filtered_signal._skip_regeneration = False
    with pytest.warns(UserWarning, match="Regeneration returned no data"):
        filtered_signal.apply_operation("filter_lowpass", cutoff=5.0, inplace=True)
    assert filtered_signal.get_data() is not None
    
    # Note: The regenerated test data has 5 rows, while the original may have more
    # We just verify that we have data with the right structure (correct columns)
    assert 'value' in filtered_signal.get_data().columns
    assert len(filtered_signal.get_data()) > 0

def test_signal_metadata_optional_fields():
    """Test that SignalMetadata handles optional fields correctly."""
    # Create metadata with only required fields
    metadata = SignalMetadata(signal_id="minimal_signal")
    assert metadata.signal_id == "minimal_signal"
    assert metadata.start_time is None
    assert metadata.operations == []
    assert not metadata.temporary
    assert metadata.framework_version == __version__

def test_register_operation_warning():
    """Test that registering an existing operation issues a warning."""
    class TestSignalWithRegistry(SignalData):
        signal_type = SignalType.PPG
        
        def get_data(self):
            return self._data
            
        def apply_operation(self, operation_name, inplace=False, **parameters):
            return self
    
    # Register an operation
    @TestSignalWithRegistry.register("test_op")
    def op1(data):
        return data
    
    # Register another operation with the same name
    import warnings
    with warnings.catch_warnings(record=True) as w:
        @TestSignalWithRegistry.register("test_op")
        def op2(data):
            return data
        
        assert len(w) > 0
        assert "already registered" in str(w[0].message)
