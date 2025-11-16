"""
Unit tests for enhanced workflow step validation.
"""

import pytest
import pandas as pd
from src.sleep_analysis.workflows.workflow_executor import WorkflowExecutor
from src.sleep_analysis.core.signal_collection import SignalCollection
from src.sleep_analysis.signals.time_series_signal import TimeSeriesSignal


class TestWorkflowStepValidation:
    """Tests for comprehensive workflow step validation."""

    @pytest.fixture
    def executor_with_collection(self):
        """Create a WorkflowExecutor with a SignalCollection."""
        executor = WorkflowExecutor()
        executor.container = SignalCollection()
        return executor

    @pytest.fixture
    def sample_signal(self):
        """Create a sample TimeSeriesSignal for testing."""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1s'))

        return TimeSeriesSignal(
            data=data,
            metadata={
                'name': 'test_signal',
                'signal_id': 'test_id',
                'sampling_rate': 1.0
            }
        )

    def test_validate_step_missing_operation(self, executor_with_collection):
        """Test validation fails when 'operation' field is missing."""
        step = {'parameters': {}}

        with pytest.raises(ValueError, match="Step missing required 'operation' field"):
            executor_with_collection._validate_step(step)

    def test_validate_step_invalid_operation_type(self, executor_with_collection):
        """Test validation fails when 'operation' is not a string."""
        step = {'operation': 123}

        with pytest.raises(ValueError, match="'operation' must be a non-empty string"):
            executor_with_collection._validate_step(step)

    def test_validate_step_empty_operation(self, executor_with_collection):
        """Test validation fails when 'operation' is empty."""
        step = {'operation': ''}

        with pytest.raises(ValueError, match="'operation' must be a non-empty string"):
            executor_with_collection._validate_step(step)

    def test_validate_step_invalid_step_type(self, executor_with_collection):
        """Test validation fails for invalid step type."""
        step = {
            'operation': 'test_op',
            'type': 'invalid_type'
        }

        with pytest.raises(ValueError, match="Invalid step type 'invalid_type'"):
            executor_with_collection._validate_step(step)

    def test_validate_step_invalid_parameters_type(self, executor_with_collection):
        """Test validation fails when parameters is not a dict."""
        step = {
            'operation': 'test_op',
            'parameters': 'not_a_dict'
        }

        with pytest.raises(TypeError, match="'parameters' must be a dictionary"):
            executor_with_collection._validate_step(step)

    def test_validate_step_invalid_inplace_type(self, executor_with_collection):
        """Test validation fails when inplace is not a boolean."""
        step = {
            'operation': 'test_op',
            'inplace': 'true'  # String instead of boolean
        }

        with pytest.raises(TypeError, match="'inplace' must be a boolean"):
            executor_with_collection._validate_step(step)

    def test_validate_step_conflicting_inputs(self, executor_with_collection):
        """Test validation fails with both 'input' and 'inputs' fields."""
        step = {
            'operation': 'test_op',
            'input': 'signal_a',
            'inputs': ['signal_b', 'signal_c']
        }

        with pytest.raises(ValueError, match="cannot have both 'input' and 'inputs'"):
            executor_with_collection._validate_step(step)

    def test_validate_step_missing_input_non_collection(self, executor_with_collection):
        """Test validation requires input for non-collection operations."""
        step = {
            'operation': 'filter_lowpass',  # Non-collection operation
            'parameters': {'cutoff': 10.0}
        }

        with pytest.raises(ValueError, match="requires 'input' or 'inputs' field"):
            executor_with_collection._validate_step(step)

    def test_validate_step_invalid_input_type(self, executor_with_collection):
        """Test validation fails with invalid input type."""
        step = {
            'operation': 'filter_lowpass',
            'input': 123  # Should be string or dict
        }

        with pytest.raises(TypeError, match="'input' must be a string or dictionary"):
            executor_with_collection._validate_step(step)

    def test_validate_step_empty_inputs_list(self, executor_with_collection):
        """Test validation fails with empty inputs list."""
        step = {
            'operation': 'test_op',
            'inputs': []
        }

        with pytest.raises(ValueError, match="'inputs' list cannot be empty"):
            executor_with_collection._validate_step(step)

    def test_validate_step_invalid_inputs_element(self, executor_with_collection):
        """Test validation fails with invalid element in inputs list."""
        step = {
            'operation': 'test_op',
            'inputs': ['signal_a', 123, 'signal_b']  # 123 is invalid
        }

        with pytest.raises(TypeError, match="'inputs\\[1\\]' must be a string or dictionary"):
            executor_with_collection._validate_step(step)

    def test_validate_step_missing_output_for_non_inplace(self, executor_with_collection):
        """Test validation requires output for non-inplace operations."""
        step = {
            'operation': 'filter_lowpass',
            'input': 'signal_a',
            'parameters': {'cutoff': 10.0},
            'inplace': False
        }

        with pytest.raises(ValueError, match="requires 'output' key"):
            executor_with_collection._validate_step(step)

    def test_validate_step_invalid_output_type(self, executor_with_collection):
        """Test validation fails with invalid output type."""
        step = {
            'operation': 'test_op',
            'input': 'signal_a',
            'output': 123  # Should be string or list
        }

        with pytest.raises(TypeError, match="'output' must be a string or list"):
            executor_with_collection._validate_step(step)

    def test_validate_step_empty_output_list(self, executor_with_collection):
        """Test validation fails with empty output list."""
        step = {
            'operation': 'test_op',
            'input': 'signal_a',
            'output': []
        }

        with pytest.raises(ValueError, match="'output' list cannot be empty"):
            executor_with_collection._validate_step(step)

    def test_validate_step_valid_collection_operation(self, executor_with_collection):
        """Test validation succeeds for valid collection operation."""
        step = {
            'operation': 'generate_epoch_grid',
            'type': 'collection',
            'parameters': {}
        }

        # Should not raise exception
        # Need to set up epoch_grid_config first
        executor_with_collection.container.metadata.epoch_grid_config = {
            'window_length': pd.Timedelta('30s'),
            'step_size': pd.Timedelta('15s')
        }

        validated = executor_with_collection._validate_step(step)
        assert validated['operation_name'] == 'generate_epoch_grid'
        assert validated['step_type'] == 'collection'

    def test_validate_step_valid_time_series_operation(self, executor_with_collection):
        """Test validation succeeds for valid time-series operation."""
        step = {
            'operation': 'filter_lowpass',
            'input': 'signal_a',
            'output': 'signal_a_filtered',
            'parameters': {'cutoff': 10.0}
        }

        validated = executor_with_collection._validate_step(step)
        assert validated['operation_name'] == 'filter_lowpass'
        assert validated['inplace'] is False
        assert validated['output_key'] == 'signal_a_filtered'


class TestOperationRequirementsValidation:
    """Tests for operation-specific requirements validation."""

    @pytest.fixture
    def executor_with_collection(self):
        """Create a WorkflowExecutor with a SignalCollection."""
        executor = WorkflowExecutor()
        executor.container = SignalCollection()
        return executor

    def test_feature_extraction_requires_epoch_grid(self, executor_with_collection):
        """Test feature extraction operations require epoch grid."""
        step = {
            'operation': 'feature_statistics',
            'input': 'signal_a',
            'parameters': {'aggregations': ['mean', 'std']}
        }

        # Should fail because epoch grid not generated
        with pytest.raises(ValueError, match="requires epoch grid to be generated first"):
            executor_with_collection._validate_step(step)

        # Mark epoch grid as calculated
        executor_with_collection.container._epoch_grid_calculated = True

        # Now should succeed
        validated = executor_with_collection._validate_step(step)
        assert validated is not None

    def test_feature_extraction_validates_aggregations(self, executor_with_collection):
        """Test validation of aggregation parameters."""
        executor_with_collection.container._epoch_grid_calculated = True

        # Invalid aggregation type (not a list)
        step = {
            'operation': 'feature_statistics',
            'input': 'signal_a',
            'parameters': {'aggregations': 'mean'}  # Should be list
        }

        with pytest.raises(TypeError, match="'aggregations' parameter must be a list"):
            executor_with_collection._validate_step(step)

        # Empty aggregations list
        step['parameters']['aggregations'] = []
        with pytest.raises(ValueError, match="'aggregations' list cannot be empty"):
            executor_with_collection._validate_step(step)

        # Invalid aggregation function
        step['parameters']['aggregations'] = ['mean', 'invalid_agg']
        with pytest.raises(ValueError, match="Invalid aggregations:"):
            executor_with_collection._validate_step(step)

    def test_combine_features_requires_existing_features(self, executor_with_collection):
        """Test combine_features validation requires features to exist."""
        step = {
            'operation': 'combine_features',
            'type': 'collection',
            'inputs': ['feature_a', 'feature_b']
        }

        # Should fail because no features exist
        with pytest.raises(ValueError, match="No features available to combine"):
            executor_with_collection._validate_step(step)

    def test_alignment_operations_require_alignment_grid(self, executor_with_collection):
        """Test alignment operations require alignment grid."""
        step = {
            'operation': 'apply_grid_alignment',
            'type': 'collection'
        }

        # Should fail because alignment grid not generated
        with pytest.raises(ValueError, match="requires alignment grid to be generated first"):
            executor_with_collection._validate_step(step)

        # Mark alignment params as calculated
        executor_with_collection.container._alignment_params_calculated = True

        # Now should succeed
        validated = executor_with_collection._validate_step(step)
        assert validated is not None

    def test_generate_epoch_grid_validates_config(self, executor_with_collection):
        """Test generate_epoch_grid validation checks epoch_grid_config."""
        step = {
            'operation': 'generate_epoch_grid',
            'type': 'collection'
        }

        # Should fail because epoch_grid_config not set
        with pytest.raises(ValueError, match="'epoch_grid_config' not set"):
            executor_with_collection._validate_step(step)

        # Set incomplete config (missing step_size)
        executor_with_collection.container.metadata.epoch_grid_config = {
            'window_length': pd.Timedelta('30s')
        }

        with pytest.raises(ValueError, match="missing required fields"):
            executor_with_collection._validate_step(step)

        # Set complete config
        executor_with_collection.container.metadata.epoch_grid_config = {
            'window_length': pd.Timedelta('30s'),
            'step_size': pd.Timedelta('15s')
        }

        # Now should succeed
        validated = executor_with_collection._validate_step(step)
        assert validated is not None

    def test_step_size_parameter_warning(self, executor_with_collection, caplog):
        """Test warning when step_size parameter is provided (now global)."""
        import logging
        executor_with_collection.container._epoch_grid_calculated = True

        step = {
            'operation': 'feature_statistics',
            'input': 'signal_a',
            'parameters': {
                'aggregations': ['mean'],
                'step_size': '15s'  # Should trigger warning
            }
        }

        with caplog.at_level(logging.WARNING):
            executor_with_collection._validate_step(step)

        # Check for warning about step_size being ignored
        warnings = [r for r in caplog.records if 'step_size' in r.message]
        assert len(warnings) > 0
        assert 'ignored' in warnings[0].message.lower()


class TestExecuteStepWithValidation:
    """Tests for execute_step method with validation."""

    @pytest.fixture
    def executor_with_signal(self):
        """Create executor with a sample signal."""
        executor = WorkflowExecutor()
        executor.container = SignalCollection()

        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1s'))

        signal = TimeSeriesSignal(
            data=data,
            metadata={
                'name': 'test_signal',
                'signal_id': 'test_id',
                'sampling_rate': 1.0
            }
        )
        executor.container.add_time_series_signal('test_signal', signal)
        return executor

    def test_execute_step_validation_failure_strict(self, executor_with_signal):
        """Test execute_step with strict validation on invalid step."""
        executor_with_signal.strict_validation = True

        invalid_step = {
            'operation': '',  # Invalid empty operation
            'input': 'test_signal'
        }

        with pytest.raises(ValueError):
            executor_with_signal.execute_step(invalid_step)

    def test_execute_step_validation_failure_non_strict(self, executor_with_signal, caplog):
        """Test execute_step with non-strict validation skips invalid step."""
        import logging
        executor_with_signal.strict_validation = False

        invalid_step = {
            'operation': '',  # Invalid empty operation
            'input': 'test_signal'
        }

        with caplog.at_level(logging.WARNING):
            executor_with_signal.execute_step(invalid_step)

        # Should log warning and skip
        warnings = [r for r in caplog.records if 'Skipping invalid step' in r.message]
        assert len(warnings) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
