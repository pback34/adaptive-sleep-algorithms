"""
Unit tests for enhanced logging utilities.
"""

import pytest
import logging
import time
from src.sleep_analysis.utils.logging import log_operation, OperationLogger


class TestLogOperation:
    """Tests for the log_operation context manager."""

    @pytest.fixture
    def test_logger(self):
        """Create a test logger with a captured handler."""
        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        return logger

    def test_log_operation_success(self, test_logger, caplog):
        """Test log_operation with successful operation."""
        with caplog.at_level(logging.INFO):
            with log_operation("test_operation", test_logger):
                time.sleep(0.01)  # Simulate work

        records = [r for r in caplog.records if 'test_operation' in r.message]
        assert len(records) >= 2  # Start and completion messages

        start_msg = records[0].message
        complete_msg = records[-1].message

        assert "Starting operation: test_operation" in start_msg
        assert "Completed operation: test_operation" in complete_msg
        assert "in" in complete_msg and "s" in complete_msg  # Duration logged

    def test_log_operation_with_context(self, test_logger, caplog):
        """Test log_operation with contextual parameters."""
        with caplog.at_level(logging.INFO):
            with log_operation("filter_signal", test_logger,
                             cutoff_freq=10.0, order=4):
                pass

        start_msg = caplog.records[0].message
        assert "cutoff_freq=10.0" in start_msg
        assert "order=4" in start_msg

    def test_log_operation_with_results(self, test_logger, caplog):
        """Test log_operation with result context."""
        with caplog.at_level(logging.INFO):
            with log_operation("process_data", test_logger) as ctx:
                ctx['rows_processed'] = 100
                ctx['errors_found'] = 2

        complete_msg = caplog.records[-1].message
        assert "rows_processed=100" in complete_msg
        assert "errors_found=2" in complete_msg

    def test_log_operation_with_exception(self, test_logger, caplog):
        """Test log_operation when exception is raised."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="test error"):
                with log_operation("failing_operation", test_logger):
                    raise ValueError("test error")

        # Should have error log with timing
        error_records = [r for r in caplog.records
                        if r.levelname == 'ERROR' and 'failing_operation' in r.message]
        assert len(error_records) > 0

        error_msg = error_records[0].message
        assert "Failed operation: failing_operation" in error_msg
        assert "ValueError: test error" in error_msg


class TestOperationLogger:
    """Tests for the OperationLogger class."""

    @pytest.fixture
    def op_logger(self):
        """Create an OperationLogger instance."""
        return OperationLogger()

    def test_log_step_success(self, op_logger, caplog):
        """Test logging a successful step."""
        with caplog.at_level(logging.INFO):
            op_logger.log_step("load_data", status="success",
                              duration=1.5, rows=1000)

        assert len(op_logger.operation_history) == 1
        record = op_logger.operation_history[0]

        assert record['step'] == 'load_data'
        assert record['status'] == 'success'
        assert record['duration'] == 1.5
        assert record['rows'] == 1000
        assert 'timestamp' in record

        # Check log message
        log_msg = caplog.records[0].message
        assert 'load_data' in log_msg
        assert 'success' in log_msg
        assert '1.500s' in log_msg

    def test_log_step_failed(self, op_logger, caplog):
        """Test logging a failed step."""
        with caplog.at_level(logging.ERROR):
            op_logger.log_step("validate_data", status="failed",
                              error="Missing column")

        record = op_logger.operation_history[0]
        assert record['status'] == 'failed'
        assert record['error'] == 'Missing column'

        # Should log as ERROR
        assert caplog.records[0].levelname == 'ERROR'

    def test_log_step_skipped(self, op_logger, caplog):
        """Test logging a skipped step."""
        with caplog.at_level(logging.WARNING):
            op_logger.log_step("optional_step", status="skipped",
                              reason="Not needed")

        # Should log as WARNING
        assert caplog.records[0].levelname == 'WARNING'

    def test_get_history(self, op_logger):
        """Test retrieving operation history."""
        op_logger.log_step("step1", status="success")
        op_logger.log_step("step2", status="success")
        op_logger.log_step("step3", status="failed")

        history = op_logger.get_history()
        assert len(history) == 3
        assert history[0]['step'] == 'step1'
        assert history[2]['status'] == 'failed'

        # Should return a copy
        history.append({'step': 'fake'})
        assert len(op_logger.operation_history) == 3

    def test_summarize_empty(self, op_logger):
        """Test summarize with no operations."""
        summary = op_logger.summarize()
        assert summary['total_steps'] == 0

    def test_summarize_with_operations(self, op_logger):
        """Test summarize with multiple operations."""
        op_logger.log_step("step1", status="success", duration=1.0)
        op_logger.log_step("step2", status="success", duration=2.0)
        op_logger.log_step("step3", status="failed", duration=0.5)
        op_logger.log_step("step4", status="skipped")

        summary = op_logger.summarize()

        assert summary['total_steps'] == 4
        assert summary['total_duration'] == 3.5
        assert summary['status_counts'] == {
            'success': 2,
            'failed': 1,
            'skipped': 1
        }
        assert summary['steps'] == ['step1', 'step2', 'step3', 'step4']

    def test_print_summary(self, op_logger, caplog):
        """Test print_summary method."""
        op_logger.log_step("step1", status="success", duration=1.0)
        op_logger.log_step("step2", status="failed", duration=0.5)

        with caplog.at_level(logging.INFO):
            op_logger.print_summary()

        # Check that summary was logged
        summary_logs = [r.message for r in caplog.records
                       if 'OPERATION SUMMARY' in r.message or
                          'Total Steps' in r.message or
                          'Total Duration' in r.message]

        assert len(summary_logs) > 0
        total_steps_log = [m for m in summary_logs if 'Total Steps' in m]
        assert len(total_steps_log) > 0
        assert '2' in total_steps_log[0]

    def test_operation_logger_tracks_multiple_sequences(self, op_logger):
        """Test that OperationLogger correctly tracks sequences of operations."""
        # Simulate a workflow
        op_logger.log_step("load_config", status="success", duration=0.1)
        op_logger.log_step("validate_config", status="success", duration=0.2)
        op_logger.log_step("load_signals", status="success", duration=2.0, signal_count=3)
        op_logger.log_step("filter_signals", status="success", duration=1.5)
        op_logger.log_step("extract_features", status="failed", duration=0.8, error="Invalid params")

        history = op_logger.get_history()
        assert len(history) == 5

        # Check metadata is preserved
        assert history[2]['signal_count'] == 3
        assert history[4]['error'] == "Invalid params"

        summary = op_logger.summarize()
        assert summary['status_counts']['success'] == 4
        assert summary['status_counts']['failed'] == 1
        assert summary['total_duration'] == pytest.approx(4.6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
