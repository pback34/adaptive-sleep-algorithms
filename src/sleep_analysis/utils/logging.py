"""
Logging configuration for the sleep analysis framework.

This module configures Python's standard logging module to provide consistent
and configurable logging across all components of the framework.
"""

import logging
import os
import time
from typing import Union, Optional, Dict, Any
from contextlib import contextmanager

def setup_logging(output_dir: Optional[str] = None, 
                 log_level: Union[int, str] = logging.INFO) -> None:
    """
    Configure the root logger for the framework.
    
    Sets up logging with a consistent format that includes module name and line number
    for precise traceability. Configures both file and console handlers.
    
    Args:
        output_dir: Directory for log files. If provided, logs will be written to 
                    <output_dir>/logs/workflow.log. If the directory doesn't exist,
                    it will be created automatically.
        log_level: Logging level (DEBUG, INFO, WARN, ERROR). Can be provided as a string
                   or as a logging module constant (e.g., logging.INFO).
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Convert string log level to logging constant if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Set the minimum level for logging
    logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Define log message format
    formatter = logging.Formatter('%(module)s:%(lineno)d %(funcName)s - %(levelname)s - %(message)s')
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Configure file handler if output_dir is provided
    if output_dir:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Set up file handler
        log_file = os.path.join(logs_dir, 'workflow.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a convenience function that returns a logger that inherits the
    configuration from the root logger.

    Args:
        name: Name of the logger, typically __name__ for module-specific loggers.

    Returns:
        A logger instance with the specified name.
    """
    return logging.getLogger(name)


@contextmanager
def log_operation(operation_name: str, logger: logging.Logger = None,
                  level: int = logging.INFO, **context):
    """
    Context manager for logging operation execution with timing.

    Automatically logs start and end of an operation, including execution time
    and any contextual information provided. Handles exceptions gracefully.

    Usage:
        >>> with log_operation("filter_lowpass", logger, cutoff_freq=10.0):
        ...     signal.filter_lowpass(cutoff_freq=10.0)

    Args:
        operation_name: Name of the operation being performed.
        logger: Logger instance to use. If None, uses root logger.
        level: Logging level for operation messages.
        **context: Additional context to include in log messages (e.g., parameters).

    Yields:
        Dictionary that can be used to store operation results or additional context.
    """
    if logger is None:
        logger = logging.getLogger()

    # Format context for logging
    context_str = ""
    if context:
        ctx_items = [f"{k}={v}" for k, v in context.items()]
        context_str = f" ({', '.join(ctx_items)})"

    # Create a result dictionary that can be updated during operation
    result_context = {}

    # Log operation start
    logger.log(level, f"Starting operation: {operation_name}{context_str}")
    start_time = time.time()

    try:
        yield result_context

        # Log successful completion
        elapsed = time.time() - start_time
        result_str = ""
        if result_context:
            result_items = [f"{k}={v}" for k, v in result_context.items()]
            result_str = f" | Results: {', '.join(result_items)}"

        logger.log(level,
                  f"Completed operation: {operation_name} in {elapsed:.3f}s{result_str}")

    except Exception as e:
        # Log failure with exception details
        elapsed = time.time() - start_time
        logger.error(
            f"Failed operation: {operation_name} after {elapsed:.3f}s | "
            f"Error: {type(e).__name__}: {str(e)}"
        )
        raise  # Re-raise the exception


class OperationLogger:
    """
    Helper class for tracking and logging sequences of operations.

    Maintains a history of operations performed and their outcomes,
    useful for debugging workflows and understanding processing pipelines.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the operation logger.

        Args:
            logger: Logger instance to use. If None, creates a new logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.operation_history: list[Dict[str, Any]] = []

    def log_step(self, step_name: str, status: str = "success",
                 duration: Optional[float] = None, **metadata):
        """
        Log a single operation step.

        Args:
            step_name: Name/description of the step.
            status: Status of the step ("success", "failed", "skipped").
            duration: Optional execution duration in seconds.
            **metadata: Additional metadata about the step.
        """
        record = {
            "step": step_name,
            "status": status,
            "timestamp": time.time(),
        }

        if duration is not None:
            record["duration"] = duration

        record.update(metadata)
        self.operation_history.append(record)

        # Log the step
        msg = f"Step: {step_name} - Status: {status}"
        if duration:
            msg += f" - Duration: {duration:.3f}s"
        if metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            msg += f" - {meta_str}"

        if status == "success":
            self.logger.info(msg)
        elif status == "failed":
            self.logger.error(msg)
        else:
            self.logger.warning(msg)

    def get_history(self) -> list[Dict[str, Any]]:
        """
        Get the full operation history.

        Returns:
            List of operation records with step names, statuses, and metadata.
        """
        return self.operation_history.copy()

    def summarize(self) -> Dict[str, Any]:
        """
        Generate a summary of all logged operations.

        Returns:
            Dictionary containing summary statistics.
        """
        if not self.operation_history:
            return {"total_steps": 0}

        total_duration = sum(
            record.get("duration", 0)
            for record in self.operation_history
        )

        status_counts = {}
        for record in self.operation_history:
            status = record["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_steps": len(self.operation_history),
            "total_duration": total_duration,
            "status_counts": status_counts,
            "steps": [r["step"] for r in self.operation_history]
        }

    def print_summary(self):
        """Print a formatted summary of operations."""
        summary = self.summarize()

        self.logger.info("=" * 60)
        self.logger.info("OPERATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Steps: {summary['total_steps']}")
        self.logger.info(f"Total Duration: {summary.get('total_duration', 0):.3f}s")

        if "status_counts" in summary:
            self.logger.info("Status Breakdown:")
            for status, count in summary["status_counts"].items():
                self.logger.info(f"  {status}: {count}")

        self.logger.info("=" * 60)
