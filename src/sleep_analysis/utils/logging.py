"""
Logging configuration for the sleep analysis framework.

This module configures Python's standard logging module to provide consistent
and configurable logging across all components of the framework.
"""

import logging
import os
from typing import Union, Optional

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
