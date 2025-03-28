"""
Command-line interface for executing workflows.

This module provides a command-line script for running workflow files
against specified data directories.
"""

import argparse
import logging
import os
import sys
import yaml
from typing import Dict, Any

from ..workflows.workflow_executor import WorkflowExecutor
from ..utils.logging import setup_logging


def parse_args():
    """
    Parse command-line arguments for the workflow runner.
    
    Returns:
        Namespace of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Execute a workflow file with a specified data directory.",
        epilog="""
Examples:
  python -m sleep_analysis.cli.run_workflow -w workflow.yaml -d ./data
  python -m sleep_analysis.cli.run_workflow --workflow analysis.yaml --data-dir ./data -o ./results -v
  python -m sleep_analysis.cli.run_workflow -w workflow.yaml -d ./data -o ./results -l DEBUG
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-w", "--workflow", 
        required=True,
        help="Path to the workflow YAML file."
    )
    parser.add_argument(
        "-d", "--data-dir", 
        required=True,
        help="Base directory containing the data files referenced in the workflow."
    )
    parser.add_argument(
        "-o", "--output-dir", 
        default="./output",
        help="Directory for output files (overrides the output_dir in export section if specified)."
    )
    parser.add_argument(
        "-l", "--log-level", 
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "-v", 
        action="store_const", 
        const="DEBUG", 
        dest="log_level",
        help="Set logging level to DEBUG (shorthand for --log-level DEBUG)"
    )
    
    return parser.parse_args()


def load_workflow(workflow_path: str) -> Dict[str, Any]:
    """
    Load a workflow configuration from a YAML file.
    
    Args:
        workflow_path: Path to the workflow YAML file.
        
    Returns:
        Dictionary containing the workflow configuration.
        
    Raises:
        FileNotFoundError: If the workflow file does not exist.
        yaml.YAMLError: If the workflow file is not valid YAML.
    """
    if not os.path.isfile(workflow_path):
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    
    try:
        with open(workflow_path, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing workflow YAML: {e}")


def run_workflow():
    """
    Run a workflow from the command line.
    
    Parses command-line arguments, loads the workflow configuration,
    and executes it with the specified data directory.
    
    Returns:
        0 on success, non-zero on error.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging early, before any other operations
    setup_logging(args.output_dir, args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting workflow execution")
        
        # Log the arguments
        logger.debug(f"Arguments: workflow={args.workflow}, data_dir={args.data_dir}, "
                     f"output_dir={args.output_dir}, log_level={args.log_level}")
        
        # Load workflow configuration
        workflow_config = load_workflow(args.workflow)
        logger.debug(f"Loaded workflow configuration from {args.workflow}")
        
        # Override output directory if specified
        if args.output_dir:
            logger.debug(f"Overriding output directory to {args.output_dir}")
            
            # Override export output directory
            if "export" in workflow_config:
                workflow_config["export"]["output_dir"] = args.output_dir
            
            # Override visualization output paths
            if "visualization" in workflow_config:
                for vis_config in workflow_config["visualization"]:
                    if "output" in vis_config:
                        original_path = vis_config["output"]
                        filename = os.path.basename(original_path)
                        
                        # Extract subdirectories (if any) after the first level
                        dirname = os.path.dirname(original_path)
                        if dirname:
                            parts = dirname.split(os.sep, 1)
                            if len(parts) > 1:
                                # Preserve subdirectories after the first level
                                vis_config["output"] = os.path.join(args.output_dir, parts[1], filename)
                            else:
                                vis_config["output"] = os.path.join(args.output_dir, filename)
                        else:
                            vis_config["output"] = os.path.join(args.output_dir, filename)
        
        # Check if data directory exists
        if not os.path.isdir(args.data_dir):
            logger.error(f"Data directory not found: {args.data_dir}")
            raise ValueError(f"Data directory not found: {args.data_dir}")
        
        # Create and execute workflow
        logger.info("Initializing workflow executor")
        executor = WorkflowExecutor(data_dir=args.data_dir)
        logger.info("Executing workflow")
        executor.execute_workflow(workflow_config)
        
        logger.info("Workflow executed successfully")
        return 0
        
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}", exc_info=True)
        print(f"File not found: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}", exc_info=True)
        print(f"Invalid workflow YAML: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        logger.error(f"Import error: {e}", exc_info=True)
        print(f"Failed to import required module: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error executing workflow: {e}", exc_info=True)
        print(f"Unexpected error executing workflow: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_workflow())
