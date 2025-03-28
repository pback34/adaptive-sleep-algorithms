"""Main entry point for the sleep_analysis package."""

from .cli.run_workflow import run_workflow
import sys

if __name__ == "__main__":
    sys.exit(run_workflow())
