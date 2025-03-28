# Sleep Analysis Framework

A flexible, extensible framework for processing sleep-related signals, designed for researchers and developers working with physiological data. The framework provides a robust foundation for signal processing with an emphasis on reproducibility, type safety, and memory efficiency.

## Key Features

- **Type-Safe Signal Processing**: Enum-based type safety ensures operations match signal types
- **Complete Traceability**: Full metadata and operation history for reproducibility
- **Memory Optimization**: Smart memory management for processing large datasets efficiently
- **Flexible Workflows**: Support for both structured workflows and ad-hoc processing
- **Extensible Design**: Easy to add new signal types and processing operations
- **Import Flexibility**: Convert signals from various sources to a standardized format

## Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd sleep_analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Project Structure

- `src/sleep_analysis/core/`: Base classes and metadata structures
- `src/sleep_analysis/signals/`: Signal type implementations
- `src/sleep_analysis/importers/`: Data import modules
- `src/sleep_analysis/operations/`: Signal processing operations
- `src/sleep_analysis/workflows/`: Workflow execution
- `src/sleep_analysis/utils/`: Utility functions
