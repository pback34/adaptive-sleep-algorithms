# Sleep Analysis Framework

A flexible, extensible framework for processing sleep-related signals, designed for researchers and developers working with physiological data. The framework provides a robust foundation for signal processing with an emphasis on reproducibility, type safety, and memory efficiency.

## Key Features

- **Type-Safe Signal Processing**: Enum-based type safety ensures operations match signal types
- **Complete Traceability**: Full metadata and operation history for reproducibility
- **Memory Optimization**: Smart memory management for processing large datasets efficiently
- **Flexible Workflows**: Support for both structured workflows and ad-hoc processing
- **Extensible Design**: Easy to add new signal types and processing operations
- **Import Flexibility**: Convert signals from various sources to a standardized format

## Installation

### Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd sleep_analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Regular Installation

```bash
# From PyPI (when available)
pip install sleep-analysis

# From local directory
pip install .
```

## Usage

### Command Line Interface

The package provides a command-line tool to run workflow files:

```bash
# Run using the installed entry point
sleep-analysis --workflow workflows/polar_workflow.yaml --data-dir data

# Run using the Python module
python -m sleep_analysis --workflow workflows/polar_workflow.yaml --data-dir data

# Run using the specific CLI module
python -m sleep_analysis.cli.run_workflow --workflow workflows/polar_workflow.yaml --data-dir data
```

### Options

```
-w, --workflow      Path to the workflow YAML file (required)
-d, --data-dir      Base directory containing the data files (required)
-o, --output-dir    Directory for output files (default: ./output)
-l, --log-level     Set logging level (DEBUG, INFO, WARN, ERROR)
-v                  Set logging level to DEBUG (shorthand)
```

### Creating Workflow Files

Workflow files are YAML documents with three main sections:
1. `import` - Data import specifications
2. `steps` - Processing operations to apply
3. `export` - Output format and location

Example:
```yaml
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      timestamp_col: "Phone timestamp"
    base_name: "hr_h10_merged"

steps:
  - type: signal
    input: "hr_h10_merged_0"
    operation: "filter_lowpass"
    parameters:
      cutoff_frequency: 0.5
    output: "hr_h10_filtered"

export:
  formats: ["csv"]
  output_dir: "results/polar_data"
  include_combined: true
```

## Project Structure

- `src/sleep_analysis/core/`: Base classes and metadata structures
- `src/sleep_analysis/signals/`: Signal type implementations
- `src/sleep_analysis/importers/`: Data import modules
- `src/sleep_analysis/operations/`: Signal processing operations
- `src/sleep_analysis/workflows/`: Workflow execution
- `src/sleep_analysis/utils/`: Utility functions
