# Sleep Analysis Framework

A flexible, extensible Python framework for processing and analyzing sleep-related physiological signals. Designed for researchers and developers working with multi-sensor sleep data, the framework provides a robust foundation for signal processing, feature extraction, and machine learning-based sleep analysis.

## Key Features

- **Type-Safe Signal Processing** - Enum-based type safety ensures operations match signal types
- **Complete Traceability** - Full metadata and operation history for reproducibility
- **Memory Optimization** - Smart memory management for processing large datasets efficiently
- **Flexible Workflows** - Support for both structured workflows and ad-hoc processing
- **Extensible Design** - Easy to add new signal types, features, and processing operations
- **Multi-Sensor Support** - Import and synchronize data from various sensor types and formats
- **Epoch-Based Feature Extraction** - Generate statistical, frequency, and categorical features aligned to common time grids
- **Interactive Visualization** - Backend-agnostic visualization with support for Bokeh and Plotly
- **Machine Learning Ready** - Built-in support for sleep staging algorithms (Random Forest, XGBoost, etc.)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd adaptive-sleep-algorithms

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all features
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

### Your First Analysis

Create a simple workflow file `my_workflow.yaml`:

```yaml
# Import heart rate data
import:
  - signal_type: "heart_rate"
    importer: "PolarCSVImporter"
    source: "data/polar_hr.csv"
    base_name: "hr"

# Extract features
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

steps:
  - type: collection
    operation: "generate_epoch_grid"

  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]
    parameters:
      aggregations: ["mean", "std", "min", "max"]
    output: "hr_features"

# Export results
export:
  - formats: ["csv"]
    output_dir: "results"
    content: ["all_features"]
```

Run the workflow:

```bash
sleep-analysis --workflow my_workflow.yaml --data-dir data --output-dir results
```

See [QUICK-START.md](QUICK-START.md) for a detailed walkthrough.

## Documentation

### Getting Started
- **[QUICK-START.md](QUICK-START.md)** - Installation, setup, and your first analysis workflow
- **[docs/data-preparation.md](docs/data-preparation.md)** - Supported file formats and data requirements
- **[docs/troubleshooting.md](docs/troubleshooting.md)** - Common issues and solutions

### Usage Guides
- **[USER-GUIDE.md](USER-GUIDE.md)** - Comprehensive guide to using the framework
  - Creating workflows
  - Signal processing operations
  - Feature extraction
  - Visualization
  - Machine learning workflows
- **[docs/feature_extraction_plan.md](docs/feature_extraction_plan.md)** - Complete feature extraction reference
- **[docs/feature-extraction-examples.md](docs/feature-extraction-examples.md)** - Feature extraction examples

### Architecture & Development
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Framework architecture and design patterns
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines and development setup
- **[docs/coding_guidelines.md](docs/coding_guidelines.md)** - Coding standards and best practices

### Advanced Topics
- **[docs/requirements/requirements.md](docs/requirements/requirements.md)** - Detailed requirements documentation
- **[docs/designs/](docs/designs/)** - Design documents for specific features

## Project Structure

```
adaptive-sleep-algorithms/
├── src/sleep_analysis/
│   ├── core/                   # Core classes (SignalCollection, SignalData)
│   ├── signals/                # Signal type implementations
│   ├── features/               # Feature classes
│   ├── importers/              # Data import modules
│   ├── operations/             # Signal processing operations
│   ├── workflows/              # Workflow execution engine
│   ├── visualization/          # Visualization backends
│   ├── export/                 # Data export modules
│   ├── algorithms/             # Machine learning algorithms
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
├── workflows/                  # Example workflow files
├── docs/                       # Documentation
└── README.md                   # This file
```

## Supported Signal Types

- **Cardiac**: Heart rate, PPG, HRV, R-R intervals
- **Movement**: Accelerometer (3-axis), gyroscope, activity counts
- **Respiratory**: Respiratory rate, breathing patterns
- **Sleep Stages**: EEG-based sleep staging (Wake, REM, N1, N2, N3)
- **Environmental**: Light, temperature, sound
- **Custom**: Easily extensible for new signal types

## Supported Importers

- **Polar Devices**: H10, Verity Sense (CSV format)
- **Generic CSV**: Flexible column mapping
- **Merging Importer**: Combine fragmented data files
- **Custom**: Create importers for proprietary formats

## Example Workflows

### Basic Feature Extraction

```bash
# Extract HRV and movement features from Polar data
sleep-analysis -w workflows/basic_features.yaml -d data/polar_session -o results/
```

### Sleep Staging with Random Forest

```bash
# Train and apply sleep staging model
sleep-analysis -w workflows/sleep_staging_rf.yaml -d data/labeled_data -o results/
```

### Multi-Sensor Alignment and Analysis

```bash
# Align chest and wrist sensors, extract synchronized features
sleep-analysis -w workflows/multi_sensor_alignment.yaml -d data/multi_sensor -o results/
```

See `workflows/` directory for complete examples.

## CLI Usage

```bash
# Basic usage
sleep-analysis --workflow <workflow.yaml> --data-dir <data_path> --output-dir <output_path>

# Options
-w, --workflow      Path to workflow YAML file (required)
-d, --data-dir      Base directory containing data files (required)
-o, --output-dir    Directory for output files (default: ./output)
-l, --log-level     Set logging level (DEBUG, INFO, WARN, ERROR)
-v                  Enable verbose logging (DEBUG level)
```

## Python API

```python
from sleep_analysis.core.signal_collection import SignalCollection
from sleep_analysis.workflows.workflow_executor import WorkflowExecutor

# Create collection and executor
collection = SignalCollection()
executor = WorkflowExecutor(collection)

# Execute workflow
executor.execute_workflow_from_file(
    workflow_path="my_workflow.yaml",
    data_dir="data"
)

# Access results
features = collection.get_stored_combined_features()
print(f"Extracted {len(features)} epochs with {len(features.columns)} features")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src.sleep_analysis --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_signals.py -v
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up your development environment
- Code style and standards
- Testing requirements
- Submitting pull requests

## Development Status

### Recently Completed
- ✅ **Phase 5 Refactoring** - Completed modular architecture transformation
- ✅ **Service-Based Architecture** - Broke monolithic SignalCollection into 13 focused services
- ✅ **Comprehensive Test Suite** - 231+ unit tests for all services
- ✅ **Visualization System** - Backend-agnostic visualization with Bokeh and Plotly support

### Roadmap
- 🔄 Enhanced sleep staging algorithms (Gradient Boosting, LSTM)
- 🔄 Real-time streaming signal processing
- 📋 Additional sensor support (Fitbit, Apple Watch, etc.)
- 📋 Advanced feature engineering pipelines
- 📋 Cloud deployment support

See [docs-dev/TECHNICAL-DEBT.md](docs-dev/TECHNICAL-DEBT.md) for detailed development notes.

## License

[Add license information here]

## Citation

If you use this framework in your research, please cite:

```
[Add citation information here]
```

## Support and Contact

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: See the `docs/` directory for detailed guides
- **Development**: See `docs-dev/` for development notes and architecture documentation

## Acknowledgments

This framework was developed to support sleep research and analysis. Special thanks to all contributors and the open-source community.

---

**Version**: 1.0.0
**Last Updated**: 2025-11-18
