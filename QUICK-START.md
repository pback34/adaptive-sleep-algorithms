# Quick Start Guide

Get started with the Sleep Analysis Framework in 5 minutes. This guide will walk you through installation, setup, and your first sleep analysis workflow.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Basic familiarity with command line

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd adaptive-sleep-algorithms
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install the Framework

```bash
# Install with all features (recommended for getting started)
pip install -e ".[dev]"

# Or install just the core framework
pip install -e .

# Or install with specific features
pip install -e ".[vis,algorithms]"  # Visualization + ML algorithms
```

### Verify Installation

```bash
# Check that the CLI is available
sleep-analysis --help

# Run tests to ensure everything works
pytest tests/ -v
```

**Success Criteria**: You should see the help message and all tests should pass (211 tests passing).

## Hello World: Your First Sleep Analysis

This example demonstrates the complete workflow: Import data → Extract features → Export results.

### Step 1: Prepare Your Data

Create a directory with your Polar sensor data files:

```bash
mkdir -p data
# Place your Polar_H10_*_HR.txt and Polar_H10_*_ACC.txt files in data/
```

**Don't have data yet?** See the [Data Preparation Guide](data-preparation.md) for file format requirements.

### Step 2: Create a Simple Workflow

Create a file called `my_first_workflow.yaml`:

```yaml
# Simple Sleep Analysis Workflow
# Extracts basic features from Polar heart rate data

# Global settings for feature extraction
collection_settings:
  epoch_grid_config:
    window_length: "30s"  # 30-second epochs (standard for sleep analysis)
    step_size: "30s"      # Non-overlapping windows

# Import heart rate data
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    base_name: "hr"

# Processing steps
steps:
  # Step 1: Generate epoch grid for feature extraction
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # Step 2: Extract HRV features (heart rate variability)
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]
    parameters:
      hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
      use_rr_intervals: false
    output: "hrv_features"

  # Step 3: Extract basic statistics
  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]
    parameters:
      aggregations: ["mean", "std", "min", "max"]
    output: "hr_stats"

  # Step 4: Combine all features
  - type: collection
    operation: "combine_features"
    inputs: ["hrv_features", "hr_stats"]
    parameters: {}

# Export results
export:
  - formats: ["csv", "excel"]
    output_dir: "results/features"
    content: ["combined_features"]

  - formats: ["csv"]
    output_dir: "results/raw_data"
    content: ["all_ts"]
```

### Step 3: Run Your Workflow

```bash
# Run the workflow
python -m sleep_analysis.cli.run_workflow \
  --workflow my_first_workflow.yaml \
  --data-dir data \
  --output-dir results

# Or use the short form
sleep-analysis -w my_first_workflow.yaml -d data -o results
```

### Step 4: Check Your Results

```bash
# View the output structure
ls -R results/

# Expected output:
# results/
#   features/
#     combined_features.csv
#     combined_features.xlsx
#   raw_data/
#     hr_merged_0.csv
```

**Success Criteria**:
- No errors in the console output
- Files created in `results/features/` directory
- CSV/Excel files contain feature data with timestamps

### Step 5: Inspect Your Features

Open `results/features/combined_features.csv` in Excel or any spreadsheet application. You should see:

- **Index column**: Timestamps (epoch start times)
- **Feature columns**: Multiple columns for HRV metrics and statistics
  - `hr_mean`, `hr_std`, `hr_cv`, `hr_range` (HRV features)
  - `mean`, `std`, `min`, `max` (basic statistics)

Each row represents one 30-second epoch of your sleep data.

## What's Next?

Now that you've run your first analysis, you can:

1. **Add More Features**: See [Feature Extraction Guide](feature_extraction_plan.md) for all available features
2. **Use Sleep Staging**: Add machine learning sleep stage classification (see [Algorithms README](../src/sleep_analysis/algorithms/README.md))
3. **Visualize Results**: Add plots to your workflow (see README visualization section)
4. **Process Multiple Sensors**: Combine data from chest and wrist sensors

## Common Next Steps

### Add Movement Features

Add this to your workflow's `steps` section:

```yaml
# Import accelerometer data first (in import section)
- signal_type: "accelerometer"
  importer: "MergingImporter"
  source: "."
  config:
    file_pattern: "Polar_H10_*_ACC.txt"
    time_column: "Phone timestamp"
    sort_by: "timestamp"
    delimiter: ";"
  base_name: "accel"

# Then extract movement features (in steps section)
- type: multi_signal
  operation: "compute_movement_features"
  inputs: ["accel"]
  parameters:
    movement_metrics: "all"
  output: "movement_features"

# Add to combine_features step
- type: collection
  operation: "combine_features"
  inputs: ["hrv_features", "hr_stats", "movement_features"]  # Added movement
  parameters: {}
```

### Add Visualization

Add this `visualization` section to your workflow:

```yaml
visualization:
  - type: time_series
    signals: ["hr_merged_0"]
    title: "Heart Rate Over Time"
    output: "results/plots/heart_rate.html"
    backend: plotly
    parameters:
      width: 1200
      height: 400
      max_points: 10000
```

Then open `results/plots/heart_rate.html` in your browser to see an interactive plot.

### Train a Sleep Staging Model

See the complete example workflow at `workflows/sleep_staging_with_rf.yaml` for a full sleep staging pipeline using Random Forest.

## Python API (Advanced)

You can also use the framework programmatically:

```python
from sleep_analysis.core.signal_collection import SignalCollection
from sleep_analysis.workflows.workflow_executor import WorkflowExecutor

# Load and execute a workflow
collection = SignalCollection()
executor = WorkflowExecutor(collection)

# Execute from YAML file
executor.execute_workflow_from_file(
    workflow_path="my_first_workflow.yaml",
    data_dir="data"
)

# Access results
combined_features = collection.get_stored_combined_features()
print(f"Extracted {len(combined_features)} epochs")
print(f"Features: {list(combined_features.columns)}")

# Export programmatically
from sleep_analysis.export.export_module import ExportModule
exporter = ExportModule(collection)
exporter.export(
    formats=["csv"],
    output_dir="results",
    content=["combined_features"]
)
```

## Troubleshooting

Having issues? Check the [Troubleshooting Guide](troubleshooting.md) for common problems and solutions.

### Quick Fixes

**Import Error: No module named 'sleep_analysis'**
```bash
# Make sure you installed with -e flag
pip install -e .
```

**No data files found**
```bash
# Check your file pattern matches your files
ls data/Polar_H10_*_HR.txt

# Make sure source path is correct in workflow (use "." for data_dir)
```

**Empty results / No features extracted**
```bash
# Enable debug logging to see what's happening
sleep-analysis -w workflow.yaml -d data -v
```

## Getting Help

- **Documentation**: See [README.md](../README.md) for comprehensive documentation
- **Examples**: Check the `workflows/` directory for complete examples
- **Troubleshooting**: See [troubleshooting.md](troubleshooting.md)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) (coming soon)

## Summary Checklist

After completing this guide, you should be able to:

- [ ] Install the Sleep Analysis Framework
- [ ] Create a simple workflow YAML file
- [ ] Import Polar sensor data
- [ ] Extract HRV and statistical features
- [ ] Export results to CSV/Excel
- [ ] Locate and inspect your output files
- [ ] Know where to find more advanced examples

**Congratulations!** You're now ready to analyze sleep data with the framework.
