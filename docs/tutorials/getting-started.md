# Getting Started with Adaptive Sleep Algorithms

**Welcome!** This guide will help you get up and running with the Adaptive Sleep Algorithms framework for analyzing sleep-related sensor data.

## Table of Contents

1. [What is this framework?](#what-is-this-framework)
2. [Installation](#installation)
3. [Quick Start: Your First Workflow](#quick-start-your-first-workflow)
4. [Understanding the Basic Concepts](#understanding-the-basic-concepts)
5. [Next Steps](#next-steps)

---

## What is this framework?

The Adaptive Sleep Algorithms framework is a flexible Python toolkit for processing physiological signals (heart rate, accelerometer, etc.) to extract features for sleep analysis and staging. It's designed for researchers and developers who need to:

- **Import** sensor data from various sources (Polar devices, CSV files, etc.)
- **Process** time-series signals (filtering, alignment, feature extraction)
- **Extract** sleep-relevant features (HRV metrics, movement patterns, correlations)
- **Export** results for machine learning or statistical analysis
- **Visualize** signals and features for quality control

**Key Features:**
- 📊 Multi-sensor support (heart rate, accelerometer, PPG, sleep stages)
- 🔄 Reproducible workflows via YAML configuration
- 🎯 Comprehensive metadata tracking for all operations
- 📈 Built-in visualization with Bokeh and Plotly
- 🧩 Extensible architecture for custom operations
- 📦 Multiple export formats (CSV, Excel, HDF5, Pickle)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

Currently, the framework is installed from source:

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-sleep-algorithms.git
cd adaptive-sleep-algorithms

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Verify Installation

```bash
# Test the CLI is available
python -m sleep_analysis.cli.run_workflow --help

# You should see the help message for the workflow runner
```

---

## Quick Start: Your First Workflow

Let's run a simple workflow to import heart rate data and extract basic features.

### Step 1: Prepare Sample Data

You'll need sensor data files. For this example, we'll use Polar heart rate data. The framework expects CSV files with timestamps and sensor values.

**Example file structure:**
```
data/
  └── Polar_H10_ABC123_20231115_120000_HR.txt
```

**File format (semicolon-delimited):**
```
Phone timestamp;HR [bpm]
2023-11-15T12:00:00.000;65
2023-11-15T12:00:01.000;66
2023-11-15T12:00:02.000;64
...
```

### Step 2: Create Your First Workflow

Create a file called `my_first_workflow.yaml`:

```yaml
# my_first_workflow.yaml
# A simple workflow to import heart rate and extract basic statistics

# Collection settings
collection_settings:
  epoch_grid_config:
    window_length: "30s"  # 30-second windows (standard for sleep)
    step_size: "30s"      # Non-overlapping windows

# Import heart rate data
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."  # Look in the data directory
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "EKG"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr"

# Processing steps
steps:
  # Generate the epoch grid for feature extraction
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # Extract basic statistics for each 30-second window
  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]
    parameters:
      aggregations: ["mean", "std", "min", "max"]
    output: "hr_stats"

  # Print a summary
  - type: collection
    operation: "summarize_signals"
    parameters:
      print_summary: true

# Export results
export:
  - formats: ["csv", "excel"]
    output_dir: "results"
    content: ["hr_stats"]
```

### Step 3: Run the Workflow

```bash
# Run the workflow
python -m sleep_analysis.cli.run_workflow \
  --workflow my_first_workflow.yaml \
  --data-dir data

# The framework will:
# 1. Import your heart rate data
# 2. Create 30-second epochs
# 3. Calculate mean, std, min, max for each epoch
# 4. Export results to results/hr_stats.csv and results/hr_stats.xlsx
```

### Step 4: View the Results

Check the `results/` directory for your output files:

```
results/
  ├── hr_stats.csv       # Feature matrix in CSV format
  └── hr_stats.xlsx      # Feature matrix in Excel format
```

**Example output (hr_stats.csv):**
```csv
timestamp,hr_mean,hr_std,hr_min,hr_max
2023-11-15T12:00:00,65.2,2.1,62,68
2023-11-15T12:00:30,64.8,1.9,61,67
2023-11-15T12:01:00,66.1,2.3,63,70
...
```

**🎉 Congratulations!** You've successfully run your first workflow.

---

## Understanding the Basic Concepts

### 1. Signals

A **signal** is a time-series of sensor data. The framework supports:

- **TimeSeriesSignal**: Raw sensor data (heart rate, accelerometer, PPG)
- **Feature**: Derived features extracted from time-series (statistics, HRV metrics)

Each signal has:
- **Data**: A pandas DataFrame with timestamps and values
- **Metadata**: Information about the signal (sensor type, operations applied, etc.)

### 2. SignalCollection

A **SignalCollection** is a container that holds all your signals. It provides:
- Storage for time-series signals and features
- Operations for processing signals
- Metadata tracking for reproducibility

### 3. Workflows

A **workflow** is a YAML file that defines a complete analysis pipeline:

```yaml
# What data to import
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    ...

# How to process it
steps:
  - operation: "feature_statistics"
    ...

# Where to save results
export:
  - formats: ["csv"]
    ...
```

Workflows are:
- **Declarative**: You describe *what* you want, not *how* to do it
- **Reproducible**: Same workflow + same data = same results
- **Version-controlled**: Track workflows alongside your code

### 4. Operations

**Operations** transform signals:

- **Collection operations**: Work on the entire collection (e.g., `generate_epoch_grid`)
- **Multi-signal operations**: Process one or more signals (e.g., `feature_statistics`, `compute_hrv_features`)
- **Single-signal operations**: Transform a single signal (e.g., `filter_lowpass`)

### 5. Epochs

**Epochs** are time windows used for feature extraction. For sleep analysis:
- Standard epoch: **30 seconds** (matches sleep staging conventions)
- Non-overlapping: `step_size = window_length`
- Overlapping: `step_size < window_length` (for more granular analysis)

### 6. Features

**Features** are summary statistics computed over epochs:

| Feature Type | Examples | Use Case |
|-------------|----------|----------|
| Statistical | mean, std, min, max | General patterns |
| HRV Metrics | RMSSD, SDNN, pNN50 | Autonomic activity |
| Movement | magnitude, variance | Activity detection |
| Spectral | power in frequency bands | Periodicity analysis |
| Correlation | cross-correlation | Multi-sensor relationships |

---

## Next Steps

Now that you understand the basics, explore more advanced topics:

1. **[Common Workflows Tutorial](common-workflows.md)** - Learn HRV analysis, sleep staging, and multi-sensor fusion
2. **[Feature Extraction Guide](feature-extraction-guide.md)** - Deep dive into available features
3. **[Best Practices](best-practices.md)** - Tips for production-quality analysis
4. **[Troubleshooting](../troubleshooting.md)** - Solutions to common issues
5. **[Python API Guide](python-api-guide.md)** - Use the framework programmatically

### Example Workflows to Try

The framework includes several example workflows in the `workflows/` directory:

- **`polar_workflow.yaml`**: Import Polar sensor data
- **`complete_sleep_analysis.yaml`**: Full feature extraction pipeline
- **`sleep_staging_with_rf.yaml`**: Apply a trained sleep staging model
- **`train_sleep_staging_model.yaml`**: Train your own sleep staging model

Try running them:

```bash
python -m sleep_analysis.cli.run_workflow \
  --workflow workflows/complete_sleep_analysis.yaml \
  --data-dir data
```

---

## Getting Help

If you run into issues:

1. Check the **[Troubleshooting Guide](../troubleshooting.md)**
2. Review the **[Documentation](../README.md)**
3. Look at **[Example Workflows](../../workflows/)**
4. Open an issue on GitHub

**Happy analyzing!** 🎉
