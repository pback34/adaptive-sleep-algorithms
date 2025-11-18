# User Guide - Sleep Analysis Framework

This comprehensive guide covers all aspects of using the Sleep Analysis Framework, from basic workflows to advanced features.

## Table of Contents

1. [Workflow Basics](#workflow-basics)
2. [Importing Data](#importing-data)
3. [Signal Processing Operations](#signal-processing-operations)
4. [Signal Alignment](#signal-alignment)
5. [Feature Extraction](#feature-extraction)
6. [Visualization](#visualization)
7. [Exporting Results](#exporting-results)
8. [Python API](#python-api)
9. [Advanced Topics](#advanced-topics)

## Workflow Basics

### Workflow File Structure

Workflows are defined in YAML files with the following sections:

```yaml
# Optional: Global timezone settings
default_input_timezone: "America/New_York"
target_timezone: "UTC"

# Optional: Collection settings
collection_settings:
  index_config: ["signal_type", "name"]
  feature_index_config: ["source", "feature_name"]
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

# Required: Data import
import:
  - signal_type: "heart_rate"
    importer: "PolarCSVImporter"
    source: "data/hr.csv"
    base_name: "hr"

# Required: Processing steps
steps:
  - type: collection
    operation: "generate_epoch_grid"

  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]
    output: "hr_features"

# Required: Export results
export:
  - formats: ["csv"]
    output_dir: "results"
    content: ["all_features"]

# Optional: Visualization
visualization:
  - type: time_series
    signals: ["hr_0"]
    output: "results/plots/hr.html"
```

### Workflow Execution

```bash
# Basic execution
sleep-analysis --workflow my_workflow.yaml --data-dir data --output-dir results

# With verbose logging
sleep-analysis -w workflow.yaml -d data -o results -v

# Custom log level
sleep-analysis -w workflow.yaml -d data -l INFO
```

## Importing Data

### Supported Importers

#### PolarCSVImporter

Import data from Polar devices (H10, Verity Sense):

```yaml
import:
  - signal_type: "heart_rate"
    importer: "PolarCSVImporter"
    source: "data/polar_hr.csv"
    config:
      time_column: "Phone timestamp"
      delimiter: ";"
      origin_timezone: "America/New_York"
    base_name: "hr"
```

#### MergingImporter

Combine multiple files into a single signal:

```yaml
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "data/hr_files"
    config:
      importer_name: "PolarCSVImporter"
      file_pattern: "Polar_H10_*_HR.txt"
      sort_by: "timestamp"
      delimiter: ";"
    base_name: "hr_merged"
```

#### CSVImporterBase

Generic CSV importer with flexible column mapping:

```yaml
import:
  - signal_type: "accelerometer"
    importer: "CSVImporterBase"
    source: "data/accel.csv"
    config:
      column_mapping:
        timestamp: "time"
        x: "accel_x"
        y: "accel_y"
        z: "accel_z"
      time_format: "%Y-%m-%d %H:%M:%S.%f"
      delimiter: ","
    base_name: "accel"
```

### Timezone Handling

The framework handles timezones through three settings:

1. **`origin_timezone`**: Timezone of the source data (per importer)
2. **`default_input_timezone`**: Fallback timezone if origin not specified
3. **`target_timezone`**: Target timezone for all signals (default: "UTC")

```yaml
# Top-level defaults
default_input_timezone: "America/New_York"
target_timezone: "UTC"

import:
  - signal_type: "heart_rate"
    importer: "PolarCSVImporter"
    source: "data/hr.csv"
    config:
      origin_timezone: "Europe/Berlin"  # Override default
    base_name: "hr"
```

**Special Values**:
- `"system"` - Use local machine timezone
- `"UTC"` - Universal Coordinated Time (default)

## Signal Processing Operations

### Signal-Level Operations

#### Filtering

**Lowpass Filter**:
```yaml
steps:
  - type: signal
    operation: "lowpass_filter"
    inputs: ["hr_0"]
    parameters:
      cutoff: 0.5  # Hz
      order: 4
    output: "hr_filtered"
```

**Bandpass Filter**:
```yaml
steps:
  - type: signal
    operation: "bandpass_filter"
    inputs: ["accel_x_0"]
    parameters:
      low_cutoff: 0.3
      high_cutoff: 3.0
      order: 4
    output: "accel_filtered"
```

#### Resampling

```yaml
steps:
  - type: signal
    operation: "resample"
    inputs: ["ppg_0"]
    parameters:
      target_rate: 25.0  # Hz
      method: "linear"   # or "cubic", "nearest"
    output: "ppg_resampled"
```

### Batch Operations

Apply operations to multiple signals:

```yaml
steps:
  - type: batch
    operation: "lowpass_filter"
    inputs: ["hr_0", "hr_1", "hr_2"]
    parameters:
      cutoff: 0.5
    outputs: ["hr_0_filt", "hr_1_filt", "hr_2_filt"]
```

## Signal Alignment

Alignment synchronizes signals to a common time grid.

### Method 1: In-Place Alignment + Combination

```yaml
steps:
  # Step 1: Calculate alignment grid
  - type: collection
    operation: "generate_alignment_grid"
    parameters:
      target_sample_rate: 10.0  # Optional: specify rate

  # Step 2: Apply alignment to signals
  - type: collection
    operation: "apply_grid_alignment"
    parameters:
      method: "nearest"  # or "ffill", "bfill", "linear"

  # Step 3: Combine aligned signals
  - type: collection
    operation: "combine_aligned_signals"
```

### Method 2: merge_asof Alignment

```yaml
steps:
  # Step 1: Calculate alignment grid
  - type: collection
    operation: "generate_alignment_grid"

  # Step 2: Align and combine in one step
  - type: collection
    operation: "align_and_combine_signals"
```

### Alignment Methods

- **`nearest`**: Use closest data point
- **`ffill`**: Forward fill (propagate last value)
- **`bfill`**: Backward fill (propagate next value)
- **`linear`**: Linear interpolation between points

## Feature Extraction

### Epoch Grid Configuration

Define global epoch parameters:

```yaml
collection_settings:
  epoch_grid_config:
    window_length: "30s"  # Epoch duration
    step_size: "30s"      # Non-overlapping windows
    # step_size: "15s"    # 50% overlap
```

### Statistical Features

```yaml
steps:
  - type: collection
    operation: "generate_epoch_grid"

  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]  # Base name or list of keys
    parameters:
      aggregations: ["mean", "std", "min", "max", "median"]
      # Optional: Override window for this operation
      # window_length: "60s"
    output: "hr_stats"
```

### HRV Features

```yaml
steps:
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr_0"]
    parameters:
      hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
      use_rr_intervals: false
    output: "hrv_features"
```

### Movement Features

```yaml
steps:
  - type: multi_signal
    operation: "compute_movement_features"
    inputs: ["accel_0"]
    parameters:
      movement_metrics: "all"  # or list specific metrics
    output: "movement_features"
```

### Sleep Stage Mode

For categorical data (sleep stages):

```yaml
steps:
  - type: multi_signal
    operation: "compute_sleep_stage_mode"
    inputs: ["sleep_stage_0"]
    output: "sleep_stage_mode"
```

### Combining Features

Merge multiple feature sets into a matrix:

```yaml
steps:
  # After extracting multiple feature sets...
  - type: collection
    operation: "combine_features"
    inputs: ["hr_stats", "hrv_features", "movement_features"]
    parameters:
      feature_index_config: ["source", "feature_name"]  # Optional
```

## Visualization

### Time Series Plots

#### Single Signal

```yaml
visualization:
  - type: time_series
    signals: ["hr_0"]
    title: "Heart Rate Over Time"
    output: "results/plots/hr.html"
    backend: plotly
    parameters:
      width: 1200
      height: 400
      line_color: "blue"
      max_points: 10000  # Downsample for performance
```

#### Multiple Signals (Overlay)

```yaml
visualization:
  - type: time_series
    signals: ["hr_chest_0", "hr_wrist_0"]
    layout: overlay
    title: "Chest vs Wrist HR"
    output: "results/plots/hr_comparison.html"
    backend: plotly
```

#### Multiple Signals (Subplots)

```yaml
visualization:
  - type: time_series
    signals: ["hr_0", "accel_magnitude_0", "resp_rate_0"]
    layout: vertical
    title: "Multi-Sensor Dashboard"
    output: "results/plots/dashboard.html"
    backend: bokeh
    parameters:
      link_x_axes: true
      height: 300  # Per subplot
```

### Hypnograms

Specialized visualization for sleep stages:

```yaml
visualization:
  - type: hypnogram
    signals: ["sleep_stage_0"]
    title: "Sleep Hypnogram"
    output: "results/plots/hypnogram.html"
    backend: plotly
    parameters:
      add_statistics: true
      stage_colors:
        Awake: "#FF5733"
        REM: "#AA33FF"
        N1: "#33AAFF"
        N2: "#33FF57"
        N3: "#000080"
```

### Sleep Stage Overlay

Add sleep stage background shading to time series:

```yaml
visualization:
  - type: time_series
    signals: ["hr_0"]
    layout: single
    output: "results/plots/hr_with_stages.html"
    backend: bokeh
    parameters:
      overlay_sleep_stages: "sleep_stage_0"
      alpha: 0.15  # Transparency
```

### Scatter Plots

Compare two signals:

```yaml
visualization:
  - type: scatter
    x_signal: "heart_rate_0"
    y_signal: "resp_rate_0"
    title: "HR vs Respiratory Rate"
    output: "results/plots/scatter.html"
    backend: plotly
    parameters:
      marker_size: 8
      marker_color: "red"
```

### Grid Layouts

```yaml
visualization:
  - type: time_series
    signals: ["sig1", "sig2", "sig3", "sig4"]
    layout: grid
    title: "4-Panel Grid"
    output: "results/plots/grid.html"
    backend: bokeh
    parameters:
      subplot_titles: ["Signal 1", "Signal 2", "Signal 3", "Signal 4"]
      link_x_axes: true
```

### Backend Comparison

| Feature | Bokeh | Plotly |
|---------|-------|--------|
| Interactive tools | Pan, zoom, hover | Pan, zoom, hover, lasso |
| Export formats | HTML, PNG, SVG, PDF | HTML, PNG, SVG, PDF |
| Styling | Moderate customization | Extensive customization |
| Performance | Fast for medium datasets | Better for large datasets |
| Linked axes | Yes | Yes |
| Categorical data | Good | Excellent |

## Exporting Results

### Export Formats

Supported formats:
- **CSV** - Comma-separated values
- **Excel** - Microsoft Excel (.xlsx)
- **HDF5** - Hierarchical Data Format (.h5)
- **Pickle** - Python pickle format (.pkl)

### Export Content Types

```yaml
export:
  # Export all individual time-series signals
  - formats: ["csv"]
    output_dir: "results/signals"
    content: ["all_ts"]

  # Export all individual features
  - formats: ["excel"]
    output_dir: "results/features"
    content: ["all_features"]

  # Export combined aligned time-series
  - formats: ["csv", "hdf5"]
    output_dir: "results/combined"
    content: ["combined_ts"]

  # Export combined feature matrix
  - formats: ["csv", "excel"]
    output_dir: "results/features"
    content: ["combined_features"]

  # Export summary table
  - formats: ["csv"]
    output_dir: "results"
    content: ["summary"]

  # Export specific signals/features by key
  - formats: ["pickle"]
    output_dir: "results/specific"
    content: ["hr_0", "hr_features_0"]

  # Export multiple content types together
  - formats: ["csv"]
    output_dir: "results/all_data"
    content: ["all_ts", "combined_features", "summary"]
```

### MultiIndex Columns

Configure hierarchical column names:

```yaml
collection_settings:
  # For combined time-series
  index_config: ["signal_type", "name"]

  # For combined features
  feature_index_config: ["source_signal", "feature_name"]
```

Result:
```
| signal_type | heart_rate | heart_rate | accelerometer |
| name        | hr_0       | hr_1       | accel_0       |
|-------------|------------|------------|---------------|
| 2024-01-01  | 65.2       | 68.1       | 0.15          |
```

## Python API

### Basic Usage

```python
from sleep_analysis.core.signal_collection import SignalCollection
from sleep_analysis.workflows.workflow_executor import WorkflowExecutor

# Create collection
collection = SignalCollection()

# Execute workflow
executor = WorkflowExecutor(collection)
executor.execute_workflow_from_file(
    workflow_path="my_workflow.yaml",
    data_dir="data"
)

# Access results
features = collection.get_stored_combined_features()
print(f"Extracted {len(features)} epochs")
```

### Manual Signal Processing

```python
import pandas as pd
from sleep_analysis.signals.heart_rate_signal import HeartRateSignal

# Create signal
data = pd.DataFrame({
    'hr': [60, 62, 64, 63, 61, 59]
}, index=pd.date_range('2024-01-01', periods=6, freq='1s'))

signal = HeartRateSignal(
    data=data,
    metadata={
        'name': 'manual_hr',
        'signal_id': 'hr_001',
        'sampling_rate': 1.0
    }
)

# Apply operations
filtered = signal.apply_operation(
    'lowpass_filter',
    cutoff=0.5,
    inplace=False
)

# Add to collection
collection.add_time_series_signal("hr_manual", signal)
```

### Feature Extraction Programmatically

```python
# Generate epoch grid
collection.apply_operation(
    'generate_epoch_grid',
    parameters={}
)

# Extract features
hr_stats = collection.apply_multi_signal_operation(
    operation_name='feature_statistics',
    inputs=['hr'],
    output_key='hr_stats',
    parameters={
        'aggregations': ['mean', 'std', 'min', 'max']
    }
)

# Access feature data
feature_data = hr_stats.get_data()
print(feature_data.head())
```

### Export Programmatically

```python
from sleep_analysis.export.export_module import ExportModule

exporter = ExportModule(collection)

# Export specific content
exporter.export(
    formats=['csv', 'excel'],
    output_dir='results',
    content=['combined_features', 'summary']
)
```

### Visualization Programmatically

```python
from sleep_analysis.visualization import PlotlyVisualizer

visualizer = PlotlyVisualizer()

# Get signal
signal = collection.get_signal("hr_0")

# Create plot
figure = visualizer.create_time_series_plot(
    signal,
    title="Heart Rate",
    width=1200,
    height=400
)

# Save
visualizer.save(figure, "hr_plot.html")
```

## Advanced Topics

### Custom Signal Types

```python
from sleep_analysis.core.signal_data import TimeSeriesSignal
from sleep_analysis.core.signal_types import SignalType

# Define enum value (add to signal_types.py)
class SignalType(Enum):
    CUSTOM_SENSOR = "custom_sensor"

# Create signal class
class CustomSensorSignal(TimeSeriesSignal):
    signal_type = SignalType.CUSTOM_SENSOR
    required_columns = ['value', 'quality']

    def validate(self):
        super().validate()
        # Custom validation logic
        ...
```

### Custom Operations

```python
import pandas as pd

# Define operation
@TimeSeriesSignal.register("custom_operation", output_class=TimeSeriesSignal)
def custom_operation(data_list, parameters):
    """Custom signal processing operation."""
    df = data_list[0]

    # Your processing logic
    result = df.copy()
    result['processed'] = df['value'] * parameters.get('factor', 1.0)

    return result

# Use in workflow
steps:
  - type: signal
    operation: "custom_operation"
    inputs: ["signal_0"]
    parameters:
      factor: 2.0
    output: "processed"
```

### Custom Feature Extractors

```python
from typing import List, Dict, Any
from sleep_analysis.signals.time_series_signal import TimeSeriesSignal
from sleep_analysis.features.feature import Feature

def custom_feature_extractor(
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    global_window_length: pd.Timedelta,
    global_step_size: pd.Timedelta
) -> Feature:
    """Extract custom features over epochs."""

    signal = signals[0]
    data = signal.get_data()

    # Extract features for each epoch
    features = []
    for epoch_start in epoch_grid_index:
        epoch_end = epoch_start + global_window_length
        epoch_data = data.loc[epoch_start:epoch_end]

        # Calculate custom metrics
        custom_metric = epoch_data.mean() * 2  # Example

        features.append({
            'custom_metric': custom_metric
        })

    feature_df = pd.DataFrame(features, index=epoch_grid_index)

    return Feature(
        data=feature_df,
        metadata={...}
    )

# Register
SignalCollection.multi_signal_registry["custom_features"] = (
    custom_feature_extractor,
    Feature
)
```

### Metadata Management

```python
# Update signal metadata
collection.update_time_series_metadata(
    "hr_0",
    updates={
        'sensor_location': 'chest',
        'subject_id': 'S001'
    }
)

# Query by metadata
chest_signals = collection.get_signals(
    criteria={'sensor_location': 'chest'}
)

# Access operation history
signal = collection.get_signal("hr_0")
print(signal.metadata.operations)
```

### Error Handling

```python
from sleep_analysis.core.exceptions import SignalNotFoundError, ValidationError

try:
    collection.add_time_series_signal("hr_0", signal)
except ValidationError as e:
    print(f"Validation failed: {e}")

try:
    signal = collection.get_signal("nonexistent")
except SignalNotFoundError as e:
    print(f"Signal not found: {e}")
```

## See Also

- [QUICK-START.md](QUICK-START.md) - Quick start guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Framework architecture
- [docs/feature_extraction_plan.md](docs/feature_extraction_plan.md) - Complete feature reference
- [docs/troubleshooting.md](docs/troubleshooting.md) - Troubleshooting guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines

---

**Version**: 1.0.0
**Last Updated**: 2025-11-18
