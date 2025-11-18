# Python API Guide

This guide shows you how to use the Adaptive Sleep Algorithms framework programmatically in Python scripts or Jupyter notebooks.

## Table of Contents

1. [Why Use the Python API?](#why-use-the-python-api)
2. [Basic Usage](#basic-usage)
3. [Working with Signals](#working-with-signals)
4. [Feature Extraction](#feature-extraction)
5. [Combining Features](#combining-features)
6. [Export and Visualization](#export-and-visualization)
7. [Complete Examples](#complete-examples)

---

## Why Use the Python API?

**Use YAML workflows when:**
- You have a standard analysis pipeline
- You want declarative, version-controlled workflows
- You need to run batch processing via command line

**Use Python API when:**
- You need dynamic, conditional logic
- You want to integrate with existing Python code
- You're doing exploratory analysis in Jupyter notebooks
- You need fine-grained control over processing

---

## Basic Usage

### 1. Import the Framework

```python
from sleep_analysis.core import SignalCollection
from sleep_analysis.importers import MergingImporter, PolarCSVImporter
from sleep_analysis.signals import SignalType, SensorType, SensorModel, BodyPosition
from sleep_analysis.export_module import ExportModule
from sleep_analysis.visualization import BokehVisualizer, PlotlyVisualizer
import pandas as pd
```

### 2. Create a SignalCollection

The `SignalCollection` is the main container for all your signals and features.

```python
# Create a collection
collection = SignalCollection(
    metadata={
        'subject_id': 'subject_001',
        'session_id': '20231115',
        'timezone': 'America/New_York'
    }
)

print(f"Created collection for {collection.metadata.subject_id}")
```

### 3. Import Data

```python
# Create an importer
importer = MergingImporter(config={
    'file_pattern': 'Polar_H10_*_HR.txt',
    'time_column': 'Phone timestamp',
    'sort_by': 'timestamp',
    'delimiter': ';'
})

# Import heart rate signals
hr_signals = collection.import_signals_from_source(
    importer_instance=importer,
    source='data/',
    spec={
        'signal_type': SignalType.HEART_RATE,
        'sensor_type': SensorType.EKG,
        'sensor_model': SensorModel.POLAR_H10,
        'body_position': BodyPosition.CHEST,
        'base_name': 'hr'
    }
)

print(f"Imported {len(hr_signals)} heart rate signal(s)")
```

---

## Working with Signals

### Accessing Signals

```python
# Get a specific signal
hr_signal = collection.get_time_series_signal('hr_merged_0')

# Access the signal data (pandas DataFrame)
hr_data = hr_signal.get_data()
print(hr_data.head())

# Output:
#                           hr
# 2023-11-15 22:00:00.000   65
# 2023-11-15 22:00:01.000   66
# 2023-11-15 22:00:02.000   64
# ...
```

### Accessing Signal Metadata

```python
# Get metadata
metadata = hr_signal.metadata

print(f"Signal ID: {metadata.signal_id}")
print(f"Signal Type: {metadata.signal_type}")
print(f"Sample Rate: {metadata.sample_rate}")
print(f"Duration: {metadata.end_time - metadata.start_time}")
print(f"Sensor: {metadata.sensor_model}")
print(f"Operations applied: {len(metadata.operations)}")
```

### Querying Multiple Signals

```python
# Get all heart rate signals
all_hr = collection.get_signals(
    signal_type=SignalType.HEART_RATE
)
print(f"Found {len(all_hr)} heart rate signals")

# Get signals by sensor model
chest_signals = collection.get_signals(
    criteria={'sensor_model': SensorModel.POLAR_H10}
)
print(f"Found {len(chest_signals)} signals from Polar H10")

# Get signals by base name pattern
hr_signals = collection.get_signals(
    base_name='hr'
)
print(f"Found {len(hr_signals)} signals with base_name='hr'")
```

### Single-Signal Operations

```python
# Apply lowpass filter to heart rate
hr_signal = collection.get_time_series_signal('hr_merged_0')

filtered_data = hr_signal.apply_operation(
    operation_name='filter_lowpass',
    inplace=False,  # Create new signal, don't modify original
    cutoff=0.5,     # 0.5 Hz cutoff
    order=4
)

# The operation is recorded in metadata
print(hr_signal.metadata.operations[-1])
# Output: OperationInfo(operation_name='filter_lowpass', parameters={'cutoff': 0.5, 'order': 4})
```

---

## Feature Extraction

### 1. Generate Epoch Grid

Before extracting features, create a common time grid for all features:

```python
# Set epoch grid configuration
collection.metadata.epoch_grid_config = {
    'window_length': '30s',
    'step_size': '30s'
}

# Generate the epoch grid
collection.generate_epoch_grid()

print(f"Generated {len(collection.epoch_grid_index)} epochs")
print(f"Epoch window: {collection.global_epoch_window_length}")
print(f"Epoch step: {collection.global_epoch_step_size}")
```

### 2. Extract HRV Features

```python
# Extract HRV features
hrv_feature = collection.apply_multi_signal_operation(
    operation_name='compute_hrv_features',
    signal_keys=['hr_merged_0'],
    parameters={
        'hrv_metrics': ['hr_mean', 'hr_std', 'hr_cv', 'hr_range'],
        'use_rr_intervals': False
    }
)

# Add to collection
collection.add_feature('hrv_features', hrv_feature)

# Access feature data
hrv_data = hrv_feature.get_data()
print(hrv_data.head())

# Output (MultiIndex columns):
#                      (hr_merged_0, hr_mean)  (hr_merged_0, hr_std)  ...
# 2023-11-15 22:00:00  65.2                    2.1                    ...
# 2023-11-15 22:00:30  64.8                    1.9                    ...
# ...
```

### 3. Extract Movement Features

```python
# Import accelerometer data first
accel_importer = MergingImporter(config={
    'file_pattern': 'Polar_H10_*_ACC.txt',
    'time_column': 'Phone timestamp',
    'sort_by': 'timestamp',
    'delimiter': ';'
})

accel_signals = collection.import_signals_from_source(
    importer_instance=accel_importer,
    source='data/',
    spec={
        'signal_type': SignalType.ACCELEROMETER,
        'sensor_type': SensorType.ACCEL,
        'sensor_model': SensorModel.POLAR_H10,
        'body_position': BodyPosition.CHEST,
        'base_name': 'accel'
    }
)

# Extract movement features
movement_feature = collection.apply_multi_signal_operation(
    operation_name='compute_movement_features',
    signal_keys=['accel_merged_0'],
    parameters={
        'movement_metrics': 'all'
    }
)

collection.add_feature('movement_features', movement_feature)
```

### 4. Extract Correlation Features

```python
# Compute correlation between HR and movement
corr_feature = collection.apply_multi_signal_operation(
    operation_name='compute_correlation_features',
    signal_keys=['hr_merged_0', 'accel_merged_0'],
    parameters={
        'signal1_column': 'hr',
        'signal2_column': 'x',  # X-axis acceleration
        'method': 'pearson',
        'window_length': '60s'
    }
)

collection.add_feature('hr_movement_corr', corr_feature)
```

### 5. Extract Statistical Features

```python
# Basic statistics
stats_feature = collection.apply_multi_signal_operation(
    operation_name='feature_statistics',
    signal_keys=['hr_merged_0'],
    parameters={
        'aggregations': ['mean', 'std', 'min', 'max', 'median']
    }
)

collection.add_feature('hr_stats', stats_feature)
```

---

## Combining Features

### Combine Multiple Features into One Matrix

```python
# Combine all features
collection.combine_features(
    inputs=['hrv_features', 'movement_features', 'hr_movement_corr', 'hr_stats'],
    feature_index_config=['sensor_model', 'feature_type']
)

# Access combined feature matrix
combined = collection._combined_feature_matrix
print(f"Combined matrix shape: {combined.shape}")
print(f"Columns: {combined.columns.tolist()[:5]}...")  # First 5 columns

# Output:
# Combined matrix shape: (960, 24)
# Columns: [(PolarH10, HRV, hr_mean), (PolarH10, HRV, hr_std), ...]
```

### Access Combined Features

```python
# Get as DataFrame
combined_df = collection._combined_feature_matrix

# Access specific feature column
hr_mean_col = combined_df[('PolarH10', 'HRV', 'hr_mean')]
print(f"Average HR across all epochs: {hr_mean_col.mean():.2f} bpm")

# Filter by time
night_features = combined_df.between_time('22:00', '06:00')
print(f"Features during night: {len(night_features)} epochs")
```

---

## Export and Visualization

### Export to Files

```python
from sleep_analysis.export_module import ExportModule

# Create exporter
exporter = ExportModule(collection=collection)

# Export combined features to CSV and Excel
exporter.export(
    formats=['csv', 'excel'],
    output_dir='results/features',
    content=['combined_features']
)

# Export individual features
exporter.export(
    formats=['excel'],
    output_dir='results/features/individual',
    content=['hrv_features', 'movement_features']
)

# Export raw time-series
exporter.export(
    formats=['csv'],
    output_dir='results/raw_signals',
    content=['all_ts']
)

# Export summary table
exporter.export(
    formats=['csv'],
    output_dir='results/summary',
    content=['summary']
)

print("Export complete!")
```

### Visualization with Plotly

```python
from sleep_analysis.visualization import PlotlyVisualizer

# Create visualizer
viz = PlotlyVisualizer()

# Visualize heart rate signal
fig = viz.visualize_signal(
    signal=collection.get_time_series_signal('hr_merged_0'),
    title='Heart Rate Over Time',
    width=1400,
    height=400
)

# Save to HTML
viz.save(fig, 'results/plots/heart_rate.html')

# Or display in Jupyter
# viz.show(fig)
```

### Multi-Signal Visualization

```python
# Visualize multiple signals in one plot
fig = viz.visualize_collection(
    collection=collection,
    signals=['hr_merged_0', 'accel_merged_0'],
    layout='vertical',
    title='Heart Rate and Accelerometer',
    width=1400,
    height=800,
    link_x_axes=True  # Synchronized zooming
)

viz.save(fig, 'results/plots/multi_signal.html')
```

### Feature Visualization

```python
# Visualize extracted features
hrv_feature = collection.get_feature('hrv_features')

fig = viz.visualize_signal(
    signal=hrv_feature,
    title='HRV Features Over Time',
    width=1400,
    height=600
)

viz.save(fig, 'results/plots/hrv_features.html')
```

---

## Complete Examples

### Example 1: Quick HRV Analysis

```python
from sleep_analysis.core import SignalCollection
from sleep_analysis.importers import MergingImporter
from sleep_analysis.signals import SignalType, SensorType, SensorModel, BodyPosition
from sleep_analysis.export_module import ExportModule

# Create collection
collection = SignalCollection(metadata={
    'subject_id': 'subject_001',
    'session_id': '20231115',
    'timezone': 'America/New_York'
})

# Import heart rate
importer = MergingImporter(config={
    'file_pattern': 'Polar_H10_*_HR.txt',
    'time_column': 'Phone timestamp',
    'sort_by': 'timestamp',
    'delimiter': ';'
})

collection.import_signals_from_source(
    importer_instance=importer,
    source='data/',
    spec={
        'signal_type': SignalType.HEART_RATE,
        'sensor_type': SensorType.EKG,
        'sensor_model': SensorModel.POLAR_H10,
        'body_position': BodyPosition.CHEST,
        'base_name': 'hr'
    }
)

# Generate epochs and extract HRV
collection.metadata.epoch_grid_config = {
    'window_length': '30s',
    'step_size': '30s'
}
collection.generate_epoch_grid()

hrv = collection.apply_multi_signal_operation(
    operation_name='compute_hrv_features',
    signal_keys=['hr_merged_0'],
    parameters={
        'hrv_metrics': ['hr_mean', 'hr_std', 'hr_cv', 'hr_range'],
        'use_rr_intervals': False
    }
)
collection.add_feature('hrv', hrv)

# Export
exporter = ExportModule(collection=collection)
exporter.export(
    formats=['csv', 'excel'],
    output_dir='results/hrv',
    content=['hrv']
)

print(f"✓ Extracted HRV features: {hrv.get_data().shape}")
```

### Example 2: Multi-Sensor Analysis in Jupyter Notebook

```python
# Cell 1: Setup
from sleep_analysis.core import SignalCollection
from sleep_analysis.importers import MergingImporter
from sleep_analysis.signals import SignalType, SensorType, SensorModel, BodyPosition
from sleep_analysis.visualization import PlotlyVisualizer
import pandas as pd

collection = SignalCollection(metadata={
    'subject_id': 'subject_001',
    'session_id': '20231115'
})
```

```python
# Cell 2: Import data
hr_importer = MergingImporter(config={
    'file_pattern': 'Polar_H10_*_HR.txt',
    'time_column': 'Phone timestamp',
    'sort_by': 'timestamp',
    'delimiter': ';'
})

accel_importer = MergingImporter(config={
    'file_pattern': 'Polar_H10_*_ACC.txt',
    'time_column': 'Phone timestamp',
    'sort_by': 'timestamp',
    'delimiter': ';'
})

# Import both signals
collection.import_signals_from_source(
    importer_instance=hr_importer,
    source='data/',
    spec={
        'signal_type': SignalType.HEART_RATE,
        'sensor_type': SensorType.EKG,
        'sensor_model': SensorModel.POLAR_H10,
        'body_position': BodyPosition.CHEST,
        'base_name': 'hr'
    }
)

collection.import_signals_from_source(
    importer_instance=accel_importer,
    source='data/',
    spec={
        'signal_type': SignalType.ACCELEROMETER,
        'sensor_type': SensorType.ACCEL,
        'sensor_model': SensorModel.POLAR_H10,
        'body_position': BodyPosition.CHEST,
        'base_name': 'accel'
    }
)

print(f"Imported {len(collection.time_series_signals)} signals")
```

```python
# Cell 3: Visualize raw data
viz = PlotlyVisualizer()

fig = viz.visualize_collection(
    collection=collection,
    signals=['hr_merged_0', 'accel_merged_0'],
    layout='vertical',
    title='Raw Sensor Data',
    width=1400,
    height=800,
    link_x_axes=True
)

viz.show(fig)
```

```python
# Cell 4: Extract features
collection.metadata.epoch_grid_config = {
    'window_length': '30s',
    'step_size': '30s'
}
collection.generate_epoch_grid()

# HRV features
hrv = collection.apply_multi_signal_operation(
    operation_name='compute_hrv_features',
    signal_keys=['hr_merged_0'],
    parameters={
        'hrv_metrics': ['hr_mean', 'hr_std', 'hr_cv', 'hr_range'],
        'use_rr_intervals': False
    }
)
collection.add_feature('hrv', hrv)

# Movement features
movement = collection.apply_multi_signal_operation(
    operation_name='compute_movement_features',
    signal_keys=['accel_merged_0'],
    parameters={'movement_metrics': 'all'}
)
collection.add_feature('movement', movement)

print(f"✓ HRV features: {hrv.get_data().shape}")
print(f"✓ Movement features: {movement.get_data().shape}")
```

```python
# Cell 5: Combine and analyze
collection.combine_features(
    inputs=['hrv', 'movement'],
    feature_index_config=['feature_type', 'sensor_model']
)

combined = collection._combined_feature_matrix

# Quick analysis
print("Feature Matrix Summary:")
print(f"  Shape: {combined.shape}")
print(f"  Time range: {combined.index.min()} to {combined.index.max()}")
print(f"  Duration: {combined.index.max() - combined.index.min()}")
print(f"\nAverage HR: {combined[('HRV', 'PolarH10', 'hr_mean')].mean():.2f} bpm")
print(f"HR variability: {combined[('HRV', 'PolarH10', 'hr_std')].mean():.2f} bpm")
```

```python
# Cell 6: Export results
from sleep_analysis.export_module import ExportModule

exporter = ExportModule(collection=collection)
exporter.export(
    formats=['csv', 'excel'],
    output_dir='results/notebook_analysis',
    content=['combined_features', 'summary']
)

print("✓ Results exported to results/notebook_analysis/")
```

### Example 3: Conditional Processing

```python
# Process differently based on signal quality
from sleep_analysis.core import SignalCollection
from sleep_analysis.importers import MergingImporter
from sleep_analysis.signals import SignalType, SensorType, SensorModel, BodyPosition

collection = SignalCollection()

# Import HR
# ... (import code here)

# Get HR signal
hr_signal = collection.get_time_series_signal('hr_merged_0')
hr_data = hr_signal.get_data()

# Check signal quality
mean_hr = hr_data['hr'].mean()
std_hr = hr_data['hr'].std()

if mean_hr < 40 or mean_hr > 120:
    print("⚠ Warning: Unusual mean HR detected. Applying stricter filtering.")
    # Apply more aggressive filtering
    hr_signal.apply_operation(
        operation_name='filter_lowpass',
        inplace=True,
        cutoff=0.3,  # Lower cutoff for noisy data
        order=6
    )
else:
    print("✓ HR in normal range. Using standard filtering.")
    hr_signal.apply_operation(
        operation_name='filter_lowpass',
        inplace=True,
        cutoff=0.5,
        order=4
    )

# Continue with feature extraction...
```

---

## Tips for Python API Usage

### 1. Use Context Managers for Cleanup

```python
class AnalysisSession:
    def __init__(self, subject_id):
        self.collection = SignalCollection(metadata={'subject_id': subject_id})

    def __enter__(self):
        return self.collection

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup temporary data
        for key, signal in self.collection.time_series_signals.items():
            if signal.metadata.temporary:
                signal.clear_data()

# Usage
with AnalysisSession('subject_001') as collection:
    # Do analysis
    pass
# Automatic cleanup when done
```

### 2. Create Reusable Functions

```python
def extract_sleep_features(collection, hr_key, accel_key):
    """Extract standard sleep features from HR and accelerometer."""

    # Generate epoch grid
    collection.metadata.epoch_grid_config = {
        'window_length': '30s',
        'step_size': '30s'
    }
    collection.generate_epoch_grid()

    # HRV
    hrv = collection.apply_multi_signal_operation(
        operation_name='compute_hrv_features',
        signal_keys=[hr_key],
        parameters={
            'hrv_metrics': ['hr_mean', 'hr_std', 'hr_cv', 'hr_range'],
            'use_rr_intervals': False
        }
    )
    collection.add_feature('hrv', hrv)

    # Movement
    movement = collection.apply_multi_signal_operation(
        operation_name='compute_movement_features',
        signal_keys=[accel_key],
        parameters={'movement_metrics': 'all'}
    )
    collection.add_feature('movement', movement)

    # Correlation
    corr = collection.apply_multi_signal_operation(
        operation_name='compute_correlation_features',
        signal_keys=[hr_key, accel_key],
        parameters={
            'signal1_column': 'hr',
            'signal2_column': 'x',
            'method': 'pearson',
            'window_length': '60s'
        }
    )
    collection.add_feature('hr_accel_corr', corr)

    # Combine
    collection.combine_features(
        inputs=['hrv', 'movement', 'hr_accel_corr'],
        feature_index_config=['feature_type']
    )

    return collection._combined_feature_matrix

# Usage
features = extract_sleep_features(collection, 'hr_merged_0', 'accel_merged_0')
```

### 3. Error Handling

```python
try:
    # Import data
    signals = collection.import_signals_from_source(...)

    if not signals:
        raise ValueError("No signals imported. Check data directory and file pattern.")

    # Process
    collection.generate_epoch_grid()
    features = collection.apply_multi_signal_operation(...)

except FileNotFoundError as e:
    print(f"❌ Data files not found: {e}")
except ValueError as e:
    print(f"❌ Invalid data: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
```

---

## Next Steps

- **[Common Workflows](common-workflows.md)** - See YAML workflow equivalents
- **[Best Practices](best-practices.md)** - Production-quality analysis
- **[Feature Extraction Guide](feature-extraction-guide.md)** - Detailed feature docs
- **[API Reference](../api-reference.md)** - Complete API documentation

**Happy coding!** 💻
