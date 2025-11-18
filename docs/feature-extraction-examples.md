# Feature Extraction Examples

This document provides practical usage examples for all feature extraction operations in the Sleep Analysis Framework.

## Table of Contents

1. [Basic Statistical Features](#basic-statistical-features)
2. [Sleep Stage Mode](#sleep-stage-mode)
3. [HRV (Heart Rate Variability) Features](#hrv-heart-rate-variability-features)
4. [Movement Features](#movement-features)
5. [Correlation Features](#correlation-features)

---

## Basic Statistical Features

Computes statistical aggregations (mean, std, min, max, median, var) over epochs for any numeric signal.

### Workflow YAML Example

```yaml
# Extract basic statistics from heart rate signal
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      delimiter: ";"
    base_name: "hr"

steps:
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]  # Base name matches hr_0, hr_1, etc.
    parameters:
      aggregations: ["mean", "std", "min", "max"]
    output: "hr_stats"

export:
  - formats: ["csv"]
    output_dir: "results"
    content: ["hr_stats"]
```

### Python API Example

```python
from sleep_analysis.core.signal_collection import SignalCollection
from sleep_analysis.operations.feature_extraction import compute_feature_statistics
import pandas as pd

# Create collection and import signals
collection = SignalCollection()
# ... import signals ...
collection.generate_epoch_grid()

# Get heart rate signals
hr_signals = [collection.get_signal("hr_0")]

# Compute features
features = compute_feature_statistics(
    signals=hr_signals,
    epoch_grid_index=collection.epoch_grid_index,
    parameters={
        "aggregations": ["mean", "std", "min", "max", "median"]
    },
    global_window_length=pd.Timedelta("30s"),
    global_step_size=pd.Timedelta("30s")
)

# Access feature data
print(features.data.head())
# Output: MultiIndex columns with format (signal_key, feature_name)
# e.g., ('hr_0', 'mean'), ('hr_0', 'std'), ('hr_0', 'min'), etc.

# Get specific epoch statistics
epoch_start = features.data.index[0]
mean_hr = features.data.loc[epoch_start, ('hr_0', 'mean')]
print(f"Mean HR at {epoch_start}: {mean_hr} bpm")
```

### Custom Window Length

```yaml
# Override global window length for specific features
- type: multi_signal
  operation: "feature_statistics"
  inputs: ["hr"]
  parameters:
    window_length: "60s"  # Override: use 60s windows instead of 30s
    aggregations: ["mean", "std"]
  output: "hr_stats_60s"
```

---

## Sleep Stage Mode

Computes the most frequent sleep stage within each epoch from EEG sleep stage signals.

### Workflow YAML Example

```yaml
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

import:
  - signal_type: "EEG_SLEEP_STAGE"
    importer: "EnchantedWaveImporter"
    source: "sleep_data"
    config:
      filename_pattern: "Session_(?P<session_id>\\d+).csv"
    base_name: "sleep_stages"

steps:
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  - type: multi_signal
    operation: "compute_sleep_stage_mode"
    inputs: ["sleep_stages"]
    parameters: {}
    output: "sleep_stage_mode"

export:
  - formats: ["csv"]
    output_dir: "results"
    content: ["sleep_stage_mode"]
```

### Python API Example

```python
from sleep_analysis.operations.feature_extraction import compute_sleep_stage_mode

# Get sleep stage signal
sleep_signal = collection.get_signal("sleep_stages_0")

# Compute modal sleep stage per epoch
features = compute_sleep_stage_mode(
    signals=[sleep_signal],
    epoch_grid_index=collection.epoch_grid_index,
    parameters={},
    global_window_length=pd.Timedelta("30s"),
    global_step_size=pd.Timedelta("30s")
)

# Access results
print(features.data.head())
# Output: MultiIndex DataFrame with modal sleep stage per epoch

# Access stage column (assuming single signal input 'sleep_stages_0')
# MultiIndex format: (signal_key, feature_name)
stage_mode_col = ('sleep_stages_0', 'sleep_stage_mode')
stage_counts = features.data[stage_mode_col].value_counts()
print("Sleep stage distribution:")
print(stage_counts)
```

---

## HRV (Heart Rate Variability) Features

Computes heart rate variability metrics crucial for sleep stage classification.

### Workflow YAML Example

```yaml
# Option 1: Use RR intervals (most accurate)
- type: multi_signal
  operation: "compute_hrv_features"
  inputs: ["rr_intervals"]
  parameters:
    hrv_metrics: ["sdnn", "rmssd", "pnn50", "sdsd"]
    use_rr_intervals: true  # Expecting RR interval data
  output: "hrv_features"

# Option 2: Use heart rate (approximation when RR not available)
- type: multi_signal
  operation: "compute_hrv_features"
  inputs: ["hr"]
  parameters:
    hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
    use_rr_intervals: false  # Using heart rate approximation
  output: "hrv_approx"

# Option 3: Compute all available metrics
- type: multi_signal
  operation: "compute_hrv_features"
  inputs: ["hr"]
  parameters:
    hrv_metrics: "all"  # Computes all applicable metrics
    use_rr_intervals: false
  output: "hrv_complete"
```

### Python API Example

```python
from sleep_analysis.operations.feature_extraction import compute_hrv_features

# Get heart rate signal
hr_signals = [collection.get_signal("hr_chest_0")]

# Compute HRV features
hrv_features = compute_hrv_features(
    signals=hr_signals,
    epoch_grid_index=collection.epoch_grid_index,
    parameters={
        "hrv_metrics": ["hr_mean", "hr_std", "hr_cv", "hr_range"],
        "use_rr_intervals": False
    },
    global_window_length=pd.Timedelta("30s"),
    global_step_size=pd.Timedelta("30s")
)

# Access HRV metrics
print(hrv_features.data.columns)
# Output: MultiIndex with columns like:
# ('hr_chest_0', 'hr_mean'), ('hr_chest_0', 'hr_std'), etc.

# Get mean HR for first epoch
first_epoch = hrv_features.data.index[0]
mean_hr = hrv_features.data.loc[first_epoch, ('hr_chest_0', 'hr_mean')]
print(f"Mean HR: {mean_hr:.1f} bpm")

# Calculate average HRV over all epochs
avg_hr_std = hrv_features.data[('hr_chest_0', 'hr_std')].mean()
print(f"Average HR variability (std): {avg_hr_std:.2f} bpm")
```

### Complete Sleep Analysis Example

```yaml
# Comprehensive HRV + Movement analysis for sleep staging
steps:
  - type: collection
    operation: "generate_epoch_grid"

  # Chest sensor HRV
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr_chest"]
    parameters:
      hrv_metrics: "all"
      use_rr_intervals: false
    output: "hrv_chest"

  # Wrist sensor HRV (comparison)
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr_wrist"]
    parameters:
      hrv_metrics: "all"
      use_rr_intervals: false
    output: "hrv_wrist"

  # Combine for ML input
  - type: collection
    operation: "combine_features"
    inputs: ["hrv_chest", "hrv_wrist"]
```

---

## Movement Features

Computes activity and movement metrics from accelerometer data, essential for detecting sleep/wake states.

### Workflow YAML Example

```yaml
# Extract movement features from accelerometer
- type: multi_signal
  operation: "compute_movement_features"
  inputs: ["accel"]  # Accelerometer signal with x, y, z columns
  parameters:
    movement_metrics: "all"  # All available metrics
  output: "movement_features"

# Or select specific metrics
- type: multi_signal
  operation: "compute_movement_features"
  inputs: ["accel"]
  parameters:
    movement_metrics: ["magnitude_mean", "activity_count", "stillness_ratio"]
  output: "movement_selected"
```

### Python API Example

```python
from sleep_analysis.operations.feature_extraction import compute_movement_features

# Get accelerometer signal (must have x, y, z columns)
accel_signals = [collection.get_signal("accel_chest_0")]

# Compute movement features
movement_features = compute_movement_features(
    signals=accel_signals,
    epoch_grid_index=collection.epoch_grid_index,
    parameters={
        "movement_metrics": "all"
        # Available: ['magnitude_mean', 'magnitude_std', 'magnitude_max',
        #             'activity_count', 'stillness_ratio', 'x_std', 'y_std', 'z_std']
    },
    global_window_length=pd.Timedelta("30s"),
    global_step_size=pd.Timedelta("30s")
)

# Access movement metrics
print(movement_features.data.columns)
# Output: MultiIndex with format (signal_key, feature_name)
# e.g., ('accel_chest_0', 'magnitude_mean'), ('accel_chest_0', 'stillness_ratio'), etc.

# Detect sleep periods (low activity)
stillness = movement_features.data[('accel_chest_0', 'stillness_ratio')]
sleep_epochs = stillness > 0.8  # 80% stillness threshold
print(f"Potential sleep epochs: {sleep_epochs.sum()} of {len(sleep_epochs)}")

# Get active vs. still periods
active_epochs = movement_features.data[('accel_chest_0', 'activity_count')] > 50
print(f"Active epochs: {active_epochs.sum()}")
```

### Movement Feature Descriptions

- **magnitude_mean**: Average magnitude of 3D acceleration vector
- **magnitude_std**: Variability in acceleration magnitude
- **magnitude_max**: Peak acceleration in epoch
- **activity_count**: Number of samples above adaptive threshold
- **stillness_ratio**: Proportion of samples below threshold (higher = more still)
- **x_std, y_std, z_std**: Variability per axis (useful for posture detection)

---

## Correlation Features

Computes correlation between two signals over epochs, useful for detecting relationships like HR-movement coupling.

### Workflow YAML Example

```yaml
# Compute correlation between HR and movement
- type: multi_signal
  operation: "compute_correlation_features"
  inputs: ["hr", "accel"]  # Exactly 2 signals required
  parameters:
    signal1_column: "hr"  # Column from first signal
    signal2_column: "x"   # Column from second signal
    method: "pearson"     # Options: pearson, spearman, kendall
    window_length: "60s"  # Optional: override for more stable correlation
  output: "hr_accel_corr"

# Spearman correlation for non-linear relationships
- type: multi_signal
  operation: "compute_correlation_features"
  inputs: ["hr", "movement_magnitude"]
  parameters:
    signal1_column: "hr"
    signal2_column: "magnitude"
    method: "spearman"
  output: "hr_movement_corr"
```

### Python API Example

```python
from sleep_analysis.operations.feature_extraction import compute_correlation_features

# Get two signals to correlate
hr_signal = collection.get_signal("hr_0")
accel_signal = collection.get_signal("accel_0")

# Compute correlation features
corr_features = compute_correlation_features(
    signals=[hr_signal, accel_signal],
    epoch_grid_index=collection.epoch_grid_index,
    parameters={
        "signal1_column": "hr",
        "signal2_column": "x",
        "method": "pearson",
        "window_length": "60s"  # Longer window for stable correlation
    },
    global_window_length=pd.Timedelta("30s"),
    global_step_size=pd.Timedelta("30s")
)

# Access correlation values
print(corr_features.data.head())
# Output: MultiIndex DataFrame with correlation coefficient per epoch (-1 to 1)

# Analyze correlation patterns
# MultiIndex format: (combined_signal_key, feature_name)
corr_col = (corr_features.metadata.source_signal_keys[0], 'pearson_corr')
corr_values = corr_features.data[corr_col]

print(f"Average correlation: {corr_values.mean():.3f}")
print(f"High correlation epochs: {(corr_values.abs() > 0.7).sum()}")

# Detect sleep based on low HR-movement correlation
sleep_indicator = corr_values.abs() < 0.3  # Low correlation suggests sleep
print(f"Potential sleep epochs (low corr): {sleep_indicator.sum()}")
```

### Correlation Methods

- **Pearson**: Linear correlation, assumes normal distribution
- **Spearman**: Rank-based, handles monotonic non-linear relationships
- **Kendall**: Rank-based, more robust to outliers (requires scipy)

### Multi-Sensor Correlation Example

```yaml
# Comprehensive correlation analysis
steps:
  - type: collection
    operation: "generate_epoch_grid"

  # Chest: HR vs. Movement
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr_chest", "accel_chest"]
    parameters:
      signal1_column: "hr"
      signal2_column: "x"
      method: "pearson"
    output: "chest_hr_accel_corr"

  # Wrist: HR vs. Movement
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr_wrist", "accel_wrist"]
    parameters:
      signal1_column: "hr"
      signal2_column: "x"
      method: "pearson"
    output: "wrist_hr_accel_corr"

  # Cross-sensor: Chest HR vs. Wrist HR
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr_chest", "hr_wrist"]
    parameters:
      signal1_column: "hr"
      signal2_column: "hr"
      method: "pearson"
    output: "chest_wrist_hr_corr"

  # Combine all correlations
  - type: collection
    operation: "combine_features"
    inputs: ["chest_hr_accel_corr", "wrist_hr_accel_corr", "chest_wrist_hr_corr"]
```

---

## Complete Workflow Example

Combining all feature types for comprehensive sleep analysis:

```yaml
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"
  feature_index_config: ["sensor_model", "feature_type"]

import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      delimiter: ";"
    sensor_model: "PolarH10"
    base_name: "hr"

  - signal_type: "accelerometer"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_ACC.txt"
      delimiter: ";"
    sensor_model: "PolarH10"
    base_name: "accel"

steps:
  # 1. Generate epoch grid
  - type: collection
    operation: "generate_epoch_grid"

  # 2. HRV features
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]
    parameters:
      hrv_metrics: "all"
      use_rr_intervals: false
    output: "hrv_features"

  # 3. Movement features
  - type: multi_signal
    operation: "compute_movement_features"
    inputs: ["accel"]
    parameters:
      movement_metrics: "all"
    output: "movement_features"

  # 4. Correlation features
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr", "accel"]
    parameters:
      signal1_column: "hr"
      signal2_column: "x"
      method: "pearson"
      window_length: "60s"
    output: "hr_accel_corr"

  # 5. Basic statistics
  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]
    parameters:
      aggregations: ["mean", "std"]
    output: "hr_stats"

  # 6. Combine all features
  - type: collection
    operation: "combine_features"
    inputs: ["hrv_features", "movement_features", "hr_accel_corr", "hr_stats"]

export:
  - formats: ["csv", "excel"]
    output_dir: "results/features"
    content: ["combined_features"]
```

---

## Tips and Best Practices

### 1. Window Length Selection

- **30 seconds**: Standard for sleep stage classification
- **60 seconds**: More stable for correlation and HRV
- **5-10 seconds**: High temporal resolution for rapid events

### 2. Feature Selection

Choose features based on your analysis goals:

- **Sleep stage classification**: HRV + Movement + Correlation
- **Sleep quality**: HRV variability, stillness ratio
- **Sleep/wake detection**: Activity count, magnitude statistics
- **Multi-sensor validation**: Cross-sensor correlations

### 3. Handling Missing Data

All feature operations handle missing data gracefully:
- Return NaN for epochs with insufficient data
- Continue processing other epochs
- Filter NaN rows in post-processing if needed

### 4. Memory Optimization

For large datasets:
- Process signals in smaller time chunks
- Use longer step sizes (overlapping windows = more memory)
- Export individual features before combining

### 5. Debugging

Enable debug logging to see feature extraction details:
```bash
sleep-analysis -w workflow.yaml -d data -v
```

Look for:
- Number of epochs processed vs. skipped
- Missing data warnings
- Feature name verification

---

**Last Updated**: 2025-11-18

**See Also**:
- [Quick Start Guide](quick-start.md) - Getting started
- [Data Preparation Guide](data-preparation.md) - Preparing your data
- [Troubleshooting Guide](troubleshooting.md) - Common issues
