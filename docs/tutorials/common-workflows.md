# Common Workflows Tutorial

This tutorial covers the most common use cases for sleep analysis. Each example is a complete, ready-to-use workflow.

## Table of Contents

1. [Heart Rate Variability (HRV) Analysis](#1-heart-rate-variability-hrv-analysis)
2. [Movement and Activity Detection](#2-movement-and-activity-detection)
3. [Multi-Sensor Feature Extraction](#3-multi-sensor-feature-extraction)
4. [Sleep Staging Preparation](#4-sleep-staging-preparation)
5. [Correlation Analysis](#5-correlation-analysis)

---

## 1. Heart Rate Variability (HRV) Analysis

HRV is a key indicator of autonomic nervous system activity and sleep quality. This workflow extracts HRV features from heart rate data.

### Use Case
Extract HRV metrics (mean HR, HR variability, HR range) from overnight heart rate recordings.

### Workflow: `hrv_analysis.yaml`

```yaml
# HRV Analysis Workflow
# Extracts heart rate variability metrics for sleep analysis

collection_settings:
  epoch_grid_config:
    window_length: "30s"  # 30-second epochs (sleep staging standard)
    step_size: "30s"      # Non-overlapping

import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "EKG"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr"

steps:
  # Generate epoch grid
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # Extract HRV features
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]
    parameters:
      hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
      use_rr_intervals: false  # Use HR approximation
    output: "hrv_features"

  # Also extract basic statistics for comparison
  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]
    parameters:
      aggregations: ["mean", "std", "min", "max", "median"]
    output: "hr_stats"

export:
  - formats: ["csv", "excel"]
    output_dir: "results/hrv"
    content: ["hrv_features", "hr_stats"]

visualization:
  # Plot heart rate over time
  - type: time_series
    signals: ["hr_merged_0"]
    title: "Heart Rate Over Time"
    output: "results/hrv/heart_rate_plot.html"
    backend: plotly
    parameters:
      width: 1400
      height: 400
      max_points: 10000
```

### Run It

```bash
python -m sleep_analysis.cli.run_workflow \
  --workflow hrv_analysis.yaml \
  --data-dir data
```

### Output Features

The `hrv_features.csv` will contain:

| Timestamp | hr_mean | hr_std | hr_cv | hr_range |
|-----------|---------|--------|-------|----------|
| 2023-11-15T22:00:00 | 65.2 | 2.1 | 0.032 | 8 |
| 2023-11-15T22:00:30 | 64.8 | 1.9 | 0.029 | 7 |
| ... | ... | ... | ... | ... |

**Feature Meanings:**
- `hr_mean`: Average heart rate in epoch
- `hr_std`: Standard deviation (variability)
- `hr_cv`: Coefficient of variation (normalized variability)
- `hr_range`: Max - Min heart rate in epoch

### Interpretation

- **Lower HR + Higher HRV** → Deep sleep (N3)
- **Higher HR + Lower HRV** → REM sleep or wake
- **Moderate HR + Moderate HRV** → Light sleep (N1/N2)

---

## 2. Movement and Activity Detection

Movement patterns are strong indicators of sleep/wake states. This workflow extracts movement features from accelerometer data.

### Use Case
Detect periods of activity and stillness from wrist or chest accelerometer data.

### Workflow: `movement_analysis.yaml`

```yaml
# Movement Analysis Workflow
# Extracts activity features from accelerometer data

collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

import:
  - signal_type: "accelerometer"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_ACC.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "ACCEL"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "accel"

steps:
  # Generate epoch grid
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # Extract all movement features
  - type: multi_signal
    operation: "compute_movement_features"
    inputs: ["accel"]
    parameters:
      movement_metrics: "all"  # All available movement features
    output: "movement_features"

  # Also get basic statistics on each axis
  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["accel"]
    parameters:
      aggregations: ["mean", "std", "var"]
    output: "accel_stats"

export:
  - formats: ["csv", "excel"]
    output_dir: "results/movement"
    content: ["movement_features", "accel_stats"]

visualization:
  # Plot accelerometer magnitude
  - type: time_series
    signals: ["accel_merged_0"]
    title: "Accelerometer Activity"
    output: "results/movement/activity_plot.html"
    backend: plotly
    parameters:
      width: 1400
      height: 600
```

### Run It

```bash
python -m sleep_analysis.cli.run_workflow \
  --workflow movement_analysis.yaml \
  --data-dir data
```

### Output Features

Movement features typically include:

- **magnitude_mean**: Average movement intensity
- **magnitude_std**: Movement variability
- **magnitude_max**: Peak movement
- **zero_crossing_rate**: Frequency of direction changes
- **activity_count**: Number of activity events

### Interpretation

- **Low magnitude, low variance** → Still/sleeping
- **High magnitude, high variance** → Active/awake
- **Intermittent spikes** → Restless sleep or stage transitions

---

## 3. Multi-Sensor Feature Extraction

Combining multiple sensors provides richer features for sleep staging. This workflow extracts features from both heart rate and accelerometer.

### Use Case
Prepare a comprehensive feature set for machine learning-based sleep staging.

### Workflow: `multi_sensor_features.yaml`

```yaml
# Multi-Sensor Feature Extraction
# Combines heart rate and accelerometer features

collection_settings:
  feature_index_config: ["sensor_model", "feature_type"]
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

import:
  # Heart rate from chest
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "EKG"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr"

  # Accelerometer from chest
  - signal_type: "accelerometer"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_ACC.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "ACCEL"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "accel"

steps:
  # Generate epoch grid
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # Extract HRV features
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]
    parameters:
      hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
      use_rr_intervals: false
    output: "hrv"

  # Extract movement features
  - type: multi_signal
    operation: "compute_movement_features"
    inputs: ["accel"]
    parameters:
      movement_metrics: "all"
    output: "movement"

  # Extract correlation between HR and movement
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr", "accel"]
    parameters:
      signal1_column: "hr"
      signal2_column: "x"
      method: "pearson"
      window_length: "60s"  # Longer window for stable correlation
    output: "hr_movement_corr"

  # Combine all features into one matrix
  - type: collection
    operation: "combine_features"
    inputs: ["hrv", "movement", "hr_movement_corr"]
    parameters: {}

export:
  - formats: ["csv", "excel"]
    output_dir: "results/combined_features"
    content: ["combined_features"]

visualization:
  # Plot both signals together
  - type: time_series
    signals: ["hr_merged_0", "accel_merged_0"]
    layout: vertical
    title: "Heart Rate and Movement"
    output: "results/combined_features/sensors_plot.html"
    backend: plotly
    parameters:
      width: 1400
      height: 800
      link_x_axes: true  # Synchronized zooming
```

### Run It

```bash
python -m sleep_analysis.cli.run_workflow \
  --workflow multi_sensor_features.yaml \
  --data-dir data
```

### Output

The combined feature matrix will have MultiIndex columns:

```
timestamp | (PolarH10, HRV, hr_mean) | (PolarH10, HRV, hr_std) | (PolarH10, MOVEMENT, magnitude_mean) | ...
```

This format is ideal for:
- Machine learning models (scikit-learn, TensorFlow)
- Statistical analysis (R, SPSS)
- Feature selection and correlation analysis

---

## 4. Sleep Staging Preparation

Prepare data for sleep staging algorithms by extracting comprehensive features and aligning with ground truth labels.

### Use Case
Create a feature matrix ready for training or applying sleep staging models.

### Workflow: `sleep_staging_prep.yaml`

```yaml
# Sleep Staging Feature Preparation
# Extracts all relevant features for sleep classification

collection_settings:
  feature_index_config: ["feature_type", "sensor_model"]
  epoch_grid_config:
    window_length: "30s"  # Match sleep staging standard
    step_size: "30s"

import:
  # Heart rate data
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "EKG"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr"

  # Accelerometer data
  - signal_type: "accelerometer"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_ACC.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "ACCEL"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "accel"

  # (Optional) Ground truth sleep stages from EEG or expert scoring
  # - signal_type: "eeg_sleep_stage"
  #   importer: "EnchantedWaveImporter"
  #   source: "sleep_stages.csv"
  #   ...

steps:
  # Generate epoch grid
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # HRV features (autonomic activity)
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]
    parameters:
      hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
      use_rr_intervals: false
    output: "hrv_features"

  # Movement features (activity level)
  - type: multi_signal
    operation: "compute_movement_features"
    inputs: ["accel"]
    parameters:
      movement_metrics: "all"
    output: "movement_features"

  # Statistical features from HR
  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]
    parameters:
      aggregations: ["mean", "std", "min", "max", "median", "q25", "q75"]
    output: "hr_stats"

  # Statistical features from accelerometer
  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["accel"]
    parameters:
      aggregations: ["mean", "std", "var", "max"]
    output: "accel_stats"

  # HR-movement correlation
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr", "accel"]
    parameters:
      signal1_column: "hr"
      signal2_column: "x"
      method: "pearson"
      window_length: "60s"
    output: "hr_accel_corr"

  # (Optional) Extract sleep stage labels if available
  # - type: multi_signal
  #   operation: "compute_sleep_stage_mode"
  #   inputs: ["sleep_stages"]
  #   parameters: {}
  #   output: "sleep_labels"

  # Combine all features
  - type: collection
    operation: "combine_features"
    inputs: [
      "hrv_features",
      "movement_features",
      "hr_stats",
      "accel_stats",
      "hr_accel_corr"
    ]
    parameters: {}

  # Print summary
  - type: collection
    operation: "summarize_signals"
    parameters:
      print_summary: true

export:
  # Export feature matrix for ML
  - formats: ["csv", "excel"]
    output_dir: "results/sleep_staging"
    content: ["combined_features"]

  # Export individual features for inspection
  - formats: ["excel"]
    output_dir: "results/sleep_staging/individual"
    content: ["hrv_features", "movement_features", "hr_accel_corr"]

  # Export summary
  - formats: ["csv"]
    output_dir: "results/sleep_staging"
    content: ["summary"]
```

### Run It

```bash
python -m sleep_analysis.cli.run_workflow \
  --workflow sleep_staging_prep.yaml \
  --data-dir data
```

### Use the Output with scikit-learn

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the feature matrix
features = pd.read_csv('results/sleep_staging/combined_features.csv', index_col=0)

# Load labels (if you have them)
# labels = pd.read_csv('results/sleep_staging/sleep_labels.csv', index_col=0)

# Train a model
# model = RandomForestClassifier()
# model.fit(features, labels)
```

---

## 5. Correlation Analysis

Analyze relationships between different physiological signals to understand coupling and interactions.

### Use Case
Examine how heart rate and movement correlate over time, which can reveal sleep-wake patterns.

### Workflow: `correlation_analysis.yaml`

```yaml
# Correlation Analysis Workflow
# Computes correlations between heart rate and movement

collection_settings:
  epoch_grid_config:
    window_length: "60s"  # Longer window for stable correlation
    step_size: "30s"      # Overlapping for smooth correlation

import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "EKG"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr"

  - signal_type: "accelerometer"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_ACC.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "ACCEL"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "accel"

steps:
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # Pearson correlation (linear relationship)
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr", "accel"]
    parameters:
      signal1_column: "hr"
      signal2_column: "x"  # X-axis acceleration
      method: "pearson"
      window_length: "60s"
    output: "pearson_corr"

  # Spearman correlation (monotonic relationship)
  - type: multi_signal
    operation: "compute_correlation_features"
    inputs: ["hr", "accel"]
    parameters:
      signal1_column: "hr"
      signal2_column: "x"
      method: "spearman"
      window_length: "60s"
    output: "spearman_corr"

export:
  - formats: ["csv", "excel"]
    output_dir: "results/correlation"
    content: ["pearson_corr", "spearman_corr"]

visualization:
  - type: time_series
    signals: ["pearson_corr", "spearman_corr"]
    title: "HR-Movement Correlation Over Time"
    output: "results/correlation/correlation_plot.html"
    backend: plotly
    parameters:
      width: 1400
      height: 400
```

### Run It

```bash
python -m sleep_analysis.cli.run_workflow \
  --workflow correlation_analysis.yaml \
  --data-dir data
```

### Interpretation

- **Positive correlation** → HR and movement increase together (awake, active)
- **Near-zero correlation** → HR and movement independent (deep sleep)
- **Negative correlation** → Movement decreases as HR changes (rare, may indicate artifacts)

---

## Tips for All Workflows

### 1. Adjust Epoch Length Based on Use Case

```yaml
# For sleep staging (standard)
epoch_grid_config:
  window_length: "30s"
  step_size: "30s"

# For detailed analysis (overlapping)
epoch_grid_config:
  window_length: "60s"
  step_size: "10s"

# For long-term trends
epoch_grid_config:
  window_length: "5m"
  step_size: "5m"
```

### 2. Use Visualization for Quality Control

Always visualize your data before and after processing:

```yaml
visualization:
  - type: time_series
    signals: ["your_signal"]
    output: "results/plots/signal_check.html"
```

### 3. Export Multiple Formats

Export to both CSV (for Python/R) and Excel (for manual inspection):

```yaml
export:
  - formats: ["csv", "excel"]
    output_dir: "results"
```

### 4. Use Descriptive Base Names

```yaml
base_name: "hr_chest_h10"  # Clear
# vs
base_name: "hr"  # Ambiguous if you have multiple HR sources
```

### 5. Print Summaries for Debugging

```yaml
steps:
  - type: collection
    operation: "summarize_signals"
    parameters:
      print_summary: true
```

This prints a table showing all signals, their types, and sample rates.

---

## Next Steps

- **[Feature Extraction Guide](feature-extraction-guide.md)** - Detailed documentation of all available features
- **[Best Practices](best-practices.md)** - Production-quality analysis tips
- **[Python API Guide](python-api-guide.md)** - Use workflows programmatically
- **[Troubleshooting](../troubleshooting.md)** - Common issues and solutions

**Happy analyzing!** 📊
