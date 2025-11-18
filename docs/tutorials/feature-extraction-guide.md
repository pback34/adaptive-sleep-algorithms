# Feature Extraction Guide

Complete reference for all feature extraction operations in the Adaptive Sleep Algorithms framework.

## Table of Contents

1. [Overview](#overview)
2. [Statistical Features](#statistical-features)
3. [HRV Features](#hrv-features)
4. [Movement Features](#movement-features)
5. [Correlation Features](#correlation-features)
6. [Sleep Stage Features](#sleep-stage-features)
7. [Custom Features](#custom-features)

---

## Overview

Features are summary statistics computed over time windows (epochs). All feature extraction operations:

- **Require an epoch grid**: Call `generate_epoch_grid` first
- **Return Feature objects**: With FeatureMetadata and MultiIndex DataFrames
- **Support multiple inputs**: Can process multiple signals simultaneously
- **Track provenance**: Record source signals and parameters in metadata

### Common Parameters

All feature extraction operations accept:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `inputs` | List[str] | Signal keys to process | Required |
| `output` | str | Name for extracted feature | Auto-generated |

---

## Statistical Features

Extract basic statistical summaries over each epoch.

### Operation: `feature_statistics`

**Use Cases:**
- Baseline feature extraction
- Data quality assessment
- Simple trend analysis

### Parameters

```yaml
- type: multi_signal
  operation: "feature_statistics"
  inputs: ["signal_key"]
  parameters:
    aggregations: ["mean", "std", "min", "max"]  # Statistics to compute
  output: "signal_stats"
```

### Available Aggregations

| Aggregation | Description | Use Case |
|-------------|-------------|----------|
| `mean` | Average value | Central tendency |
| `std` | Standard deviation | Variability |
| `var` | Variance | Spread |
| `min` | Minimum value | Range detection |
| `max` | Maximum value | Peak detection |
| `median` | 50th percentile | Robust central tendency |
| `q25` | 25th percentile | Lower quartile |
| `q75` | 75th percentile | Upper quartile |
| `iqr` | Interquartile range (q75 - q25) | Robust spread |
| `range` | max - min | Full range |
| `skew` | Skewness | Distribution asymmetry |
| `kurtosis` | Kurtosis | Distribution tailedness |

### Example Output

For a heart rate signal with 30-second epochs:

```
timestamp            | (hr_0, mean) | (hr_0, std) | (hr_0, min) | (hr_0, max)
---------------------|--------------|-------------|-------------|-------------
2023-11-15 22:00:00  | 65.2         | 2.1         | 62          | 68
2023-11-15 22:00:30  | 64.8         | 1.9         | 61          | 67
2023-11-15 22:01:00  | 66.1         | 2.3         | 63          | 70
```

### Python API

```python
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

## HRV Features

Extract heart rate variability metrics, indicators of autonomic nervous system activity.

### Operation: `compute_hrv_features`

**Use Cases:**
- Sleep/wake classification
- Autonomic activity assessment
- Stress and recovery monitoring
- Sleep stage discrimination (especially REM vs. N3)

### Parameters

```yaml
- type: multi_signal
  operation: "compute_hrv_features"
  inputs: ["hr_signal_key"]
  parameters:
    hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
    use_rr_intervals: false  # true if RR interval data available
  output: "hrv_features"
```

### Available HRV Metrics

#### Time-Domain Metrics

| Metric | Full Name | Description | Physiological Meaning |
|--------|-----------|-------------|-----------------------|
| `hr_mean` | Mean HR | Average heart rate | Overall cardiac workload |
| `hr_std` | HR StdDev | Standard deviation of HR | HR variability |
| `hr_cv` | HR CoV | Coefficient of variation (std/mean) | Normalized variability |
| `hr_range` | HR Range | max - min HR | Total variation |
| `hr_min` | Minimum HR | Lowest HR in epoch | Basal heart rate |
| `hr_max` | Maximum HR | Highest HR in epoch | Peak heart rate |

#### When `use_rr_intervals: true` (requires RR interval data)

| Metric | Description | Physiological Meaning |
|--------|-------------|-----------------------|
| `rmssd` | Root mean square of successive differences | Short-term HRV, parasympathetic activity |
| `sdnn` | Standard deviation of NN intervals | Overall HRV |
| `pnn50` | % of successive NN differences > 50ms | Parasympathetic activity |

### Physiological Interpretation

**Deep Sleep (N3):**
- **Low HR** (50-60 bpm)
- **High HRV** (high RMSSD, high SDNN)
- Parasympathetic dominance

**REM Sleep:**
- **Variable HR** (high hr_range)
- **Moderate HRV**
- Autonomic instability

**Wake:**
- **High HR** (70-90 bpm)
- **Low HRV** (low RMSSD)
- Sympathetic dominance

### Example Output

```
timestamp            | (hr_0, hr_mean) | (hr_0, hr_std) | (hr_0, hr_cv) | (hr_0, hr_range)
---------------------|-----------------|----------------|---------------|------------------
2023-11-15 22:00:00  | 65.2            | 2.1            | 0.032         | 8
2023-11-15 22:00:30  | 64.8            | 1.9            | 0.029         | 7
2023-11-15 22:01:00  | 66.1            | 2.3            | 0.035         | 9
```

### Python API

```python
hrv_feature = collection.apply_multi_signal_operation(
    operation_name='compute_hrv_features',
    signal_keys=['hr_merged_0'],
    parameters={
        'hrv_metrics': ['hr_mean', 'hr_std', 'hr_cv', 'hr_range'],
        'use_rr_intervals': False  # Set to True if you have RR data
    }
)
collection.add_feature('hrv', hrv_feature)
```

---

## Movement Features

Extract activity and movement patterns from accelerometer data.

### Operation: `compute_movement_features`

**Use Cases:**
- Sleep/wake discrimination
- Restlessness detection
- Sleep fragmentation analysis
- Activity level assessment

### Parameters

```yaml
- type: multi_signal
  operation: "compute_movement_features"
  inputs: ["accel_signal_key"]
  parameters:
    movement_metrics: "all"  # or list of specific metrics
  output: "movement_features"
```

### Available Movement Metrics

| Metric | Description | Physiological Meaning |
|--------|-------------|-----------------------|
| `magnitude_mean` | Average movement intensity | Overall activity level |
| `magnitude_std` | Movement variability | Activity consistency |
| `magnitude_var` | Movement variance | Activity spread |
| `magnitude_max` | Peak movement | Maximum activity |
| `magnitude_min` | Minimum movement | Baseline activity |
| `magnitude_range` | max - min movement | Activity range |
| `zero_crossing_rate` | Frequency of direction changes | Movement oscillation |
| `activity_count` | Number of activity events | Discrete movements |
| `stillness_ratio` | % of time below threshold | Stillness periods |

### Thresholds

Default activity threshold: 10 mg (milli-g)
- Below threshold → stillness
- Above threshold → activity

### Physiological Interpretation

**Deep Sleep:**
- **Low magnitude_mean** (< 20 mg)
- **High stillness_ratio** (> 0.9)
- Minimal movement

**Light Sleep:**
- **Moderate magnitude_mean** (20-50 mg)
- **Occasional spikes**
- Small adjustments

**Wake:**
- **High magnitude_mean** (> 100 mg)
- **High activity_count**
- Frequent movements

### Example Output

```
timestamp            | (accel_0, magnitude_mean) | (accel_0, magnitude_std) | (accel_0, stillness_ratio)
---------------------|---------------------------|--------------------------|---------------------------
2023-11-15 22:00:00  | 15.3                      | 8.2                      | 0.85
2023-11-15 22:00:30  | 12.1                      | 6.5                      | 0.92
2023-11-15 22:01:00  | 45.7                      | 25.3                     | 0.23
```

### Python API

```python
movement_feature = collection.apply_multi_signal_operation(
    operation_name='compute_movement_features',
    signal_keys=['accel_merged_0'],
    parameters={
        'movement_metrics': 'all'
        # Or specific metrics:
        # 'movement_metrics': ['magnitude_mean', 'magnitude_std', 'stillness_ratio']
    }
)
collection.add_feature('movement', movement_feature)
```

---

## Correlation Features

Compute correlations between different physiological signals.

### Operation: `compute_correlation_features`

**Use Cases:**
- Multi-sensor fusion
- Cardio-respiratory coupling analysis
- HR-movement relationship
- Sleep stage discrimination

### Parameters

```yaml
- type: multi_signal
  operation: "compute_correlation_features"
  inputs: ["signal1_key", "signal2_key"]
  parameters:
    signal1_column: "hr"      # Column name from signal 1
    signal2_column: "x"       # Column name from signal 2
    method: "pearson"         # pearson, spearman, or kendall
    window_length: "60s"      # Correlation window (optional, uses epoch by default)
  output: "correlation"
```

### Correlation Methods

| Method | Type | Use Case | Robust to Outliers? |
|--------|------|----------|---------------------|
| `pearson` | Linear correlation | Normal distributions | No |
| `spearman` | Rank correlation | Monotonic relationships | Yes |
| `kendall` | Rank correlation | Small samples | Yes |

### Common Signal Pairs

#### HR-Movement Correlation

```yaml
- type: multi_signal
  operation: "compute_correlation_features"
  inputs: ["hr_signal", "accel_signal"]
  parameters:
    signal1_column: "hr"
    signal2_column: "x"
    method: "pearson"
```

**Interpretation:**
- **Positive correlation** → HR increases with movement (wake)
- **Near-zero** → HR independent of movement (sleep)
- **Negative** → Unusual, may indicate artifacts

#### HR-Respiration Correlation (if respiratory signal available)

```yaml
- type: multi_signal
  operation: "compute_correlation_features"
  inputs: ["hr_signal", "resp_signal"]
  parameters:
    signal1_column: "hr"
    signal2_column: "resp_rate"
    method: "pearson"
```

**Interpretation:**
- **Respiratory sinus arrhythmia** (RSA)
- Higher in deep sleep
- Lower in wake and REM

### Example Output

```
timestamp            | (hr_0_accel_0, correlation)
---------------------|----------------------------
2023-11-15 22:00:00  | 0.65
2023-11-15 22:00:30  | 0.12
2023-11-15 22:01:00  | 0.78
```

### Python API

```python
corr_feature = collection.apply_multi_signal_operation(
    operation_name='compute_correlation_features',
    signal_keys=['hr_merged_0', 'accel_merged_0'],
    parameters={
        'signal1_column': 'hr',
        'signal2_column': 'x',
        'method': 'pearson',
        'window_length': '60s'  # Optional
    }
)
collection.add_feature('hr_movement_corr', corr_feature)
```

---

## Sleep Stage Features

Extract modal sleep stage for each epoch (when ground truth labels available).

### Operation: `compute_sleep_stage_mode`

**Use Cases:**
- Preparing labels for supervised learning
- Comparing algorithmic staging to ground truth
- Sleep stage distribution analysis

### Parameters

```yaml
- type: multi_signal
  operation: "compute_sleep_stage_mode"
  inputs: ["sleep_stage_signal_key"]
  parameters: {}
  output: "sleep_labels"
```

### Sleep Stage Encoding

Standard AASM stages:

| Stage | Code | Description |
|-------|------|-------------|
| Wake | 0 | Awake |
| N1 | 1 | Light sleep stage 1 |
| N2 | 2 | Light sleep stage 2 |
| N3 | 3 | Deep sleep (slow-wave) |
| REM | 4 | Rapid eye movement sleep |

### Mode Calculation

For each epoch, the **most frequent** sleep stage is selected:

```
Epoch 1 (30 seconds of second-by-second labels):
  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ...]  # All N2
  → Mode = 2 (N2)

Epoch 2 (transition):
  [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...]  # More N3 than N2
  → Mode = 3 (N3)
```

### Example Output

```
timestamp            | (sleep_0, sleep_stage_mode)
---------------------|----------------------------
2023-11-15 22:00:00  | 0  # Wake
2023-11-15 22:00:30  | 1  # N1
2023-11-15 22:01:00  | 2  # N2
2023-11-15 22:01:30  | 2  # N2
2023-11-15 22:02:00  | 3  # N3
```

### Python API

```python
# First, import sleep stage labels
sleep_importer = EnchantedWaveImporter()
collection.import_signals_from_source(
    importer_instance=sleep_importer,
    source='ground_truth_stages.csv',
    spec={
        'signal_type': SignalType.EEG_SLEEP_STAGE,
        'base_name': 'sleep_stages'
    }
)

# Extract modal stage for each epoch
stage_feature = collection.apply_multi_signal_operation(
    operation_name='compute_sleep_stage_mode',
    signal_keys=['sleep_stages_0'],
    parameters={}
)
collection.add_feature('sleep_labels', stage_feature)
```

---

## Custom Features

### Extending the Framework

You can register custom feature extraction functions:

```python
from sleep_analysis.operations.feature_extraction import register_feature_extractor
from sleep_analysis.core.feature import Feature, FeatureMetadata
from sleep_analysis.signals.signal_types import FeatureType
import pandas as pd

@register_feature_extractor('custom_spectral_features')
def compute_custom_spectral_features(
    signals,
    epoch_grid_index,
    parameters,
    global_window_length,
    global_step_size,
    **kwargs
):
    """
    Custom feature extractor for spectral features.

    Parameters:
        signals: List[TimeSeriesSignal] - Input signals
        epoch_grid_index: pd.DatetimeIndex - Epoch grid
        parameters: Dict - User parameters from workflow
        global_window_length: pd.Timedelta - Epoch window
        global_step_size: pd.Timedelta - Epoch step
    """

    # Your custom feature extraction logic here
    results = []

    for epoch_start in epoch_grid_index:
        epoch_end = epoch_start + global_window_length

        # Extract data for this epoch
        for signal in signals:
            data = signal.get_data()
            epoch_data = data.loc[epoch_start:epoch_end]

            # Compute custom features (example: FFT power in specific bands)
            # ... your logic here ...

            result_row = {
                'timestamp': epoch_start,
                'signal_key': signal.metadata.name,
                'delta_power': delta_power,
                'theta_power': theta_power,
                'alpha_power': alpha_power
            }
            results.append(result_row)

    # Create feature DataFrame
    feature_df = pd.DataFrame(results)
    feature_df = feature_df.set_index('timestamp')

    # Create FeatureMetadata
    metadata = FeatureMetadata(
        feature_id='custom_spectral',
        name='custom_spectral',
        feature_type=FeatureType.SPECTRAL,
        epoch_window_length=global_window_length,
        epoch_step_size=global_step_size,
        operations=[],
        feature_names=['delta_power', 'theta_power', 'alpha_power'],
        source_signal_keys=[s.metadata.name for s in signals],
        source_signal_ids=[s.metadata.signal_id for s in signals]
    )

    # Return Feature object
    return Feature(metadata=metadata, data=feature_df)
```

### Using Custom Features

```yaml
# In workflow
steps:
  - type: multi_signal
    operation: "custom_spectral_features"
    inputs: ["eeg_signal"]
    parameters:
      freq_bands:
        delta: [0.5, 4]
        theta: [4, 8]
        alpha: [8, 13]
    output: "spectral_features"
```

---

## Feature Selection Tips

### For Sleep/Wake Classification

**Minimum features:**
- HR mean and std (HRV)
- Movement magnitude mean and std

**Recommended features:**
- HR mean, std, cv, range
- Movement magnitude mean, std, max, stillness_ratio
- HR-movement correlation

### For Sleep Stage Classification

**Minimum features:**
- All HRV features
- All movement features
- HR-movement correlation

**Recommended features:**
- All of the above
- Spectral features (if available)
- Multiple sensors (chest + wrist)

### For Sleep Fragmentation Analysis

**Key features:**
- Movement activity_count
- Movement stillness_ratio
- HR range (variability)
- Stage transition frequency (from sleep_stage_mode)

---

## Next Steps

- **[Common Workflows](common-workflows.md)** - See features in complete workflows
- **[Python API Guide](python-api-guide.md)** - Extract features programmatically
- **[Best Practices](best-practices.md)** - Feature extraction guidelines
- **[Data Preparation](../data-preparation.md)** - Prepare data for feature extraction

**Happy feature engineering!** 🔬
