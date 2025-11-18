# Best Practices for Sleep Signal Analysis

This guide provides recommendations for production-quality sleep analysis using the Adaptive Sleep Algorithms framework.

## Table of Contents

1. [Data Organization](#data-organization)
2. [Workflow Design](#workflow-design)
3. [Feature Extraction](#feature-extraction)
4. [Quality Control](#quality-control)
5. [Performance Optimization](#performance-optimization)
6. [Reproducibility](#reproducibility)
7. [Common Pitfalls](#common-pitfalls)

---

## Data Organization

### 1. Organize Data by Subject and Session

```
data/
├── subject_001/
│   ├── session_20231115/
│   │   ├── Polar_H10_*_HR.txt
│   │   ├── Polar_H10_*_ACC.txt
│   │   └── metadata.json
│   └── session_20231116/
│       └── ...
├── subject_002/
│   └── ...
```

**Benefits:**
- Easy batch processing
- Clear subject-session mapping
- Metadata stored alongside data

### 2. Use Consistent File Naming

```
# Good: Includes sensor, device ID, date, time, data type
Polar_H10_ABC123_20231115_220000_HR.txt

# Bad: Ambiguous
heart_rate.txt
```

**Include in filename:**
- Sensor model
- Device ID (if multiple devices)
- Date and time (ISO 8601 format)
- Data type (HR, ACC, PPG, etc.)

### 3. Store Metadata Separately

Create a `metadata.json` for each session:

```json
{
  "subject_id": "subject_001",
  "session_id": "20231115",
  "recording_date": "2023-11-15",
  "recording_start": "22:00:00",
  "recording_end": "06:00:00",
  "sensors": {
    "chest": "Polar H10 (ABC123)",
    "wrist": "Polar Sense (XYZ789)"
  },
  "notes": "Good quality recording, subject reported restful sleep"
}
```

---

## Workflow Design

### 1. Start Simple, Then Expand

**Phase 1: Data Import**
```yaml
# workflow_phase1.yaml - Just import and export
import:
  - signal_type: "heart_rate"
    ...
export:
  - formats: ["csv"]
    content: ["all_ts"]
```

**Phase 2: Add Basic Features**
```yaml
# workflow_phase2.yaml - Add simple features
steps:
  - operation: "feature_statistics"
    ...
```

**Phase 3: Full Pipeline**
```yaml
# workflow_phase3.yaml - Complete analysis
steps:
  - operation: "compute_hrv_features"
  - operation: "compute_movement_features"
  - operation: "compute_correlation_features"
  - operation: "combine_features"
```

### 2. Use Descriptive Names for Outputs

```yaml
# Good: Clear purpose
output: "hrv_chest_30s"
output: "movement_wrist_overlapping"

# Bad: Ambiguous
output: "features1"
output: "output"
```

### 3. Modularize Complex Workflows

Split large workflows into stages:

```bash
# Stage 1: Import and preprocess
python -m sleep_analysis.cli.run_workflow \
  --workflow 01_import_preprocess.yaml

# Stage 2: Feature extraction
python -m sleep_analysis.cli.run_workflow \
  --workflow 02_feature_extraction.yaml

# Stage 3: ML model application
python -m sleep_analysis.cli.run_workflow \
  --workflow 03_apply_model.yaml
```

### 4. Document Your Workflows

Add comments explaining your choices:

```yaml
# Epoch Grid Configuration
# Using 30s epochs to match standard sleep staging practice (AASM guidelines)
# Non-overlapping to maintain independence between epochs
epoch_grid_config:
  window_length: "30s"
  step_size: "30s"

# HRV Feature Extraction
# Using HR approximation (not RR intervals) because:
# - Polar H10 provides HR, not individual R-peaks
# - HR-based HRV is sufficient for sleep/wake classification
- type: multi_signal
  operation: "compute_hrv_features"
  parameters:
    use_rr_intervals: false  # No RR interval data available
```

---

## Feature Extraction

### 1. Choose Epoch Length Appropriately

| Use Case | Window Length | Step Size | Rationale |
|----------|---------------|-----------|-----------|
| Sleep staging | 30s | 30s | AASM standard |
| Detailed transitions | 30s | 10s | Capture rapid changes |
| Long-term trends | 5min | 5min | Reduce noise |
| Real-time monitoring | 10s | 5s | Quick response |

### 2. Extract Features Hierarchically

**Level 1: Basic Statistics**
- Always include mean, std, min, max
- Foundation for understanding data distribution

**Level 2: Domain-Specific Features**
- HRV metrics for autonomic activity
- Movement features for activity detection
- Spectral features for periodicity

**Level 3: Interaction Features**
- Correlations between signals
- Cross-spectral features
- Multi-sensor fusion

### 3. Handle Missing Data Appropriately

```yaml
# Configure importers to handle missing values
config:
  missing_value_strategy: "interpolate"  # or "drop", "ffill"
  interpolation_method: "linear"
  max_gap_seconds: 60  # Don't interpolate gaps > 1 minute
```

**Strategies:**
- **Interpolate**: For brief gaps (<1 min)
- **Drop**: For long gaps or noisy sections
- **Flag**: Mark epochs with missing data for exclusion

### 4. Validate Feature Ranges

Check that extracted features make physiological sense:

```python
import pandas as pd

features = pd.read_csv('results/hrv_features.csv')

# Check HR is in valid range (40-200 bpm for sleep)
assert (features['hr_mean'] >= 40).all(), "HR below 40 bpm detected"
assert (features['hr_mean'] <= 200).all(), "HR above 200 bpm detected"

# Check for NaN values
assert not features.isnull().any().any(), "NaN values in features"
```

---

## Quality Control

### 1. Always Visualize Raw Data First

```yaml
visualization:
  # Before processing: Check for artifacts
  - type: time_series
    signals: ["hr_merged_0"]
    title: "Raw Heart Rate (Pre-Processing)"
    output: "results/qc/raw_hr.html"
```

**Look for:**
- Sensor disconnections (flat lines or gaps)
- Unrealistic values (HR > 200 bpm, HR < 30 bpm)
- Sudden jumps (artifacts)
- Periodicity (normal for HR, abnormal for others)

### 2. Plot Features Alongside Raw Signals

```yaml
visualization:
  # Raw signal on top, features below
  - type: time_series
    signals: ["hr_merged_0", "hrv_features"]
    layout: vertical
    title: "HR and Extracted HRV Features"
    output: "results/qc/hr_with_features.html"
    parameters:
      link_x_axes: true  # Synchronized zooming
```

### 3. Use Summary Statistics

```yaml
steps:
  - type: collection
    operation: "summarize_signals"
    parameters:
      print_summary: true
```

**Example output:**
```
Signal Summary:
  hr_merged_0: 28800 samples, 8.00 hours, 1.000Hz, 40.0-180.0 bpm
  hrv_features: 960 epochs, mean hr_mean=65.2, std hr_std=2.1
```

### 4. Check for Gaps in Time Series

```python
import pandas as pd

# Load signal
hr = pd.read_csv('results/raw_signals/hr_merged_0.csv', index_col=0, parse_dates=True)

# Check for gaps > 5 seconds
time_diffs = hr.index.to_series().diff()
large_gaps = time_diffs[time_diffs > pd.Timedelta('5s')]

if not large_gaps.empty:
    print(f"Warning: {len(large_gaps)} gaps > 5s detected")
    print(large_gaps)
```

---

## Performance Optimization

### 1. Process Multiple Subjects in Parallel

```bash
# Create a script to run workflows in parallel
# process_batch.sh

for subject_dir in data/subject_*/; do
    subject=$(basename $subject_dir)
    python -m sleep_analysis.cli.run_workflow \
        --workflow workflows/complete_analysis.yaml \
        --data-dir "$subject_dir" \
        --output-prefix "$subject" &
done

wait  # Wait for all processes to complete
```

### 2. Use Efficient Data Formats

**For intermediate results:**
```yaml
export:
  - formats: ["pickle"]  # Fast, preserves dtypes
    content: ["all_features"]
```

**For final results:**
```yaml
export:
  - formats: ["csv", "excel"]  # Human-readable
    content: ["combined_features"]
```

### 3. Downsample High-Frequency Signals for Visualization

```yaml
visualization:
  - type: time_series
    signals: ["hr_merged_0"]
    parameters:
      max_points: 10000  # Downsample to 10k points for fast rendering
```

### 4. Clear Temporary Signals

```python
# In Python API
collection.time_series_signals['temp_signal'].metadata.temporary = True
collection.time_series_signals['temp_signal'].clear_data()
```

Temporary signals are excluded from exports and can be regenerated if needed.

---

## Reproducibility

### 1. Version Control Your Workflows

```bash
git init
git add workflows/*.yaml
git commit -m "Initial workflow for HRV analysis"
```

### 2. Document Software Versions

Create a `requirements.txt`:

```txt
numpy==1.24.0
pandas==2.0.0
scipy==1.10.0
scikit-learn==1.2.0
```

Or use a `environment.yml` for conda:

```yaml
name: sleep-analysis
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy=1.24
  - pandas=2.0
  - scipy=1.10
```

### 3. Include Metadata in Exports

The framework automatically includes metadata in exports:

```yaml
export:
  - formats: ["excel"]  # Metadata sheet included
    content: ["combined_features"]
```

**Excel output:**
- Sheet 1: `combined_features` (data)
- Sheet 2: `metadata` (operations applied, versions, timestamps)

### 4. Use Consistent Random Seeds

If using randomized operations:

```python
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
```

---

## Common Pitfalls

### ❌ Pitfall 1: Ignoring Time Zones

**Problem:**
```yaml
# No timezone specified
import:
  - signal_type: "heart_rate"
    source: "data/hr.csv"
```

**Solution:**
```yaml
# Specify timezone
collection_settings:
  timezone: "America/New_York"  # or "UTC", "Europe/London", etc.
```

---

### ❌ Pitfall 2: Mismatched Epoch Grids

**Problem:**
```yaml
# Different epoch lengths for different features
- operation: "compute_hrv_features"
  parameters:
    window_length: "30s"  # ← Different

- operation: "compute_movement_features"
  parameters:
    window_length: "60s"  # ← Different
```

**Solution:**
```yaml
# Use global epoch grid
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

steps:
  - type: collection
    operation: "generate_epoch_grid"  # ← Generate once

  # Both operations use the same grid
  - operation: "compute_hrv_features"
    inputs: ["hr"]
    output: "hrv"

  - operation: "compute_movement_features"
    inputs: ["accel"]
    output: "movement"
```

---

### ❌ Pitfall 3: Not Validating Sensor Alignment

**Problem:**
```python
# Assuming sensors are aligned, but they may have different start times
features_hr = pd.read_csv('hrv_features.csv')
features_accel = pd.read_csv('movement_features.csv')
combined = pd.concat([features_hr, features_accel], axis=1)  # ← May misalign!
```

**Solution:**
```yaml
# Use combine_features operation (handles alignment)
steps:
  - type: collection
    operation: "combine_features"
    inputs: ["hrv", "movement"]
```

---

### ❌ Pitfall 4: Over-Smoothing Features

**Problem:**
```yaml
# Very long window smooths out sleep stage transitions
epoch_grid_config:
  window_length: "5m"  # ← Too long for sleep staging
```

**Solution:**
```yaml
# Use standard 30s for sleep staging
epoch_grid_config:
  window_length: "30s"
```

---

### ❌ Pitfall 5: Forgetting to Export Summaries

**Problem:**
```yaml
# Only export features, lose signal-level metadata
export:
  - content: ["combined_features"]
```

**Solution:**
```yaml
# Also export summary
export:
  - content: ["combined_features", "summary"]
```

The summary includes: signal names, sample rates, durations, sensor info.

---

## Checklist for Production Workflows

Before deploying your analysis pipeline, verify:

- [ ] **Data Quality**
  - [ ] Visualized all raw signals
  - [ ] Checked for missing data and gaps
  - [ ] Validated feature ranges (no NaN, no outliers)

- [ ] **Workflow Design**
  - [ ] Documented rationale for epoch length
  - [ ] Used descriptive output names
  - [ ] Added comments explaining non-obvious choices

- [ ] **Reproducibility**
  - [ ] Version-controlled workflows
  - [ ] Documented software versions
  - [ ] Set random seeds if applicable

- [ ] **Performance**
  - [ ] Used efficient formats for intermediate data
  - [ ] Downsampled for visualization
  - [ ] Parallelized batch processing

- [ ] **Quality Control**
  - [ ] Exported summaries
  - [ ] Validated feature statistics
  - [ ] Spot-checked results for multiple subjects

---

## Additional Resources

- **[Feature Extraction Guide](feature-extraction-guide.md)** - Detailed feature documentation
- **[Common Workflows](common-workflows.md)** - Ready-to-use examples
- **[Troubleshooting](../troubleshooting.md)** - Common issues and fixes
- **[Python API Guide](python-api-guide.md)** - Programmatic usage

**Happy analyzing!** 🎯
