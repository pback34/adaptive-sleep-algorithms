# Data Preparation Guide

This guide explains how to prepare your sensor data for use with the Sleep Analysis Framework, with a focus on Polar device data and best practices for data quality.

## Table of Contents

1. [Overview](#overview)
2. [Polar File Format Requirements](#polar-file-format-requirements)
3. [Timezone Handling](#timezone-handling)
4. [File Organization](#file-organization)
5. [Data Quality Considerations](#data-quality-considerations)
6. [Custom Data Formats](#custom-data-formats)
7. [Troubleshooting Data Issues](#troubleshooting-data-issues)

---

## Overview

The Sleep Analysis Framework processes time-series physiological data from wearable sensors. The most common use case is analyzing Polar device data (H10 chest strap, Sense wrist device), but the framework supports custom formats through extensible importers.

### Supported Signal Types

- **Heart Rate (HR)**: BPM measurements from ECG or optical sensors
- **Heart Rate Variability (HRV)**: RR interval data (milliseconds)
- **Accelerometer**: 3-axis acceleration data (x, y, z)
- **Sleep Stages**: Categorical sleep stage labels (Wake, Light, Deep, REM)
- **Other**: PPG, magnitude, angle, and custom signal types

---

## Polar File Format Requirements

### Standard Polar Export Format

Polar devices export data as CSV files with semicolon or comma delimiters. The framework expects specific column names and formats.

### Heart Rate Files

**Typical filename**: `Polar_H10_12345678_HR.txt` or `Polar_Sense_12345678_HR.txt`

**Required columns**:
- `Phone timestamp`: Timestamp in format `YYYY-MM-DD HH:MM:SS` or ISO 8601
- `HR [bpm]`: Heart rate in beats per minute (integer or float)
- `HRV [ms]` (optional): Heart rate variability in milliseconds

**Example file content**:
```csv
Phone timestamp;HR [bpm];HRV [ms]
2024-01-15 22:30:00;65;850
2024-01-15 22:30:01;64;870
2024-01-15 22:30:02;66;
2024-01-15 22:30:03;65;860
```

**Notes**:
- Missing HRV values are common and expected (empty fields or NaN)
- Delimiter is typically `;` (semicolon) for Polar files
- Framework automatically handles missing HRV data

### Accelerometer Files

**Typical filename**: `Polar_H10_12345678_ACC.txt` or `Polar_Sense_12345678_ACC.txt`

**Required columns**:
- `Phone timestamp`: Timestamp in format `YYYY-MM-DD HH:MM:SS`
- `Accelerometer X [mg]`: X-axis acceleration in milligravity
- `Accelerometer Y [mg]`: Y-axis acceleration
- `Accelerometer Z [mg]`: Z-axis acceleration

**Example file content**:
```csv
Phone timestamp;Accelerometer X [mg];Accelerometer Y [mg];Accelerometer Z [mg]
2024-01-15 22:30:00.000;-125;850;-50
2024-01-15 22:30:00.040;-130;845;-52
2024-01-15 22:30:00.080;-128;848;-51
```

**Notes**:
- Higher sampling rate than HR (typically 25-50 Hz)
- More frequent timestamps (subsecond precision)
- Values can be negative

### Column Mapping

If your Polar files use different column names, map them in your workflow:

```yaml
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      delimiter: ";"
      column_mapping:
        "timestamp": "Phone timestamp"  # Map to standard name
        "hr": "HR [bpm]"
        "hrv": "HRV [ms]"
```

### Common Column Name Variations

If your files have different headers, use these mappings:

```yaml
# Variation 1: Simple headers
column_mapping:
  "timestamp": "Time"
  "hr": "Heart_Rate"

# Variation 2: No units in brackets
column_mapping:
  "timestamp": "Timestamp"
  "hr": "HR"
  "hrv": "HRV"

# Variation 3: Different accelerometer names
column_mapping:
  "timestamp": "Time"
  "x": "Acc_X"
  "y": "Acc_Y"
  "z": "Acc_Z"
```

---

## Timezone Handling

Proper timezone handling is crucial for aligning data from multiple sensors and ensuring correct temporal analysis.

### Understanding Timezone Parameters

The framework uses three timezone concepts:

1. **`origin_timezone`**: The timezone where data was recorded (local time)
2. **`default_input_timezone`**: Fallback for naive timestamps
3. **`target_timezone`**: The timezone for all processed data (typically UTC)

### Best Practices

#### 1. Always Specify Origin Timezone

**Recommendation**: Explicitly set the timezone where data was recorded.

```yaml
# Top-level: applies to all imports unless overridden
default_input_timezone: "America/New_York"

# Per-importer: for specific data sources
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      origin_timezone: "Europe/London"  # Override for this import
      file_pattern: "Polar_H10_*_HR.txt"
```

#### 2. Use Standard Timezone Names

Use IANA timezone database names (e.g., "America/New_York", not "EST").

**Valid examples**:
- `"UTC"` - Coordinated Universal Time
- `"America/New_York"` - Eastern Time (handles DST automatically)
- `"Europe/London"` - UK time
- `"Asia/Tokyo"` - Japan time
- `"system"` - Use local machine timezone

**Avoid**:
- `"EST"`, `"PST"` - Ambiguous (no DST handling)
- `"GMT+5"` - Not standard IANA format

#### 3. Convert to UTC for Processing

**Recommendation**: Use UTC as `target_timezone` for consistent processing.

```yaml
# Recommended: Convert all signals to UTC
default_input_timezone: "America/New_York"  # Where data was recorded
target_timezone: "UTC"  # Standard for processing

# Alternative: Keep in local time
target_timezone: "America/New_York"

# Or: Use machine's local time
target_timezone: "system"
```

### Timezone Scenarios

#### Scenario 1: Single Location, Single Session

Simplest case - all data from one location at one time:

```yaml
default_input_timezone: "America/Denver"
target_timezone: "UTC"

import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
```

#### Scenario 2: Multiple Sensors, Same Location

Data from chest and wrist sensors in the same location:

```yaml
default_input_timezone: "Europe/Paris"
target_timezone: "UTC"

import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
    base_name: "hr_chest"

  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_Sense_*_HR.txt"
    base_name: "hr_wrist"
```

#### Scenario 3: Multi-Day Recording with DST Transition

If your recording spans a DST transition, the framework handles it automatically:

```yaml
# March 2024: DST transition on March 10
default_input_timezone: "America/New_York"  # Handles DST automatically
target_timezone: "UTC"  # No DST issues
```

#### Scenario 4: Timezone-Aware Timestamps

If your CSV files already include timezone info (ISO 8601 with offset):

```csv
timestamp;HR [bpm]
2024-01-15T22:30:00-05:00;65
2024-01-15T22:30:01-05:00;64
```

The framework uses the embedded timezone and converts to `target_timezone`. No `origin_timezone` needed.

---

## File Organization

### Recommended Structure

Organize files by subject, session, and sensor type:

```
data/
├── subject_001/
│   ├── session_2024-01-15/
│   │   ├── Polar_H10_12345678_HR.txt
│   │   ├── Polar_H10_12345678_ACC.txt
│   │   ├── Polar_Sense_87654321_HR.txt
│   │   └── Polar_Sense_87654321_ACC.txt
│   └── session_2024-01-16/
│       ├── Polar_H10_12345678_HR.txt
│       └── ...
├── subject_002/
│   └── session_2024-01-15/
│       └── ...
└── README.txt  # Document collection details
```

### Workflow Configuration for Organized Data

```yaml
# Process subject_001, session_2024-01-15
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "subject_001/session_2024-01-15"  # Relative to data_dir
    config:
      file_pattern: "Polar_H10_*_HR.txt"
    base_name: "hr"
```

### Fragmented Data Files

If data is split across multiple files (e.g., hourly exports), use `MergingImporter`:

```
data/
├── Polar_H10_12345678_HR_part1.txt
├── Polar_H10_12345678_HR_part2.txt
├── Polar_H10_12345678_HR_part3.txt
└── ...
```

```yaml
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR_part*.txt"  # Matches all parts
      sort_by: "timestamp"  # Ensures chronological order
    base_name: "hr"
```

The framework automatically:
1. Finds all matching files
2. Loads each file
3. Sorts by timestamp
4. Merges into a single signal
5. Names it `hr_merged_0`

---

## Data Quality Considerations

### 1. Sampling Rate Consistency

**Heart Rate**: Typically 1 Hz (1 sample per second)
- Most Polar devices: 1 Hz
- Some devices: Variable rate (framework handles this)

**Accelerometer**: Typically 25-50 Hz
- Polar H10: 25 Hz or 50 Hz
- Polar Sense: 25 Hz

**What to check**:
- Files should have relatively consistent time intervals
- Large gaps indicate sensor disconnection or battery issues

### 2. Missing Data

**Common causes**:
- Sensor not in contact with skin
- Low battery
- Bluetooth disconnection
- Device malfunction

**How the framework handles it**:
- Preserves timestamp gaps (doesn't interpolate by default)
- Feature extraction returns NaN for epochs with insufficient data
- You can filter NaN values in post-processing

**Example: Check for gaps**:
```python
# After import
import pandas as pd
signal = collection.get_signal("hr_merged_0")
df = signal.data

# Find gaps > 10 seconds
time_diffs = df.index.to_series().diff()
large_gaps = time_diffs[time_diffs > pd.Timedelta("10s")]
print(f"Found {len(large_gaps)} gaps > 10 seconds")
```

### 3. Outliers and Noise

**Heart Rate outliers**:
- Physically impossible values: < 30 or > 220 bpm
- Motion artifacts: Sudden spikes
- Poor sensor contact: Erratic readings

**Accelerometer noise**:
- Sensor saturation (values at limits)
- Electronic noise (high-frequency jitter)

**Recommendations**:
1. **Pre-processing**: Filter outliers before feature extraction
2. **Visual inspection**: Plot raw signals to identify issues
3. **Feature-level filtering**: Remove epochs with suspicious features

**Example: Remove HR outliers**:
```yaml
steps:
  # Add filtering step after import
  - type: signal
    operation: "filter_lowpass"
    input: "hr_merged_0"
    parameters:
      cutoff_frequency: 0.5  # Remove high-frequency noise
    output: "hr_filtered"
    inplace: true
```

### 4. Temporal Alignment

When combining multiple sensors, ensure they're time-aligned:

**Check alignment**:
- Compare start/end times of different signals
- Look for systematic offsets (e.g., one sensor starts 30 seconds later)

**Framework handles**:
- Automatic time grid alignment
- Resampling to common timestamps
- Outer join for combining signals (keeps all timestamps)

### 5. File Integrity

**Before processing, verify**:

```bash
# Check file sizes (empty files indicate problems)
ls -lh data/*.txt

# Check for CSV parsing issues
head -n 5 data/Polar_H10_*_HR.txt

# Count lines (should be consistent across related files)
wc -l data/Polar_H10_*_HR.txt
```

**Common issues**:
- Empty files (0 bytes)
- Truncated files (partial header or data)
- Encoding issues (non-UTF-8 characters)

---

## Custom Data Formats

### Creating a Custom Importer

If you have a non-Polar data source:

**Step 1: Create importer class**

```python
# my_importers/my_device.py
from sleep_analysis.importers.formats.csv import CSVImporterBase
import pandas as pd

class MyDeviceImporter(CSVImporterBase):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

    def _parse_csv(self, source: str) -> pd.DataFrame:
        # Custom parsing logic
        df = pd.read_csv(source, delimiter=self.config.get('delimiter', ','))

        # Your custom transformations
        # ...

        return df
```

**Step 2: Register importer**

```python
# In your workflow execution code
from my_importers.my_device import MyDeviceImporter
from sleep_analysis.importers import importer_registry

importer_registry["MyDeviceImporter"] = MyDeviceImporter
```

**Step 3: Use in workflow**

```yaml
import:
  - signal_type: "heart_rate"
    importer: "MyDeviceImporter"
    source: "data/my_device_data.csv"
    config:
      delimiter: ","
    base_name: "hr"
```

### Non-CSV Formats

For non-CSV formats (JSON, binary, databases):

1. Inherit from `SignalImporter` base class
2. Implement `import_signal()` method
3. Return `SignalData` object with standardized structure

See `src/sleep_analysis/importers/base.py` for the interface.

---

## Troubleshooting Data Issues

### Issue: "No files found matching pattern"

**Cause**: File pattern doesn't match your filenames.

**Solution**:
```bash
# Check actual filenames
ls data/

# Update pattern to match
file_pattern: "your_actual_pattern_*.txt"
```

### Issue: "Missing required column 'hr'"

**Cause**: Column names don't match expected names.

**Solution**: Add `column_mapping`:
```yaml
config:
  column_mapping:
    "hr": "Heart Rate (bpm)"  # Map your column to expected name
```

### Issue: "Failed to parse timestamp"

**Cause**: Timestamp format differs from expected.

**Solutions**:

1. Check actual format in your file:
   ```bash
   head -n 2 data/your_file.txt
   ```

2. Specify correct format:
   ```yaml
   config:
     time_format: "%m/%d/%Y %H:%M:%S"  # Match your format
   ```

3. Or let pandas auto-detect (default behavior).

### Issue: "Empty feature DataFrames"

**Cause**: No data overlap between signal time range and epoch grid.

**Solution**:

1. Check signal time range:
   ```python
   signal = collection.get_signal("hr_merged_0")
   print(f"Start: {signal.data.index.min()}")
   print(f"End: {signal.data.index.max()}")
   ```

2. Verify epoch grid:
   ```python
   print(f"Epoch grid: {collection.epoch_grid_index.min()} to {collection.epoch_grid_index.max()}")
   ```

3. Adjust epoch grid if needed:
   ```yaml
   steps:
     - type: collection
       operation: "generate_epoch_grid"
       parameters:
         start_time: "2024-01-15 22:00:00"  # Manual override
         end_time: "2024-01-16 08:00:00"
   ```

### Issue: "Too many NaN values in features"

**Cause**: Insufficient data points per epoch for calculations.

**Solutions**:

1. **Increase window length**:
   ```yaml
   collection_settings:
     epoch_grid_config:
       window_length: "60s"  # Longer windows capture more data
   ```

2. **Check for data gaps**: Use debug logging
   ```bash
   sleep-analysis -w workflow.yaml -d data -v
   ```

3. **Filter post-processing**:
   ```python
   features = collection.get_stored_combined_features()
   # Remove rows with any NaN
   clean_features = features.dropna()
   # Or remove rows with > 50% NaN
   clean_features = features.dropna(thresh=len(features.columns) * 0.5)
   ```

---

## Data Validation Checklist

Before running your workflow, verify:

- [ ] Files are in correct directory
- [ ] File naming pattern is consistent
- [ ] CSV delimiter matches configuration (`,` or `;`)
- [ ] Column names match expected or mapped correctly
- [ ] Timestamps are in consistent format
- [ ] Timezone is specified for naive timestamps
- [ ] Files are non-empty and complete
- [ ] No obvious outliers in data (spot check a few files)
- [ ] Related files (HR + ACC) have overlapping time ranges

---

## Example: Complete Data Setup

**Directory structure**:
```
my_sleep_study/
├── data/
│   ├── Polar_H10_ABCD1234_HR.txt
│   ├── Polar_H10_ABCD1234_ACC.txt
│   ├── Polar_Sense_WXYZ5678_HR.txt
│   └── Polar_Sense_WXYZ5678_ACC.txt
└── my_workflow.yaml
```

**Workflow**:
```yaml
# Timezone setup
default_input_timezone: "America/Los_Angeles"
target_timezone: "UTC"

# Collection settings
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

# Import all sensors
import:
  # Chest HR
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      delimiter: ";"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr_chest"

  # Chest Accelerometer
  - signal_type: "accelerometer"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_ACC.txt"
      delimiter: ";"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "accel_chest"

  # Wrist HR
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_Sense_*_HR.txt"
      delimiter: ";"
    sensor_model: "PolarSense"
    body_position: "wrist"
    base_name: "hr_wrist"

  # Wrist Accelerometer
  - signal_type: "accelerometer"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_Sense_*_ACC.txt"
      delimiter: ";"
    sensor_model: "PolarSense"
    body_position: "wrist"
    base_name: "accel_wrist"

# Process features
steps:
  - type: collection
    operation: "generate_epoch_grid"

  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr_chest"]
    output: "hrv_features"
    parameters:
      hrv_metrics: "all"

# Export
export:
  - formats: ["csv"]
    output_dir: "results"
    content: ["combined_features"]
```

**Run**:
```bash
cd my_sleep_study
sleep-analysis -w my_workflow.yaml -d data -o results
```

---

**Last Updated**: 2025-11-17

**Next Steps**: See [Quick Start Guide](quick-start.md) for workflow examples and [Troubleshooting Guide](troubleshooting.md) for common issues.
