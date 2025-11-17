# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the Sleep Analysis Framework.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Import and Data Loading Errors](#import-and-data-loading-errors)
3. [Workflow Execution Errors](#workflow-execution-errors)
4. [Feature Extraction Issues](#feature-extraction-issues)
5. [Export Problems](#export-problems)
6. [Visualization Errors](#visualization-errors)
7. [Debug Logging](#debug-logging)
8. [FAQ](#faq)

---

## Installation Issues

### Error: `No module named 'sleep_analysis'`

**Problem**: Python cannot find the installed package.

**Solutions**:

1. Ensure you installed with the `-e` flag for development mode:
   ```bash
   pip install -e .
   ```

2. Verify your virtual environment is activated:
   ```bash
   # Check which Python is being used
   which python
   # Should point to your venv directory
   ```

3. Reinstall the package:
   ```bash
   pip uninstall sleep_analysis
   pip install -e .
   ```

### Error: `ModuleNotFoundError: No module named 'sklearn'`

**Problem**: Optional dependencies for algorithms are not installed.

**Solution**: Install the algorithms dependency group:
```bash
pip install -e ".[algorithms]"
# Or install individually
pip install scikit-learn scipy joblib matplotlib seaborn
```

### Error: `ModuleNotFoundError: No module named 'bokeh'` or `'plotly'`

**Problem**: Visualization dependencies are not installed.

**Solution**: Install visualization dependencies:
```bash
pip install -e ".[vis]"
# Or install individually
pip install bokeh plotly
```

### Test Failures After Installation

**Problem**: Some tests fail, especially HDF5 export tests.

**Solution**: Install optional export dependencies:
```bash
pip install -e ".[export]"
# Or specifically
pip install h5py tables
```

**Note**: The test `test_export_hdf5` will be skipped if `tables` (pytables) is not installed. This is expected and not an error.

---

## Import and Data Loading Errors

### Error: `FileNotFoundError: Workflow file not found`

**Problem**: The workflow YAML file path is incorrect.

**Solution**:
```bash
# Use absolute path
sleep-analysis -w /full/path/to/workflow.yaml -d data

# Or relative path from current directory
sleep-analysis -w workflows/my_workflow.yaml -d data

# Check file exists
ls -la workflows/my_workflow.yaml
```

### Error: `Data directory not found`

**Problem**: The `--data-dir` path doesn't exist or is incorrect.

**Solution**:
```bash
# Create the directory
mkdir -p data

# Or use correct path
sleep-analysis -w workflow.yaml -d /path/to/your/data

# Verify directory exists
ls -la data/
```

### Warning: `No files found matching pattern`

**Problem**: The file pattern in your workflow doesn't match any files.

**Solution**:

1. Check your file pattern matches your actual files:
   ```bash
   # List files to see actual names
   ls data/Polar*.txt
   ```

2. Update your workflow's `file_pattern`:
   ```yaml
   import:
     - signal_type: "heart_rate"
       importer: "MergingImporter"
       source: "."
       config:
         file_pattern: "Polar_H10_*_HR.txt"  # Must match actual filenames
   ```

3. Make sure the `source` path is correct:
   - Use `"."` if files are directly in `data-dir`
   - Use `"subfolder"` if files are in `data-dir/subfolder`

### Error: `Failed to parse timestamp`

**Problem**: Timestamp format in your data files doesn't match expected format.

**Solution**:

1. Specify the correct time column:
   ```yaml
   config:
     time_column: "Phone timestamp"  # Match your file's column name
   ```

2. Add timezone information if timestamps are naive:
   ```yaml
   default_input_timezone: "America/New_York"  # Top-level setting
   # Or per-importer
   config:
     origin_timezone: "America/New_York"
   ```

3. Check your CSV delimiter matches:
   ```yaml
   config:
     delimiter: ";"  # Use "," for comma-separated, "\t" for tab
   ```

### Error: `Missing required column(s)`

**Problem**: The imported file is missing expected columns (e.g., 'hr' for heart rate).

**Solution**:

1. Check your file's column names:
   ```bash
   head -n 1 data/Polar_H10_*_HR.txt
   ```

2. Add column mapping in your workflow:
   ```yaml
   config:
     column_mapping:
       "Heart Rate": "hr"  # Map file column to expected name
   ```

3. For accelerometer data, ensure x, y, z columns exist or map them:
   ```yaml
   config:
     column_mapping:
       "Acc_X": "x"
       "Acc_Y": "y"
       "Acc_Z": "z"
   ```

---

## Workflow Execution Errors

### Error: `Step missing required 'operation' field`

**Problem**: A step in your workflow YAML is missing the `operation` key.

**Solution**: Ensure every step has an `operation`:
```yaml
steps:
  - type: collection
    operation: "generate_epoch_grid"  # Required
    parameters: {}
```

### Error: `Invalid step type 'xxx'`

**Problem**: The `type` field contains an invalid value.

**Valid types**: `signal`, `multi_signal`, `collection`

**Solution**:
```yaml
steps:
  # For signal operations
  - type: signal
    operation: "filter_lowpass"
    input: "hr_0"

  # For feature extraction (multiple signals)
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]

  # For collection-level operations
  - type: collection
    operation: "generate_epoch_grid"
```

### Error: `Step cannot have both 'input' and 'inputs' fields`

**Problem**: A step specifies both singular and plural input fields.

**Solution**: Use only one:
```yaml
# For single signal operations
- type: signal
  operation: "filter_lowpass"
  input: "hr_0"  # Singular

# For multi-signal operations
- type: multi_signal
  operation: "compute_hrv_features"
  inputs: ["hr"]  # Plural (list)
```

### Error: `'parameters' must be a dictionary`

**Problem**: The `parameters` field is not a dict/object.

**Solution**:
```yaml
# Correct
parameters:
  hrv_metrics: ["hr_mean", "hr_std"]
  use_rr_intervals: false

# Incorrect
parameters: "some_string"  # ❌
```

### Error: `Signal 'xxx' not found in collection`

**Problem**: You're referencing a signal that doesn't exist.

**Solution**:

1. Check the signal was imported successfully (look at console output)

2. Use correct signal names:
   - Base names: `"hr"` matches `hr_0`, `hr_1`, etc.
   - Specific names: `"hr_0"` matches exactly
   - After merging: Names may be `"hr_merged_0"`

3. Enable debug logging to see available signals:
   ```bash
   sleep-analysis -w workflow.yaml -d data -v
   ```

---

## Feature Extraction Issues

### Error: `Epoch grid not generated`

**Problem**: Trying to extract features before creating the epoch grid.

**Solution**: Add `generate_epoch_grid` as the first step:
```yaml
steps:
  # Must be first!
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {}

  # Then extract features
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]
    output: "hrv_features"
```

### Error: `epoch_grid_config not set`

**Problem**: Missing epoch configuration in collection settings.

**Solution**: Add to top of workflow:
```yaml
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"
```

### Warning: `Insufficient data for HRV calculation`

**Problem**: Not enough data points in an epoch to calculate HRV metrics.

**This is expected behavior** when:
- Epochs have missing data
- Signal sampling rate is very low
- Data has gaps

**Solution**:
- The framework handles this gracefully by returning NaN
- You can filter out NaN rows in post-processing
- Consider using longer window lengths if needed

### Empty Feature DataFrames

**Problem**: Features are calculated but contain no data.

**Possible Causes**:

1. **Signal has no overlap with epoch grid**:
   - Check your signal's time range
   - Verify epoch grid covers signal timestamps

2. **All epochs filtered out**:
   - Enable debug logging to see details
   ```bash
   sleep-analysis -w workflow.yaml -d data -v
   ```

3. **Incorrect input signal names**:
   - Verify signal names in debug output
   - Use correct base names or specific keys

---

## Export Problems

### Error: `Cannot export 'combined_features': not created`

**Problem**: Trying to export combined features before creating them.

**Solution**: Add `combine_features` step before export:
```yaml
steps:
  # ... feature extraction steps ...

  # Combine before exporting
  - type: collection
    operation: "combine_features"
    inputs: ["hrv_features", "movement_features"]
    parameters: {}

export:
  - formats: ["csv"]
    content: ["combined_features"]  # Now available
```

### Excel Export Issues

**Problem**: Excel files are corrupted or can't be opened.

**Solutions**:

1. Install/upgrade openpyxl:
   ```bash
   pip install --upgrade openpyxl
   ```

2. Try CSV export as alternative:
   ```yaml
   export:
     - formats: ["csv"]  # Instead of "excel"
       output_dir: "results"
   ```

3. Check for special characters in column names

### HDF5 Export Fails

**Problem**: HDF5 export gives errors or is not supported.

**Solution**: Install HDF5 dependencies:
```bash
pip install h5py tables
```

If still failing, use CSV or Excel instead.

---

## Visualization Errors

### Error: `No module named 'bokeh'` during visualization

**Problem**: Visualization backend not installed.

**Solution**:
```bash
# Install both backends
pip install bokeh plotly

# Or just the one you need
pip install plotly  # Recommended
```

### Error: `Signal 'xxx' not found for visualization`

**Problem**: Signal referenced in visualization doesn't exist.

**Solution**:

1. Check signal names after import (they may have suffixes like `_merged_0`)

2. Update visualization section:
   ```yaml
   visualization:
     - type: time_series
       signals: ["hr_merged_0"]  # Use actual signal names
   ```

3. Use `strict: false` to skip missing signals:
   ```yaml
   visualization:
     - type: time_series
       signals: ["hr_0", "hr_1", "hr_2"]
       parameters:
         strict: false  # Skip any missing signals
   ```

### Plots Are Slow or Hang

**Problem**: Too many data points causing performance issues.

**Solution**: Add downsampling:
```yaml
visualization:
  - type: time_series
    signals: ["hr_0"]
    parameters:
      max_points: 10000  # Downsample to 10k points
      # Or use time-based downsampling
      downsample: "1S"  # 1-second intervals
```

---

## Debug Logging

### Enable Verbose Logging

Add `-v` flag or set log level:

```bash
# Method 1: Use -v flag
sleep-analysis -w workflow.yaml -d data -v

# Method 2: Set explicit log level
sleep-analysis -w workflow.yaml -d data --log-level DEBUG

# Available levels: DEBUG, INFO, WARN, ERROR
```

### What to Look For in Debug Output

1. **Import section**: Verify files are found and loaded
   ```
   INFO: Found 3 files matching pattern 'Polar_H10_*_HR.txt'
   INFO: Loaded signal 'hr_merged_0' with 86400 samples
   ```

2. **Signal names**: Note the actual names for use in steps
   ```
   DEBUG: Added signal to collection: hr_merged_0
   ```

3. **Epoch grid**: Check grid parameters
   ```
   INFO: Generated epoch grid: 2880 epochs, window=30s, step=30s
   ```

4. **Feature extraction**: See if features are created
   ```
   INFO: Computed HRV features: 2880 epochs, 4 features
   ```

5. **Export**: Verify file creation
   ```
   INFO: Exported combined_features to results/features/combined_features.csv
   ```

### Python Logging in Code

If using the Python API, enable logging:

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or for specific module
logger = logging.getLogger('sleep_analysis')
logger.setLevel(logging.DEBUG)
```

---

## FAQ

### Q: How do I know if my workflow succeeded?

**A**: Check for these indicators:
- No ERROR messages in output
- Expected files created in output directories
- Final message: "Workflow execution completed"
- Return code 0 (check with `echo $?` on Linux/Mac)

### Q: What's the difference between `input` and `inputs`?

**A**:
- `input`: Single signal (string) - for signal operations
- `inputs`: Multiple signals (list) - for multi-signal operations
- Never use both in the same step

### Q: Can I use Windows paths in workflows?

**A**: Yes, but use forward slashes or escaped backslashes:
```yaml
# Good
source: "C:/Users/Me/data"
source: "data/subfolder"

# Avoid
source: "C:\Users\Me\data"  # May cause issues
```

### Q: Why are my features empty or all NaN?

**A**: Common causes:
1. Epoch grid doesn't overlap signal time range
2. Window length is too long for your data
3. Signal has too many gaps
4. Wrong signal input names

**Solution**: Enable debug logging and check overlap.

### Q: Can I run multiple workflows in parallel?

**A**: Yes, but use different output directories:
```bash
# Terminal 1
sleep-analysis -w workflow1.yaml -d data -o results1

# Terminal 2
sleep-analysis -w workflow2.yaml -d data -o results2
```

### Q: How do I process multiple subjects?

**A**: Options:
1. Create separate workflows for each subject
2. Use different data directories
3. Script it:
   ```bash
   for subject in subject1 subject2 subject3; do
     sleep-analysis -w workflow.yaml -d data/$subject -o results/$subject
   done
   ```

### Q: What if my sensor uses different file formats?

**A**: Create a custom importer:
1. See `src/sleep_analysis/importers/sensors/` for examples
2. Inherit from `CSVImporterBase` or `SignalImporter`
3. Implement file parsing logic
4. Register your importer

### Q: Can I modify features after extraction?

**A**: Yes, in Python:
```python
# Access feature object
hrv_features = collection.get_feature("hrv_features_0")

# Get DataFrame
df = hrv_features.data

# Modify
df['new_column'] = df['hr_mean'] * 2

# Use in further processing
```

### Q: How do I report a bug?

**A**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for bug report guidelines. Include:
- Full error message
- Minimal workflow YAML to reproduce
- Python version and installed packages (`pip list`)
- Debug log output (`-v` flag)

---

## Still Having Issues?

If you couldn't find your issue here:

1. **Check the examples**: Look in `workflows/` directory for working examples
2. **Review documentation**: See [README.md](../README.md) for detailed docs
3. **Run tests**: `pytest tests/ -v` to verify installation
4. **Enable debug logging**: Always use `-v` when troubleshooting
5. **Search for error messages**: Error messages often contain hints

## Common Error Message Patterns

| Error Pattern | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| `No module named...` | Missing dependency | `pip install <module>` |
| `not found in collection` | Signal doesn't exist | Check signal names with `-v` |
| `must be a dictionary` | YAML syntax error | Check indentation and colons |
| `Missing required field` | Incomplete workflow | Add required fields |
| `Cannot convert` | Type mismatch | Check parameter types |
| `File not found` | Wrong path | Use absolute path or verify |
| `Empty DataFrame` | No data overlap | Check time ranges |

---

**Last Updated**: 2025-11-17
