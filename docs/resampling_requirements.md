# Signal Alignment and Resampling

## Overview

When data is imported from various sources with different timestamps and sampling rates, combining these signals into a common dataframe creates challenges:
- Different timestamps for each signal result in interleaved rows
- Excessive NaN values where signals don't share the same timestamps
- Clock precision mismatches (e.g., millisecond precision with 1 Hz, 50 Hz, or 100 Hz sampling rates)

The framework implements an **alignment grid** system to create a common temporal index shared by all signals, eliminating unnecessary missing data and enabling efficient signal combination.

## Current Implementation

### Alignment Strategy

The framework uses a two-phase approach:

1. **Generate Alignment Grid** (`generate_alignment_grid`)
   - Calculates optimal grid parameters based on all time-series signals
   - Determines target sample rate, reference time, and grid index
   - Stores parameters in the SignalCollection

2. **Apply Grid Alignment** (`apply_grid_alignment`)
   - Reindexes time-series signals to the common grid
   - Uses configurable alignment methods (nearest, linear, etc.)
   - Performs in-place modification of signal data

### Target Sample Rate Selection

The target sample rate can be determined in two ways:

1. **Auto-detection** (default): Uses the largest standard rate ≤ maximum signal rate
   - Standard rates: [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000] Hz
   - Example: If max signal rate is 75 Hz, target rate becomes 50 Hz

2. **Manual specification**: User provides explicit `target_sample_rate` parameter
   - Signals faster than target rate are downsampled
   - Signals slower than target rate remain unchanged (with NaNs at missing grid points)

### Grid Generation Process

1. **Find time range**: Determine the union of all signal start/end times
2. **Calculate reference time**: Align to epoch (1970-01-01) and target period for stability
3. **Generate grid index**: Create `pd.DatetimeIndex` with calculated frequency
4. **Set merge tolerance**: `merge_tolerance = 0.4 * period` for combining aligned signals

### Alignment Methods

When applying grid alignment, signals are reindexed using one of these methods:

- **`nearest`** (default): Use nearest timestamp (good for discrete sensors)
- **`linear`**: Linear interpolation between points (good for continuous signals)
- **`ffill`**: Forward fill (carry last value forward)
- **`bfill`**: Backward fill (use next value)

## Usage Examples

### Basic Workflow

```yaml
collection_settings:
  index_config: ["signal_type", "sensor_model"]

steps:
  # Step 1: Generate alignment grid (auto-detect rate)
  - type: collection
    operation: "generate_alignment_grid"
    parameters: {}

  # Step 2: Apply alignment to all signals
  - type: collection
    operation: "apply_grid_alignment"
    parameters:
      method: "nearest"

  # Step 3: Combine aligned signals
  - type: collection
    operation: "align_and_combine_signals"
    parameters: {}
```

### Manual Target Rate

```yaml
steps:
  # Specify target rate of 10 Hz
  - type: collection
    operation: "generate_alignment_grid"
    parameters:
      target_sample_rate: 10.0

  - type: collection
    operation: "apply_grid_alignment"
    parameters:
      method: "linear"  # Use interpolation for smooth signals
```

### Python API

```python
from sleep_analysis.core.signal_collection import SignalCollection

collection = SignalCollection()
# ... import signals ...

# Generate alignment grid (auto-detect)
collection.generate_alignment_grid()
print(f"Target rate: {collection.target_rate} Hz")
print(f"Grid size: {len(collection.grid_index)}")

# Apply alignment to all time-series signals
collection.apply_grid_alignment(method='nearest')

# Combine aligned signals
combined_df = collection.align_and_combine_signals()
print(f"Combined shape: {combined_df.shape}")
```

## Behavior Details

### Signals Slower Than Target Rate

- **Not downsampled**: Original data preserved
- **NaN values**: Grid points without data are filled with NaN
- Example: 1 Hz heart rate signal aligned to 10 Hz grid has 9 NaNs between each data point

### Signals Faster Than Target Rate

- **Downsampled**: Resampled to target rate using specified method
- **No NaN values**: All grid points have values (from downsampling)
- Example: 100 Hz accelerometer downsampled to 50 Hz grid

### Edge Cases

1. **Empty signals**: Skipped during alignment (warning logged)
2. **Single-point signals**: Aligned but may have limited usefulness
3. **Non-overlapping time ranges**: Combined DataFrame spans full range with NaNs where signals don't overlap

## Design Rationale

### Why This Approach?

1. **Efficiency**: Single common grid eliminates redundant timestamps
2. **Flexibility**: Supports signals with different sampling rates
3. **Precision**: Aligns to epoch for numerical stability
4. **Standard rates**: Uses widely-accepted sampling frequencies
5. **Explicit control**: Users can override auto-detection when needed

### Alternative Approaches Considered

1. **Per-signal grids**: More flexible but complicates combination logic
2. **Highest rate always**: Would upsample slow signals (wasteful, introduces artificial precision)
3. **Asynchronous merging**: Keeps original timestamps but creates sparse, inefficient DataFrames

The current implementation balances these tradeoffs by:
- Using standard rates (predictable, familiar)
- Not upsampling slower signals (avoids artificial precision)
- Providing explicit control (manual target rate when needed)
- Supporting various alignment methods (nearest, linear, etc.)

## Troubleshooting

### Issue: Combined DataFrame is very large

**Cause**: Target rate is too high for your data needs.

**Solution**: Specify a lower target rate:
```yaml
- operation: "generate_alignment_grid"
  parameters:
    target_sample_rate: 1.0  # Use 1 Hz instead of auto-detected rate
```

### Issue: Too many NaN values after alignment

**Cause**: Signals have large gaps or non-overlapping time ranges.

**Solution**:
1. Check signal time ranges: `collection.summarize_signals(print_summary=True)`
2. Filter out signals with poor coverage
3. Use `dropna()` in post-processing if appropriate

### Issue: Alignment changes signal characteristics

**Cause**: Downsampling or interpolation method doesn't match signal type.

**Solution**: Choose appropriate method:
- Discrete/categorical: Use `method='nearest'` (no interpolation)
- Continuous/smooth: Use `method='linear'` (interpolates)
- Step functions: Use `method='ffill'` or `method='bfill'`

## See Also

- [Data Preparation Guide](data-preparation.md) - Preparing data for alignment
- [Quick Start Guide](quick-start.md) - Basic workflow examples
- [Troubleshooting Guide](troubleshooting.md) - Common alignment issues

---

**Last Updated**: 2025-11-18
