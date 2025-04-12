# Feature Extraction: Requirements, Design, and Implementation Plan

## 1. Introduction

This document outlines the requirements, design, and implementation plan for the epoch-based feature extraction functionality within the Flexible Signal Processing Framework for Sleep Analysis. The goal is to enable the transformation of continuous derived signals (e.g., heart rate, respiratory rate) into structured datasets of features computed over specified time windows (epochs), facilitating tasks like sleep stage classification and event detection.

## 2. Requirements

### 2.1 Functional Requirements (FR-FE)

-   **FR-FE.1 Epoch Generation**: The framework must allow users to define epochs based on continuous time-series signals using configurable parameters:
    -   `window_length`: The duration of each epoch window (e.g., 30 seconds).
    -   `step_size`: The time interval between the start of consecutive epochs (e.g., 10 seconds). The `step_size` can be less than or equal to the `window_length`, allowing for overlapping epochs.
-   **FR-FE.2 Feature Computation**: The framework must support computing aggregation functions over the data within each epoch for one or more input signals. Supported functions should include, but not be limited to:
    -   Basic statistics: mean, standard deviation, median, minimum, maximum, variance.
    -   Signal-specific metrics: e.g., Heart Rate Variability (HRV) features for heart rate signals, spectral power bands for EEG signals.
-   **FR-FE.3 Single and Multi-Signal Features**: Support computation of features derived from:
    -   A single input signal (e.g., mean heart rate over epoch).
    -   Multiple input signals (e.g., correlation between heart rate and accelerometer magnitude over epoch).
-   **FR-FE.4 Input Signals**: Feature extraction operations must accept one or more derived `TimeSeriesSignal` instances from the `SignalCollection` as input.
-   **FR-FE.5 Output Signal**: Feature extraction operations must produce a new signal instance of type `FeatureSignal` (a subclass of `SignalData`).
    -   The `FeatureSignal`'s data must be a `pandas.DataFrame` where:
        -   The index represents the start timestamp of each epoch.
        -   Each column represents a computed feature (e.g., `hr_mean`, `accel_std_x`). Column names should clearly indicate the source signal and the feature computed.
-   **FR-FE.6 Metadata and Traceability**: The `FeatureSignal` metadata must include:
    -   Links to the source `SignalData` instance(s) (`derived_from`).
    -   The parameters used for epoch generation (`window_length`, `step_size`).
    -   The name(s) of the aggregation function(s) applied.
    -   A list of the feature names (column names) present in the data.
-   **FR-FE.7 Workflow Integration**: Feature extraction steps must be configurable within the workflow YAML files executed by the `WorkflowExecutor`. Operations should be registered appropriately (likely in `SignalCollection.multi_signal_registry`).
-   **FR-FE.8 Extensibility**: The framework must allow users to define and register their own custom feature extraction functions easily.

### 2.2 Non-Functional Requirements (NFR-FE)

-   **NFR-FE.1 Performance**: Feature extraction should be optimized for performance on large datasets, leveraging vectorized operations (e.g., via `pandas`, `numpy`) where possible.
-   **NFR-FE.2 Memory Efficiency**: Minimize memory footprint, especially when dealing with long signals and overlapping epochs.
-   **NFR-FE.3 Usability**: Provide an intuitive API for specifying feature extraction parameters both programmatically and within workflow configurations.
-   **NFR-FE.4 Maintainability**: The design should be modular to allow easy addition of new feature functions without modifying core framework components.

## 3. Design

### 3.1 Core Components

1.  **`FeatureSignal` Class**:
    -   A new subclass of `SignalData` located in `src/sleep_analysis/signals/feature_signal.py`.
    -   `signal_type` attribute set to `SignalType.FEATURES`.
    -   `data`: A `pandas.DataFrame` indexed by epoch start timestamps. Columns represent features.
    -   `metadata`: Extends `SignalMetadata` with feature-specific fields:
        -   `epoch_window_length`: `pd.Timedelta` representing the duration of each epoch.
        -   `epoch_step_size`: `pd.Timedelta` representing the step between epoch starts.
        -   `feature_names`: List of strings corresponding to the DataFrame column names.
        -   `source_signal_keys`: List of keys from the `SignalCollection` identifying the input signals.
    -   Does not require `sample_rate` or `units` in the same way as `TimeSeriesSignal`.
    -   Instances of `FeatureSignal` are stored within the main `SignalCollection` alongside other signal types, identified by their assigned keys.

2.  **Feature Computation Functions**:
    -   Located in a new module, e.g., `src/sleep_analysis/operations/feature_extraction.py`.
    -   Functions will accept `data_list: List[pd.DataFrame]` (containing data segments for one epoch from input signals) and `parameters: Dict[str, Any]`.
    -   They will return a dictionary or `pd.Series` containing the computed feature values for that single epoch.

3.  **Epoch Generation Logic**:
    -   Implemented likely within the `WorkflowExecutor` or a helper function called by registered feature operations.
    -   Takes input `TimeSeriesSignal`(s), `window_length`, and `step_size`.
    -   Iterates through time, creating epochs (defined by start/end timestamps).
    -   For each epoch, extracts the corresponding data segments from the input signal(s).

4.  **Feature Combination Operation**:
    -   A dedicated collection-level operation (e.g., `combine_features`) responsible for merging multiple `FeatureSignal` instances into a single DataFrame.
    -   Located likely in `src/sleep_analysis/operations/collection_operations.py` or directly as a method in `SignalCollection`.
    -   Takes a list of `FeatureSignal` keys as input.
    -   Verifies that all input `FeatureSignal`s share the exact same `DatetimeIndex` (epoch start times).
    -   Concatenates the DataFrames column-wise (`pd.concat(axis=1)`).
    -   Outputs a new `SignalData` instance (potentially another `FeatureSignal` or a dedicated `CombinedFeatureSignal`) containing the unified feature matrix.

5.  **Registration**:
    -   *Feature extraction* operations (creating individual `FeatureSignal`s) will be registered in `SignalCollection.multi_signal_registry`.
    -   The registration key should indicate it's a feature operation (e.g., `"feature_mean"`, `"feature_hrv"`).
    -   The registered function will likely be a wrapper that handles epoch generation and calls the specific feature computation function for each epoch.
    -   The registered output class will be `FeatureSignal`.

### 3.2 Data Structure and Traceability

-   **Input**: One or more `TimeSeriesSignal` instances identified by their keys in the `SignalCollection`.
-   **Epochs**: Defined by `window_length` and `step_size` (e.g., "30s", "10s"). These are converted to `pd.Timedelta`. Epochs are indexed by their start time.
-   **Output `FeatureSignal.data` DataFrame**:
    ```
                         hr_mean  hr_std  accel_x_max
    epoch_start_time
    2023-01-01 00:00:00     75.2     3.1          1.2
    2023-01-01 00:00:10     76.0     3.5          1.1
    2023-01-01 00:00:20     75.5     3.3          1.3
    ...
    ```
-   **Output `FeatureSignal.metadata`**:
    -   `signal_id`: Unique ID for the `FeatureSignal`.
    -   `name`: Key assigned in the workflow (e.g., "hr_features").
    -   `signal_type`: `SignalType.FEATURES`.
    -   `derived_from`: List of `(source_signal_id, operation_index)` tuples linking back to the input `TimeSeriesSignal`(s).
    -   `operations`: List containing one `OperationInfo` for the feature extraction step (e.g., `operation_name="feature_mean"`).
    -   `epoch_window_length`: `pd.Timedelta("30s")`.
    -   `epoch_step_size`: `pd.Timedelta("10s")`.
    -   `feature_names`: `["hr_mean", "hr_std", "accel_x_max"]`.
    -   `source_signal_keys`: `["hr_filtered", "accel_processed"]`.
    -   `framework_version`: Version of the framework used.

### 3.3 Workflow Integration

Feature extraction steps are defined in the `steps` section of the workflow YAML:

```yaml
steps:
  # Example 1: Mean and Std Dev for a single signal
  - operation: "feature_statistics" # A potential operation computing multiple stats
    inputs: ["hr_filtered"]         # List containing one signal key
    parameters:
      window_length: "30s"
      step_size: "10s"
      aggregations: ["mean", "std"] # Specify which stats to compute
    output: "hr_features"           # Key for the resulting FeatureSignal

  # Example 2: Correlation between two signals
  - operation: "feature_correlation"
    inputs: ["hr_filtered", "accel_magnitude"] # List containing two signal keys
    parameters:
      window_length: "60s"
      step_size: "30s"
      method: "pearson"             # Parameter for the correlation function
    output: "hr_accel_corr_features"

  # Example 3: Signal-specific feature (e.g., HRV)
  - operation: "feature_hrv"        # Specific operation for HRV
    inputs: ["rr_interval_signal"]  # Requires an RR-interval signal
    parameters:
      window_length: "5m"           # 5-minute windows
      step_size: "5m"               # Non-overlapping
      hrv_features: ["sdnn", "rmssd"] # Specify which HRV features
    output: "hrv_features"

  # Example 4: Combine multiple feature signals
  - operation: "combine_features"   # Registered collection operation
    inputs: ["hr_features", "hr_accel_corr_features", "hrv_features"] # List of FeatureSignal keys
    # No specific parameters typically needed, unless handling column name conflicts
    output: "final_feature_matrix" # Key for the combined FeatureSignal
```

The `WorkflowExecutor` will:
1.  Identify feature extraction operations (like `feature_statistics`) in `SignalCollection.multi_signal_registry`.
2.  Retrieve the input `TimeSeriesSignal`(s) using the keys from the `inputs` list.
3.  Call the registered wrapper function, passing the signals and parameters (`window_length`, `step_size`, and any function-specific params).
4.  The wrapper function handles epoch generation and calls the core feature computation logic for each epoch.
5.  The wrapper function assembles the results into the `FeatureSignal` DataFrame and metadata.
6.  The `WorkflowExecutor` adds the resulting `FeatureSignal` to the `SignalCollection` using the `output` key for feature extraction steps.
7.  For the `combine_features` step, the `WorkflowExecutor` identifies it in the `SignalCollection.collection_operation_registry`, retrieves the input `FeatureSignal` objects, calls the combination logic, and adds the resulting combined signal to the collection.

### 3.4 Epoch Generation Details (Revised for Consistency)

-   **Prerequisite:** A `generate_alignment_grid` step must be executed *before* feature extraction steps to calculate the `SignalCollection.grid_index`. This index defines the definitive start and end time for all subsequent epoch generation within that collection instance.
-   **Epoch Range:** The epoch generation logic within each feature operation wrapper will use `collection.grid_index.min()` and `collection.grid_index.max()` as the boundaries for generating the sequence of potential epoch start times.
-   **Epoch Sequence Generation:** A sequence of epoch start times is generated using the collection's overall start time (`grid_index.min()`), the specified `step_size`, and ensuring the last epoch does not extend beyond `grid_index.max()`. A common way is `pd.date_range(start=grid_index.min(), end=grid_index.max() - window_length + step_size, freq=step_size, inclusive='left')`.
-   **Data Slicing:** For each generated epoch interval `[start_time, end_time = start_time + window_length)`, data segments are extracted from the specific `TimeSeriesSignal`(s) listed as `inputs` for *that particular feature operation*.
-   **Handling Missing Data:** If an input signal does not have data covering a specific epoch interval (because the interval falls outside that signal's range, even though it's within the collection's global range), the slicing will yield an empty DataFrame for that signal/epoch. Feature computation functions must handle empty inputs gracefully (typically returning `NaN`).
-   **Passing Data:** The potentially empty or partial data segments for the current epoch are passed to the core feature computation function.

## 4. Implementation Plan

1.  **Create `FeatureSignal` Class**:
    -   Define `src/sleep_analysis/signals/feature_signal.py`.
    -   Inherit from `SignalData`.
    -   Set `signal_type = SignalType.FEATURES`.
    -   Define `__init__` to handle feature-specific metadata (`epoch_window_length`, `epoch_step_size`, `feature_names`, `source_signal_keys`).
    -   Implement `get_data()` to return the feature DataFrame.
    -   Add `SignalType.FEATURES` to `src/sleep_analysis/signal_types.py`.

2.  **Create Feature Computation Functions**:
    -   Define `src/sleep_analysis/operations/feature_extraction.py`.
    -   Implement core logic functions (e.g., `_compute_mean_std(segment: pd.DataFrame) -> Dict`, `_compute_correlation(segment1: pd.DataFrame, segment2: pd.DataFrame, method: str) -> Dict`). These functions operate on data *within a single epoch*.

3.  **Implement Epoch Generation and Wrapper Logic**:
    -   Create a helper function `_generate_epochs(signals: List[TimeSeriesSignal], window_length: pd.Timedelta, step_size: pd.Timedelta)` that yields `(epoch_start, epoch_end, List[pd.DataFrame_segment])`.
    -   Create wrapper functions for each feature operation (e.g., `compute_feature_statistics`, `compute_feature_correlation`).
        -   These wrappers will be registered in `SignalCollection.multi_signal_registry`.
        -   Input signature: `(signals: List[SignalData], parameters: Dict[str, Any]) -> FeatureSignal`.
        -   Inside the wrapper:
            -   Parse `window_length`, `step_size`, and other parameters.
            -   Call `_generate_epochs`.
            -   For each yielded epoch:
                -   Call the appropriate core feature computation function(s) (from step 2) with the data segments.
            -   Collect results from all epochs.
            -   Assemble the final `FeatureSignal` DataFrame (index=epoch start times, columns=features).
            -   Construct the `FeatureSignal` metadata.
            -   Instantiate and return the `FeatureSignal`.
            -   **Crucially:** Ensure the wrapper retrieves the `collection.grid_index` and uses its min/max times to define the epoch generation range.

4.  **Register Operations**:
    -   In `SignalCollection` or a dedicated registration point, add entries to `multi_signal_registry` for feature extraction operations:
        ```python
        from .operations.feature_extraction import compute_feature_statistics, compute_feature_correlation
        from .signals.feature_signal import FeatureSignal

        SignalCollection.multi_signal_registry.update({
            "feature_statistics": (compute_feature_statistics, FeatureSignal),
            "feature_correlation": (compute_feature_correlation, FeatureSignal),
            # Add other feature operations here
        })
        ```
    -   Register the `combine_features` operation in `SignalCollection.collection_operation_registry`.

5.  **Implement `combine_features` Operation**:
    -   Create the function/method for `combine_features`.
    -   Implement logic to retrieve input `FeatureSignal`s.
    -   Add validation to check for identical `DatetimeIndex` across inputs.
    -   Perform `pd.concat(axis=1)`.
    -   Create and return the resulting combined `SignalData` instance with appropriate metadata (`derived_from` pointing to input `FeatureSignal` IDs).

6.  **Update `WorkflowExecutor`**:
    -   Ensure the executor correctly handles steps with feature extraction `operation` keys (likely via `multi_signal_registry`).
    -   Ensure the executor correctly handles the `combine_features` step (via `collection_operation_registry`).
    -   It should correctly parse `inputs` (list of signal keys) and `parameters`.
    -   It should call the appropriate wrapper function from `multi_signal_registry`.
    -   It should add the resulting `FeatureSignal` (individual or combined) to the collection using the `output` key.

7.  **Update Documentation**:
    -   Add `FeatureSignal` and the `combine_features` operation to relevant documentation sections.
    -   Document the available feature extraction operations and their parameters in `README.md` or dedicated docs.
    -   Provide clear examples showing `generate_alignment_grid` preceding feature extraction and combination steps in workflow YAML files.

8.  **Implement Tests**:
    -   **Unit Tests**:
        -   Test core feature computation functions (`_compute_mean_std`, etc.).
        -   Test the `_generate_epochs` helper function.
        -   Test the `FeatureSignal` class initialization and metadata handling.
        -   Test the feature operation wrapper functions.
        -   Test the `combine_features` operation logic (including index validation and concatenation).
    -   **Integration Tests**:
        -   Test the end-to-end feature extraction *and combination* process within `WorkflowExecutor` using sample workflow YAML files.
        -   Verify the structure and content of the individual and final combined `FeatureSignal` DataFrames and metadata.
        -   Test with overlapping and non-overlapping epochs.
        -   Test with single and multiple input signals feeding into the combination step.
        -   Test that `combine_features` correctly validates matching indices generated using the grid range.
        -   Test feature generation when `generate_alignment_grid` hasn't been run (should likely raise an error or warn, depending on implementation choice).

## 5. Considerations

-   **Workflow Prerequisite:** Users must remember to include `generate_alignment_grid` before feature extraction steps if consistent, combinable feature sets are desired. This should be clearly documented.
-   **Performance**: For very long signals or small step sizes, generating epochs over the full `grid_index` range can be intensive. `pandas.DataFrame.rolling()` might offer optimization for *some* basic statistical features on *single* signals if the signal is already on the grid, but the proposed epoch iteration approach is more general for custom functions and multi-signal features.
-   **Handling Missing Data / NaN Features:** The approach of using the global grid range means feature functions *must* robustly handle cases where input data for an epoch is empty or partial (due to the epoch falling outside a specific signal's original range). Returning `NaN` for features in such cases is standard.
-   **Edge Cases**: The use of `grid_index.min()` and `grid_index.max()` naturally handles the start and end of the collection's valid data range. Partial windows at the edges are implicitly handled by slicing; feature functions decide how to compute on partial data (e.g., require minimum data points or return NaN).
-   **Signal Alignment:** While epoch *intervals* are aligned by this design, the recommendation remains strong: align the underlying `TimeSeriesSignal` data to the `grid_index` *before* feature extraction for the most meaningful results, especially for multi-signal features.
