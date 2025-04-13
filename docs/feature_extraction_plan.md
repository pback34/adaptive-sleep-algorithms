# Feature Extraction and Refactoring Plan

## 1. Introduction

This document outlines the requirements, design, and refactoring plan for the epoch-based feature extraction functionality within the Flexible Signal Processing Framework for Sleep Analysis.

**Core Refactoring Decision:** To improve clarity, maintainability, and conceptual accuracy, the framework will be refactored to treat Time Series Signals and epoch-based Features as distinct entities. This involves creating separate metadata classes and storage containers within the `SignalCollection` for each type.

The goal remains to enable the transformation of continuous time-series signals (e.g., heart rate, accelerometer magnitude) into structured datasets of features computed over specified time windows (epochs), facilitating tasks like sleep stage classification and event detection, but within this revised, more robust structure.

The current scope assumes all input signals are time-series (`TimeSeriesSignal`), fitting sleep analysis needs. Future extensions could support non-time-series data (e.g., static metadata) via new signal types or adapters. The mandatory global epoch grid (FR-FE.10) ensures temporal consistency for features within a collection analysis run, though it offers less flexibility than per-feature-set grids, which could be considered in future extensions.

## 2. Requirements

### 2.1 Functional Requirements (FR-FE)

-   **FR-FE.1 Epoch Generation**: The framework must allow users to define epochs based on continuous time-series signals using configurable parameters:
    -   `window_length`: The duration of each epoch window (e.g., 30 seconds).
    -   `step_size`: The time interval between the start of consecutive epochs (e.g., 10 seconds). The `step_size` can be less than or equal to the `window_length`, allowing for overlapping epochs.
-   **FR-FE.2 Feature Computation**: The framework must support computing aggregation functions over the data within each epoch for one or more input signals. Supported functions should include, but not be limited to:
    -   Basic statistics: mean, standard deviation, median, minimum, maximum, variance.
    -   Signal-specific metrics: e.g., Heart Rate Variability (HRV) features for heart rate signals, spectral power bands for EEG signals.
-   **FR-FE.3 Single and Multi-Signal Features**: Support computation of features derived from:
    -   A single input `TimeSeriesSignal` (e.g., mean heart rate over epoch).
    -   Multiple input `TimeSeriesSignal`s (e.g., correlation between heart rate and accelerometer magnitude over epoch).
-   **FR-FE.4 Input Signals**: Feature extraction operations must accept one or more `TimeSeriesSignal` instances from the `SignalCollection.time_series_signals` container as input.
-   **FR-FE.5 Output Feature Object**: Feature extraction operations must produce a new `Feature` object (or similar distinct class).
    -   The `Feature` object's data must be a `pandas.DataFrame` where:
        -   The index represents the start timestamp of each epoch.
        -   Each column represents a computed feature (e.g., `hr_mean`, `accel_std_x`). Column names should clearly indicate the source signal and the feature computed.
    -   The resulting `Feature` object will be stored in a separate container within the `SignalCollection` (e.g., `SignalCollection.features`).
-   **FR-FE.6 Feature Metadata and Traceability**: A dedicated `FeatureMetadata` class must store information specific to features, including:
    -   Links to the source `TimeSeriesSignal` instance(s) (using their unique IDs).
    -   The parameters used for epoch generation (`epoch_window_length`, `epoch_step_size`).
    -   Details of the operation performed (e.g., operation name, aggregation functions).
    -   Details of the operation performed (`operations` list containing `OperationInfo` with `operation_name` and `parameters`). This identifies the *type* of feature generation method used (e.g., statistical, spectral).
    -   A list of the exact feature column names present in the data (`feature_names`). These names must follow a consistent, interpretable convention (e.g., `<source_signal_name>_<source_column_name>_<feature_identifier>`) to ensure traceability.
    -   The keys/names of the source signals from the collection (`source_signal_keys`).
    -   An explicit `feature_type` string (e.g., "statistical", "spectral", "hrv") indicating the category of features, useful for filtering and indexing.
-   **FR-FE.7 Workflow Integration**: Feature extraction steps must be configurable within the workflow YAML files. The `WorkflowExecutor` must correctly dispatch these operations to act on `TimeSeriesSignal` inputs and store results in the `features` container within the `SignalCollection`.
-   **FR-FE.8 Extensibility**: The framework must allow users to define and register their own custom feature extraction functions easily, adhering to the new structure (input: `TimeSeriesSignal`, output: `Feature`), setting the appropriate `feature_type`, and generating appropriately named feature columns.
-   **FR-FE.9 Combined Feature Matrix Index Configuration**: The framework must allow users to specify a separate configuration (`feature_index_config`) to define the structure of the MultiIndex columns for the final combined feature matrix (`_combined_feature_matrix`). This configuration should be independent of the `index_config` used for combined time-series signals and should use metadata fields from the source `Feature` objects (and potentially their source `TimeSeriesSignal`s) to create traceable column headers.
-   **FR-FE.10 Global Epoch Grid**: The framework must support the calculation of a global epoch grid (`epoch_grid_index`) based on the overall time range of the collection's time-series signals and globally defined epoch parameters (`epoch_grid_config` containing `window_length` and `step_size`). This grid defines the common start times for all feature extraction epochs within the collection. Feature extraction steps must use this grid, removing the need for per-step `step_size` configuration. The framework must robustly check that this grid has been generated before allowing feature extraction operations to proceed.

### 2.2 Non-Functional Requirements (NFR-FE)

-   **NFR-FE.1 Performance**: Feature extraction should be optimized for performance on large datasets, leveraging vectorized operations (e.g., via `pandas`, `numpy`) where possible.
-   **NFR-FE.2 Memory Efficiency**: Minimize memory footprint, especially when dealing with long signals and overlapping epochs.
-   **NFR-FE.3 Usability**: Provide an intuitive API for specifying feature extraction parameters both programmatically and within workflow configurations.
-   **NFR-FE.4 Maintainability**: The design should be modular to allow easy addition of new feature functions without modifying core framework components.

## 3. Design

### 3.1 Core Components (Refactored Design)

1.  **Metadata Classes**:
    *   **`TimeSeriesMetadata`**: (Refactored from `SignalMetadata`) Located in `src/sleep_analysis/core/metadata.py`. Contains fields relevant only to time-series signals (e.g., `signal_id`, `name`, `signal_type`, `sample_rate`, `units`, `start_time`, `end_time`, `sensor_type`, `sensor_model`, `body_position`, `source_files`, `operations`, `derived_from`). *Excludes* feature-specific fields.
    *   **`FeatureMetadata`**: New class in `src/sleep_analysis/core/metadata.py`. Contains fields relevant only to features:
        *   `feature_id`: New unique UUID.
        *   `name`: Key assigned in the workflow.
        *   `feature_type`: Optional `FeatureType` Enum (e.g., `STATISTICAL`, `SPECTRAL`, `HRV`) identifying the category of features. Set by the generating operation. Useful for filtering and MultiIndex creation. A `FeatureType` Enum will be defined to ensure consistency.
        *   `epoch_window_length`, `epoch_step_size`: Global epoch parameters used for the `epoch_grid_index`.
        *   `operations`: List of `OperationInfo` detailing the specific generation method (e.g., `feature_statistics`) and its parameters (e.g., `aggregations`, `window_length` if overridden). Provides detailed provenance.
        *   `feature_names`: List of the *simple* feature names computed by the operation (e.g., `["mean", "std"]` for statistics on a single-column input, `["X_mean", "X_std", "Y_mean", "Y_std"]` for a two-column input, `["sdnn", "rmssd"]` for HRV). These names represent the specific metrics calculated and should *not* include source signal prefixes or other metadata. The final complex column structure (often MultiIndex) is created by `combine_features`.
        *   `feature_type`: Optional `FeatureType` Enum (e.g., `STATISTICAL`, `SPECTRAL`, `HRV`) identifying the category of features. Set by the generating operation. Useful for filtering and MultiIndex creation. A `FeatureType` Enum will be defined to ensure consistency.
        *   `source_signal_keys`: List of keys of the source `TimeSeriesSignal`(s) (e.g., `["hr_0"]`).
        *   `source_signal_ids`: List of UUIDs of the source `TimeSeriesSignal`(s) for robust provenance.
        *   `sensor_type`, `sensor_model`, `body_position`: Propagated from source `TimeSeriesSignal`(s) if applicable and potentially needed for `feature_index_config`. See propagation rules below.
        *   `framework_version`.
        *   *Note:* This class must contain all metadata fields that might be referenced by the `feature_index_config` (see FR-FE.9).
        *   **Metadata Propagation Rules for Multi-Input Features:**
            *   **Single Input:** If a feature is derived from a single `TimeSeriesSignal`, identifying metadata fields (e.g., `sensor_type`, `sensor_model`, `body_position`) are directly copied from the source signal's metadata to the `FeatureMetadata`.
            *   **Multiple Inputs:** If a feature is derived from multiple `TimeSeriesSignal`s:
                *   If all input signals share the *same* value for a given metadata field (e.g., all have `sensor_type=SensorType.ACCEL`), that common value is propagated to the `FeatureMetadata`.
                *   If input signals have *different* values for a field, the `FeatureMetadata` will store a special value indicating heterogeneity, such as the string `"mixed"` or `None`. The exact handling might depend on the field and configuration. This ensures the `feature_index_config` can still group features but acknowledges the mixed source characteristics. Alternatively, specific operations could define custom aggregation logic for certain metadata fields.
    *   **`CollectionMetadata`**: Existing class in `src/sleep_analysis/core/metadata.py`. Will be extended to include:
        *   `index_config: List[str]`: Configuration for combined *time-series* signal MultiIndex columns.
        *   `feature_index_config: List[str]`: Configuration for combined *feature* matrix MultiIndex columns.
        *   `epoch_grid_config: Dict[str, str]`: New field. Contains global `window_length` and `step_size` (e.g., `{"window_length": "30s", "step_size": "10s"}`).

2.  **Data Classes**:
    *   **`TimeSeriesSignal`**: Remains largely the same structure but now uses `TimeSeriesMetadata`. Located in `src/sleep_analysis/signals/time_series_signal.py`. Base class for specific signal types (ACCEL, HR, etc.).
    *   **`Feature`**: (Refactored and renamed from `FeatureSignal`) Class located in `src/sleep_analysis/features/feature.py` (after renaming `src/sleep_analysis/signals/feature_signal.py`).
        *   Holds feature data (`pandas.DataFrame` indexed by epoch start time).
        *   Uses `FeatureMetadata`.
        *   Does *not* inherit from `SignalData` or `TimeSeriesSignal`. It's a distinct concept.

3.  **`SignalCollection` Refactoring**:
    *   Located in `src/sleep_analysis/core/signal_collection.py`.
    *   **Storage:** Will contain two separate primary storage dictionaries:
        *   `self.time_series_signals: Dict[str, TimeSeriesSignal]` for time-domain signals.
        *   `self.features: Dict[str, Feature]` for epoch-based feature sets.
    *   **Alignment & Epoch Grids:**
        *   `grid_index: pd.DatetimeIndex`: Common time grid for signal alignment.
        *   `_alignment_params_calculated: bool`: Flag indicating if `grid_index` is calculated.
        *   `epoch_grid_index: pd.DatetimeIndex`: New. Common grid of epoch start times for feature extraction.
        *   `global_epoch_window_length: pd.Timedelta`: New. Global window length used for epoch grid generation.
        *   `global_epoch_step_size: pd.Timedelta`: New. Global step size used for epoch grid generation.
        *   `_epoch_grid_calculated: bool`: New flag indicating if `epoch_grid_index` is calculated.
    *   **Access:** Methods like `add_signal`, `get_signal` will be adapted or replaced by type-specific methods (e.g., `add_time_series_signal`, `add_feature`, `get_time_series_signal`, `get_feature`). `get_signals` will be updated to search the appropriate dictionary based on criteria.
    *   **Operations:** Operation registries (`multi_signal_registry`, `collection_operation_registry`) will be adapted.
        *   New collection operation `generate_epoch_grid` calculates and stores `epoch_grid_index`, `global_epoch_window_length`, `global_epoch_step_size`, and sets `_epoch_grid_calculated = True`.
        *   Feature extraction operations will now be functions that take `TimeSeriesSignal`(s) and `epoch_grid_index` as input and return a `Feature` object to be stored in `self.features`.
        *   `apply_multi_signal_operation` will check `_epoch_grid_calculated` before dispatching feature extraction operations, raising an error if `False`. It will also handle propagating identifying metadata (e.g., sensor type, model, position) from the input `TimeSeriesSignal` to the output `Feature` object's metadata *if* the feature is derived from a single input signal. This is crucial for populating fields used by `feature_index_config`.

4.  **Feature Computation Functions**:
    *   Remain in `src/sleep_analysis/operations/feature_extraction.py`.
    *   Input signature changes: Accept `signals: List[TimeSeriesSignal]`, `epoch_grid_index: pd.DatetimeIndex`, and parameters (excluding `step_size`, `window_length` is optional override). For multi-signal features, align input signals to `collection.grid_index` within each epoch before computation.
    *   Output signature changes: Return a `Feature` object (containing the data DataFrame and populated `FeatureMetadata`). The DataFrame index must match the `epoch_grid_index`.
    *   **Optimizations**:
        *   Use vectorized operations (e.g., `pandas` groupby, `rolling`).
        *   Implement a `@cache_features` decorator: Caches the resulting `Feature` object based on input signal IDs, operation name, and parameters. Cache invalidation occurs if input signals change.
        *   Implement optional lazy evaluation: `Feature` objects store metadata and computation parameters upon creation. `get_data()` triggers computation and caching only when first called.
        *   Optimize overlapping epochs: Primarily leverage `pandas.DataFrame.rolling()` methods. For complex functions, explore caching intermediate segment calculations.
    *   **Scalability**: To handle potentially large combined feature matrices (`_combined_feature_matrix`), strategies like lazy column access or chunk-based processing during export will be considered.

5.  **Epoch Generation Logic**:
    *   Remains conceptually similar, likely implemented within the feature computation wrapper functions.
    *   Takes input `TimeSeriesSignal`(s), `window_length`, `step_size`, and the collection's `grid_index`.
    *   Extracts data segments from the input `TimeSeriesSignal`(s) for each epoch.

6.  **Feature Combination Operation (`combine_features`)**:
    *   Refactored as a collection-level operation.
    *   Input: List of keys identifying `Feature` objects in `self.features`. Accepts `feature_index_config` parameter (list of metadata field names from `FeatureMetadata` or its source `TimeSeriesMetadata`).
    *   Logic: Retrieves `Feature` objects. Validates that all indices match `collection.epoch_grid_index` exactly (using `pandas.Index.equals()`). Concatenates the DataFrames column-wise (`pd.concat(axis=1)`). Builds MultiIndex columns for the resulting DataFrame using the provided `feature_index_config` (retrieving values from each source `Feature` object's metadata) and the *simple* feature names stored in `FeatureMetadata.feature_names` as the final level(s) of the index.
    *   Output: Stores the resulting combined feature `pandas.DataFrame` (with MultiIndex columns if `feature_index_config` was provided) in a dedicated `SignalCollection` attribute (e.g., `self._combined_feature_matrix`). It does *not* add the result back into `self.features`.

7.  **Registration**:
    *   Feature extraction operations (e.g., `compute_feature_statistics`) will be registered (perhaps in a new registry or adapted existing ones) as functions that produce `Feature` objects.
    *   The `combine_features` operation remains a collection-level operation.

### 3.2 Data Structure and Traceability (Refactored Design)

-   **Input**: One or more `TimeSeriesSignal` instances identified by their keys in `SignalCollection.time_series_signals`.
-   **Epochs**: Defined by `window_length` and `step_size` (e.g., "30s", "10s"). These are converted to `pd.Timedelta`. Epochs are indexed by their start time, aligned to the collection's `grid_index`.
-   **Output `Feature.data` DataFrame**: Contains simple feature columns. Example for statistical features from a single-column signal:
    ```
                         mean  std   min   max
    epoch_start_time
    2023-01-01 00:00:00  75.2  3.1  70.0  80.0
    2023-01-01 00:00:10  76.0  3.5  71.0  81.0
    2023-01-01 00:00:20  75.5  3.3  70.5  80.5
    ...
    ```
-   **Output `Feature.metadata` (using `FeatureMetadata` class)**: Example for the above data, assuming it was generated from `hr_0` and stored with key `hr_stats_0`.
    -   `feature_id`: New unique UUID for the `Feature` object.
    -   `name`: Key assigned in the workflow (e.g., "hr_stats_0").
    -   `signal_type`: Should be `SignalType.FEATURES`.
    -   `sensor_type`, `sensor_model`, `body_position`: Inherited or derived from source `TimeSeriesSignal`(s) if applicable and needed for `feature_index_config`.
    -   `epoch_window_length`: `pd.Timedelta("30s")`. (Global value used)
    -   `epoch_step_size`: `pd.Timedelta("10s")`. (Global value used)
    -   `feature_names`: `["mean", "std", "min", "max"]` (Simple metric names).
    -   `feature_type`: `FeatureType.STATISTICAL` (Category of features).
    -   `source_signal_keys`: `["hr_0"]` (Key of the source `TimeSeriesSignal`).
    -   `source_signal_ids`: List containing the UUID of the `hr_0` signal.
    -   `operations`: List containing `OperationInfo` detailing the generation, e.g., `[OperationInfo(operation_name='feature_statistics', parameters={'aggregations': ['mean', 'std', 'min', 'max']})]`. Note: `window_length` might be omitted if global was used.
    -   `framework_version`: Version of the framework used.
-   **Column Naming Convention:** Feature columns within the `Feature.data` DataFrame should have *simple names* representing the metric (e.g., `mean`, `std`, `sdnn`). The connection to the source signal is maintained through `FeatureMetadata` (`name`, `source_signal_keys`). The final complex column structure (often MultiIndex) is created by `combine_features` using `feature_index_config`.

### 3.3 Workflow Integration (Refactored Design)

Feature extraction steps are defined in the `steps` section of the workflow YAML. The `type` field will implicitly determine whether the operation targets time-series signals or features. The `feature_index_config` is specified globally.

```yaml
# Example Collection Settings
collection_settings:
  index_config: ["signal_type", "sensor_model", "body_position", "name"] # For combined signals
  feature_index_config: ["name", "feature_type", "sensor_model"] # For combined features
  epoch_grid_config: # New section for global feature epoch settings
    window_length: "30s" # Global default window
    step_size: "10s"     # Global step size (defines epoch_grid_index frequency)

steps:
  # Prerequisite: Generate signal alignment grid (if needed for preprocessing)
  - type: collection
    operation: "generate_alignment_grid"
    parameters:
      target_sample_rate: 10.0

  # Prerequisite: Generate the common epoch grid for features
  - type: collection
    operation: "generate_epoch_grid" # New step

  # Example 1: Mean and Std Dev for a single time-series signal
  - type: time_series # Implicitly targets time_series_signals container
    operation: "feature_statistics" # Operation that produces a Feature object
    inputs: ["hr_filtered"]         # Key(s) from time_series_signals
    parameters:
      # window_length: "30s" # Optional: Uses global "30s" from epoch_grid_config if omitted
      # step_size: "10s"     # REMOVED: Step size is now global, defined by epoch_grid_index
      aggregations: ["mean", "std"] # Specify which stats to compute
    output: "hr_stats"              # Key for the resulting Feature object (e.g., "hr_stats_0")

  # Example 2: Correlation between two time-series signals (using a longer window)
  - type: time_series # Implicitly targets time_series_signals container
    operation: "feature_correlation" # Operation that produces a Feature object
    inputs: ["hr_filtered", "accel_magnitude"] # Keys from time_series_signals
    parameters:
      window_length: "60s"          # Overrides global "30s" for this calculation
      # step_size: REMOVED
      method: "pearson"             # Parameter for the correlation function
    output: "hr_accel_corr"         # Key for the resulting Feature object (e.g., "hr_accel_corr_0")

  # Example 3: Signal-specific feature (e.g., HRV) from a time-series signal (non-overlapping)
  - type: time_series # Implicitly targets time_series_signals container
    operation: "feature_hrv"        # Specific operation producing a Feature object
    inputs: ["rr_interval_signal"]  # Key from time_series_signals
    parameters:
      window_length: "5m"           # 5-minute windows (overrides global)
      # step_size: REMOVED (Epoch start times still determined by global step_size via epoch_grid_index)
      hrv_features: ["sdnn", "rmssd"] # Specify which HRV features
    output: "hrv_metrics"            # Key for the resulting Feature object (e.g., "hrv_metrics_0")

  # Example 4: Combine multiple Feature objects into the final matrix
  - type: collection # Targets the collection itself
    operation: "combine_features"   # Registered collection operation
    inputs: ["hr_stats", "hr_accel_corr", "hrv_metrics"] # List of Feature object keys (base names)
    # No 'output' key needed, result stored in collection._combined_feature_matrix
    # Parameters might include handling for column name conflicts if feature_index_config isn't used or is ambiguous
    parameters: {} # e.g., conflict_resolution: 'rename'

  # --- Full Example with Multi-Signal Features ---
  # Here’s a complete example combining statistical and correlation features:
  collection_settings:
    index_config: ["signal_type", "name"]
    feature_index_config: ["name", "feature_type"]
    epoch_grid_config:
      window_length: "30s"
      step_size: "10s"
  steps:
    - type: collection
      operation: "generate_epoch_grid"
    - type: time_series
      operation: "feature_statistics"
      inputs: ["hr_0"]
      parameters:
        aggregations: ["mean", "std"]
      output: "hr_stats"
    - type: time_series
      operation: "feature_correlation"
      inputs: ["hr_0", "accel_0"]
      parameters:
        method: "pearson"
      output: "hr_accel_corr"
    - type: collection
      operation: "combine_features"
      inputs: ["hr_stats", "hr_accel_corr"]
```

The `WorkflowExecutor` will:
1.  Determine the target container (`time_series_signals` or `features`) based on the step `type` or operation name convention.
2.  For feature extraction steps (e.g., `feature_statistics`):
    *   Retrieve input `TimeSeriesSignal`(s) from `collection.time_series_signals`.
    *   Retrieve the pre-calculated `epoch_grid_index` from the collection (checking `_epoch_grid_calculated` flag first).
    *   Call the appropriate feature extraction function (passing signals, `epoch_grid_index`, parameters).
    *   Add the resulting `Feature` object to `collection.features` using the `output` key.
3.  For the `generate_epoch_grid` step:
    *   Identify it as a collection operation.
    *   Call `collection.generate_epoch_grid()`.
4.  For the `combine_features` step:
    *   Identify it as a collection operation.
    *   Retrieve input `Feature` objects from `collection.features`.
    *   Call the `collection.combine_features` method, passing the `feature_index_config` read from the collection settings.
    *   The result (DataFrame with MultiIndex columns) is stored internally in `collection._combined_feature_matrix`.

### 3.4 Epoch Generation Details (Revised for Epoch Grid)

-   **Recommended Workflow:** The standard workflow should be:
    1.  `generate_alignment_grid`: Calculate the common time grid (`grid_index`) for signals.
    2.  (Optional) `apply_grid_alignment`: Reindex relevant `TimeSeriesSignal`s to `grid_index`.
    3.  `generate_epoch_grid`: Calculate the common epoch start times (`epoch_grid_index`) based on global config.
    4.  Feature Extraction Steps (e.g., `feature_statistics`): These steps operate on `TimeSeriesSignal` data (potentially aligned) using the `epoch_grid_index`.
-   **Prerequisites:**
    *   `generate_alignment_grid` must run before `generate_epoch_grid` if the epoch grid needs to be based on the aligned time range.
    *   `generate_epoch_grid` must run before any feature extraction steps. The framework enforces this by checking the `_epoch_grid_calculated` flag in `SignalCollection` before dispatching feature operations.
-   **Epoch Definition**: The `epoch_grid_index` is a `DatetimeIndex` of epoch start times, calculated as a regular grid from the earliest start time to the latest end time across all `TimeSeriesSignal` objects, using `epoch_grid_config` parameters (`window_length`, `step_size`). Optional `start_time` and `end_time` parameters can override the range.
-   **Iteration:** Feature functions iterate through each `epoch_start_time` in the `epoch_grid_index`.
-   **Epoch Interval:** For each `epoch_start_time`, the corresponding epoch interval is `[epoch_start_time, epoch_start_time + effective_window_length)`, where `effective_window_length` is either the optional `window_length` parameter passed to the step or the `collection.global_epoch_window_length`.
-   **Data Slicing:** Data segments are extracted from the input `TimeSeriesSignal`(s) over the calculated epoch interval. Slicing logic depends on whether signals were aligned to `grid_index` beforehand.
-   **Handling Missing Data:** Feature computation functions must handle empty or all-NaN data segments gracefully (typically returning `NaN` for the corresponding features for that epoch). They must also define their behavior when an epoch contains *partially* missing data (e.g., require a minimum percentage of valid data points, otherwise return NaN). The resulting `Feature` object's DataFrame will have an index matching the `epoch_grid_index`.

## 3.5 Error Handling Strategy

-   **Within Feature Computation:** Errors encountered during the calculation for a specific epoch (e.g., due to invalid data) should ideally be handled gracefully by the computation function. The function should log a warning and return `NaN` for the affected features for that epoch, allowing the overall step to continue.
-   **Step-Level Errors:** Errors occurring at the step level (e.g., invalid parameters, failure to load input signals, critical computation failure across all epochs) will, by default, cause the `WorkflowExecutor` to halt execution and raise the error.
-   **Workflow Continuation (Optional):** A `continue_on_error: true` flag could be added to workflow steps. If set, the `WorkflowExecutor` would log the error, potentially store a placeholder/error marker for the step's output (if applicable), and proceed to the next step. This allows for partial results but requires careful consideration of downstream dependencies.

## 4. Refactoring Plan (Based on Current Codebase)

This plan outlines the steps to refactor the existing codebase to implement the separation of Time Series Signals and Features.

1.  **Define Metadata Classes**:
    *   **`FeatureMetadata`**: Create `src/sleep_analysis/core/feature_metadata.py` (or add to `metadata.py`). Define the `FeatureMetadata` dataclass including the new `feature_type: Optional[str]` field. Key fields also include `operations` (detailed provenance) and `feature_names` (listing columns following a convention). See Design section 3.1 for full field list.
    *   **`TimeSeriesMetadata`**: Rename the existing `SignalMetadata` class in `src/sleep_analysis/core/metadata.py` to `TimeSeriesMetadata`. Remove feature-specific fields. Ensure all existing time-series metadata fields are present. Update imports and references to `SignalMetadata` throughout the codebase to use `TimeSeriesMetadata`.

2.  **Define `Feature` Class**:
    *   Create `src/sleep_analysis/features/feature.py` (or similar location).
    *   Rename the file `src/sleep_analysis/signals/feature_signal.py` to `src/sleep_analysis/features/feature.py`.
    *   Rename the class `FeatureSignal` to `Feature` within the renamed file.
    *   Modify the `Feature` class:
        *   Remove inheritance from `SignalData`.
        *   Update `__init__` to accept `data: pd.DataFrame` and `metadata: FeatureMetadata`. Ensure it calls `super().__init__()` if any base class functionality (like basic attribute setting) is desired from `object`, otherwise remove the `super` call.
        *   Update type hints and docstrings.
        *   Remove methods specific to `SignalData` inheritance (e.g., `apply_operation`, `get_sampling_rate`, `_update_sample_rate_metadata`). Keep `get_data()`.
    *   Update all imports referencing the old path/class name.

3.  **Refactor `SignalCollection`**:
    *   In `src/sleep_analysis/core/signal_collection.py`:
        *   **Storage:** Replace `self.signals: Dict[str, SignalData]` with:
            *   `self.time_series_signals: Dict[str, TimeSeriesSignal]`
            *   `self.features: Dict[str, Feature]`
        *   **Attributes:** Add `epoch_grid_index: Optional[pd.DatetimeIndex] = None`, `global_epoch_window_length: Optional[pd.Timedelta] = None`, `global_epoch_step_size: Optional[pd.Timedelta] = None`, `_epoch_grid_calculated: bool = False`.
        *   Update `__init__` to initialize these new dictionaries and attributes.
        *   **Access:** Refactor `add_signal` into `add_time_series_signal(key, signal: TimeSeriesSignal)` and `add_feature(key, feature: Feature)`. Update `add_signal_with_base_name` accordingly or create type-specific versions. Refactor `get_signal` into `get_time_series_signal(key)` and `get_feature(key)`.
        *   Update `get_signals` (and `get_signals_from_input_spec`) to search the appropriate dictionary (`time_series_signals` or `features`) based on criteria (e.g., `signal_type` vs. `feature_type`). Adapt logic for base names and lists of keys.
        *   Update `update_signal_metadata` to `update_time_series_metadata` using `TimeSeriesMetadata`. Create `update_feature_metadata` for `FeatureMetadata`.
        *   **Operations:**
            *   Add new collection operation method `generate_epoch_grid()`. Calculate `epoch_grid_index` as a regular grid from the earliest `start_time` to the latest `end_time` across all `time_series_signals` (union of ranges), using `epoch_grid_config`. Allow optional `start_time` and `end_time` overrides. Set `_epoch_grid_calculated = True`. Register it.
            *   Adapt operation registries (`multi_signal_registry`, `collection_operation_registry`). Feature extraction operations registered in `multi_signal_registry` should now point to functions that accept `List[TimeSeriesSignal]`, `epoch_grid_index: pd.DatetimeIndex`, and parameters. The `output_class` in the registry tuple becomes `Feature`.
            *   Update `apply_multi_signal_operation` to clarify its flow and responsibilities:
                *   **Input Resolution:** Resolve the `inputs` list (containing keys or base names) into a list of specific `TimeSeriesSignal` keys present in `self.time_series_signals`. Raise an error if resolution fails for any input specifier.
                *   **Registry Lookup:** Look up the `operation_name` in `self.multi_signal_registry` to get the target function and expected output class (`Feature`).
                *   **Prerequisite Check (Features):** If the `operation_name` indicates a feature extraction operation (e.g., starts with `"feature_"`):
                    *   Verify that `self._epoch_grid_calculated` is `True`. If not, raise `RuntimeError` (as the required epoch grid is missing).
                    *   Retrieve the `self.epoch_grid_index`.
                *   **Function Execution:** Call the target function retrieved from the registry.
                    *   For feature operations, pass the list of input `TimeSeriesSignal` objects, the `epoch_grid_index`, and the `parameters`.
                    *   For other potential multi-signal operations, adapt the call signature as needed.
                *   **Result Handling:** Receive the result object (expected to be a `Feature` instance for feature operations).
                *   **Metadata Propagation:**
                    *   The called function (e.g., `compute_feature_statistics`) is responsible for setting core `FeatureMetadata` like `operations`, `feature_names`, `source_signal_keys`, `source_signal_ids`, and the global `epoch_window_length`/`epoch_step_size`.
                    *   `apply_multi_signal_operation` is then responsible for propagating *identifying* metadata (fields listed in `feature_index_config`, e.g., `sensor_type`, `sensor_model`, `body_position`) from the input `TimeSeriesSignal`(s) to the resulting `Feature` object's metadata, following the defined propagation rules (copy if single input, check for common value or mark as "mixed"/None if multiple inputs).
                *   **Return:** Return the fully processed result object (`Feature` instance).
            *   Update `generate_alignment_grid`, `apply_grid_alignment`, `align_and_combine_signals`, `combine_aligned_signals` to operate exclusively on `self.time_series_signals`.
        *   Refactor `combine_features`:
            *   Accept `feature_index_config: List[str]` as an argument.
            *   Retrieve input `Feature` objects from `self.features`.
            *   **Strict Index Validation:** Verify that the `pandas.DatetimeIndex` of *each* input `Feature` object is *exactly equal* to the `self.epoch_grid_index` using `pandas.Index.equals()`. If any index does not match, raise a `ValueError`, as this indicates a deviation from the expected workflow (feature functions must produce data aligned to the global epoch grid). Do *not* attempt to align or fill NaNs at this validation stage; mismatches are considered errors.
            *   Perform direct concatenation (`pd.concat(axis=1)`). Because indices are guaranteed to match, this effectively performs an outer join where missing values for specific features in certain epochs will naturally become `NaN`.
            *   Build the MultiIndex columns for the resulting DataFrame using the `feature_index_config`.
            *   Store the resulting DataFrame in `self._combined_feature_matrix`.
            *   Remove the `output` parameter from the method signature and workflow step definition.
        *   Refactor `summarize_signals`:
            *   Generate separate summaries for `time_series_signals` and `features`.
            *   Update logic for counts and shapes.
            *   Optionally add info about `_combined_feature_matrix`.
            *   Store results appropriately.
            *   Update printing logic.
        *   Update `_validate_timestamp_index` to operate on `TimeSeriesSignal`.

4.  **Refactor `TimeSeriesSignal` and Subclasses**:
    *   In `src/sleep_analysis/signals/time_series_signal.py` and subclasses (e.g., `accelerometer_signal.py`):
        *   Update `__init__` and other methods to use `TimeSeriesMetadata` instead of `SignalMetadata`.
        *   Remove any handling related to feature-specific metadata.

5.  **Refactor Feature Extraction Operations**:
    *   In `src/sleep_analysis/operations/feature_extraction.py`:
        *   Update function signatures (e.g., `compute_feature_statistics`) to accept `signals: List[TimeSeriesSignal]`, `epoch_grid_index: pd.DatetimeIndex`, `parameters: Dict`.
        *   Remove `step_size` from the `parameters` dictionary within the function. Make `window_length` optional, falling back to `collection.global_epoch_window_length` if not provided in `parameters`.
        *   Update the implementation to:
            *   Iterate through `epoch_grid_index` for epoch start times.
            *   Determine `effective_window_length` (from parameters or global).
            *   Calculate epoch end time (`epoch_start_time + effective_window_length`).
            *   Retrieve data from input `TimeSeriesSignal`s for the calculated interval.
            *   Perform calculations.
            *   Assemble the feature DataFrame, ensuring its index matches `epoch_grid_index`. Column names should represent the *simple* metric identifiers (e.g., `mean`, `std`, `X_mean`). If multiple input signals are processed, the DataFrame should ideally use a MultiIndex for columns, like `(signal_key, simple_feature_name)`.
            *   For multi-signal features (like correlation), align input signals to `collection.grid_index` within each epoch. Optimize overlapping epochs using vectorized methods (e.g., `pandas.rolling`). Add a `@cache_features` decorator and support lazy evaluation optionally.
            *   Create a `FeatureMetadata` instance, populating `feature_type` (e.g., `FeatureType.STATISTICAL`), `operations`, `feature_names` (with the *list* of simple column names generated, e.g., `['mean', 'std']`), `source_signal_ids`, `source_signal_keys`, etc. Use the *global* step/window sizes (`collection.global_epoch_step_size`, `collection.global_epoch_window_length`) for the metadata fields `epoch_window_length` and `epoch_step_size` to reflect the grid used.
            *   Instantiate and return a `Feature` object containing the data DataFrame and the populated `FeatureMetadata`.
        *   **Performance Implementation:** Ensure implementations leverage `pandas.rolling` for overlapping epochs where possible. Implement the `@cache_features` decorator and lazy evaluation logic within the `Feature` class or its factory functions.

6.  **Refactor `WorkflowExecutor`**:
    *   In `src/sleep_analysis/workflows/workflow_executor.py`:
        *   Update `execute_workflow` to read `epoch_grid_config` from `collection_settings` and store it (likely on the executor or pass to collection).
        *   Update `execute_step` logic:
            *   Read `index_config`, `feature_index_config`, and `epoch_grid_config` from the workflow's `collection_settings` and set/pass them to the `SignalCollection` instance as needed.
            *   Handle the new `generate_epoch_grid` collection operation step.
            *   When calling feature extraction operations via `apply_multi_signal_operation`, ensure the pre-calculated `epoch_grid_index` is retrieved from the collection and passed correctly (the check for `_epoch_grid_calculated` happens inside `apply_multi_signal_operation`). Remove `step_size` from parameters passed to feature steps.
            *   Determine the target container and operation type primarily based on the `operation_name` (e.g., operations starting with `feature_` target `TimeSeriesSignal` inputs and produce `Feature` outputs stored in `collection.features`). The step `type` field (`collection`, `time_series`) will serve as a secondary indicator.
            *   Call the appropriate `SignalCollection` methods (e.g., `get_time_series_signal`, `get_feature`, `add_time_series_signal`, `add_feature`).
            *   Handle the `combine_features` step correctly (no `output` key expected), passing the `feature_index_config` to the collection method.
        *   Update `_process_import_section` to use `add_time_series_signal`.
        *   Update `_process_export_section` and `_process_visualization_section` to handle retrieving data from the correct containers (`time_series_signals`, `features`, `_aligned_dataframe`, `_combined_feature_matrix`).

7.  **Refactor `ExportModule`**:
    *   In `src/sleep_analysis/export/export_module.py`:
        *   Update `export` method and helpers (`_export_csv`, `_export_excel`, etc.).
        *   Adapt logic to handle exporting `TimeSeriesSignal`s, `Feature` objects, the combined time-series DataFrame (`_aligned_dataframe`), the combined feature matrix (`_combined_feature_matrix`), and the potentially separate summary tables. Ensure MultiIndex columns are handled correctly during export (especially for CSV/Excel).
        *   Modify `_serialize_metadata` to handle both `TimeSeriesMetadata` and `FeatureMetadata`. Simplify the metadata serialization for the combined feature matrix; primarily include its column names (which will be MultiIndex tuples if configured) and potentially the `feature_index_config` used. Rely on the MultiIndex itself for detailed provenance within the exported data file.

8.  **Update Tests**:
    *   Review and update all unit and integration tests affected by these changes.
    *   Test `SignalCollection` methods with separate containers.
    *   Test feature extraction workflow steps ensuring `Feature` objects are created and stored correctly.
    *   Test `combine_features` operation and retrieval of `_combined_feature_matrix`.
    *   Test export functionality for all data types.
    *   Test summary generation.
    *   Test specific edge cases: empty signals, signals with gaps, mismatched time ranges, and failed feature computations (returning NaNs).

9.  **Update Documentation**:
    *   Reflect the new architecture, classes (`TimeSeriesMetadata`, `FeatureMetadata`, `Feature`), and `SignalCollection` structure throughout the documentation.
    *   Include a complete workflow example (see Section 3.3) with multi-signal feature extraction and combination, alongside updated class and method descriptions.
    *   Update workflow examples.

## 5. Considerations (Post-Refactoring)

-   **Workflow Prerequisites:**
    *   `generate_alignment_grid` (optional, before feature steps if signal alignment needed).
    *   `generate_epoch_grid` (mandatory, before any feature extraction steps). The framework enforces this via the `_epoch_grid_calculated` flag check in `SignalCollection.apply_multi_signal_operation`.
-   **Performance**: Epoch generation performance considerations remain.
-   **Handling Missing Data / NaN Features:** Feature computation functions must handle empty/partial data segments gracefully, returning `NaN` for the corresponding epoch start time in the `epoch_grid_index`.
-   **Epoch Grid:** All feature extraction steps use the common `epoch_grid_index`. `step_size` is globally defined. `window_length` can be overridden per step.
-   **Combined Feature Matrix:** The `combine_features` operation assumes input `Feature` objects share the same index (`epoch_grid_index`), simplifying concatenation. It stores its result in `collection._combined_feature_matrix`. Point 4 concern is addressed.
-   **Traceability & Feature Type:** `FeatureMetadata` captures `source_signal_ids` (UUIDs) and `operations`. The `epoch_window_length` and `epoch_step_size` fields in `FeatureMetadata` reflect the *global* values used for the `epoch_grid_index`. The `feature_index_config` provides traceability for the *combined* matrix through its MultiIndex structure. The `feature_type` field provides a high-level category.
-   **Column Naming:** Feature columns *within* an individual `Feature` object's DataFrame should have simple names representing the metric (e.g., `mean`, `std`, `sdnn`). These simple names are listed in `FeatureMetadata.feature_names`. The `combine_features` operation uses `feature_index_config` and metadata (from the `Feature` object and its sources) to construct the potentially complex MultiIndex columns in the final `_combined_feature_matrix`.
-   **Metadata Propagation:** For features derived from a single `TimeSeriesSignal`, identifying metadata (like sensor type, model, position) will be propagated from the input signal to the output `Feature` object's metadata by `apply_multi_signal_operation`. This ensures this information is available for use in the `feature_index_config` during combination.
-   **Storage:** `TimeSeriesSignal` objects in `collection.time_series_signals`, `Feature` objects in `collection.features`. Combined matrices (`_aligned_dataframe`, `_combined_feature_matrix`) and grids (`grid_index`, `epoch_grid_index`) are stored as separate attributes.
-   **Performance**: For overlapping epochs, use vectorized operations (e.g., `pandas.rolling`) or batch processing to optimize performance. Profile with large datasets to ensure scalability. Implement caching and lazy evaluation as detailed in the design.
-   **Usability**: Provide utility functions to simplify filtering, selecting, and manipulating the MultiIndex columns of the `_combined_feature_matrix`. Consider implementing common utility functions needed by both `TimeSeriesSignal` and `Feature` via composition or dedicated helper modules to avoid code duplication.
