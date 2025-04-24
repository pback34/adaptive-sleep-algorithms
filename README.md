# Sleep Analysis Framework

A flexible, extensible framework for processing sleep-related signals, designed for researchers and developers working with physiological data. The framework provides a robust foundation for signal processing with an emphasis on reproducibility, type safety, and memory efficiency.

## Key Features

- **Type-Safe Signal Processing**: Enum-based type safety ensures operations match signal types.
- **Complete Traceability**: Full metadata and operation history for reproducibility.
- **Memory Optimization**: Smart memory management for processing large datasets efficiently.
- **Flexible Workflows**: Support for both structured workflows and ad-hoc processing.
- **Extensible Design**: Easy to add new signal types, features, and processing operations.
- **Import Flexibility**: Convert signals from various sources to a standardized format.
- **Modular Importer System**: Easily add support for new file formats or sensor types, including handling fragmented data files via `MergingImporter`.
- **Robust Timestamp Handling**: Consistent management of timezones across import, processing, and export.
- **Epoch-Based Feature Extraction**: Generate features (statistical, categorical mode, etc.) aligned to a common time grid.
- **Flexible Data Combination**: Combine aligned time-series signals or feature sets into unified DataFrames.
- **Interactive Visualization**: Backend-agnostic visualization layer with support for Bokeh and Plotly, including specialized plots like hypnograms and sleep stage overlays.
- **Comprehensive Summarization**: Generate summary tables of all signals and features in the collection.

## Installation

### Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd sleep_analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Regular Installation

```bash
# From PyPI (when available)
pip install sleep-analysis

# From local directory
pip install .
```

## Usage

### Command Line Interface

The package provides a command-line tool to run workflow files:

```bash
# Run using the installed entry point
sleep-analysis --workflow workflows/polar_workflow.yaml --data-dir data

# Run using the Python module
python -m sleep_analysis --workflow workflows/polar_workflow.yaml --data-dir data

# Run using the specific CLI module
python -m sleep_analysis.cli.run_workflow --workflow workflows/polar_workflow.yaml --data-dir data
```

### Options

```
-w, --workflow      Path to the workflow YAML file (required)
-d, --data-dir      Base directory containing the data files (required)
-o, --output-dir    Directory for output files (default: ./output)
-l, --log-level     Set logging level (DEBUG, INFO, WARN, ERROR)
-v                  Set logging level to DEBUG (shorthand)
```

### Creating Workflow Files

Workflow files are YAML documents with several key sections:

1.  **Top-Level Settings**: Define global parameters like timezones.
    *   `default_input_timezone`: (Optional) The assumed timezone for naive timestamps in source files if not specified per importer (e.g., "America/New_York"). If omitted, naive timestamps might be treated as UTC.
    *   `target_timezone`: (Optional) The target timezone for all processed signals (e.g., "UTC", "Europe/London", or "system" to use the local machine's timezone). Defaults to "UTC".
2.  `collection_settings`: (Optional) Configure global settings for the collection.
    *   `index_config`: List of metadata fields for MultiIndex columns in combined *time-series* exports.
    *   `feature_index_config`: List of metadata fields for MultiIndex columns in combined *feature* exports.
    *   `epoch_grid_config`: Dictionary defining the global `window_length` and `step_size` for epoch-based feature extraction.
3.  `import`: Data import specifications. Define how raw data is loaded.
4.  `steps`: Processing operations to apply to signals or the entire collection. Includes alignment, feature extraction, combination, and summarization.
5.  `export`: Output format and location specifications.
6.  `visualization`: Data visualization specifications.

Example workflow demonstrating timezone handling and importer wrapping:

```yaml
# Top-level settings
default_input_timezone: "America/New_York" # Assume naive timestamps are US Eastern
target_timezone: "UTC"                     # Convert all signals to UTC

import:
  - signal_type: "EEG_SLEEP_STAGE"
    importer: "EnchantedWaveImporter"      # Importer for specific device format
    source: "session_data/subject1"
    config:
      # No origin_timezone needed here, EnchantedWaveImporter reads it from file
      # or uses default_input_timezone if file doesn't specify offset.
      filename_pattern: "Session_(?P<session_id>\\d+).csv" # Extract metadata from filename
    base_name: "eeg_stages"

  - signal_type: "heart_rate"
    importer: "MergingImporter"            # Use MergingImporter to combine files
    source: "hr_data/subject1"
    config:
      importer_name: "PolarCSVImporter"    # Wrap Polar importer to handle file format
      file_pattern: "Polar_H10_.*_HR.txt"  # Pattern for files to merge
      # origin_timezone: "Europe/Berlin"   # Override default_input_timezone for this source
      # column_mapping, delimiter etc. are passed to the underlying PolarCSVImporter config
    base_name: "hr_polar" # Signals will be named hr_polar_0, hr_polar_1, etc.

steps:
  # --- Alignment Workflow ---
  # Step 1: Calculate alignment grid parameters for time-series signals
  - type: collection
    operation: "generate_alignment_grid"
    parameters:
      target_sample_rate: 10.0 # Optional: Specify target rate (Hz).

  # Step 2: Apply the calculated alignment grid to all time-series signals in place
  - type: collection
    operation: "apply_grid_alignment"
    parameters:
      method: "nearest" # Method used by the underlying 'reindex_to_grid' signal operation

  # Step 3: Combine the modified (aligned) signals using outer join + reindex
  # The result is stored internally and can be exported using "combined_ts" content type.
  - type: collection
    operation: "combine_aligned_signals"
    parameters: {} # No parameters needed

  # --- Feature Extraction Workflow ---
  # Step 4: Generate the common epoch grid for all feature extraction steps
  # Uses epoch_grid_config from collection_settings (e.g., window="30s", step="15s")
  - type: collection
    operation: "generate_epoch_grid"
    parameters: {} # Optional: start_time, end_time overrides

  # Step 5: Compute basic statistics for Heart Rate signals over epochs
  - type: multi_signal # Feature operations are typically multi-signal
    operation: "feature_statistics"
    inputs: ["hr"] # Use base name to get all HR signals (hr_0, hr_1, ...)
    parameters:
      # window_length: "60s" # Optional: Override global window length for this step
      aggregations: ["mean", "std", "min", "max"]
    output: "hr_features" # Base name for the resulting Feature object(s)

  # Step 6: Compute modal sleep stage over epochs
  - type: multi_signal
    operation: "compute_sleep_stage_mode"
    inputs: ["eeg_stages"] # Use base name for the sleep stage signal(s)
    parameters: {} # No specific parameters needed for mode calculation
    output: "sleep_stage_mode"

  # Step 7: Combine all generated feature objects into a single matrix
  # The result is stored internally and can be exported using "combined_features" content type.
  - type: collection
    operation: "combine_features"
    inputs: ["hr_features", "sleep_stage_mode"] # List of feature keys/base names
    parameters: {} # No parameters needed currently

  # --- Summarization ---
  # Step 8: Generate and store a summary of all signals and features
  # The result is stored internally and can be exported using "summary" content type.
  - type: collection
    operation: "summarize_signals"
    parameters:
      # Optional: Specify exact fields to include in the stored summary DataFrame
      # fields_to_include: ["signal_type", "name", "sample_rate", "data_shape"]
      # Optional: Control printing of the formatted summary to console (default is true)
      print_summary: true

export:
  # Export ONLY the combined aligned time-series data
  - formats: ["csv"]
    output_dir: "results/subject1_analysis/aligned_timeseries"
    content: ["combined_ts"] # Export the result of combine_aligned_signals

  # Export ONLY the final combined feature matrix and the summary table
  - formats: ["csv", "excel"]
    output_dir: "results/subject1_analysis/features_and_summary"
    content: ["combined_features", "summary"] # Export results of combine_features and summarize_signals

  # Export ONLY all individual non-temporary TimeSeriesSignals to CSV
  - formats: ["csv"]
    output_dir: "results/subject1_analysis/individual_timeseries"
    content: ["all_ts"] # Export all signals in collection.time_series_signals

  # Export ONLY all individual Features to Excel
  - formats: ["excel"]
    output_dir: "results/subject1_analysis/individual_features"
    content: ["all_features"] # Export all features in collection.features

  # Export specific signals/features by key
  - formats: ["pickle"]
    output_dir: "results/subject1_analysis/specific_items"
    content: ["hr_polar_0", "sleep_stage_mode_0"] # Use specific keys

  # Example: Export individual TS AND combined features in one task
  - formats: ["csv"]
    output_dir: "results/subject1_analysis/individual_ts_and_combined_features"
    content: ["all_ts", "combined_features"]
```

### Visualization

The framework provides a powerful visualization abstraction layer that supports multiple backends (currently Bokeh and Plotly) and various plot types. Visualizations can be configured in workflow files to automatically generate plots during analysis.

#### Supported Visualization Types

- **Time Series Plots**: Display one or more signals over time. Automatically handles numerical and categorical data (e.g., sleep stages plotted as stepped lines). Supports downsampling via `max_points` or time frequency string (e.g., `'1S'`).
- **Hypnograms**: Specialized visualization for sleep stage data (`EEGSleepStageSignal`), showing stages over time with optional sleep statistics.
- **Scatter Plots**: Compare two signals against each other.
- **Heatmaps**: Visualize 2D data like spectrograms (requires appropriate signal type).
- **Multi-panel Layouts**: Arrange multiple plots in grid, vertical, or horizontal layouts, with options for linked axes.
- **Sleep Stage Overlay**: Add sleep stage background shading to time series plots for context.

#### Visualization Configuration

In your workflow YAML, add a `visualization` section like this:

```yaml
visualization:
  - type: time_series           # Plot type (time_series, scatter, etc.)
    signals: ["hr_0", "hr_1"]   # Signals to visualize
    layout: vertical            # Layout type (vertical, horizontal, grid)
    title: "Heart Rate Comparison"
    output: "results/plots/heart_rate.html"  # Output path
    backend: bokeh              # Visualization backend (bokeh or plotly)
    parameters:                 # Optional styling and config parameters
      width: 1200
      height: 600
      x_label: "Time"
      y_label: "Heart Rate (bpm)"
      line_color: "blue"
      strict: false             # Skip missing signals with warning
  
  - type: scatter
    x_signal: "heart_rate_0"    # Signal for x-axis
    y_signal: "resp_rate_0"     # Signal for y-axis  
    title: "HR vs Respiratory Rate"
    output: "results/plots/hr_vs_rr.html"
    backend: plotly
    parameters:
      marker_size: 10
      marker_color: "red"

  # Create a hypnogram plot for a sleep stage signal
  - type: hypnogram # Specific type for sleep stage visualization
    signals: ["sleep_stage_0"] # Key(s) of the EEGSleepStageSignal(s)
    title: "Sleep Hypnogram - Subject 1"
    output: "results/plots/hypnogram_subject1.html"
    backend: plotly
    parameters:
      width: 1000
      height: 400
      add_statistics: true # Display calculated sleep statistics on the plot
      # Optional: Customize stage colors and order
      # stage_colors: { Awake: 'red', REM: 'purple', ... }
      # stage_order: [ Awake, REM, N1, N2, N3, Unknown ]
```

#### Backend-Specific Features

**Bokeh**:
- Interactive tools: pan, zoom, hover tooltips
- Export formats: HTML, PNG, SVG, PDF
- Synchronized axes when using grid layouts

**Plotly**:
- Modern interactive interface
- Built-in export functionality to PNG/SVG/PDF
- Extensive customization options

#### Running Visualizations

Visualizations are automatically generated when running the workflow:

```bash
python -m sleep_analysis.cli.run_workflow -w workflow.yaml -d ./data -o ./results
```

The visualizer will create the specified plots and save them to the output paths defined in your workflow file.

## Project Structure

- `src/sleep_analysis/core/`: Base classes (`SignalData`, `SignalCollection`), metadata structures (`SignalMetadata`, `CollectionMetadata`), and `MetadataHandler`.
- `src/sleep_analysis/signals/`: Concrete time-series signal type implementations (e.g., `PPGSignal`, `TimeSeriesSignal`, `EEGSleepStageSignal`).
- `src/sleep_analysis/features/`: Feature class definition (`feature.py`).
- `src/sleep_analysis/importers/`: Data import modules (`base.py`, `formats/csv.py`, `sensors/polar.py`, `merging.py`).
- `src/sleep_analysis/operations/`: Signal processing operations (e.g., `filters.py`, `feature_extraction.py`). Operations are registered with `TimeSeriesSignal` or `SignalCollection`.
- `src/sleep_analysis/workflows/`: Workflow execution logic (`WorkflowExecutor`).
- `src/sleep_analysis/visualization/`: Visualization infrastructure (`base.py`, `BokehVisualizer`, `PlotlyVisualizer`).
- `src/sleep_analysis/export/`: Data export module (`ExportModule`).
- `src/sleep_analysis/utils/`: Utility functions (logging, `standardize_timestamp`, enum conversion).
- `src/sleep_analysis/cli/`: Command-line interface (`run_workflow.py`).

## Advanced Usage

### Timestamp Handling and Timezones

The framework employs a robust strategy for handling timestamps and timezones:

1.  **Internal Representation**: All `TimeSeriesSignal` objects internally use timezone-aware `pandas.DatetimeIndex`.
2.  **Target Timezone**: The `target_timezone` setting in the workflow (or the default "UTC") defines the timezone all signals will be converted to during import or processing. You can use specific timezones (e.g., "Europe/Paris") or "system" to use the local machine's timezone.
3.  **Input Timezones**:
    *   **Aware Sources**: If source data includes timezone information (e.g., ISO 8601 format with offset), it's used directly.
    *   **Naive Sources**: If source data has naive timestamps (no timezone info), the framework needs guidance:
        *   `origin_timezone`: Specify this in the importer's `config` section within the workflow to declare the source's local timezone (e.g., `origin_timezone: "America/Denver"`).
        *   `default_input_timezone`: Set this at the top level of the workflow as a fallback if an importer doesn't specify `origin_timezone`.
        *   **Ambiguity**: If timestamps are naive and neither `origin_timezone` nor `default_input_timezone` is provided, the framework will likely assume UTC, which might lead to incorrect alignment if the data originated elsewhere. A warning will be issued.
4.  **Standardization**: A central `standardize_timestamp` utility (`utils/__init__.py`) handles parsing, localization of naive timestamps (using `origin_timezone`), and conversion to the `target_timezone`.
5.  **Export**: Timestamps are formatted appropriately for export (e.g., timezone removed for Excel, ISO format with offset for CSV/JSON).

### Signal Alignment and Combined Time-Series DataFrames

The framework provides flexible options for aligning time-series signals to a common time grid and generating a combined DataFrame suitable for export or further analysis. Alignment parameters are calculated once and stored on the `SignalCollection`. Two main combination strategies are available:

1.  **In-Place Alignment + Combination**:
    *   **`generate_alignment_grid`**: (Collection Operation) Calculates the common `target_rate`, `ref_time`, and `grid_index` based on all time-series signals. Stores these parameters on the collection instance but does *not* modify the signals.
    *   **`apply_grid_alignment`**: (Collection Operation) Modifies signals *in place*. It iterates through specified (or all) `TimeSeriesSignal` objects and calls their `apply_operation('reindex_to_grid', inplace=True, ...)` method, passing the stored `grid_index` and an alignment `method` (e.g., 'nearest'). This snaps the signal's data points to the grid, potentially making the signal's index sparse if points don't align perfectly or if NaN rows are dropped by `reindex_to_grid`. The operation is recorded in each signal's metadata.
    *   **`combine_aligned_signals`**: (Collection Operation) Assumes signals have been modified by `apply_grid_alignment`. It performs an `outer join` on the (potentially sparse) data of the modified signals and then `reindex`es the result to the full `grid_index` calculated earlier. This creates the final combined DataFrame with NaNs where signals didn't have data at a specific grid point. The result is stored internally in the collection.

2.  **`merge_asof` Alignment + Combination**:
    *   **`generate_alignment_grid`**: (Collection Operation) Same as above, calculates and stores alignment parameters.
    *   **`align_and_combine_signals`**: (Collection Operation) Aligns signals using `pandas.merge_asof`. For each signal, it merges its *original* data with the `grid_index` using `direction='nearest'` and a calculated `tolerance` (half the grid period). This finds the closest original data point for each grid timestamp within the tolerance window. The results for all signals are then concatenated. This method does *not* modify the original signals in place. The result is stored internally in the collection.

**Choosing a Strategy**:

*   Use **In-Place Alignment + Combination** (`apply_grid_alignment` + `combine_aligned_signals`) if you need the individual signals to be modified to conform strictly to the grid *before* combining, or if you need to access these modified intermediate signals.
*   Use **`merge_asof` Alignment + Combination** (`align_and_combine_signals`) if you want to preserve the original signal data and perform alignment only during the combination step, finding the nearest original value for each grid point. This is often preferred for preserving original measurements.

**Accessing the Result**:

*   Both combination strategies store the resulting combined DataFrame internally within the `SignalCollection`.
*   This stored DataFrame is automatically used by the `ExportModule` when `include_combined: true` is set in the `export` section of the workflow.
*   Programmatically, you can access it via `collection.get_stored_combined_dataframe()`.
*   The parameters used for generation can be accessed via `collection.get_stored_combination_params()`.

**Combined DataFrame Structure**:

*   The resulting DataFrame will have the calculated `grid_index` as its index.
*   Columns can be simple (`key_colname`) or a `pandas.MultiIndex` if `index_config` is set in the `collection_settings` section of the workflow (e.g., `index_config: ["signal_type", "name"]`).

### Feature Extraction

The framework supports epoch-based feature extraction from `TimeSeriesSignal` objects.

1.  **Global Epoch Grid**: A common grid of epoch start times is defined for the entire collection using the `generate_epoch_grid` collection operation. This operation uses the `epoch_grid_config` (defining `window_length` and `step_size`) from the `collection_settings` in the workflow YAML and the overall time range of the signals. The resulting `epoch_grid_index` is stored on the `SignalCollection`.
2.  **Feature Operations**: Functions like `feature_statistics` or `compute_sleep_stage_mode` (defined in `operations/feature_extraction.py`) calculate features for each epoch defined by the `epoch_grid_index`. These operations are registered in the `SignalCollection.multi_signal_registry`.
3.  **Workflow Integration**: Feature extraction is invoked in the workflow `steps` section using `type: multi_signal`. The operation takes a list of input signal keys/base names and produces a `Feature` object.
    *   The `window_length` parameter can optionally be provided in the step's `parameters` to override the global setting for that specific feature calculation.
    *   The `step_size` is always determined by the global `epoch_grid_index`.
4.  **`Feature` Class**: The results are stored in `Feature` objects (`features/feature.py`). These objects contain the feature data (a DataFrame indexed by epoch start time) and `FeatureMetadata` (including epoch parameters, source signal IDs, and feature names).

### Feature Combination

Multiple `Feature` objects (e.g., statistics from different signal types) can be combined into a single feature matrix.

1.  **`combine_features` Operation**: This collection operation takes a list of input `Feature` keys/base names.
2.  **Validation**: It ensures all input features share the same `epoch_grid_index` as the collection.
3.  **Concatenation**: The data from the input features is concatenated column-wise.
4.  **MultiIndex Columns**: A `pandas.MultiIndex` is created for the columns based on the `feature_index_config` specified in the `collection_settings` (or overridden in the step parameters). This allows organizing columns by feature set, source signal, and feature name.
5.  **Storage**: The resulting combined feature matrix is stored internally in the `SignalCollection` (`_combined_feature_matrix`) and can be exported using the `"combined_features"` content type in the `export` section.

### Signal Summarization

A summary table of all signals (`TimeSeriesSignal`) and features (`Feature`) in the collection can be generated.

1.  **`summarize_signals` Operation**: This collection operation gathers metadata and basic information (like data shape) for all items.
2.  **Parameters**:
    *   `fields_to_include`: (Optional) List of specific metadata fields to include in the summary. Defaults to a comprehensive list.
    *   `print_summary`: (Optional) Boolean flag (default `True`) to control whether the formatted summary is printed to the console.
3.  **Storage**: The raw summary data is stored internally as a DataFrame in the `SignalCollection` (`_summary_dataframe`) and can be exported using the `"summary"` content type in the `export` section.

### Metadata Management

-   **`TimeSeriesMetadata`, `FeatureMetadata`, `CollectionMetadata`**: Dataclasses defined in `core/metadata.py` store structured information about individual time-series signals, features (including epoch parameters), and the overall collection.
-   **`MetadataHandler`**: A helper class (`core/metadata_handler.py`) ensures consistent creation and updating of metadata, handling defaults and unique ID generation. Each `TimeSeriesSignal` and `Feature` instance holds a reference to a handler.
-   **Traceability**: Operations applied are recorded in the `operations` list within the respective metadata object. Source signal/feature IDs are tracked (`source_signal_ids`, `derived_from`).

### Importers

-   **Hierarchy**: Importers inherit from `SignalImporter` (`importers/base.py`). Format-specific bases like `CSVImporterBase` (`importers/formats/csv.py`) provide common logic. Concrete importers like `PolarCSVImporter` or `EnchantedWaveImporter` (`importers/sensors/`) handle specific device formats.
-   **Configuration**: Importers are typically configured via the `config` section in the workflow `import` step. This allows specifying column mappings, time formats, delimiters, etc.
-   **`MergingImporter`**: A specialized importer (`importers/merging.py`) that can wrap another importer (specified via `importer_name` in its config). It finds files matching a `file_pattern`, uses the underlying importer to read each file, and then merges the resulting data into a single `SignalData` instance. This is useful for data fragmented across multiple files.
-   **Timestamp Handling**: Importers use the centralized `standardize_timestamp` utility (`utils/__init__.py`) to parse timestamps, handle naive timestamps using `origin_timezone` and `default_input_timezone`, and convert everything to the `target_timezone`.

### Extensibility

-   **Adding New Signal Types**:
    1.  Define a new `Enum` value in `SignalType` (`signal_types.py`).
    2.  Create a new class inheriting from `SignalData` or `TimeSeriesSignal` (`signals/`).
    3.  Set the `signal_type` class attribute to the new enum value.
    4.  Define `required_columns` for the signal.
    5.  Implement any specific methods needed for this signal type.
-   **Adding New Operations**:
    1.  Define the core logic as a Python function (e.g., in `operations/filters.py`).
    2.  Register the function with the `TimeSeriesSignal` class using the `@TimeSeriesSignal.register("operation_name", output_class=...)` decorator. The `output_class` specifies the type of signal the operation produces (defaults to `TimeSeriesSignal`).
    3.  The operation can then be invoked via `signal.apply_operation("operation_name", ...)` or used in workflow `steps` with `type: signal`.
-   **Multi-Signal Operations (Feature Extraction)**: To add operations that take multiple `TimeSeriesSignal` inputs and produce a `Feature` output:
    1.  Define the function (e.g., in `operations/feature_extraction.py`). It should accept `signals: List[TimeSeriesSignal]`, `epoch_grid_index: pd.DatetimeIndex`, `parameters: Dict[str, Any]`, `global_window_length: pd.Timedelta`, and `global_step_size: pd.Timedelta`. It should return a `Feature` instance.
    2.  Register the function and the output type (`Feature`) in the `SignalCollection.multi_signal_registry` dictionary (at the bottom of `core/signal_collection.py`).
    3.  The operation can then be invoked via `collection.apply_multi_signal_operation(...)` or used in workflow `steps` with `type: multi_signal`.
-   **Collection Operations**: To add operations that act on the `SignalCollection` itself (e.g., alignment, combination, summarization):
    1.  Implement the logic as a method within the `SignalCollection` class.
    2.  Decorate the method with `@register_collection_operation("operation_name")`.
    3.  The operation can then be invoked via `collection.apply_operation("operation_name", ...)` or used in workflow `steps` with `type: collection`.

### Visualization Configuration Options

#### Common Options for All Plot Types
- `title`: Main plot title
- `width`, `height`: Dimensions in pixels
- `output`: Output file path
- `format`: Output format (html, png, svg, pdf)
- `backend`: Visualization backend ("bokeh" or "plotly")

#### Time Series Plots
```yaml
- type: time_series
  signals: ["signal_key1", "signal_key2"]  # Signal keys or base names
  layout: "vertical"  # vertical, horizontal, grid, or overlay
  parameters:
    link_x_axes: true  # Synchronize x-axes across subplots
    time_range: ["2023-01-01 00:00:00", "2023-01-02 00:00:00"]  # Optional time restriction
    max_points: 10000  # Downsample for better performance
    line_width: 2
    line_color: "blue"  # Only applies if not using multiple signals
    downsample: 10000 # Optional: Max points per plot
    # downsample: '1S' # Optional: Resample frequency string
```

#### Scatter Plots
```yaml
- type: scatter
  x_signal: "heart_rate"  # Signal for x-axis
  y_signal: "resp_rate"   # Signal for y-axis
  parameters:
    marker_size: 8
    marker_symbol: "circle"  # circle, square, etc.
    fill_color: "blue"
    opacity: 0.7
```

#### Hypnogram Plots
```yaml
- type: hypnogram
  signal: "sleep_stage_signal_key" # Key of the EEGSleepStageSignal
  parameters:
    add_statistics: true # Show sleep stats (Total time, time in stage, %)
    stage_colors: { Awake: '#FF5733', REM: '#AA33FF', ... } # Optional color overrides
    stage_order: [ Awake, REM, N1, N2, N3, Unknown ] # Optional y-axis order
```

#### Categorical Data (e.g., Sleep Stages) in Time Series Plots
- Time series plots automatically handle categorical data (like `EEGSleepStageSignal`) by default:
  - Plotting as a stepped line (`line_shape='hv'` in Plotly).
  - Mapping categories to numerical values for plotting.
  - Setting appropriate y-axis labels (e.g., "Stage") and tick marks based on category names.
- You can customize the appearance using `category_map` in the `parameters` section to define specific colors for each category, although the hypnogram plot type is generally preferred for dedicated sleep stage visualization.

#### Grid Layouts
```yaml
- type: time_series
  signals: ["signal1", "signal2", "signal3", "signal4"]
  layout: "grid"  # Will create a 2x2 grid automatically
  parameters:
    subplot_titles: ["Signal 1", "Signal 2", "Signal 3", "Signal 4"]
    link_x_axes: true  # Synchronize time axes

  # Time series plot with sleep stage background overlay (Bokeh example)
  - type: time_series
    signals: ["hr_0", "accel_magnitude_0"] # Signals to plot
    layout: vertical
    title: "Signals with Sleep Stage Background"
    output: "results/plots/signals_with_overlay.html"
    backend: bokeh
    parameters:
      link_x_axes: true
      overlay_sleep_stages: "sleep_stage_0" # Key of the EEGSleepStageSignal for background
      alpha: 0.15 # Opacity of the background regions
      # Optional: Customize background stage colors
      # stage_colors: { Awake: 'rgba(255, 0, 0, 0.5)', REM: 'rgba(0, 0, 255, 0.5)', ... }
```

### Programmatic API

If you want to create visualizations from Python code:

```python
from sleep_analysis.visualization import BokehVisualizer
from sleep_analysis.core.signal_collection import SignalCollection

# Initialize visualizer and collection
collection = SignalCollection()
visualizer = BokehVisualizer()

# Create time series plot for a single signal
signal = collection.get_signal("hr_0")
figure = visualizer.create_time_series_plot(signal, title="Heart Rate")
visualizer.save(figure, "heart_rate.html")

# Create a multi-signal dashboard
dashboard = visualizer.visualize_collection(
    collection,
    signals=["hr_0", "accel_magnitude_0"],
    layout="vertical"
)
visualizer.save(dashboard, "dashboard.html")
```

## Contributing and Development

This section provides guidance for developers looking to contribute to or modify the framework.

### Coding Guidelines

All contributions must adhere to the rules outlined in `docs/coding_guidelines.md`. Please review this document before making changes to ensure consistency and maintainability. Key principles include favoring declarative design, using common utilities, and respecting signal encapsulation.

### Adding Operations

-   **Signal Operations**: To add a new processing step for a specific signal type (e.g., `TimeSeriesSignal`, `PPGSignal`):
    1.  Define the core logic as a Python function or an instance method within the signal class.
    2.  **Instance Methods**: If implementing as a method (preferred for core functionality), ensure it handles `inplace` logic and returns the correct signal instance. The `apply_operation` method will find and call it directly.
    3.  **Registered Functions**: If implementing as a standalone function (useful for optional or plugin-like operations):
        *   The function should typically accept `data_list: List[pd.DataFrame]` and `parameters: Dict[str, Any]` and return a `pd.DataFrame`.
        *   Register it with the appropriate signal class using the `@SignalClass.register("operation_name", output_class=...)` decorator. `output_class` specifies the type of `SignalData` the operation produces.
    4.  The operation can then be invoked via `signal.apply_operation("operation_name", ...)` or used in workflow `steps`.
-   **Collection Operations**: To add operations that act on the `SignalCollection` itself (e.g., alignment, combination):
    1.  Implement the logic as a method within the `SignalCollection` class.
    2.  Decorate the method with `@register_collection_operation("operation_name")`.
    3.  The operation can then be invoked via `collection.apply_operation("operation_name", ...)` or used in workflow `steps` with `type: collection`.

### Metadata Management

-   The framework automatically manages metadata updates when operations are applied correctly through the `apply_operation` methods of `SignalData` and `SignalCollection`.
-   `MetadataHandler` ensures consistent metadata creation and updates.
-   `SignalMetadata.operations` tracks the history of applied operations.
-   `SignalMetadata.derived_from` links new signals created by non-inplace operations back to their source signals.
-   Directly modifying `signal._data` bypasses this crucial metadata tracking and should be avoided. Refer to Rule 6 and Rule 7 in the coding guidelines.

### Running Tests

Ensure all tests pass before submitting changes. Run the test suite using `pytest` from the project root directory:

```bash
pytest
```
