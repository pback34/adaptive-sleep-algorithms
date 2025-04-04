# Sleep Analysis Framework

A flexible, extensible framework for processing sleep-related signals, designed for researchers and developers working with physiological data. The framework provides a robust foundation for signal processing with an emphasis on reproducibility, type safety, and memory efficiency.

## Key Features

- **Type-Safe Signal Processing**: Enum-based type safety ensures operations match signal types.
- **Complete Traceability**: Full metadata and operation history for reproducibility.
- **Memory Optimization**: Smart memory management for processing large datasets efficiently.
- **Flexible Workflows**: Support for both structured workflows and ad-hoc processing.
- **Extensible Design**: Easy to add new signal types and processing operations.
- **Import Flexibility**: Convert signals from various sources to a standardized format.
- **Modular Importer System**: Easily add support for new file formats or sensor types, including handling fragmented data files via `MergingImporter`.
- **Robust Timestamp Handling**: Consistent management of timezones across import, processing, and export.
- **Interactive Visualization**: Backend-agnostic visualization layer with support for Bokeh and Plotly, including specialized plots like hypnograms.

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
2.  `import`: Data import specifications. Define how raw data is loaded.
3.  `steps`: Processing operations to apply to signals or the entire collection.
4.  `export`: Output format and location.
5.  `visualization`: Data visualization specifications.

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
  # Align all time-series signals to a common grid (calculates alignment parameters)
  - type: collection
    operation: align_signals
    # parameters:
    #   target_sample_rate: 100 # Optional: force a specific rate

  # Apply a filter to a specific signal
  - type: signal
    input: "hr_polar_merged_0"
    operation: "filter_lowpass"
    parameters:
      cutoff: 0.5 # Parameter name might vary based on operation
    output: "hr_polar_filtered"

export:
  formats: ["csv", "excel"]
  output_dir: "results/subject1_analysis"
  include_combined: true # Export a combined dataframe of all non-temporary signals
  index_config: ["name", "signal_type"] # Configure MultiIndex for combined export
```

### Visualization

The framework provides a powerful visualization abstraction layer that supports multiple backends (currently Bokeh and Plotly) and various plot types. Visualizations can be configured in workflow files to automatically generate plots during analysis.

#### Supported Visualization Types

- **Time Series Plots**: Display one or more signals over time. Automatically handles numerical and categorical data (e.g., sleep stages plotted as stepped lines). Supports downsampling via `max_points` or time frequency string (e.g., `'1S'`).
- **Hypnograms**: Specialized visualization for sleep stage data (`EEGSleepStageSignal`), showing stages over time with optional sleep statistics.
- **Scatter Plots**: Compare two signals against each other.
- **Heatmaps**: Visualize 2D data like spectrograms.
- **Multi-panel Layouts**: Arrange multiple plots in grid, vertical, or horizontal layouts, with options for linked axes.

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
    signal: "sleep_stage_0" # Key of the EEGSleepStageSignal
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
- `src/sleep_analysis/signals/`: Concrete signal type implementations (e.g., `PPGSignal`, `TimeSeriesSignal`, `EEGSleepStageSignal`).
- `src/sleep_analysis/importers/`: Data import modules (`base.py`, `formats/csv.py`, `sensors/polar.py`, `merging.py`).
- `src/sleep_analysis/operations/`: Signal processing operations (registered with signal classes via `@SignalClass.register`).
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
4.  **Standardization**: A central `standardize_timestamp` utility handles parsing, localization of naive timestamps (using `origin_timezone`), and conversion to the `target_timezone`.
5.  **Export**: Timestamps are formatted appropriately for export (e.g., timezone removed for Excel, ISO format with offset for CSV/JSON).

### Signal Alignment and Combined DataFrames

-   **`align_signals` Step**: This collection-level operation (run via the `steps` section) **calculates** alignment parameters (`target_rate`, `ref_time`, `grid_index`) based on all time-series signals present. It determines a common sampling grid but **does not modify** the signals themselves. The calculated parameters are stored on the `SignalCollection` instance.
-   **`apply_grid_alignment` Step**: This collection-level operation **applies** the previously calculated grid alignment to the specified signals (or all time-series signals if none specified) **in place**. It modifies the internal data of each `TimeSeriesSignal` by calling its `reindex_to_grid` operation, which uses the collection's `grid_index` and a specified method (e.g., 'nearest', 'ffill'). The operation is recorded in each signal's metadata.
-   **`generate_and_store_aligned_dataframe` Step**: This collection-level operation generates the combined, aligned dataframe using the parameters from `align_signals` and **stores it persistently** within the `SignalCollection` instance. This avoids recalculating the potentially large dataframe multiple times (e.g., for export and visualization). The stored dataframe can be accessed programmatically via `collection.get_stored_aligned_dataframe()`.
-   **`get_combined_dataframe()`**: This method generates and returns the combined dataframe **on the fly** each time it's called. It uses the alignment parameters from `align_signals` if available. This is used internally by the `ExportModule` when `include_combined: true` if the dataframe hasn't been pre-generated and stored.
    *   Alignment uses `pandas.merge_asof` with `direction='nearest'` and a tolerance (half the grid period, if a regular grid exists) to align each signal's data points to the common grid index. This method avoids simple resampling and preserves original values by finding the nearest data point within the tolerance window.
    *   The resulting DataFrame can have simple columns (`key_colname`) or a `pandas.MultiIndex` if `index_config` is set in the `export` section.

### Metadata Management

-   **`SignalMetadata` & `CollectionMetadata`**: Dataclasses defined in `core/metadata.py` store structured information about individual signals (type, sensor, operations, source files) and the overall collection (subject, session, timezone).
-   **`MetadataHandler`**: A helper class (`core/metadata_handler.py`) ensures consistent creation and updating of `SignalMetadata`, handling defaults and unique ID generation. Each `SignalData` instance holds a reference to a handler.
-   **Traceability**: Operations applied to signals are recorded in `SignalMetadata.operations`, providing a history for reproducibility. The `derived_from` field links signals to their parent signals and the operation that created them.

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
    1.  Define a function that takes a list of DataFrames (`data_list`) and a parameters dictionary (`parameters`) and returns the resulting DataFrame.
    2.  Register the function with the relevant signal class using the `@SignalClass.register("operation_name", output_class=...)` decorator. The `output_class` specifies the type of signal the operation produces (defaults to the class it's registered with).
    3.  The operation can then be called using `signal.apply_operation("operation_name", ...)` or specified in the `steps` section of a workflow. Example: `reindex_to_grid` is registered on `TimeSeriesSignal`.

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
