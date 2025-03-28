# Sleep Analysis Framework

A flexible, extensible framework for processing sleep-related signals, designed for researchers and developers working with physiological data. The framework provides a robust foundation for signal processing with an emphasis on reproducibility, type safety, and memory efficiency.

## Key Features

- **Type-Safe Signal Processing**: Enum-based type safety ensures operations match signal types
- **Complete Traceability**: Full metadata and operation history for reproducibility
- **Memory Optimization**: Smart memory management for processing large datasets efficiently
- **Flexible Workflows**: Support for both structured workflows and ad-hoc processing
- **Extensible Design**: Easy to add new signal types and processing operations
- **Import Flexibility**: Convert signals from various sources to a standardized format
- **Interactive Visualization**: Backend-agnostic visualization layer with support for Bokeh and Plotly

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

Workflow files are YAML documents with four main sections:
1. `import` - Data import specifications
2. `steps` - Processing operations to apply
3. `export` - Output format and location
4. `visualization` - Data visualization specifications

Example of basic workflow:
```yaml
import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      timestamp_col: "Phone timestamp"
    base_name: "hr_h10_merged"

steps:
  - type: signal
    input: "hr_h10_merged_0"
    operation: "filter_lowpass"
    parameters:
      cutoff_frequency: 0.5
    output: "hr_h10_filtered"

export:
  formats: ["csv"]
  output_dir: "results/polar_data"
  include_combined: true
```

### Visualization

The framework provides a powerful visualization abstraction layer that supports multiple backends (currently Bokeh and Plotly) and various plot types. Visualizations can be configured in workflow files to automatically generate plots during analysis.

#### Supported Visualization Types

- **Time Series Plots**: Display one or more signals over time
- **Scatter Plots**: Compare two signals against each other
- **Heatmaps**: Visualize 2D data like spectrograms
- **Multi-panel Layouts**: Arrange multiple plots in grid, vertical, or horizontal layouts

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

- `src/sleep_analysis/core/`: Base classes and metadata structures
- `src/sleep_analysis/signals/`: Signal type implementations
- `src/sleep_analysis/importers/`: Data import modules
- `src/sleep_analysis/operations/`: Signal processing operations
- `src/sleep_analysis/workflows/`: Workflow execution
- `src/sleep_analysis/visualization/`: Visualization infrastructure
  - `base.py`: Base abstract class defining visualization interface
  - `bokeh_visualizer.py`: Bokeh implementation
  - `plotly_visualizer.py`: Plotly implementation
- `src/sleep_analysis/utils/`: Utility functions

## Advanced Usage

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
