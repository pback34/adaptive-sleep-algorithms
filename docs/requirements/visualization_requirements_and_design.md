# Visualization Abstraction Layer: Requirements and Design Specification

## 1. Introduction

This document outlines the requirements and design for a visualization abstraction layer within the Flexible Signal Processing Framework for Sleep Analysis. The layer provides a declarative, flexible, and extensible system for generating interactive and static visualizations of physiological signals, leveraging Python’s Abstract Base Classes (ABC) to define a common interface. It initially supports Bokeh and Plotly, integrating seamlessly with the framework’s `SignalCollection` and `WorkflowExecutor` for visualizing signals in workflows or ad-hoc scripts. Key outputs include interactive HTML files, with support for static and vector formats.

---

## 2. Requirements

### 2.1 Core Visualization Capabilities
- **FR-V1: Plot Types**
  - Support **time-series plots** (default: one subplot per signal), **scatter plots** (e.g., heart rate vs. respiratory rate), **line plots**, **heatmaps**, and **spectrograms**.
  - Allow future extensibility for additional types (e.g., histograms, 3D plots).
- **FR-V2: Layout and Customization**
  - Enable **multi-panel layouts** (vertical, horizontal, grid) and **signal overlays** with legends.
  - Provide **time synchronization** across plots for aligned views.
  - Use signal metadata (e.g., `name`, `units`) for default titles, axis labels, and legends, with user overrides.
  - Support customization of **colors, line styles, markers, and labels**.
- **FR-V3: Interactivity**
  - Include **zooming, panning, and hover tooltips**.
  - Enable **highlighting events or regions** (e.g., sleep stages) via lines or shaded areas.

### 2.2 Export and Output Formats
- **FR-V4: Output Options**
  - Generate **interactive HTML files** for web viewing.
  - Support **static images** (PNG, JPG) and **vector formats** (SVG, PDF) for reports/publications.
  - Enable embedding in **Jupyter notebooks** for interactive analysis.

### 2.3 Integration with Framework
- **FR-V5: Signal Integration**
  - Retrieve signals from `SignalCollection` using keys (e.g., `"ppg_raw"`) or metadata (e.g., `{"signal_type": "PPG"}`).
  - Handle **raw, processed, and derived signals** with DataFrames from `SignalData.get_data()`.
  - Visualize **signal segments** by timestamps or indices.
- **FR-V6: Workflow Integration**
  - Process a `visualization` section in workflow YAML via `WorkflowExecutor`.

### 2.4 Declarative Configuration
- **FR-V7: Configuration**
  - Support declarative specification via YAML/JSON or Python dictionaries, including plot type, signals, layout, and output.
  - Provide **parameter templates** for consistent styling.
  - Enable programmatic configuration with equivalent functionality.

### 2.5 Non-Functional Requirements
- **NFR-V1: Abstraction**: Use Python’s ABC for a backend-agnostic interface.
- **NFR-V2: Extensibility**: Allow new libraries and plot types without core changes.
- **NFR-V3: Performance**: Handle large datasets efficiently with configurable downsampling (e.g., `max_points`).
- **NFR-V4: Headless Operation**: Support server-side execution without a GUI.
- **NFR-V5: Modularity**: Ensure easy extension of backends and features.
- **NFR-V6: Usability**: Align API and configuration with the framework’s declarative style.
- **NFR-V7: Memory Efficiency**: Optimize for large time-series datasets.
- **NFR-V8: Consistent Styling**: Maintain uniform visuals across backends.
- **NFR-V9: Documentation**: Provide comprehensive guides for options and parameters.

---

## 3. Design

### 3.1 Architecture

The visualization layer uses a `VisualizerBase` ABC, with concrete implementations (`BokehVisualizer`, `PlotlyVisualizer`) handling library-specific plotting. It integrates with `SignalCollection` for data access and `WorkflowExecutor` for workflow-driven visualization.

#### 3.1.1 Class Hierarchy
```
VisualizerBase (ABC)
├── BokehVisualizer
└── PlotlyVisualizer
```

#### 3.1.2 Architecture Diagram
```
+-------------------+       +-------------------+
|   SignalCollection| ← →   |  VisualizerBase   |
|   (holds signals) |       |  (abstract class) |
+-------------------+       +-------------------+
          ↑                         ↑
          |                         |
          |                +-------------------------+
          |                |       Concrete          |
          |                |    Visualizers          |
          |                |  (Bokeh, Plotly, etc.)  |
          |                +-------------------------+
          |                         ↑
          |                         |
          |                +-------------------------+
          |                |   WorkflowExecutor      |
          +-------------→  | (processes visualization|
                           |   configs and executes  |
                           |    visualizer methods)  |
                           +-------------------------+
```

- **SignalCollection**: Provides signal data.
- **VisualizerBase**: Defines the abstract interface.
- **Concrete Visualizers**: Implement plotting using specific libraries.
- **WorkflowExecutor**: Orchestrates visualization via configurations.

### 3.2 Abstract Visualizer Class (`VisualizerBase`)

**Purpose**: Defines a unified interface for visualization backends.

**Definition**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..core.signal_collection import SignalCollection
from ..core.signal_data import SignalData

class VisualizerBase(ABC):
    @abstractmethod
    def create_figure(self, **kwargs) -> Any:
        """Create a new figure/plot container."""
        pass

    @abstractmethod
    def add_line_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """Add a line plot to an existing figure."""
        pass

    @abstractmethod
    def add_scatter_plot(self, figure: Any, x: Any, y: Any, **kwargs) -> Any:
        """Add a scatter plot to an existing figure."""
        pass

    @abstractmethod
    def add_heatmap(self, figure: Any, data: Any, **kwargs) -> Any:
        """Add a heatmap to an existing figure."""
        pass

    @abstractmethod
    def create_grid_layout(self, figures: List[Any], rows: int, cols: int, **kwargs) -> Any:
        """Create a grid layout of multiple figures."""
        pass

    @abstractmethod
    def add_legend(self, figure: Any, **kwargs) -> None:
        """Add a legend to a figure."""
        pass

    @abstractmethod
    def set_title(self, figure: Any, title: str, **kwargs) -> None:
        """Set the title of a figure."""
        pass

    @abstractmethod
    def set_axis_labels(self, figure: Any, x_label: Optional[str] = None, y_label: Optional[str] = None, **kwargs) -> None:
        """Set axis labels for a figure."""
        pass

    @abstractmethod
    def set_axis_limits(self, figure: Any, x_min: Optional[Any] = None, x_max: Optional[Any] = None, y_min: Optional[Any] = None, y_max: Optional[Any] = None) -> None:
        """Set axis limits for a figure."""
        pass

    @abstractmethod
    def add_hover_tooltip(self, figure: Any, tooltips: Any, **kwargs) -> None:
        """Add hover tooltips to a figure."""
        pass

    @abstractmethod
    def add_vertical_line(self, figure: Any, x: Any, **kwargs) -> Any:
        """Add a vertical line to a figure."""
        pass

    @abstractmethod
    def add_horizontal_line(self, figure: Any, y: Any, **kwargs) -> Any:
        """Add a horizontal line to a figure."""
        pass

    @abstractmethod
    def add_region(self, figure: Any, x_start: Any, x_end: Any, **kwargs) -> Any:
        """Add a highlighted region to a figure."""
        pass

    @abstractmethod
    def save(self, figure: Any, filename: str, format: str = "html", **kwargs) -> None:
        """Save a figure to a file."""
        pass

    @abstractmethod
    def show(self, figure: Any) -> None:
        """Display a figure."""
        pass

    def visualize_signal(self, signal: SignalData, **kwargs) -> Any:
        """Visualize a single signal."""
        figure = self.create_figure(**kwargs)
        data = signal.get_data()
        self.add_line_plot(figure, data.index, data.iloc[:, 0], **kwargs)
        return figure

    def visualize_collection(self, collection: SignalCollection, signals: Optional[List[str]] = None, layout: Optional[str] = None, **kwargs) -> Any:
        """Visualize multiple signals from a collection."""
        if signals is None:
            signals = [s.metadata.signal_id for s in collection.signals]
        figures = [self.visualize_signal(collection.get_signal(s), **kwargs) for s in signals]
        if layout == "vertical":
            return self.create_grid_layout(figures, len(figures), 1)
        elif layout == "horizontal":
            return self.create_grid_layout(figures, 1, len(figures))
        else:
            return figures
```

### 3.3 Concrete Visualizer Implementations
- **BokehVisualizer**: Uses Bokeh for interactive HTML output and grid layouts via `gridplot`.
- **PlotlyVisualizer**: Uses Plotly for HTML output and subplots via `make_subplots`.

Both support subplot layouts, metadata-driven labels, and event highlighting.

### 3.4 Configuration Schema

Example YAML configuration:
```yaml
visualization:
  - type: time_series
    signals: ["ppg_raw", "ppg_filtered"]
    layout: vertical
    title: "PPG Signals"
    output: "plots/ppg.html"
    downsample: true
    max_points: 10000
  - type: scatter
    x_signal: "heart_rate"
    y_signal: "respiratory_rate"
    output: "plots/hr_vs_rr.png"
    format: "png"
```

- **type**: Plot type (e.g., `time_series`, `scatter`).
- **signals**: List of signal keys for time-series.
- **x_signal, y_signal**: Keys for scatter plots.
- **layout**: Arrangement (e.g., `vertical`, `grid`).
- **output**: File path with format inferred from extension if unspecified.

### 3.5 Workflow Integration

The `WorkflowExecutor` processes visualization steps:
```yaml
steps:
  - operation: "visualize_signal"
    input: "ppg_filtered"
    output: "ppg_plot.html"
    parameters:
      title: "Filtered PPG"
      color: "blue"
  - operation: "visualize_multiple"
    inputs: ["ppg_raw", "ppg_filtered"]
    output: "comparison.html"
    parameters:
      layout: "overlay"
```

### 3.6 Usage Examples

#### Programmatic Usage
```python
from sleep_analysis.visualization import BokehVisualizer
visualizer = BokehVisualizer()
signal = collection.get_signal("ppg_filtered")
figure = visualizer.create_figure(title="PPG Signal")
data = signal.get_data()
visualizer.add_line_plot(figure, data.index, data["value"], color="blue")
visualizer.save(figure, "ppg.html")
```

#### Multi-Signal Visualization
```python
dashboard = visualizer.visualize_collection(
    collection,
    signals=["ppg_raw", "heart_rate"],
    layout="vertical"
)
visualizer.save(dashboard, "dashboard.html")
```

---

## 4. Testing Strategy

1. **Unit Tests**:
   - Verify `BokehVisualizer` and `PlotlyVisualizer` implement all abstract methods.
   - Test configuration parsing and template application.
2. **Integration Tests**:
   - Validate signal retrieval from `SignalCollection`.
   - Ensure `WorkflowExecutor` processes visualization steps correctly.
3. **Visual Tests**:
   - Compare generated outputs against reference visualizations.

---

## 5. Future Extensions

- **Additional Backends**: Add Matplotlib or Altair support.
- **Advanced Features**: Implement 3D plots or interactive annotations.
- **Web Integration**: Support real-time dashboards with Dash or Panel.

---
