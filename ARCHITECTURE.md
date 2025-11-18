# Sleep Analysis Framework - Architecture

This document provides an overview of the framework's architecture, design patterns, and key components.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Design Patterns](#design-patterns)
5. [Signal Processing Pipeline](#signal-processing-pipeline)
6. [Service-Based Architecture](#service-based-architecture)
7. [Extensibility Points](#extensibility-points)

## Overview

The Sleep Analysis Framework follows a **service-oriented architecture** with a clear separation of concerns. The framework is built around several key principles:

- **Type Safety**: Enum-based signal types ensure type-safe operations
- **Metadata Traceability**: Complete operation history for reproducibility
- **Declarative Configuration**: YAML-based workflow definitions
- **Service Composition**: Modular services with single responsibilities
- **Extensibility**: Plugin-like architecture for new signal types and operations

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
│  ┌──────────────────┐              ┌──────────────────────────┐    │
│  │   CLI Interface  │              │   Python API             │    │
│  │  run_workflow.py │              │  SignalCollection        │    │
│  └────────┬─────────┘              └────────────┬─────────────┘    │
└───────────┼──────────────────────────────────────┼──────────────────┘
            │                                      │
            ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Workflow Execution Layer                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              WorkflowExecutor                            │      │
│  │   • Parse YAML workflows                                 │      │
│  │   • Orchestrate import → process → export                │      │
│  │   • Handle errors and logging                            │      │
│  └──────────────────────────────────────────────────────────┘      │
└───────────┬──────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Core Orchestration Layer                          │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              SignalCollection (Orchestrator)             │      │
│  │   Delegates to specialized services:                     │      │
│  │   • SignalRepository        • SignalQueryService         │      │
│  │   • MetadataManager         • AlignmentGridService       │      │
│  │   • EpochGridService        • AlignmentExecutor          │      │
│  │   • SignalCombinationService • OperationExecutor         │      │
│  │   • DataImportService       • SignalSummaryReporter      │      │
│  └──────────────────────────────────────────────────────────┘      │
└───────────┬──────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Service Layer                                 │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ SignalRepository │  │  QueryService    │  │ MetadataManager │  │
│  │  • add_signal    │  │  • get_signals   │  │  • update       │  │
│  │  • get_signal    │  │  • filter        │  │  • validate     │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ AlignmentGrid    │  │  EpochGrid       │  │ Alignment       │  │
│  │   Service        │  │   Service        │  │  Executor       │  │
│  │  • generate_grid │  │  • generate_grid │  │  • apply_align  │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Combination      │  │  Operation       │  │ Import          │  │
│  │   Service        │  │   Executor       │  │  Service        │  │
│  │  • combine       │  │  • execute_op    │  │  • import       │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                      │
└───────────┬──────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Domain Model Layer                            │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  TimeSeriesSignal│  │     Feature      │  │  SignalMetadata │  │
│  │  • PPGSignal     │  │  • FeatureSet    │  │  • Metadata     │  │
│  │  • HRSignal      │  │  • Statistics    │  │    tracking     │  │
│  │  • AccelSignal   │  │                  │  │                 │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                      │
└───────────┬──────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Infrastructure Layer                             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  Importers   │  │  Operations  │  │     Visualization        │ │
│  │  • CSV       │  │  • Filters   │  │  • BokehVisualizer       │ │
│  │  • Polar     │  │  • Features  │  │  • PlotlyVisualizer      │ │
│  │  • Merging   │  │  • Resample  │  │                          │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │   Export     │  │  Algorithms  │  │     Utilities            │ │
│  │  • CSV       │  │  • RF        │  │  • Timestamp handling    │ │
│  │  • Excel     │  │  • XGBoost   │  │  • Enum conversion       │ │
│  │  • HDF5      │  │  • Ensemble  │  │  • Logging               │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. SignalCollection (Orchestrator)

**Purpose**: Central facade that orchestrates all operations and delegates to specialized services.

**Responsibilities**:
- Maintain collection-level metadata
- Delegate CRUD operations to SignalRepository
- Coordinate workflows across services
- Provide backward-compatible API

**Key Methods**:
- `add_time_series_signal()`, `get_signal()` - Delegate to Repository
- `generate_alignment_grid()` - Delegate to AlignmentGridService
- `combine_aligned_signals()` - Delegate to CombinationService
- `apply_operation()` - Delegate to OperationExecutor

### 2. Service Layer

#### SignalRepository
**Responsibility**: Storage and retrieval of signals and features

**Operations**:
- Add/get time-series signals
- Add/get features
- Validate signal IDs and metadata
- Handle auto-incrementing names

#### SignalQueryService
**Responsibility**: Advanced querying and filtering

**Operations**:
- Filter signals by type, metadata criteria
- Base name pattern matching
- Enum-based filtering
- Complex metadata queries

#### MetadataManager
**Responsibility**: Metadata updates and validation

**Operations**:
- Update time-series metadata
- Update feature metadata
- Validate metadata fields
- Enum conversion and parsing

#### AlignmentGridService
**Responsibility**: Calculate alignment grid parameters

**Operations**:
- Generate alignment grid
- Calculate target sample rate
- Compute reference time
- Create DatetimeIndex grid

#### EpochGridService
**Responsibility**: Calculate epoch grid for feature extraction

**Operations**:
- Generate epoch grid
- Validate epoch configuration
- Handle time range overrides

#### AlignmentExecutor
**Responsibility**: Apply alignment to signals

**Operations**:
- Apply grid alignment to signals
- Support multiple reindexing methods
- In-place signal modification

#### SignalCombinationService
**Responsibility**: Combine signals into DataFrames

**Operations**:
- Combine aligned time-series signals
- Combine features into feature matrix
- Handle MultiIndex columns

#### OperationExecutor
**Responsibility**: Execute multi-signal and collection operations

**Operations**:
- Execute collection-level operations
- Execute multi-signal operations (feature extraction)
- Batch signal operations
- Metadata propagation

#### DataImportService
**Responsibility**: Import signals from external sources

**Operations**:
- Import using importer instances
- Handle file pattern matching
- Sequential key generation

#### SignalSummaryReporter
**Responsibility**: Generate summary reports

**Operations**:
- Summarize signals and features
- Format summary tables
- Pretty-print to console

### 3. Domain Model Layer

#### SignalData (Base Class)

Abstract base class for all signal types.

**Key Attributes**:
- `_data`: pandas DataFrame with signal data
- `metadata`: SignalMetadata instance
- `signal_type`: Enum value

**Key Methods**:
- `apply_operation()`: Apply registered operations
- `get_data()`: Access underlying data
- `copy()`: Create deep copy

#### TimeSeriesSignal

Concrete class for time-series signals with DatetimeIndex.

**Specialized Subclasses**:
- `PPGSignal` - Photoplethysmography
- `HeartRateSignal` - Heart rate data
- `AccelerometerSignal` - 3-axis acceleration
- `EEGSleepStageSignal` - Sleep stage annotations
- `RespiratoryRateSignal` - Breathing rate

#### Feature

Represents extracted features over epochs.

**Key Attributes**:
- `_data`: DataFrame with feature values (indexed by epoch start time)
- `metadata`: FeatureMetadata instance
- `feature_names`: List of feature column names

### 4. Operation Registry System

Operations are registered with signal classes and the collection using decorators:

**Signal Operations**:
```python
@TimeSeriesSignal.register("lowpass_filter", output_class=TimeSeriesSignal)
def lowpass_filter(data_list, parameters):
    # Filter implementation
    ...
```

**Multi-Signal Operations** (Feature Extraction):
```python
# Registered in SignalCollection.multi_signal_registry
def feature_statistics(
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    ...
) -> Feature:
    # Feature extraction logic
    ...
```

**Collection Operations**:
```python
@register_collection_operation("generate_alignment_grid")
def generate_alignment_grid(self, parameters):
    # Delegate to AlignmentGridService
    ...
```

## Design Patterns

### 1. Service-Oriented Architecture (SOA)

The framework uses SOA to break down the monolithic SignalCollection into focused services.

**Benefits**:
- Single Responsibility Principle (each service has one job)
- Easier testing (mock dependencies)
- Better maintainability
- Clear separation of concerns

### 2. Facade Pattern

`SignalCollection` acts as a facade, providing a unified interface while delegating to services.

**Example**:
```python
# User calls SignalCollection method
collection.add_time_series_signal("hr_0", signal)

# SignalCollection delegates to SignalRepository
self._repository.add_time_series_signal("hr_0", signal)
```

### 3. Registry Pattern

Operations are registered with classes for dynamic dispatch.

**Benefits**:
- Extensible (add new operations without modifying core classes)
- Declarative workflow definitions
- Type-safe dispatch

### 4. Data Class Pattern

Immutable state objects encapsulate related data.

**Examples**:
- `AlignmentGridState` - Alignment parameters
- `EpochGridState` - Epoch parameters
- `CombinationResult` - Combined DataFrame results

### 5. Metadata Handler Pattern

Centralized metadata management ensures consistency.

**Benefits**:
- Automatic ID generation
- Consistent metadata updates
- Operation history tracking

## Signal Processing Pipeline

### Typical Workflow

```
1. Import
   ↓
   ├─→ DataImportService reads files
   ├─→ Importer parses and validates
   ├─→ SignalRepository stores signals
   └─→ MetadataManager sets metadata

2. Alignment (optional)
   ↓
   ├─→ AlignmentGridService calculates grid
   ├─→ AlignmentExecutor applies to signals
   └─→ Signals reindexed to common grid

3. Feature Extraction
   ↓
   ├─→ EpochGridService generates epochs
   ├─→ OperationExecutor runs feature operations
   └─→ Features stored in collection

4. Combination
   ↓
   ├─→ SignalCombinationService combines signals
   ├─→ Creates MultiIndex DataFrame
   └─→ Stores result internally

5. Export
   ↓
   ├─→ ExportModule retrieves data
   ├─→ Formats for output (CSV, Excel, HDF5, etc.)
   └─→ Writes to disk

6. Visualization (optional)
   ↓
   ├─→ Visualizer retrieves signals
   ├─→ Generates plots (Bokeh or Plotly)
   └─→ Saves to HTML/PNG/SVG
```

### Data Flow

```
External Files
    │
    ▼
Importers → SignalData → SignalRepository
    │                          │
    ▼                          ▼
Operations             SignalCollection
(Filters,                     │
 Resample,                    ▼
 Features)            CombinationService
    │                          │
    ▼                          ▼
Modified Signals        Combined DataFrame
    │                          │
    ▼                          ▼
Export Module          Visualization
    │                          │
    ▼                          ▼
Output Files           HTML/PNG Plots
```

## Service-Based Architecture

### Before Refactoring (Monolithic)

```
SignalCollection (1,971 lines, 41 methods)
├─ Signal storage
├─ Metadata management
├─ Alignment logic
├─ Combination logic
├─ Operation execution
├─ Import coordination
└─ Summary generation
```

**Issues**:
- High complexity
- Hard to test
- Difficult to maintain
- Violates Single Responsibility Principle

### After Refactoring (Service-Based)

```
SignalCollection (1,034 lines, ~20 methods) - Orchestrator
│
├─→ SignalRepository (450 lines) - Storage
├─→ SignalQueryService (270 lines) - Querying
├─→ MetadataManager (280 lines) - Metadata
├─→ AlignmentGridService (400 lines) - Grid calculation
├─→ EpochGridService (240 lines) - Epoch grid
├─→ AlignmentExecutor (160 lines) - Grid application
├─→ SignalCombinationService (400 lines) - Combination
├─→ OperationExecutor (400 lines) - Operations
├─→ DataImportService (240 lines) - Import
└─→ SignalSummaryReporter (270 lines) - Reporting
```

**Benefits**:
- ✅ Each service has single responsibility
- ✅ Easy to test in isolation
- ✅ Clear interfaces
- ✅ Better code organization
- ✅ Maintainable and extensible

## Extensibility Points

### 1. Adding New Signal Types

```python
# 1. Define enum value
class SignalType(Enum):
    MY_NEW_SIGNAL = "my_new_signal"

# 2. Create signal class
class MyNewSignal(TimeSeriesSignal):
    signal_type = SignalType.MY_NEW_SIGNAL
    required_columns = ['value']

# 3. Use in workflows
import:
  - signal_type: "my_new_signal"
    importer: "CSVImporter"
    source: "data.csv"
```

### 2. Adding New Operations

```python
# 1. Define operation function
@TimeSeriesSignal.register("my_operation", output_class=TimeSeriesSignal)
def my_operation(data_list, parameters):
    # Implementation
    return processed_dataframe

# 2. Use in workflows
steps:
  - type: signal
    operation: "my_operation"
    inputs: ["signal_key"]
    parameters:
      param1: value1
```

### 3. Adding New Importers

```python
# 1. Create importer class
class MyImporter(CSVImporterBase):
    def __init__(self, config, ...):
        super().__init__(config, ...)

    def read_file(self, file_path):
        # Custom parsing logic
        return SignalData(...)

# 2. Register and use
import:
  - signal_type: "heart_rate"
    importer: "MyImporter"
    source: "data/my_format.csv"
```

### 4. Adding New Feature Operations

```python
# 1. Define feature extraction function
def my_feature_extractor(
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    ...
) -> Feature:
    # Feature extraction logic
    return Feature(data=feature_df, metadata=metadata)

# 2. Register in SignalCollection.multi_signal_registry
SignalCollection.multi_signal_registry["my_features"] = (
    my_feature_extractor,
    Feature
)

# 3. Use in workflows
steps:
  - type: multi_signal
    operation: "my_features"
    inputs: ["hr"]
    output: "my_extracted_features"
```

### 5. Adding New Visualization Backends

```python
# 1. Implement VisualizerBase interface
class MyVisualizer(VisualizerBase):
    def create_time_series_plot(self, signal, **kwargs):
        # Implementation
        ...

# 2. Register in visualizer factory
visualization:
  - type: time_series
    backend: "my_visualizer"
    ...
```

## Best Practices

### 1. Use Operations Instead of Direct Data Access

**Good**:
```python
signal.apply_operation("lowpass_filter", cutoff=0.5, inplace=True)
```

**Bad**:
```python
signal._data = signal._data.apply(some_filter)  # Bypasses metadata tracking
```

### 2. Leverage Metadata

```python
# Metadata automatically tracks operations
signal.metadata.operations
# [{'operation': 'lowpass_filter', 'timestamp': '...', 'parameters': {...}}]
```

### 3. Use Type Hints

```python
def my_operation(
    signal: TimeSeriesSignal,
    parameters: Dict[str, Any]
) -> TimeSeriesSignal:
    ...
```

### 4. Follow Coding Guidelines

See [docs/coding_guidelines.md](docs/coding_guidelines.md) for detailed standards.

## Testing Architecture

### Unit Tests

Each service has dedicated unit tests:
- `tests/unit/test_signal_repository.py` (34 tests)
- `tests/unit/test_signal_query_service.py` (38 tests)
- `tests/unit/test_metadata_manager.py` (28 tests)
- `tests/unit/test_alignment_grid_service.py` (24 tests)
- And more...

### Integration Tests

Workflow-level tests verify end-to-end functionality:
- `tests/integration/test_workflow_executor.py`
- `tests/integration/test_alignment_workflow.py`

### Test Structure

```python
@pytest.fixture
def sample_signal():
    """Create test signal."""
    ...

def test_operation_basic_case(sample_signal):
    """Test operation with valid input."""
    ...

def test_operation_edge_case(sample_signal):
    """Test operation with boundary conditions."""
    ...

def test_operation_error_handling():
    """Test operation error handling."""
    ...
```

## Performance Considerations

### Memory Management

- Signals use pandas DataFrames (efficient for large datasets)
- In-place operations minimize memory copies
- Combined DataFrames use sparse storage where appropriate

### Computation Optimization

- Vectorized pandas operations
- Lazy evaluation where possible
- Caching of computed grids and metadata

### Scalability

- Modular services enable parallel processing (future enhancement)
- Export supports chunking for large datasets
- Visualization supports downsampling

## Related Documentation

- [README.md](README.md) - High-level overview
- [USER-GUIDE.md](USER-GUIDE.md) - Detailed usage guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [docs/coding_guidelines.md](docs/coding_guidelines.md) - Coding standards
- [docs-dev/](docs-dev/) - Development notes and refactoring details

---

**Version**: 1.0.0
**Last Updated**: 2025-11-18
