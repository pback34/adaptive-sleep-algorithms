Below is a comprehensive **Requirements and Design Specification Document** for the **Flexible Signal Processing Framework for Sleep Analysis**. This document outlines the requirements, architecture, detailed design, and examples for the entire system, including metadata structures, key classes, workflow execution, importers, and YAML workflow examples. It provides a structured blueprint for implementing the framework.

---

# Flexible Signal Processing Framework for Sleep Analysis: Requirements and Design Specification

## 1. Introduction

This document defines the requirements and design specifications for a flexible signal processing framework tailored for sleep analysis. The framework is designed to handle various physiological signals (e.g., PPG, accelerometer) and derived metrics (e.g., heart rate, sleep stages) while ensuring full traceability, memory optimization, and type safety. It supports both structured workflows and ad-hoc processing, making it suitable for use in scripts, notebooks, and automated pipelines.

The framework is built around a hierarchy of signal classes with embedded registries, allowing for type-safe operations while maintaining flexibility. A separate importer module handles the conversion of signals from various manufacturers and formats into a standardized format.

---

## 2. Requirements

### 2.1 Functional Requirements
- **Signal Representation**: Represent various signal types (e.g., PPG, accelerometer) with appropriate metadata.
- **FR1.1**: The framework shall support importing multiple fragmented data files from the same signal source and merging them into a single unified signal.
- **FR1.2**: The merging process shall order data chronologically based on a specified timestamp column.
- **FR1.3**: The system shall automatically identify related fragmented files using filename patterns (e.g., `ppg_001.csv`, `ppg_002.csv`) and timestamp continuity.
- **FR1.4**: Merged signals shall include metadata listing all source files used in the merge.
- **FR2.1**: The framework shall allow users to configure hierarchical (multi-index) structures for exported dataframes.
- **FR2.2**: The index configuration shall support any combination of fields from `SignalMetadata`.
- **FR2.3**: Index configuration shall be specifiable via both programmatic calls and workflow YAML files.
- **FR2.4**: The system shall default to a single-level index using `signal_id` if no custom configuration is provided.
- **FR2.5**: The system shall automatically add an additional level called 'column' to the multi-index to distinguish between different data columns from the same signal.
- **Processing Traceability**: Track all operations applied to signals, including operation names, parameters, and the state of input signals at the time of derivation, sufficient to regenerate derived signals. This includes data provenance (e.g., source file details), full operation history, and the exact state of input signals used to create each derived signal.
- **Memory Optimization**: Support clearing data from intermediate signals while preserving the ability to regenerate them.
- **Type Safety**: Ensure processing operations are appropriate for specific signal types.
- **Flexibility**: Support both structured workflows (via YAML/JSON) and ad-hoc processing in scripts or notebooks.
- **Extensibility**: Easily add new signal types and processing functions.
- **Import Support**: Handle signals from various manufacturers and formats, converting them to a standard format.
- **Workflow Support**: Enable both sensor-agnostic (by signal type) and non-agnostic (by specific signal) workflows.
- **Framework Versioning**: The framework must store its version in the metadata of processed signals and collections to ensure traceability, compatibility, and support for auditing.
- **Epoch-Based Feature Extraction**: Transform continuous derived signals (e.g., heart rate, respiratory rate) into structured datasets of features computed over specified time windows (epochs), enabling tasks such as sleep stage classification and event detection.
- **Command-Line Interface**: The framework must provide a command-line interface that allows users to specify workflow files and data directories at runtime, enabling the same workflow to be applied to different datasets without modification.
- **Configurable Multi-Index Dataframe Export**: The framework shall allow users to customize hierarchical indexing of exported dataframes using metadata fields from signals.


### 2.2 Non-Functional Requirements
- **Usability**: Provide an intuitive API for users in scripts, notebooks, or workflows.
- **Performance**: Ensure efficient memory usage and processing speed for large datasets.
- **Scalability**: Handle multiple signals and complex workflows.
- **Maintainability**: Use a modular design for easy updates, debugging, and extension.
- **NFR1.1**: The merging process shall complete with a time complexity of O(n), where n is the total number of data points across all files.
- **NFR1.2**: The merging process shall support files with both overlapping and non-overlapping timestamps, defaulting to retaining the earliest occurrence in overlaps.
- **NFR2.1**: Configuring and applying a multi-index shall not increase export processing time by more than 5%.
- **NFR2.2**: The system shall support up to four levels of indexing without performance degradation.

### 2.3 Functional Requirements - Export Module
- **Export Formats**: The framework must support exporting signals and their associated metadata to the following file formats:
  - Excel (.xlsx)
  - CSV (.csv)
  - Pickle (.pkl)
  - HDF5 (.h5)
- **Metadata Inclusion**: 
  - Export collection-level metadata (`CollectionMetadata`) with all exported signals.
  - Export individual signal metadata (`SignalMetadata`) for each signal, including operation history and framework version.
- **Combined Signal Dataframe Export**:
  - Provide an option to export a combined dataframe containing all non-temporary signals, aligned by timestamps.
  - The combined dataframe must include columns for each signal's data, with headers indicating signal names or IDs (e.g., "ppg_raw", "hr_0").
  - Temporary signals (marked with `temporary=True` in `SignalMetadata`) must be excluded from the combined dataframe.
- **Output Configuration**:
  - Users must be able to specify the output directory and select one or more export formats.
  - Support exporting to multiple formats in a single operation (e.g., both CSV and HDF5).
- **Traceability**:
  - All exported files must include the framework version (stored in `framework_version` of `SignalMetadata` and `CollectionMetadata`) to ensure compatibility and auditability.

### 2.4 Functional Requirements - Importers
- **Format Flexibility**: The framework must support importing signals from various file formats:
  - CSV (.csv)
  - Excel (.xlsx)
  - Pickle (.pkl)
  - HDF5 (.h5)
  - JSON (.json)
  - Manufacturer-specific formats
- **Importer Hierarchy**: 
  - Provide an abstract base class (`SignalImporter`) defining the common interface for all importers.
  - Support format-specific importers (e.g., `CSVImporterBase`) that handle format-specific logic.
  - Implement manufacturer-specific concrete importers (e.g., `PolarCSVImporter`) using declarative configurations.
- **Metadata Extraction**:
  - Automatically extract metadata from source files when possible (e.g., sampling rate, timestamps).
  - Support extracting metadata from filename patterns using regular expressions.
  - Allow manual specification of metadata when automatic extraction is not possible.
- **Validation and Error Handling**:
  - Validate that required columns exist in source files before processing.
  - Validate that data conforms to expected formats and ranges.
  - Provide clear error messages when validation fails.
  - Log detailed diagnostic information for debugging import issues.
- **Multi-File Import**:
  - Support importing and merging multiple fragmented data files into a single signal.
  - Support batch importing of multiple files based on pattern matching.
  - Allow specification of different merge strategies (e.g., by timestamp, concatenation).
- **Extensibility**:
  - Allow easy addition of new importers for different data sources through configuration.
  - Support preprocessing of data during import (e.g., unit conversion, resampling).
  - Enable custom validation rules for specific importers.
- **Column Standardization**:
  - **FR2.4.1**: Importers shall map raw data columns to the expected columns defined by the target signal class using a declarative configuration (e.g., a `column_mapping` dictionary in the importer's config). By default, importers shall drop any columns in the raw data not specified in `column_mapping`.
  - **FR2.4.2**: Importers shall convert the timestamp column from the raw data's format to a standard format specified in the `CollectionMetadata`'s `timestamp_format` field, using a shared utility function that supports multiple input formats.
  - **FR2.4.3**: Importers shall validate that the mapped data contains all required columns specified by the target signal class before creating a `SignalData` instance, raising an error if any are missing. For merged signals from multiple files (per FR1.1–FR1.4), this validation shall ensure the combined data adheres to the same `required_columns`.

### 2.5 Functional Requirements - Feature Extraction
- **Epoch Generation**:
  - Segment derived signals into epochs with configurable parameters: **window length** (e.g., 30 seconds) and **step size** (e.g., 10 seconds for overlapping windows).
- **Feature Extraction**:
  - Compute user-defined **aggregation functions** (e.g., mean, standard deviation, maximum, minimum) over each epoch for one or more specified signals.
  - Support **single-signal features** (e.g., mean heart rate) and **multi-signal features** (e.g., correlation between heart rate and respiratory rate).
  - Allow **signal-specific feature extraction methods** (e.g., heart rate variability (HRV) for heart rate signals, spectral power for EEG signals).
- **Input**:
  - Accept one or more derived signals from the `SignalCollection` as input for feature extraction.
- **Output**:
  - Produce a **`FeatureSignal`** (a subclass of `SignalData`) containing a DataFrame where:
    - Each row corresponds to an epoch, indexed by timestamps.
    - Each column represents a computed feature (e.g., `heart_rate_mean`, `respiratory_rate_std`).
  - Include metadata in the `FeatureSignal` linking features to their source signals, epochs, and applied feature functions.
- **Integration**:
  - Register feature extraction operations in the `SignalCollection`'s `multi_signal_registry` with a `"feature_"` prefix (e.g., `"feature_mean"`).
  - Enable feature extraction steps to be specified in YAML/JSON workflows executed by the `WorkflowExecutor`.
- **Flexibility**:
  - Allow users to define and register custom feature extraction functions.
  - Support batch processing of multiple signals and features in a single operation.

### 2.6 Non-Functional Requirements - Feature Extraction
- **Performance**:
  - Optimize feature extraction for large datasets using vectorized operations (e.g., NumPy, Pandas) to ensure computational efficiency.
  - Minimize memory usage, particularly when handling overlapping epochs or long signal recordings.
- **Usability**:
  - Provide an intuitive API for specifying window length, step size, and features, both programmatically and in workflow configurations.
- **Extensibility**:
  - Design the feature extraction system to allow easy addition of new feature functions without modifying core framework components.
- **Traceability**:
  - Ensure that metadata in the `FeatureSignal` tracks the window length, step size, source signals, and feature functions applied, maintaining reproducibility.

### 2.7 Testing Requirements
- **Testing Framework**: Use pytest as the primary testing framework for both unit and integration tests.
- **Unit Testing**:
  - **Scope**: Test individual components of the framework, including:
    - Signal classes (e.g., `PPGSignal`, `AccelerometerSignal`)
    - Importers (e.g., `ManufacturerAImporter`, `CSVImporter`)
    - Operations (e.g., `PPGOperations`, `AccelerometerOperations`)
    - Workflow executor (`WorkflowExecutor`)
    - Export module (`ExportModule`)
  - **Objectives**:
    - Test that importers correctly map raw data columns to the expected columns using the declarative configuration and drop any extra columns not specified in `column_mapping`.
    - Verify that importers convert timestamps to the standard format specified in `CollectionMetadata` and handle various input formats correctly using the `convert_timestamp_format` utility.
    - Ensure that signal classes raise a `ValueError` when initialized with data missing any `required_columns`.
    - Verify type safety by ensuring operations are only applied to appropriate signal types.
    - Test metadata handling, including operation history, `derived_from` fields, and framework version.
    - Validate data clearing and regeneration for temporary signals.
    - Ensure signal import and standardization produce expected outputs.
  - **Implementation**: Use pytest fixtures to provide reusable test data and mock objects.
- **Integration Testing**:
  - **Scope**: Test the complete workflow from signal import to export.
    - Validate end-to-end data integrity and metadata consistency.
    - Test workflows with multiple signal types (e.g., PPG and accelerometer) and operations.
    - Verify export functionality across all supported formats.
  - **Objectives**:
    - Ensure that imported signals can be processed and exported correctly.
    - Confirm that metadata (including operation history and framework version) is preserved throughout the pipeline.
    - Validate the combined dataframe export excludes temporary signals and aligns data properly.
    - Validate that merged signals from multiple files conform to the `required_columns` of the target signal class and maintain the standardized `timestamp_format`.
  - **Implementation**: Simulate full workflows using YAML configurations and verify outputs against expected results.
- **Test Directory Structure**:
  - Organize tests in a `tests/` directory with subdirectories:
    ```
    tests/
      - unit/
        - test_signals.py
        - test_importers.py
        - test_operations.py
        - test_export.py
      - integration/
        - test_workflow.py
    ```
- **Example Test Cases**:
  - **Unit Test (PPG Filter Operation)**:
    ```python
    def test_ppg_filter_operation():
        ppg_signal = PPGSignal(data=pd.DataFrame({"value": [1, 2, 3]}), metadata={"signal_id": "ppg1"})
        filtered_signal = ppg_signal.apply_operation("filter_lowpass", cutoff=5)
        assert filtered_signal.metadata.operations[-1].operation_name == "filter_lowpass"
        assert filtered_signal.data.shape == ppg_signal.data.shape
    ```
  - **Integration Test (Full Workflow)**:
    ```python
    def test_full_workflow():
        collection = SignalCollection()
        # Import signal
        importer = ManufacturerAImporter()
        ppg_signal = importer.import_signal("path/to/ppg.csv", signal_type="ppg")
        collection.add_signal("ppg_raw", ppg_signal)
        # Execute workflow
        workflow_config = {
            "steps": [
                {"signal": "ppg_raw", "operation": "filter_lowpass", "output": "ppg_filtered", "parameters": {"cutoff": 5}}
            ]
        }
        executor = WorkflowExecutor(container=collection)
        executor.execute_workflow(workflow_config)
        # Export results
        exporter = ExportModule(collection)
        exporter.export(formats=["csv"], output_dir="./test_output")
        # Verify exported files
        assert os.path.exists("./test_output/signals.csv")
        assert os.path.exists("./test_output/metadata.csv")
    ```

---

## 3. Architecture

### 3.1 Project Structure
The framework is organized into the following modules:
- **`core/`**: Base classes for the framework.
  - `SignalData`: Base class for all signals.
  - `SignalCollection`: Container for all signals (imported, intermediate, derived).
  - `WorkflowExecutor`: Executes workflows by processing signals.
- **`signal_types.py`**: Enum definitions for signal types.
  - `SignalType`: Enum for different types of signals (e.g., PPG, ACCELEROMETER, HEART_RATE, FEATURES).
- **`cli/`**: Command-line interface for the framework.
  - `run_workflow.py`: Script for executing workflows with CLI arguments.
  
### 3.1.1 Signal Types Module
The `signal_types.py` module centralizes the definition of signal types using Python's `Enum` class. This provides several benefits:

- **Type Safety**: Using enums prevents the use of invalid signal types, as only defined enum values can be used.
- **Centralized Definition**: All signal types are defined in one place, making it easy to add new types.
- **IDE Support**: Modern IDEs provide auto-completion and type hinting for enums.
- **Documentation**: Each enum value can include documentation.

Example implementation:

```python
from enum import Enum

class SignalType(Enum):
    PPG = "PPG"
    ACCELEROMETER = "Accelerometer"
    HEART_RATE = "Heart Rate"
    # Add more signal types as needed
```

The enum values (e.g., "PPG") are used in the user interface and workflow files, while the enum constants (e.g., `SignalType.PPG`) are used in the code for type safety.

- **`signals/`**: Subclasses for specific signal types.
  - `TimeSeriesSignal`: Generic time-series signal.
  - `PPGSignal`: Signal specific to photoplethysmography (PPG).
  - `AccelerometerSignal`: Signal for accelerometer data.
  - `FeatureSignal`: Signal for epoch-based extracted features.
- **`importers/`**: Classes for importing signal data.
  - `SignalImporter`: Abstract base class for all importers.
  - `CSVImporterBase`: Abstract class for CSV format importers.
  - `PolarCSVImporter`: Concrete importer for Polar device CSV files.
  - `CustomDatasetCSVImporter`: Concrete importer for custom dataset CSV files.
- **`operations/`**: Registry of signal processing functions.
  - `PPGOperations`: Operations like filtering, peak detection for PPG signals.
  - `AccelerometerOperations`: Operations like motion detection.
  - `feature_extraction.py`: Module for feature extraction functions (e.g., compute_mean, compute_correlation, compute_hrv).
- **`export/`**: Module for exporting signals and metadata.
  - `ExportModule`: Class for handling signal exports to various formats.
- **`workflows/`**: Tools for defining and executing workflows.
  - `YAMLWorkflowParser`: Parses workflow definitions from YAML files.
- **`utils/`**: Helper functions.
  - `align_data`: Aligns signals by timestamps.

This structure separates concerns, making it easy to extend or modify individual components while maintaining clear organization of functionality.

### 3.2 Dependencies
- **Internal**:
  - Signal classes depend on `core`.
  - Importers depend on signal classes.
- **External**:
  - `pandas`: Data manipulation.
  - `numpy`: Numerical operations.
  - `yaml`: Workflow parsing.
  - `scipy`: Signal processing functions (optional).
  - `openpyxl`: For Excel export support.
  - `h5py`: For HDF5 export support.

### 3.3 Architecture Diagram
The following diagram illustrates the high-level flow of the framework:

```
[Start] → [CLI Arguments] → [Import Signals (via Importers)] → [SignalCollection] → [Apply Operations (Direct or Workflow)] → [Feature Extraction] → [Export Signals (via ExportModule)] → [End]
```

- **CLI Arguments**: Users specify the workflow file and data directory at runtime.
- **Import**: Signals are loaded from various sources using format-specific importers with declarative configurations and standardized into signal instances.
- **SignalCollection**: Signals are managed and processed collectively.
- **Operations**: Applied either directly or through workflows.
- **Export**: Processed signals and metadata are exported to user-specified formats.

### 3.3.1 Command-Line Interface (CLI)

The framework provides a command-line interface (CLI) script to execute workflows with user-specified parameters, enhancing usability and flexibility. The CLI must:

- Accept the following arguments:
  - `--workflow`: Path to the workflow YAML file (required).
  - `--data-dir`: Base directory containing the data files referenced in the YAML (required).
  - `--output-dir`: Directory for output files (optional, defaults to a predefined location).
- Load the specified workflow YAML configuration.
- Resolve `source` paths in the `import` section by combining `data_dir` with each `source` field, and, if a `file_pattern` is specified, identify all matching files.
- Initialize and execute the `WorkflowExecutor` with the modified configuration, passing the resolved paths and `data_dir`.
- Provide help text (e.g., via `argparse`) and validate arguments (e.g., check that `data_dir` exists and is a directory).

**Example Usage:**

```bash
python -m sleep_analysis.cli.run_workflow --workflow polar_workflow.yaml --data-dir /experiment1/data --output-dir /experiment1/output
```

This command loads `polar_workflow.yaml`, resolves data paths relative to `/experiment1/data`, executes the workflow, and saves results to `/experiment1/output`. The CLI ensures that workflows can be run on different datasets without modifying the YAML, fulfilling the requirement for reusability.

### 3.3.2 Logging Configuration

The framework utilizes Python's standard `logging` module to provide consistent and configurable logging across all components. This ensures that log messages are uniformly formatted, appropriately leveled, and easily traceable, enhancing debugging and monitoring capabilities. The logging system is designed to be shared among all modules, simplifying configuration and ensuring a cohesive logging strategy.

#### Logging Features

- **Shared Configuration**: A single logging configuration is established and utilized by all modules to maintain consistency in log output.
- **Log Format**: Each log message includes the module name and line number for precise traceability. The format is defined as:
  ```
  %(module)s:%(lineno)d - %(levelname)s - %(message)s
  ```
  Example output:
  ```
  signals:45 - INFO - Processing PPG signal
  cli.run_workflow:72 - DEBUG - Parsed CLI arguments
  ```
- **Log Levels**: The framework employs the following log levels throughout the codebase:
  - **`ERROR`**: For critical errors that prevent the program from continuing (e.g., invalid workflow file, missing data files).
  - **`WARN`**: For potential issues that allow execution to proceed (e.g., missing optional configuration, skipped signals due to invalid data).
  - **`INFO`**: For general progress and status updates (e.g., workflow started, signal imported, export completed).
  - **`DEBUG`**: For detailed debugging information (e.g., parameter values, intermediate results, detailed execution steps).
- **File Logging**: When executing workflows via the command-line interface (`run_workflow.py`), logs are written to a file at `<output_dir>/logs/workflow.log`. If the file exists, it is overwritten to ensure a clean log for each run. The `logs` directory within `<output_dir>` is created automatically if it does not exist.
- **Console Logging**: Logs are also output to the console, providing real-time feedback during CLI execution.
- **Configurable Log Level**: The log level is configurable via CLI options:
  - **`--log-level <LEVEL>`**: Sets the log level to one of `DEBUG`, `INFO`, `WARN`, or `ERROR` (case-insensitive).
  - **`-v`**: Shorthand flag to set the log level to `DEBUG`.
  - **Default**: The default log level is `INFO` if no option is specified.

#### Implementation Details

- **Logger Setup**: The logging configuration is established in the CLI script (`run_workflow.py`) as early as possible—immediately after parsing CLI arguments—to ensure that all subsequent module imports and operations use the configured logger. The root logger is configured with both a file handler (for `<output_dir>/logs/workflow.log`) and a console handler (for stdout).
- **Shared Configuration Across Modules**: In other modules, loggers are obtained using `logging.getLogger(__name__)`. This approach creates module-specific loggers that inherit the configuration (handlers, formatters, and level) from the root logger, ensuring consistency without requiring redundant setup.
- **File Handler**: The file handler writes logs to `<output_dir>/logs/workflow.log` in write mode (`mode='w'`), overwriting any existing file. The `logs` directory is created using `os.makedirs()` with `exist_ok=True` to handle cases where the directory already exists.
- **Console Handler**: A stream handler is added to output logs to the console, enhancing usability during interactive execution.
- **Log Level Configuration**: The CLI script parses the `--log-level` and `-v` options using `argparse`, converting the specified level to the corresponding `logging` module constant (e.g., `logging.DEBUG`) and applying it to the root logger and its handlers.

#### Example CLI Usage

- **Default Log Level (INFO)**:
  ```bash
  python -m sleep_analysis.cli.run_workflow --workflow workflow.yaml --data-dir /path/to/data --output-dir /path/to/output
  ```
  - Logs at `INFO` level and above are written to `/path/to/output/logs/workflow.log` and displayed in the console.

- **Debug Log Level with Shorthand**:
  ```bash
  python -m sleep_analysis.cli.run_workflow --workflow workflow.yaml --data-dir /path/to/data --output-dir /path/to/output -v
  ```
  - Sets the log level to `DEBUG`, capturing all log messages.

- **Explicit Log Level**:
  ```bash
  python -m sleep_analysis.cli.run_workflow --workflow workflow.yaml --data-dir /path/to/data --output-dir /path/to/output --log-level DEBUG
  ```
  - Equivalent to using `-v`, logs at `DEBUG` level.

#### Code Snippet for Logger Configuration

Below is the implementation of the logging setup within `run_workflow.py`:

```python
import logging
import os
import argparse

def setup_logging(output_dir, log_level):
    """Configure the root logger for the framework."""
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(log_level)    # Set the minimum level for logging
    
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure file handler
    log_file = os.path.join(logs_dir, 'workflow.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Define log message format
    formatter = logging.Formatter('%(module)s:%(lineno)d - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the root logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main():
    parser = argparse.ArgumentParser(description="Execute a sleep analysis workflow.")
    parser.add_argument('--workflow', required=True, help='Path to the workflow YAML file')
    parser.add_argument('--data-dir', required=True, help='Base directory for data files')
    parser.add_argument('--output-dir', default='./output', help='Directory for output files')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARN', 'ERROR'],
                        default='INFO', help='Set the logging level (default: INFO)')
    parser.add_argument('-v', action='store_const', const='DEBUG', dest='log_level',
                        help='Set logging level to DEBUG')
    
    args = parser.parse_args()
    
    # Configure logging early, before other imports or operations
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(args.output_dir, log_level)
    
    # Proceed with workflow execution
    logger = logging.getLogger(__name__)
    logger.info("Starting workflow execution")
    # ... (rest of the CLI script)
```

#### Usage in Other Modules

In all other modules, loggers are obtained using the module's name, inheriting the root logger's configuration:

```python
import logging

logger = logging.getLogger(__name__)

def process_signal(signal):
    logger.debug(f"Processing signal with parameters: {signal.metadata}")
    try:
        # Signal processing logic
        logger.info("Signal processed successfully")
    except ValueError as e:
        logger.error(f"Failed to process signal: {e}")
    except Exception as e:
        logger.warn(f"Unexpected issue during processing: {e}")
```

This ensures that all log messages follow the configured format, level, and output destinations (file and console) without additional setup in each module.

#### Integration with Framework

- **Early Configuration**: By configuring the logger immediately after parsing CLI arguments, all subsequent operations—including signal imports, workflow execution, and exports—benefit from the established logging setup.
- **Traceability**: The inclusion of module and line number in each log message aids in pinpointing the source of issues or tracking execution flow.
- **Flexibility**: The configurable log level allows users to adjust verbosity as needed, from minimal output (`INFO`) to detailed debugging (`DEBUG`).

This logging configuration satisfies the requirements for a shared, traceable, and flexible logging system integrated with the CLI interface.

### 3.4 Export Module Design
- **ExportModule Class**:
  - Implement an `ExportModule` class to manage the export process.
  - The class must accept a `SignalCollection` instance and export configuration parameters (e.g., formats, output directory, include_combined flag).
  - Methods:
    - `export(formats: List[str], output_dir: str, include_combined: bool)`: Executes the export process.
- **Metadata Serialization**:
  - Serialize `CollectionMetadata` and `SignalMetadata` into a JSON-compatible format for inclusion in all exports.
  - For Excel and CSV exports, store metadata in separate sheets (Excel) or files (CSV) alongside the signal data.
  - For Pickle and HDF5, embed metadata directly within the file structure.
- **Combined Dataframe Generation**:
  - Implement a method in `SignalCollection` (e.g., `get_combined_dataframe()`) to generate a combined dataframe of all non-temporary signals, aligning them by timestamps.
  - Handle missing data gracefully (e.g., with NaN values) when signals have different time ranges.
- **Multi-Index Configuration**:
  - Support configurable multi-index structures for exported dataframes based on metadata fields.
  - The `ExportModule` class must include a method to create a multi-index based on the configured metadata fields:
    ```python
    def _create_multiindex(self, signals):
        """
        Create a multi-index for the dataframe based on configured metadata fields.
        
        Args:
            signals: List of SignalData objects to export.
        
        Returns:
            pd.MultiIndex: A multi-index object for the dataframe columns.
        """
        import pandas as pd
        indexes = []
        for signal in signals:
            index_values = [getattr(signal.metadata, field) 
                           for field in self.collection.metadata.index_config]
            indexes.append(tuple(index_values))
        return pd.MultiIndex.from_tuples(
            indexes,
            names=self.collection.metadata.index_config
        )
    ```
  - Update the `get_combined_dataframe` method to use the custom multi-index:
    ```python
    def get_combined_dataframe(self):
        """
        Export all non-temporary signals into a combined dataframe with a custom multi-index.
        
        Returns:
            pd.DataFrame: Combined dataframe with configured multi-index columns.
        """
        import pandas as pd
        non_temp_signals = [s for s in self.signals if not s.metadata.temporary]
        if not non_temp_signals:
            return pd.DataFrame()
        
        combined_df = pd.concat([s.data for s in non_temp_signals], axis=1)
        combined_df.columns = self._create_multiindex(non_temp_signals)
        return combined_df
    ```
- **Timestamp Formatting**:
  - When exporting signals to formats requiring string timestamps (e.g., CSV, Excel), the export module shall format the `'timestamp'` column as strings using the `timestamp_format` specified in `CollectionMetadata`.
  - The export module shall ensure that all timestamp columns in exported signals are formatted as strings according to the `timestamp_format` specified in `CollectionMetadata`. This applies consistently to individual signal exports and the combined dataframe export, ensuring uniformity across all output formats.
- **Example Usage**:
  ```python
  exporter = ExportModule(signal_collection)
  exporter.export(
      formats=["excel", "csv"],
      output_dir="./exports",
      include_combined=True
  )
  # Results in:
  # - ./exports/signals.xlsx (individual signals + metadata)
  # - ./exports/combined.xlsx (combined dataframe)
  # - ./exports/signals.csv (individual signals)
  # - ./exports/metadata.csv (metadata)
  # - ./exports/combined.csv (combined dataframe)
  ```

#### Project Structure and Dependencies Diagram
```
+-------------------+
|   workflows/      | ← Uses YAML to define processing
| (WorkflowExecutor)| ← Executes workflows on SignalCollection
+-------------------+
          ↑
          |
+-------------------+
|     core/         | ← Base classes and utilities
| (SignalData,      |
|  SignalCollection)|
+-------------------+
          ↑
          |
+-------------------+       +-------------------+
|    signals/       | ← Inherits from core/  |  importers/       |
| (PPGSignal,       |                        | (ImporterFactory, |
|  AccelerometerSignal)                     |  ManufacturerAImporter)|
+-------------------+       +-------------------+
          ↑                          ↑
          |                          |
+-------------------+       +-------------------+
|   operations/     | ← Registers with signals/  | External Libraries |
| (Function Registry)|                        | (pandas, numpy, yaml, scipy)|
+-------------------+       +-------------------+
```

---

## 4. Metadata Structures

### 4.1 Enum Definitions
```python
from enum import Enum

class SensorType(Enum):
    PPG = "PPG"
    ACCEL = "Accel"
    EKG = "EKG"
    EEG = "EEG"
    # Additional sensor types can be added as needed

class SensorModel(Enum):
    POLAR_H10 = "PolarH10"
    POLAR_SENSE = "PolarSense"
    # Additional sensor models can be added as needed

# Import the framework version for use in metadata
from your_framework import __version__ as FRAMEWORK_VERSION

class SignalType(Enum):
    PPG = "PPG"
    ACCELEROMETER = "Accelerometer"
    HEART_RATE = "Heart Rate"
    FEATURES = "Features"
    # Additional signal types can be added as needed

class BodyPosition(Enum):
    CHEST = "chest"
    HEAD = "head"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    # Additional positions can be added as needed

class Unit(Enum):
    G = "g"
    BPM = "bpm"
    HZ = "Hz"
    # Additional units can be added as needed
```

### 4.2 SignalMetadata
```python
@dataclass
class OperationInfo:
    operation_name: str
    parameters: Dict[str, Any]

@dataclass
class SignalMetadata:
    signal_id: str  # Unique identifier (immutable)
    name: Optional[str]  # User-friendly name for reference
    signal_type: SignalType  # enum for type (e.g., PPG, ACCELEROMETER). In workflows, this is specified as a string (e.g., "ppg") and mapped case-insensitively to the enum value.
    sample_rate: Optional[str]  # e.g., "100Hz"
    units: Unit  # enum for physical units (e.g., G, BPM)
    start_time: Optional[datetime]  # Signal start time
    end_time: Optional[datetime]  # Signal end time
    derived_from: List[Tuple[str, int]]  # List of (signal_id, operation_index) tuples
    operations: List[OperationInfo]  # Processing history (append-only ordered list)
    temporary: bool  # Flag for memory optimization
    sensor_type: SensorType  # enum for sensor type (e.g., PPG, ACCEL)
    sensor_model: SensorModel  # enum for sensor model (e.g., POLAR_H10)
    body_position: BodyPosition  # enum for position (e.g., CHEST, LEFT_WRIST)
    sensor_info: Optional[Dict[str, Any]]  # Optional additional sensor details
    source_files: List[str] = field(default_factory=list)  # List of file paths contributing to the signal
    merged: bool = False  # Indicates if the signal results from a merge operation
    framework_version: str = FRAMEWORK_VERSION  # Framework version used to process the signal
```

### 4.3 CollectionMetadata
```python
@dataclass
class CollectionMetadata:
    collection_id: str  # Unique identifier
    subject_id: str  # Subject identifier
    session_id: Optional[str]  # Session identifier
    start_datetime: Optional[datetime]  # Collection start time
    end_datetime: Optional[datetime]  # Collection end time
    timezone: str  # Timezone for timestamps
    study_info: Dict[str, Any]  # Study details
    device_info: Dict[str, Any]  # Summary of all devices used in the collection
    notes: str  # Additional notes
    protocol_id: Optional[str]  # Optional: links to study protocol
    data_acquisition_notes: Optional[str]  # Optional: notes on data collection process
    index_config: List[str] = field(default_factory=lambda: ["signal_id"])  # Default index
    framework_version: str = FRAMEWORK_VERSION  # Framework version used to process the collection
    timestamp_format: str = '%Y-%m-%d %H:%M:%S'  # Standard format for timestamps in exported data
```

```python
def generate_device_summary(signals: List[SignalData]) -> Dict[str, Any]:
    """
    Generate a summary of devices used in the collection based on signal metadata.
    
    Args:
        signals: List of SignalData objects in the collection.
    
    Returns:
        Dictionary summarizing devices, keyed by device ID or sensor model.
    """
    device_summary = {}
    for signal in signals:
        metadata = signal.metadata
        device_key = f"{metadata.sensor_model.value}_{metadata.signal_id}"
        device_summary[device_key] = {
            "sensor_type": metadata.sensor_type.value,
            "sensor_model": metadata.sensor_model.value,
            "body_position": metadata.body_position.value
        }
    return device_summary
```

### 4.4 OperationInfo
```python
@dataclass
class OperationInfo:
    function_id: str  # Operation name or ID (matches the operation field in YAML workflow)
    parameters: Dict[str, Any]  # Parameters used in the operation (matches parameters in YAML workflow)
```

---

## 5. Key Classes

### 5.1 SignalData (Abstract Base Class)
- **Purpose**: Represents a generic signal with data and metadata.
- **Features**:
  - Supports operations via `apply_operation` using the class's registry.
  - Supports in-place operations for operations that preserve signal class.
  - Manages memory with `clear_data` and data regeneration for temporary signals.
  - Subclasses of `SignalData` shall define a class attribute `required_columns` listing the expected column names in the signal's data (e.g., `['timestamp', 'x', 'y', 'z']` for `AccelerometerSignal`).
  - The `SignalData` base class shall validate that the provided data contains all `required_columns` during initialization, raising a `ValueError` if any are missing.
- **Example**:
  ```python
  signal = SignalData(data=df, metadata={"signal_type": "ppg", "name": "left_wrist_ppg"})
  # Non-in-place operation (creates new signal instance)
  filtered_signal = signal.apply_operation("filter_lowpass", cutoff=5)
  # In-place operation (modifies existing signal)
  signal.apply_operation("filter_lowpass", cutoff=5, inplace=True)
  ```



### 5.2 SignalContainer (Interface)
- **Purpose**: Defines a standard interface for signal containers.
- **Features**:
  - Abstract base class with required methods for signal management.
  - Standardizes behavior for retrieving, adding, and filtering signals.
  - Enables swapping different container implementations.
- **Required Methods**:
  ```python
  def get_signal(self, key):
      """Retrieve a signal by key."""
      pass
      
  def add_signal(self, key, signal):
      """Add a signal with the given key."""
      pass
      
  def get_signals_by_type(self, signal_type):
      """Retrieve all signals of a specific type."""
      pass
  ```

### 5.3 SignalCollection
- **Purpose**: Manages multiple signals, allowing operations across them.
- **Features**:
  - Implements the `SignalContainer` interface.
  - Contains a global registry for multi-signal operations.
  - Acts as the **central hub and single source of truth** for all signals in the system.
  - Contains three types of signals:
    - **Imported signals**: Raw data loaded from external sources.
    - **Intermediate signals**: Results of partial processing (e.g., filtered data).
    - **Derived signals**: Final outputs (e.g., heart rate from PPG).
  - Provides a unified interface to access all signals regardless of type.
  - Supports importing signals by type or individually.
  - Facilitates batch operations.
  - Ensures all signals (raw, intermediate, derived) remain accessible for further processing.
  - Includes a generic signal retrieval mechanism (`get_signals`) to filter signals by any metadata field, including `signal_type`.
- **Additional Features**:
  - `get_combined_dataframe() -> pd.DataFrame`: Generates a combined dataframe of all non-temporary signals, aligned by timestamps. Excludes signals where `temporary=True`.
    ```python
    def get_combined_dataframe(self) -> pd.DataFrame:
        non_temp_signals = [s for s in self.signals if not s.metadata.temporary]
        if not non_temp_signals:
            return pd.DataFrame()
        combined = non_temp_signals[0].data.copy()
        for signal in non_temp_signals[1:]:
            combined = pd.merge(combined, signal.data, how="outer", on="timestamp")
        return combined
    ```
  - `set_index_config(index_fields: List[str])`: Configures the multi-index fields for dataframe exports.
    ```python
    def set_index_config(self, index_fields: List[str]):
        """
        Configure the multi-index fields for dataframe exports.
        
        Args:
            index_fields: List of metadata field names to use as index levels.
        
        Raises:
            ValueError: If any field is not a valid SignalMetadata attribute.
        """
        from dataclasses import fields
        valid_fields = {f.name for f in fields(SignalMetadata)}
        if not all(f in valid_fields for f in index_fields):
            raise ValueError(f"Invalid index fields: {set(index_fields) - valid_fields}")
        self.metadata.index_config = index_fields
    ```
- **Example**:
  ```python
  collection = SignalCollection()
  # Add raw signal
  collection.add_signal("ppg_raw", ppg_signal)
  # Process and store result back in collection
  filtered_ppg = collection.get_signal("ppg_raw").apply_operation("filter_lowpass", cutoff=5)
  collection.add_signal("ppg_filtered", filtered_ppg)
  # Access both original and processed signals
  raw_data = collection.get_signal("ppg_raw")
  filtered_data = collection.get_signal("ppg_filtered")
  
  # Generic retrieval by signal type
  ppg_signals = collection.get_signals({"signal_type": SignalType.PPG})
  
  # Generic retrieval by multiple criteria
  filtered_ppg_signals = collection.get_signals({
      "signal_type": SignalType.PPG,
      "operations": [{"operation_name": "filter_lowpass"}]
  })
  ```
### 5.3.1 SignalCollection Implementation Details

The `SignalCollection` class includes a flexible `get_signals` method for generic signal retrieval based on metadata criteria:

```python
class SignalCollection:
    def __init__(self):
        self.signals: List[Signal] = []  # List to hold Signal instances

    def add_signal(self, signal: Signal):
        """Add a signal to the collection."""
        self.signals.append(signal)

    def get_signals(self, criteria: Dict[str, Any]) -> List[Signal]:
        """
        Retrieve signals that match the given criteria.

        Args:
            criteria: A dictionary where keys are metadata fields and values are the desired values.
                      Example: {'signal_type': SignalType.PPG, 'id': 'ppg1'}

        Returns:
            A list of Signal instances that match all the criteria.
        """
        def matches(signal: Signal) -> bool:
            for key, value in criteria.items():
                if key not in signal.metadata:
                    return False
                if signal.metadata[key] != value:
                    return False
            return True

        return [signal for signal in self.signals if matches(signal)]
```

This implementation allows for flexible filtering of signals based on any metadata field, including `signal_type`, operations, or custom fields. It returns a list of signals that match all the specified criteria.
### 5.4 WorkflowExecutor
- **Purpose**: Executes workflows defined in YAML or JSON.
- **Features**:
  - Operates directly on any `SignalContainer` implementation, maintaining it as the single source of truth.
  - Includes validation to check for signals and operations before execution.
  - Supports both single-signal operations and multi-signal operations.
  - Supports sensor-agnostic (by signal type) and non-agnostic (by specific signal keys) operations.
  - Supports in-place operations that modify existing signals instead of creating new ones.
  - Retrieves input signals from the container and adds processed signals back to it (unless in-place).
  - Uses the signal instance's class to access the appropriate registry automatically.
  - Never returns a separate results object - all processed data remains in the collection.
  - During workflow execution, the `WorkflowExecutor` validates signal types by converting string inputs to `SignalType` enums using case-insensitive matching. If a string does not match any enum value, a `ValueError` is raised.
  - Accepts an optional `data_dir` parameter that specifies the base directory containing the data files referenced in the workflow. This enables workflows to use relative paths that are resolved at runtime.
  - During workflow execution, the `WorkflowExecutor` shall ensure that all signals processed or generated adhere to the `timestamp_format` specified in the `CollectionMetadata` of the `SignalCollection`. This includes validating that intermediate and derived signals maintain this standard when operations are applied.
- **Workflow Process**:
  1. Validate that required signals exist in the container.
  2. Retrieve input signal(s) from the container.
  3. Determine the operation type (single-signal or multi-signal).
  4. Look up and apply the specified operation with parameters.
  5. If `inplace=True` and the operation preserves signal class, update the existing signal.
  6. Otherwise, add the resulting signal back to the container with the specified output key.

- **Path Resolution**:
  - The `WorkflowExecutor` accepts an optional `data_dir` parameter, which specifies the base directory containing the data files referenced in the `import` section.
  - During execution, the executor resolves each `source` path by prepending `data_dir` (e.g., `{data_dir}/{source}`). 
  - If a `file_pattern` is present in an import specification, the executor scans the resolved directory for files matching the pattern and imports each one using the specified importer.
  - If no `file_pattern` is provided, the resolved `source` is treated as a specific file path.
  - The executor handles errors gracefully, raising clear exceptions if no files match a pattern or if a specified file does not exist.
- **Example**:
  ```python
  with open("workflow.yaml", "r") as file:
      workflow_config = yaml.safe_load(file)
  executor = WorkflowExecutor(container=signal_collection)
  # All results are stored in signal_collection, no separate results object
  executor.execute_workflow(workflow_config)
  # Access any processed signal after workflow execution
  heart_rate = signal_collection.get_signal("hr_0")
  ```

### 5.6 Timestamp Utility Function
A shared utility function, `convert_timestamp_format`, shall be implemented in the `utils` module to standardize timestamp conversion across all importers. This function shall:
- Accept a timestamp series or column, an optional `source_format` (from the importer config), and a `target_format` (from `CollectionMetadata`).
- Convert timestamps from the source format to the target format, supporting explicit format specification via `source_format`.
- If `source_format` is not provided, attempt to infer the format using robust parsing (e.g., pandas' `to_datetime` with error handling).
- Raise a `ValueError` with a descriptive message if conversion fails.
- Be used by all importers to ensure consistent timestamp standardization.

Example usage:
```python
from utils import convert_timestamp_format
standardized_timestamps = convert_timestamp_format(
    data['timestamp'],
    source_format='%Y-%m-%d %H:%M:%S',
    target_format=collection_metadata.timestamp_format
)
```

### 5.5 SignalImporter (Interface)
- **Purpose**: Defines a standard interface for signal importers with a multi-level class hierarchy to support diverse data formats and sources.
- **Features**:
  - **Abstract Base Class**: Provides a unified interface for all signal importers regardless of format or manufacturer.
  - **Type Safety**: Ensures that importers create properly typed `SignalData` instances based on the requested signal type.
  - **Standardized Metadata**: Enforces consistent metadata extraction and formatting across all importers.
  - **Error Handling**: Includes consistent validation and error reporting for import failures.
  - **Extensibility**: Supports easy extension to new data sources and formats through the class hierarchy.
  - **Batch Import**: Supports importing multiple signals from a single source or multiple files.
  - **Configuration-Driven**: Enables declarative configuration to define how source data maps to the framework's signal model.

#### 5.5.1 SignalImporter API
```python
class SignalImporter(ABC):
    """Abstract base class for signal importers."""
    
    def __init__(self):
        """Initialize the importer with a logger."""
        self.logger = get_logger(__name__)

    @abstractmethod
    def import_signal(self, source: str, signal_type: str) -> SignalData:
        """
        Import a single signal from the specified source.

        Args:
            source: Path or identifier of the data source (e.g., file path).
            signal_type: Type of the signal to import (e.g., "PPG").

        Returns:
            An instance of a SignalData subclass corresponding to the signal_type.
            
        Raises:
            ValueError: If the source is invalid or signal_type is not supported.
            IOError: If there are issues reading from the source.
        """
        pass

    @abstractmethod
    def import_signals(self, source: str, signal_type: str) -> List[SignalData]:
        """
        Import multiple signals from the specified source.

        Args:
            source: Path or identifier of the data source (e.g., file path).
            signal_type: Type of the signals to import (e.g., "PPG").

        Returns:
            A list of SignalData subclass instances.
            
        Raises:
            ValueError: If the source is invalid or signal_type is not supported.
            IOError: If there are issues reading from the source.
        """
        pass
    
    def _get_signal_class(self, signal_type: str) -> Type[SignalData]:
        """
        Get the appropriate SignalData subclass for the given signal_type.
        
        Args:
            signal_type: String identifier of the signal type (e.g., "ppg").
            
        Returns:
            The SignalData subclass corresponding to the signal_type.
            
        Raises:
            ValueError: If the signal_type is not recognized.
        """
        # Implementation to map string signal types to SignalData subclasses
        pass
    
    def _extract_metadata(self, data: Any, source: str, signal_type: str) -> Dict[str, Any]:
        """
        Extract metadata from the source and data.
        
        Args:
            data: The imported data.
            source: Original data source.
            signal_type: Type of signal being imported.
            
        Returns:
            Dictionary of metadata to initialize the SignalData instance.
        """
        # Implementation to extract and format metadata
        pass
    
    def _validate_source(self, source: str) -> None:
        """
        Validate that the source exists and is readable.
        
        Args:
            source: Path or identifier to validate.
            
        Raises:
            ValueError: If the source is invalid.
            IOError: If the source cannot be accessed.
        """
        # Implementation to validate source
        pass
```

#### 5.5.2 Importer Hierarchy Design

The importer system uses a three-level hierarchy to maximize code reuse and maintainability:

1. **SignalImporter (Abstract Base Class)**: 
   - Defines the common interface for all importers
   - Implements shared utility methods like `_get_signal_class` and logging
   - Enforces the contract that all importers must follow

2. **Format-Specific Importers (Abstract Classes)**:
   - Extend `SignalImporter` for specific file formats (e.g., `CSVImporterBase`, `HDF5ImporterBase`)
   - Implement format-specific parsing logic (e.g., CSV file reading)
   - Provide hooks for concrete subclasses to customize behavior
   - Still abstract, not meant to be instantiated directly

3. **Manufacturer/Sensor-Specific Importers (Concrete Classes)**:
   - Extend format-specific importers for particular manufacturers or datasets
   - Implement the declarative configuration approach
   - Define specific column mappings, validation rules, and metadata extraction
   - Can be instantiated and used directly
   
```
SignalImporter (ABC)
    ├── CSVImporterBase (ABC)
    │   ├── PolarCSVImporter
    │   ├── EmpaticaCSVImporter
    │   └── CustomDatasetCSVImporter
    ├── HDF5ImporterBase (ABC)
    │   ├── PhysioNetHDF5Importer
    │   └── CustomHDF5Importer
    └── JSONImporterBase (ABC)
        ├── WearableJSONImporter
        └── CustomJSONImporter
```

#### 5.5.3 Implementation Guidelines

1. **Error Handling**:
   - Importers should perform thorough validation before processing
   - Use specific exception types with clear error messages
   - Log detailed information about failures for debugging
   - Provide graceful fallbacks where appropriate

2. **Performance Considerations**:
   - Use streaming/chunked loading for large files
   - Implement memory optimization techniques for large datasets
   - Support filtering during import to limit data loaded

3. **Extending the System**:
   - To add support for a new data format, create a new abstract base class extending `SignalImporter`
   - To add support for a new manufacturer using an existing format, extend the appropriate format-specific importer
   - Use configuration classes/dictionaries to minimize code duplication

4. **Testing Requirements**:
   - Each importer should have unit tests for both valid and invalid inputs
   - Tests should verify correct metadata extraction and signal type assignment
   - Integration tests should verify interoperability with the `SignalCollection`

#### 5.5.5 Handling Multi-Level Columns (Optional)
If the framework needs to support signals with multi-level (hierarchical) columns, importers shall flatten the column structure to a single level using a configurable delimiter (e.g., '_') before applying the `column_mapping`. The `required_columns` in signal classes shall always refer to flat column names. This requirement is optional for the initial implementation but should be considered for future extensibility. Example:
- Raw data columns: `('accel', 'x')`, `('accel', 'y')`
- Flattened columns: `accel_x`, `accel_y`

#### 5.5.4 MergingImporter Subclass
```python
class MergingImporter(SignalImporter):
    """
    A specialized importer that merges multiple fragmented files into a single signal.
    """
    def __init__(self, config):
        """
        Initialize with configuration for file patterns and sorting.
        
        Args:
            config: Dict with keys 'file_pattern', 'time_column', and optional 'sort_by'.
        """
        self.file_pattern = config["file_pattern"]  # e.g., "ppg_*.csv"
        self.time_column = config["time_column"]    # e.g., "timestamp"
        self.sort_by = config.get("sort_by", "filename")  # "filename" or "timestamp"
    
    def import_signal(self, directory, signal_type):
        """
        Import and merge signals from multiple files.
        
        Args:
            directory: Path to the directory containing fragmented files.
            signal_type: Type of signal (e.g., "ppg").
        
        Returns:
            SignalData: A merged signal instance.
        """
        import glob
        import os
        import pandas as pd
        
        # Locate all files matching the pattern
        files = glob.glob(os.path.join(directory, self.file_pattern))
        
        # Sort files based on configuration
        if self.sort_by == "timestamp":
            files = self._sort_by_embedded_timestamp(files)
        else:
            files = sorted(files)  # Default to filename sort
        
        # Load and merge data
        dfs = [pd.read_csv(f) for f in files]
        merged_df = pd.concat(dfs).sort_values(self.time_column)
        
        # Return SignalData with updated metadata
        return SignalData(
            data=merged_df,
            metadata={
                "source_files": files,
                "merged": True,
                "signal_type": signal_type
                # Include other existing metadata fields as needed
            }
        )
    
    def _sort_by_embedded_timestamp(self, files):
        """
        Sort files by the earliest timestamp in each file.
        
        Args:
            files: List of file paths.
        
        Returns:
            List of file paths sorted by timestamp.
        """
        timestamps = []
        for f in files:
            df = pd.read_csv(f)
            earliest = df[self.time_column].min()
            timestamps.append((earliest, f))
        return [f for _, f in sorted(timestamps)]
```

- **Features**:
  - **Abstract Base Class**: `SignalImporter` provides the core interface for all importers.
  - **Intermediate Format-Specific Importers**: Abstract classes per data format (e.g., `CSVImporterBase`), extending `SignalImporter` to handle format-specific logic. Initially, only CSV is implemented, with plans for HDF5, JSON, etc.
  - **Concrete Sensor/Manufacturer-Specific Importers**: Classes like `PolarCSVImporter`, extending format-specific importers, tailored to specific sensors, manufacturers, datasets, or custom sources using declarative configurations.
- **Additional Notes**:
  - Format-specific importers (e.g., `CSVImporterBase`) encapsulate logic like file reading and validation.
  - Concrete importers rely on configurations to map data to `SignalData`, minimizing imperative code.
  - The hierarchy mirrors `SignalData`'s structure for consistency and extensibility.
  - Backward compatibility is maintained with the existing `SignalImporter` interface.

### 5.6 Importers
- **Purpose**: Converts raw data from various formats, sensors, manufacturers, and datasets into standardized signal instances using a declarative approach.
- **Structure**:
  - **`SignalImporter`**: Abstract base class for all importers.
  - **Format-Specific Importers**: Abstract classes like `CSVImporterBase`.
  - **Concrete Importers**: Classes like `PolarCSVImporter` that use configurations.
- **Example**:
  ```python
  config = {
      "column_mapping": {"time": "timestamp", "value": "ppg_value"},
      "time_format": "%Y-%m-%d %H:%M:%S",
      "filename_pattern": r"polar_(?P<subject_id>\w+)_(?P<session>\d+).csv"
  }
  importer = PolarCSVImporter(config)
  ppg_signal = importer.import_signal("path/to/polar_subject1_01.csv", signal_type="ppg")
  ```

### 5.7 SignalData Implementation Details
To reduce boilerplate code in child classes, the `SignalData` abstract base class includes a robust implementation of `apply_operation` with type hints:

```python
class SignalData(ABC):
    registry = {}  # Base registry - will be populated by child classes
    signal_type: SignalType = None  # To be overridden by subclasses
    
    def __init__(self, data, metadata: Dict[str, Any]):
        if self.signal_type is None:
            raise ValueError("Subclasses must define signal_type")
        metadata = metadata.copy()  # Avoid modifying the input dictionary
        metadata['signal_type'] = self.signal_type
        self.metadata = metadata
        self.data = data
    
    @abstractmethod
    def get_data(self):
        """Return the signal's data."""
        pass
    
    @abstractmethod
    def apply_operation(self, operation_name: str, inplace: bool = False, **parameters) -> 'SignalData':
        """
        Apply an operation to this signal by name.
        
        First attempts to find a matching method on the signal instance,
        then falls back to the class registry if no method is found.
        
        Args:
            operation_name: String name of the operation.
            inplace: If True and operation preserves signal class, modify this signal in place.
                     If False or operation changes signal class, create and return a new signal.
            **parameters: Keyword arguments to pass to the operation.
            
        Returns:
            Either this signal instance (if inplace=True) or a new signal instance with the operation results.
            
        Raises:
            ValueError: If operation not found in either methods or registry.
            ValueError: If inplace=True for an operation that changes signal class.
        """
        # Abstract method implementation would typically look like:
        # method = getattr(self, operation_name, None)
        # if method and callable(method):
        #     if inplace:
        #         # Check if method supports inplace (has inplace parameter)
        #         import inspect
        #         sig = inspect.signature(method)
        #         if 'inplace' in sig.parameters:
        #             return method(inplace=True, **parameters)
        #         else:
        #             raise ValueError(f"Method {operation_name} does not support inplace operation")
        #     else:
        #         return method(**parameters)
        # 
        # registry = self.__class__.get_registry()
        # if operation_name in registry:
        #     operation, output_class = registry[operation_name]
        #     
        #     # For in-place operations, check that output class matches input class
        #     if inplace and output_class != self.__class__:
        #         raise ValueError(f"Cannot perform in-place operation that changes signal class "
        #                         f"from {self.__class__.__name__} to {output_class.__name__}")
        #                         
        #     result_data = operation([self.get_data()], parameters)
        #     
        #     # If in-place and compatible, modify this signal
        #     if inplace and output_class == self.__class__:
        #         self.data = result_data
        #         self.metadata.operations.append({"function_id": operation_name, "parameters": parameters})
        #         return self
        #     else:
        #         # Otherwise create new signal
        #         # Store both signal_id and current operation_index for traceability
        #         operation_index = len(self.metadata.operations) - 1
        #         return output_class(data=result_data, metadata={
        #             "derived_from": [(self.metadata.signal_id, operation_index)],
        #             "operations": [{"function_id": operation_name, "parameters": parameters}]
        #         })
        # raise ValueError(f"Operation '{operation_name}' not found for {self.__class__.__name__}")
```

---

## 6. Workflow Execution

Workflows are defined in YAML (or JSON) and include:
- **Import Section** (optional): Specifies how to create the `SignalCollection` by importing signals. Paths in this section are relative to a user-specified `data_dir` provided via the command-line interface.
- **Processing Section**: Defines operations to apply, either by signal type (sensor-agnostic) or by specific signal keys (non-agnostic).

### 6.1.1 Import Section Details

The `import` section specifies how to populate the `SignalCollection` by importing signals from various sources. Each import specification includes:

- `signal_type`: The type of signal to import (e.g., "heart_rate"), mapped to `SignalType` enum values.
- `importer`: The name of the importer class to use (e.g., "PolarCSVImporter").
- `source`: A relative path to the data source, which can be either:
  - A directory (e.g., "polar_data") relative to a user-specified data directory.
  - A specific file path (e.g., "polar_data/hr_file.txt") relative to the data directory.
- `file_pattern`: An optional glob pattern (e.g., "*_HR.txt") to match multiple files within the `source` directory. If provided, the `source` is treated as a directory, and all matching files are imported.
- `config`: An optional dictionary for importer-specific configuration (e.g., column mappings).

When a `file_pattern` is specified, the framework imports all files in the `source` directory (resolved relative to the provided `data_dir`) that match the pattern. If no `file_pattern` is provided, the `source` is treated as a specific file path. The `WorkflowExecutor` resolves these paths at runtime using a `data_dir` parameter supplied via the CLI, enabling the YAML to remain agnostic to absolute file locations.

### 6.1 YAML Workflow Example
```yaml
import:
  - signal_type: "ppg"
    importer: "MergingImporter"
    source: "/data/subject01/raw"
    config:
      file_pattern: "ppg_*.csv"
      time_column: "timestamp"
      sort_by: "timestamp"  # Optional; defaults to "filename"
    base_name: "merged_ppg_signal"

  - signal_type: "ppg"
    importer: "PolarCSVImporter"  # Concrete importer class
    source: "polar_data/subject1_01.csv"  # Relative path resolved against data_dir
    config:
      column_mapping:
        time: "timestamp"
        value: "ppg_value"
      time_format: "%Y-%m-%d %H:%M:%S"
      filename_pattern: "polar_(?P<subject_id>\w+)_(?P<session>\d+).csv"
    sensor_type: "PPG"
    sensor_model: "PolarH10"
    body_position: "left_wrist"
    base_name: "ppg_left"
    
  - signal_type: "ppg"
    importer: "MergingImporter"
    source: "/data/subject01/raw"
    config:
      file_pattern: "ppg_*.csv"
      time_column: "timestamp"
      sort_by: "timestamp"  # Optional; defaults to "filename"
    base_name: "merged_ppg_signal"

  - signal_type: "ppg"
    importer: "PolarCSVImporter"
    source: "polar_data"  # Directory relative to data_dir
    file_pattern: "*_right_*.csv"  # Pattern to match specific files
    config:
      column_mapping:
        time: "timestamp"
        value: "ppg_value"
      time_format: "%Y-%m-%d %H:%M:%S"
      filename_pattern: "polar_(?P<subject_id>\w+)_(?P<session>\d+).csv"
    sensor_type: "PPG"
    sensor_model: "PolarH10"
    body_position: "right_wrist"
    base_name: "ppg_right"  # Custom base name to distinguish from left-wrist PPG
    
  - signal_type: "accelerometer"
    importer: "PolarCSVImporter"
    source: "accel_data"  # Directory relative to data_dir
    file_pattern: "*.csv"  # Import all CSV files in the directory
    config:
      column_mapping:
        time: "timestamp"
        x: "accel_x"
        y: "accel_y"
        z: "accel_z"
      time_format: "%Y-%m-%d %H:%M:%S"
      filename_pattern: "polar_(?P<subject_id>\w+)_(?P<session>\d+).csv"
    sensor_type: "ACCEL"
    sensor_model: "PolarH10"
    body_position: "chest"
```
- **Note**: The `importer` field specifies the concrete class, and `config` provides its declarative configuration.

# Processing section: Operate on the signals
```yaml
configuration:
  index_fields: ["body_position", "sensor_model", "signal_id"]

steps:
  # Index configuration
  - operation: "set_index_config"
    parameters:
      fields: ["body_position", "sensor_model", "signal_id"]
      
  # Applying operation to all signals with same base name
  - operation: "filter_lowpass"
    input: "ppg_left"               # Base name references all ppg_left signals
    output: "filtered_ppg_left"     # Auto-indexed as filtered_ppg_left_0, filtered_ppg_left_1, etc.
    parameters:
      cutoff: 5
  
  # Applying operation to a specific indexed signal
  - operation: "filter_lowpass"
    input: "ppg_right_0"            # Specific indexed signal
    output: "filtered_ppg_right"    # Single output (no indexing needed)
    parameters:
      cutoff: 5
  
  # Filtering by metadata criteria
  - operation: "filter_lowpass"
    input: 
      base_name: "ppg"
      criteria:
        sensor_type: "PPG"
        body_position: "chest"
    output: "filtered_ppg_chest"
    parameters:
      cutoff: 5
      
  # Multiple metadata criteria filters
  - operation: "filter_highpass"
    input:
      base_name: "accelerometer"
      criteria:
        sensor_model: "PolarH10"
        body_position: "left_wrist"
    output: "filtered_accel_left"
    parameters:
      cutoff: 0.5
  
  # In-place operation example
  - operation: "filter_highpass"
    input: "filtered_ppg_left_0"    # Operate on existing signal
    inplace: true                   # Modify signal in-place, no output needed
    parameters:
      cutoff: 0.5
  
  # List-based input/output with equal length
  - operation: "filter_bandpass"
    input: ["ppg_left_0", "ppg_right_0"]  # List of specific signals
    output: ["bandpass_left", "bandpass_right"]  # Corresponding outputs
    parameters:
      low_cutoff: 0.5
      high_cutoff: 5

  # Operation that changes signal type
  - operation: "compute_heart_rate"
    input: "filtered_ppg_left_0"
    output: "hr_left"               # Different signal type, inplace not supported
    parameters:
      window_size: 30
  
  # Multi-signal operation
  - operation: "compute_correlation"
    inputs: ["filtered_ppg_left_0", "accelerometer_0"]  # Using indexed signal names
    output: "correlation_result"
    parameters:
      method: "pearson"
  
  # Feature extraction operations
  - operation: "feature_mean"
    inputs: ["hr_left"]
    output: "hr_mean_features"
    parameters:
      window_length: 30
      step_size: 10
      
  - operation: "feature_std"
    inputs: ["hr_left"]
    output: "hr_std_features"
    parameters:
      window_length: 30
      step_size: 10
      
  - operation: "feature_correlation"
    inputs: ["hr_left", "accelerometer_0"]
    output: "hr_accel_correlation"
    parameters:
      window_length: 30
      step_size: 30
      method: "pearson"

# Export section: Export processed signals
export:
  formats: ["excel", "csv"]
  output_dir: "./exports"
  include_combined: true
```

You can also set the index configuration globally for the entire workflow:

```yaml
configuration:
  index_fields: ["body_position", "sensor_model", "signal_id"]

steps:
  # Workflow steps here
```

### 6.2 WorkflowExecutor Implementation
```python
class WorkflowExecutor:
    def __init__(self, container=None, strict_validation=True, data_dir=None):
        self.container = container or SignalCollection()
        self.strict_validation = strict_validation  # Controls validation behavior
        self.data_dir = data_dir  # Base directory for data files

    def execute_workflow(self, workflow_config):
        # Handle import section if present
        if "import" in workflow_config:
            # Dictionary to track counts for each base name
            base_name_counts = {}
            
            for spec in workflow_config["import"]:
                # Create importer instance with specified configuration
                importer_class = globals()[spec["importer"]]
                importer = importer_class(spec.get("config", {}))
                
                # Resolve the source path using data_dir if provided
                source = spec["source"]
                if self.data_dir:
                    source = os.path.join(self.data_dir, source)
                
                # Check if we should use a file pattern to match multiple files
                if "file_pattern" in spec:
                    # Source is a directory, find all matching files
                    if not os.path.isdir(source):
                        if self.strict_validation:
                            raise ValueError(f"Source directory not found: {source}")
                        else:
                            import warnings
                            warnings.warn(f"Source directory not found: {source}, skipping")
                            continue
                            
                    # Get all files matching the pattern
                    import glob
                    file_pattern = os.path.join(source, spec["file_pattern"])
                    matching_files = glob.glob(file_pattern)
                    
                    if not matching_files:
                        if self.strict_validation:
                            raise ValueError(f"No files found matching pattern: {file_pattern}")
                        else:
                            import warnings
                            warnings.warn(f"No files found matching pattern: {file_pattern}, skipping")
                            continue
                    
                    # Import each matching file
                    signals = []
                    for file_path in matching_files:
                        try:
                            file_signals = importer.import_signals(file_path, spec["signal_type"])
                            signals.extend(file_signals)
                        except Exception as e:
                            if self.strict_validation:
                                raise
                            else:
                                import warnings
                                warnings.warn(f"Error importing {file_path}: {e}, skipping")
                else:
                    # Direct import from the specified source
                    if not os.path.exists(source):
                        if self.strict_validation:
                            raise ValueError(f"Source file not found: {source}")
                        else:
                            import warnings
                            warnings.warn(f"Source file not found: {source}, skipping")
                            continue
                            
                    # Import signals from source
                    signals = importer.import_signals(source, spec["signal_type"])
                
                # Get or create base name
                if "base_name" in spec:
                    base_name = spec["base_name"]
                else:
                    # Default to lowercase signal type
                    base_name = spec["signal_type"].lower()
                    
                # Initialize counter if this is a new base name
                if base_name not in base_name_counts:
                    base_name_counts[base_name] = 0
                    
                # Add each signal with indexed key
                for signal in signals:
                    # Add metadata from import specification to the signal
                    if "sensor_type" in spec:
                        signal.metadata.sensor_type = self.str_to_enum(spec["sensor_type"], SensorType)
                    if "sensor_model" in spec:
                        signal.metadata.sensor_model = self.str_to_enum(spec["sensor_model"], SensorModel)
                    if "body_position" in spec:
                        signal.metadata.body_position = self.str_to_enum(spec["body_position"], BodyPosition)
                        
                    key = f"{base_name}_{base_name_counts[base_name]}"
                    self.container.add_signal(key, signal)
                    base_name_counts[base_name] += 1
                    
    def str_to_enum(self, value_str: str, enum_class) -> Any:
        """Convert a string to an enum value using case-insensitive matching."""
        try:
            return enum_class[value_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid {enum_class.__name__} value: {value_str}")
            
    def parse_metadata_from_config(self, config: Dict[str, Any]) -> SignalMetadata:
        """Parse metadata from YAML config, converting strings to enums."""
        signal_type_str = config.get("signal_type")
        units_str = config.get("units")
        sensor_type_str = config.get("sensor_type")
        sensor_model_str = config.get("sensor_model")
        body_position_str = config.get("body_position")

        try:
            signal_type = self.str_to_enum(signal_type_str, SignalType) if signal_type_str else None
            units = self.str_to_enum(units_str, Unit) if units_str else None
            sensor_type = self.str_to_enum(sensor_type_str, SensorType) if sensor_type_str else None
            sensor_model = self.str_to_enum(sensor_model_str, SensorModel) if sensor_model_str else None
            body_position = self.str_to_enum(body_position_str, BodyPosition) if body_position_str else None
        except ValueError as e:
            raise ValueError(f"Invalid enum value in config: {e}")

        return SignalMetadata(
            signal_id=config["signal_id"],
            signal_type=signal_type,
            units=units,
            sensor_type=sensor_type,
            sensor_model=sensor_model,
            body_position=body_position,
            # Add other required fields with defaults as needed
            name=None,
            sample_rate=None,
            start_time=None,
            end_time=None,
            derived_from=[],
            operations=[],
            temporary=False,
            sensor_info=None
        )

        # Execute processing steps
        for step in workflow_config.get("steps", []):
            self.execute_step(step)
        
        # Handle export section if present
        if "export" in workflow_config:
            exporter = ExportModule(self.container)
            exporter.export(
                formats=workflow_config["export"]["formats"],
                output_dir=workflow_config["export"]["output_dir"],
                include_combined=workflow_config["export"].get("include_combined", False)
            )

    def get_signals_by_input_specifier(self, input_specifier):
        """
        Get signals based on an input specifier which can be a base name or indexed name.
        
        Args:
            input_specifier: String specifying the signal(s) to retrieve. Can be:
                             - Base name (e.g., "ppg") to get all signals with that base name
                             - Indexed name (e.g., "ppg_0") to get a specific signal
                             
        Returns:
            List of signals matching the specifier
            
        Raises:
            Warning (not exception) if signals not found and strict_validation=False
        """
        # Check if this is an indexed name (contains underscore and ends with number)
        if "_" in input_specifier and input_specifier.split("_")[-1].isdigit():
            # This is an indexed name, get the specific signal
            if input_specifier in self.container.signals:
                return [self.container.get_signal(input_specifier)]
            else:
                if self.strict_validation:
                    raise ValueError(f"Signal {input_specifier} not found")
                else:
                    import warnings
                    warnings.warn(f"Signal {input_specifier} not found, skipping")
                    return []
        else:
            # This is a base name, get all signals with this base name
            base_name = input_specifier
            signals = []
            i = 0
            while True:
                key = f"{base_name}_{i}"
                if key in self.container.signals:
                    signals.append(self.container.get_signal(key))
                    i += 1
                else:
                    break
            
            if not signals and self.strict_validation:
                raise ValueError(f"No signals with base name {base_name} found")
            elif not signals:
                import warnings
                warnings.warn(f"No signals with base name {base_name} found, skipping")
                
            return signals

    def execute_step(self, step):
        operation_name = step["operation"]
        inplace = step.get("inplace", False)
        
        # Handle in-place operations (which modify existing signals)
        if inplace:
            # For in-place operations, no output is needed
            if "output" in step:
                raise ValueError("Output should not be specified for in-place operations")
                
            # Get input signal(s)
            input_signals = []
            if "input" in step:
                if isinstance(step["input"], list):
                    # Handle list of inputs
                    for input_spec in step["input"]:
                        input_signals.extend(self.get_signals_by_input_specifier(input_spec))
                else:
                    # Handle single input
                    input_signals = self.get_signals_by_input_specifier(step["input"])
            else:
                raise ValueError("In-place operation must specify 'input'")
                
            # Apply operation in-place to each signal
            for signal in input_signals:
                signal.apply_operation(operation_name, inplace=True, **step.get("parameters", {}))
                
            return

        # Handle non-in-place operations (which create new signals)
        if "output" not in step:
            raise ValueError("Output must be specified for non-in-place operations")
            
        output_name = step["output"]
        
        # Multi-signal operation
        if "inputs" in step:
            if operation_name in self.container.multi_signal_registry:
                # Get all input signals
                signals = []
                for input_spec in step["inputs"]:
                    input_signals = self.get_signals_by_input_specifier(input_spec)
                    signals.extend(input_signals)
                
                if not signals:
                    if self.strict_validation:
                        raise ValueError(f"No input signals found for step with operation {operation_name}")
                    else:
                        import warnings
                        warnings.warn(f"No input signals found for step with operation {operation_name}, skipping")
                        return
                
                # Apply the multi-signal operation
                func, output_class = self.container.multi_signal_registry[operation_name]
                result_data = func([s.get_data() for s in signals], step.get("parameters", {}))
                
                # For multi-signal operations, capture source signal states
                derived_from_list = []
                for signal in signals:
                    operation_index = len(signal.metadata.operations) - 1
                    derived_from_list.append((signal.metadata.signal_id, operation_index))
                
                output_signal = output_class(
                    data=result_data,
                    metadata={
                        "derived_from": derived_from_list,
                        "operations": [{"function_id": operation_name, "parameters": step.get("parameters", {})}]
                    }
                )
                self.container.add_signal(output_name, output_signal)
            else:
                raise ValueError(f"Multi-signal operation '{operation_name}' not found")
        
        # Single-signal operation(s) using input field
        elif "input" in step:
            # Handle list of inputs with list of outputs
            if isinstance(step["input"], list):
                # Validate that output is also a list of the same length
                if not isinstance(step["output"], list):
                    raise ValueError("When 'input' is a list, 'output' must also be a list")
                if len(step["input"]) != len(step["output"]):
                    raise ValueError("'input' and 'output' lists must have the same length")
                
                # Process each input-output pair
                for i, (input_spec, output_spec) in enumerate(zip(step["input"], step["output"])):
                    input_signals = self.get_signals_by_input_specifier(input_spec)
                    
                    # Apply operation to each signal and add to container
                    for j, signal in enumerate(input_signals):
                        if len(input_signals) > 1:
                            # Multiple signals for this input, add index to output
                            output_key = f"{output_spec}_{j}"
                        else:
                            # Just one signal, use output as is
                            output_key = output_spec
                            
                        output_signal = signal.apply_operation(operation_name, **step.get("parameters", {}))
                        self.container.add_signal(output_key, output_signal)
            
            # Handle single input (string)
            else:
                input_signals = self.get_signals_by_input_specifier(step["input"])
                
                # Apply operation to each signal and add to container
                for i, signal in enumerate(input_signals):
                    output_key = f"{output_name}_{i}" if len(input_signals) > 1 else output_name
                    output_signal = signal.apply_operation(operation_name, **step.get("parameters", {}))
                    self.container.add_signal(output_key, output_signal)
                    
        # Using metadata filtering
        elif "input" in step and isinstance(step["input"], dict):
            # Get base name and criteria
            base_name = step["input"].get("base_name")
            criteria = step["input"].get("criteria", {})
                
            # Convert string criteria to enum values
            for key, value in criteria.items():
                if key == "sensor_type":
                    criteria[key] = self.str_to_enum(value, SensorType)
                elif key == "sensor_model":
                    criteria[key] = self.str_to_enum(value, SensorModel)
                elif key == "body_position":
                    criteria[key] = self.str_to_enum(value, BodyPosition)
                
            # Add base_name as a filter if provided
            if base_name:
                # Get all signals with matching base name and criteria
                signals = []
                i = 0
                while True:
                    key = f"{base_name}_{i}"
                    if key in self.container.signals:
                        signal = self.container.get_signal(key)
                        # Check if signal meets all criteria
                        if all(signal.metadata.get(k) == v for k, v in criteria.items()):
                            signals.append(signal)
                        i += 1
                    else:
                        break
            else:
                # Get signals matching only the criteria
                signals = self.container.get_signals(criteria)
                
            if not signals:
                if self.strict_validation:
                    raise ValueError(f"No signals matching criteria {criteria} found")
                else:
                    import warnings
                    warnings.warn(f"No signals matching criteria {criteria} found, skipping")
                    return
                
            # Apply the operation to each signal and add to container
            for i, signal in enumerate(signals):
                output_key = f"{output_name}_{i}" if len(signals) > 1 else output_name
                output_signal = signal.apply_operation(operation_name, **step.get("parameters", {}))
                self.container.add_signal(output_key, output_signal)
        else:
            raise ValueError("Step must specify 'input' or 'inputs'")
```

### 6.3 Mapping String Representations to Enums in Workflows

In workflow configurations (YAML or JSON), string values for enum types like `sensor_type`, `sensor_model`, and `body_position` need to be mapped to their corresponding enum members internally for type safety. The mapping is case-insensitive and based on the enum's `value` field. For example:

- `"ppg"`, `"PPG"`, and `"Ppg"` all map to `SensorType.PPG`.
- `"chest"`, `"CHEST"`, and `"Chest"` all map to `BodyPosition.CHEST`.

The system does not support additional alternate representations unless explicitly required and implemented in the future.

The `WorkflowExecutor` uses a generic `str_to_enum` method to perform this conversion. If an invalid string is provided (i.e., it does not match any enum value case-insensitively), a `ValueError` is raised with a descriptive message.

---

## 6.4 Testing the Generic Signal Retrieval

The generic signal retrieval mechanism should be thoroughly tested to ensure it works correctly. Here's an example of unit tests for this functionality:

```python
import unittest
from signal_collection import SignalCollection
from signals import PPGSignal, AccelerometerSignal
from signal_types import SignalType

class TestSignalProcessing(unittest.TestCase):
    def setUp(self):
        self.collection = SignalCollection()
        self.collection.add_signal(PPGSignal(data=[1, 2, 3], metadata={"id": "ppg1"}))
        self.collection.add_signal(AccelerometerSignal(data=[4, 5, 6], metadata={"id": "acc1"}))

    def test_signal_type_in_metadata(self):
        ppg_signal = self.collection.signals[0]
        self.assertEqual(ppg_signal.metadata["signal_type"], SignalType.PPG)

    def test_get_signals_by_type(self):
        ppg_signals = self.collection.get_signals({"signal_type": SignalType.PPG})
        self.assertEqual(len(ppg_signals), 1)
        self.assertEqual(ppg_signals[0].metadata["id"], "ppg1")

    def test_get_signals_by_multiple_criteria(self):
        acc_signal = self.collection.get_signals({"signal_type": SignalType.ACCELEROMETER, "id": "acc1"})
        self.assertEqual(len(acc_signal), 1)
        self.assertEqual(acc_signal[0].metadata["id"], "acc1")
```

These tests verify:
1. That the `signal_type` is correctly set in the metadata during initialization.
2. That `get_signals` correctly filters by signal type.
3. That `get_signals` correctly filters by multiple criteria.

## 7. Importers
- **Purpose**: Converts raw data from various formats, sensors, manufacturers, public datasets, and custom sources into standardized `SignalData` instances using a declarative, configuration-driven approach.
- **Structure**:
  - **Abstract Base Class**: `SignalImporter` defines the interface.
  - **Intermediate Format-Specific Importers**: Abstract classes like `CSVImporterBase` handle format-specific logic (e.g., CSV parsing). Future classes (e.g., `HDF5ImporterBase`, `JSONImporterBase`) will be added as needed.
  - **Concrete Sensor/Manufacturer-Specific Importers**: Classes like `PolarCSVImporter` or `CustomDatasetCSVImporter`, extending format-specific importers, use configurations for data mapping.
- **Declarative Configuration**:
  - Importers use configuration objects (e.g., dictionaries or YAML files) to specify:
    - **Column Mappings**: Map source columns to the standardized column names required by the target signal class (e.g., `"time": "timestamp"`, `"accX": "x"`).
    - **Time Formats**: Define timestamp parsing (e.g., `"%Y-%m-%d %H:%M:%S"`).
    - **Filename Patterns**: Regular expressions to extract metadata (e.g., `r"polar_(?P<subject_id>\w+)_(?P<session>\d+).csv"`).
  - This approach reduces imperative code, enabling new importers via configuration updates.
- **Column Standardization**:
  - Each signal class defines a `required_columns` class attribute listing its expected columns.
  - Importers map raw data columns to these standard columns using declarative configuration.
  - The system validates that all required columns are present after mapping.
  - This ensures consistent column naming across all importers targeting the same signal type.
- **Timestamp Standardization**:
  - All timestamps are converted to a standard format specified in `CollectionMetadata.timestamp_format`.
  - Importers handle the conversion from various input formats to this standard format.
  - A shared utility function performs the conversion to ensure consistency.
- **Example**:
  ```python
  # Configuration for PolarCSVImporter
  config = {
      "column_mapping": {"time": "timestamp", "value": "ppg_value"},
      "time_format": "%Y-%m-%d %H:%M:%S",
      "filename_pattern": r"polar_(?P<subject_id>\w+)_(?P<session>\d+).csv"
  }
  importer = PolarCSVImporter(config)
  ppg_signal = importer.import_signal("path/to/polar_subject1_01.csv", signal_type="ppg")
  ```

### 7.1 Declarative Importer Configuration
- **Purpose**: Enable a declarative approach to defining importers, minimizing imperative code and simplifying the addition of new importers.
- **Features**:
  - **Configuration Objects**: Define importer behavior via:
    - **Column Mappings**: Map source data fields to `SignalData` attributes.
    - **Time Formats**: Specify timestamp formats for parsing.
    - **Filename Patterns**: Extract metadata (e.g., subject ID, session) from filenames using regex.
    - **Required Columns**: Specify which columns must be present in the source data.
    - **Data Validation Rules**: Define constraints that the imported data must satisfy.
    - **Preprocessing Options**: Specify transformations to apply during import (e.g., unit conversion).
  - **Benefits**:
    - Reduces code duplication and complexity in concrete importers.
    - Allows rapid addition of importers for new sensors or datasets by updating configurations.
    - Enhances robustness by standardizing data extraction.
    - Enables consistent preprocessing and validation across importers.
    - Makes importers self-documenting through their configurations.
    - Facilitates testing through dependency injection of configurations.
- **Example Configuration** (in YAML):
  ```yaml
  column_mapping:
    time: "timestamp"
    value: "ppg_value"
  time_format: "%Y-%m-%d %H:%M:%S"
  filename_pattern: "polar_(?P<subject_id>\w+)_(?P<session>\d+).csv"
  required_columns: ["timestamp", "ppg_value"]
  validation_rules:
    - type: "range_check"
      column: "ppg_value"
      min: 0
      max: 65535
  preprocessing:
    - type: "unit_conversion"
      column: "ppg_value"
      from_unit: "raw"
      to_unit: "normalized"
      factor: 0.01
  ```

#### 7.1.1 Configuration Schema
The configuration schema for importers includes the following components:

1. **Basic Configuration**:
   - `column_mapping`: Dictionary mapping framework field names to source data column names
   - `time_format`: Format string for parsing timestamps (using Python's datetime format codes)
   - `filename_pattern`: Regular expression with named capture groups for extracting metadata
   
   Importers shall retain only the columns specified in the `column_mapping` configuration. Any columns in the raw data not included in `column_mapping` shall be dropped to ensure consistency and prevent unexpected data from propagating through the framework.

2. **Validation Configuration**:
   - `required_columns`: List of column names that must be present in the source data
   - `validation_rules`: List of rule objects defining constraints on the imported data
   - `missing_value_handling`: Strategy for handling missing values ("drop", "interpolate", "zero", etc.)

3. **Preprocessing Configuration**:
   - `preprocessing`: List of preprocessing steps to apply during import
   - Each step includes: type, target column(s), and step-specific parameters

4. **Metadata Configuration**:
   - `metadata_extraction`: Rules for populating metadata fields from the source data
   - `default_metadata`: Default values for metadata fields when not found in the source

5. **Import Behavior**:
   - `chunking`: Configuration for chunked processing of large files
   - `filter_criteria`: Optional criteria to filter data during import (to avoid loading everything)
   - `merge_strategy`: How to handle multiple files or sources ("concat", "merge", etc.)

#### 7.1.2 Implementation Approach

The implementation of declarative configuration uses a combination of:

1. **Factory Methods**:
   ```python
   @classmethod
   def from_config(cls, config: Dict[str, Any]) -> 'ConcreteImporter':
       """Create an importer instance from a configuration dictionary."""
       return cls(
           column_mapping=config.get("column_mapping", {}),
           time_format=config.get("time_format"),
           # Additional configuration parameters
       )
   ```

2. **Configuration Validation**:
   ```python
   def _validate_config(self, config: Dict[str, Any]) -> None:
       """Validate that the configuration is complete and valid."""
       required_fields = ["column_mapping", "time_format"]
       missing = [f for f in required_fields if f not in config]
       if missing:
           raise ValueError(f"Missing required configuration fields: {missing}")
           
       # Additional validation logic
   ```

3. **Column Mapping and Validation**:
   ```python
   def _apply_column_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
       """Apply column mapping to standardize column names."""
       # Create a new DataFrame with mapped columns
       result = pd.DataFrame()
       
       # Apply column mapping from source to target column names
       for source_col, target_col in self.config["column_mapping"].items():
           if source_col in data.columns:
               result[target_col] = data[source_col]
           else:
               raise ValueError(f"Source column '{source_col}' not found in data")
       
       # Validate that all required columns are present
       signal_class = self._get_signal_class(self.config["signal_type"])
       missing_cols = [col for col in signal_class.required_columns if col not in result.columns]
       if missing_cols:
           raise ValueError(f"Missing required columns: {missing_cols}")
           
       return result
   ```

4. **Timestamp Conversion**:
   ```python
   def _convert_timestamps(self, data: pd.DataFrame, collection_metadata: CollectionMetadata) -> pd.DataFrame:
       """Convert timestamps to the standard format specified in collection_metadata."""
       if 'timestamp' not in data.columns:
           raise ValueError("Data does not contain a 'timestamp' column")
           
       # Use the shared timestamp conversion utility
       source_format = self.config.get("time_format")
       target_format = collection_metadata.timestamp_format
       
       # Convert timestamps to the target format
       data = data.copy()
       data['timestamp'] = convert_timestamp_format(
           data['timestamp'], 
           source_format=source_format,
           target_format=target_format
       )
       
       return data
   ```

5. **Behavioral Components**:
   ```python
   def _apply_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
       """Apply preprocessing steps defined in the configuration."""
       for step in self.config.get("preprocessing", []):
           step_type = step["type"]
           if step_type == "unit_conversion":
               data = self._apply_unit_conversion(data, step)
           elif step_type == "filtering":
               data = self._apply_filtering(data, step)
           # Additional preprocessing types
       return data
   ```

This approach ensures that concrete importers inherit common behavior while allowing for customization through configuration injection.
To reduce boilerplate code in child classes, the `SignalData` abstract base class includes a robust implementation of `apply_operation` with type hints:

```python
class SignalData(ABC):
    registry = {}  # Base registry - will be populated by child classes
    signal_type: SignalType = None  # To be overridden by subclasses
    
    def __init__(self, data, metadata: Dict[str, Any]):
        if self.signal_type is None:
            raise ValueError("Subclasses must define signal_type")
        metadata = metadata.copy()  # Avoid modifying the input dictionary
        metadata['signal_type'] = self.signal_type
        self.metadata = metadata
        self.data = data
    
    @abstractmethod
    def get_data(self):
        """Return the signal's data."""
        pass
    
    @abstractmethod
    def apply_operation(self, operation_name: str, inplace: bool = False, **parameters) -> 'SignalData':
        """
        Apply an operation to this signal by name.
        
        First attempts to find a matching method on the signal instance,
        then falls back to the class registry if no method is found.
        
        Args:
            operation_name: String name of the operation.
            inplace: If True and operation preserves signal class, modify this signal in place.
                     If False or operation changes signal class, create and return a new signal.
            **parameters: Keyword arguments to pass to the operation.
            
        Returns:
            Either this signal instance (if inplace=True) or a new signal instance with the operation results.
            
        Raises:
            ValueError: If operation not found in either methods or registry.
            ValueError: If inplace=True for an operation that changes signal class.
        """
        # Abstract method implementation would typically look like:
        # method = getattr(self, operation_name, None)
        # if method and callable(method):
        #     if inplace:
        #         # Check if method supports inplace (has inplace parameter)
        #         import inspect
        #         sig = inspect.signature(method)
        #         if 'inplace' in sig.parameters:
        #             return method(inplace=True, **parameters)
        #         else:
        #             raise ValueError(f"Method {operation_name} does not support inplace operation")
        #     else:
        #         return method(**parameters)
        # 
        # registry = self.__class__.get_registry()
        # if operation_name in registry:
        #     operation, output_class = registry[operation_name]
        #     
        #     # For in-place operations, check that output class matches input class
        #     if inplace and output_class != self.__class__:
        #         raise ValueError(f"Cannot perform in-place operation that changes signal class "
        #                         f"from {self.__class__.__name__} to {output_class.__name__}")
        #                         
        #     result_data = operation([self.get_data()], parameters)
        #     
        #     # If in-place and compatible, modify this signal
        #     if inplace and output_class == self.__class__:
        #         self.data = result_data
        #         self.metadata.operations.append({"function_id": operation_name, "parameters": parameters})
        #         return self
        #     else:
        #         # Otherwise create new signal
        #         # Store both signal_id and current operation_index for traceability
        #         operation_index = len(self.metadata.operations) - 1
        #         return output_class(data=result_data, metadata={
        #             "derived_from": [(self.metadata.signal_id, operation_index)],
        #             "operations": [{"function_id": operation_name, "parameters": parameters}]
        #         })
        # raise ValueError(f"Operation '{operation_name}' not found for {self.__class__.__name__}")
```

---

## 8. Detailed Design with Examples

### 8.1 Workflow YAML with Relative Paths and File Patterns

Below is an example of a workflow YAML with relative paths and file patterns:

```yaml
import:
  - signal_type: "heart_rate"
    importer: "PolarCSVImporter"
    source: "polar_data"  # Relative directory within data_dir
    file_pattern: "*_HR.txt"  # Matches all heart rate files
    config:
      column_mapping:
        timestamp: "Phone timestamp"
        hr: "HR [bpm]"
      time_format: "%Y-%m-%dT%H:%M:%S.%f"
    sensor_type: "EKG"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr"

  - signal_type: "accelerometer"
    importer: "PolarCSVImporter"
    source: "polar_data"  # Same directory, different pattern
    file_pattern: "*_ACC.txt"  # Matches all accelerometer files
    config:
      column_mapping:
        timestamp: "Phone timestamp"
        x: "X [mg]"
        y: "Y [mg]"
        z: "Z [mg]"
      time_format: "%Y-%m-%dT%H:%M:%S.%f"
    sensor_type: "ACCEL"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "accel"

steps:
  # Processing steps remain the same
  - operation: "filter_lowpass"
    input: "hr"
    output: "filtered_hr"
    parameters:
      cutoff: 5

export:
  formats: ["csv"]
  output_dir: "./output"
```

In this example:
- The `source` fields specify paths relative to the `data_dir` provided via the CLI
- The `file_pattern` fields specify which files to import from each directory
- The workflow can be run on different datasets by specifying different `data_dir` values without modifying the YAML

### 8.3 Signal Hierarchy
The framework uses a class hierarchy to ensure type safety:
- **`SignalData`**: Base class for all signals.
- **`TimeSeriesSignal`**: For time-indexed signals, implements generic methods like `downsample` and `filter`.
- **`WaveformSignal`**: For raw sensor signals (e.g., `PPGSignal`).
- **`MetricSignal`**: For derived metrics (e.g., `HeartRateSignal`).
- **`CompositeSignal`**: For signals derived from multiple input signals.

Each class has a registry for applicable operations, inherited from parent classes.

Each concrete signal class must define its `required_columns` by extending or overriding the `required_columns` of its parent class where applicable, ensuring that all necessary columns are consistently enforced across the hierarchy.

#### Design Principle: Method Placement
Processing methods must be defined in the highest parent class where they are universally applicable to all subclasses. This ensures minimal code duplication and maximizes reusability across the signal hierarchy. This key design principle:
- Minimizes code duplication across signal types
- Maximizes reusability of common processing functions
- Ensures methods are only available where meaningful
- Improves maintainability by centralizing common logic

For example, generic time-series operations such as filtering or downsampling should be implemented in `TimeSeriesSignal`, making them available to all subclasses like `PPGSignal` and `AccelerometerSignal`. Signal-specific operations, such as `compute_heart_rate`, should remain in their respective classes (e.g., `PPGSignal`).

Note: This placement strategy aligns with object-oriented principles of inheritance and polymorphism, ensuring that shared functionality is not redundantly implemented in multiple subclasses.

**Implementation Examples**:
```python
class TimeSeriesSignal(SignalData):
    required_columns = ['timestamp']  # All time series signals require a timestamp column
    
    # Generic methods applicable to ANY time series signal
    def downsample(self, factor=2):
        # Implementation for downsampling any time series
        pass
        
    def filter(self, filter_type, **params):
        # Generic filtering applicable to any time series
        pass

class CompositeSignal(SignalData):
    # For signals derived from multiple input signals
    def __init__(self, data, metadata):
        # Sets sensor-related metadata fields to None as they don't apply
        metadata.setdefault("sensor_type", None)
        metadata.setdefault("sensor_model", None)
        metadata.setdefault("body_position", None)
        super().__init__(data, metadata)

class PPGSignal(TimeSeriesSignal):
    signal_type = SignalType.PPG
    required_columns = ['timestamp', 'ppg_value']  # Extends parent's required_columns
    
    def __init__(self, data, metadata):
        super().__init__(data, metadata)
            
    # Methods specific to PPG signals only
    def compute_heart_rate(self, window_size=30):
        # Implementation specific to PPG
        pass
        
    # Inherits downsample() and filter() from TimeSeriesSignal
    
class AccelerometerSignal(TimeSeriesSignal):
    signal_type = SignalType.ACCELEROMETER
    required_columns = ['timestamp', 'x', 'y', 'z']  # Extends parent's required_columns
    
    # Methods specific to accelerometer signals
    def compute_activity_level(self):
        # Implementation specific to accelerometer
        pass
        
    # Also inherits downsample() and filter() from TimeSeriesSignal
    
class HeartRateSignal(TimeSeriesSignal):
    signal_type = SignalType.HEART_RATE
    
    # Methods specific to heart rate signals
    def compute_hrv(self):
        # Implementation specific to heart rate
        pass
        
    # Also inherits downsample() and filter() from TimeSeriesSignal

class FeatureSignal(SignalData):
    signal_type = SignalType.FEATURES
    
    def __init__(self, data, metadata):
        # Additional metadata fields specifically for feature signals
        metadata.setdefault("source_signals", [])  # List of signal IDs used to compute the features
        metadata.setdefault("window_length", 0)    # Length of each epoch in seconds
        metadata.setdefault("step_size", 0)        # Step size between epochs in seconds
        metadata.setdefault("features", [])        # List of feature names computed
        super().__init__(data, metadata)
```

In this hierarchy:
- Common methods like `downsample()` and `filter()` are implemented once in `TimeSeriesSignal`
- All subclasses (e.g., `PPGSignal`, `AccelerometerSignal`) automatically inherit these methods
- Type-specific methods remain in their appropriate classes
- A `PPGSignal` instance can call `signal.downsample()` directly (inherited from parent)

### 8.2 Registry Design and Operation Application

#### 8.4.1 Class-Level Registries with Proper Inheritance
The registry is a class-level attribute (not an instance attribute), ensuring memory efficiency. To ensure proper inheritance of operations, we implement explicit registry inheritance:

```python
class SignalData:
    registry = {}  # Base registry
    
    @classmethod
    def get_registry(cls):
        # Collect registries from all parent classes
        all_registries = {}
        for base in cls.__mro__:
            if hasattr(base, 'registry'):
                all_registries.update(base.registry)
        return all_registries
        
class TimeSeriesSignal(SignalData):
    registry = {}  # Additional operations specific to TimeSeriesSignal
    
class PPGSignal(TimeSeriesSignal):
    registry = {}  # Additional operations specific to PPGSignal
```

With this design:
- Each class maintains its own registry of operations specific to that class
- The `get_registry()` method collects all operations from the class hierarchy
- Child classes inherit all operations from parent classes without shadowing
- Operations registered with `SignalData` or `TimeSeriesSignal` are automatically available to `PPGSignal`

Operations are registered at the class level:
```python
@PPGSignal.register
def compute_heart_rate(data_list, parameters):
    ppg_data = data_list[0]
    window_size = parameters.get("window_size", 30)
    # Implementation
    return pd.DataFrame({"heart_rate": [70, 71, 72]})
```

#### 8.4.2 Memory Efficiency of Class-Level Registries
Using class-level registries provides significant memory benefits:
- The registry is stored **once per class**, not duplicated for each instance
- If you create 1000 `PPGSignal` instances, there's still only one `registry` object in memory
- This contrasts with instance attributes (defined in `__init__`), which would create 1000 copies

#### 8.4.3 Global Registry for Multi-Signal Operations
In addition to class-level registries, the `SignalCollection` maintains a global registry for operations that work with multiple signals:

```python
class SignalCollection(SignalContainer):
    multi_signal_registry = {
        "compute_correlation": (compute_correlation_func, CompositeSignal),
        # Feature extraction operations
        "feature_mean": (compute_mean, FeatureSignal),
        "feature_std": (compute_std, FeatureSignal),
        "feature_max": (compute_max, FeatureSignal),
        "feature_min": (compute_min, FeatureSignal),
        "feature_correlation": (compute_correlation, FeatureSignal),
        # Additional multi-signal operations
    }
```

Each entry in the registry contains:
- The operation function that accepts a list of signal data and parameters
- The output signal type to be created with the operation result

#### 8.4.4 Traceability with Operation Indexing and In-Place Operations

The framework ensures comprehensive traceability by tracking the exact state of input signals when they're used to create derived signals. This is implemented through several mechanisms:

1. The `derived_from` field in the `SignalMetadata` class, which stores tuples of `(signal_id, operation_index)`:
   ```python
   derived_from = [(input_signal.metadata.signal_id, operation_index)]
   ```

2. The `operations` list in `SignalMetadata`, which maintains a complete history of all operations applied to a signal, including their names and parameters.

3. Source file and data provenance information stored in the metadata.

This approach allows the system to:
- Identify precisely which operation version of the input signal was used
- Reconstruct the exact state of input signals at the time of derivation
- Maintain accurate lineage even when input signals are modified after derivation
- Provide full data provenance from original source files through all processing steps
- Enable reproducibility of any derived signal from its original sources

When a derived signal is created, the current operation index (the position in the operations list) of each input signal is recorded. This ensures that even if more operations are applied to the input signal later, the derived signal's metadata still points to the correct historical state. This comprehensive traceability satisfies the requirement to track all operations, parameters, and input signal states needed for full reproducibility.

##### In-Place Operation Handling

Operations that preserve the signal class (e.g., filtering a `PPGSignal` to remain a `PPGSignal`) support an optional `inplace` parameter:

```python
def filter_lowpass(signal: PPGSignal, params: Dict[str, Any], inplace: bool = False) -> PPGSignal:
    processed_data = apply_lowpass_filter(signal.data, params)
    if inplace:
        signal.data = processed_data
        signal.metadata.operations.append("filter_lowpass")
        return signal
    else:
        new_signal = PPGSignal(data=processed_data, metadata=signal.metadata.copy())
        new_signal.metadata.operations.append("filter_lowpass")
        return new_signal
```

Key aspects of in-place operations:
- Only operations that preserve signal class (same input and output type) can support `inplace=True`.
- Operations that change signal class (e.g., `PPGSignal` to `HeartRateSignal`) must always return a new instance.
- When `inplace=True`, the operation modifies the existing signal's data and metadata directly.
- When `inplace=False` (default), the operation creates a new signal instance with updated data.
- Both approaches maintain operation history in the `metadata.operations` list for traceability.
- In workflows, `inplace=True` means no `output` key is needed since the existing signal is modified.

#### 8.4.5 Hybrid Operation Application with Input/Output Naming and Metadata Filtering

The framework uses a hybrid approach for applying operations, with enhanced support for base names, indexed signals, metadata filtering, and in-place operations:

1. **Direct Method Calls**: When a method exists on the signal instance, it can be called directly:
   ```python
   heart_rate = ppg_signal.compute_heart_rate(window_size=30)
   ```

2. **Method-First apply_operation**: The `apply_operation` method first checks for a matching instance method:
   ```python
   # First tries to find ppg_signal.filter_lowpass method
   # If found, calls it with the parameters
   filtered_signal = ppg_signal.apply_operation("filter_lowpass", cutoff=5)
   
   # For in-place operations
   ppg_signal.apply_operation("filter_lowpass", cutoff=5, inplace=True)
   ```

3. **Registry Fallback with Inheritance**: If no matching method is found, falls back to the class registry, using the `get_registry()` method to include operations from parent classes:
   ```python
   # Gets the combined registry from all parent classes
   registry = ppg_signal.__class__.get_registry()
   # Looks for the operation in the combined registry
   if operation_name in registry:
       func, output_class = registry[operation_name]
       result_data = func([ppg_signal.get_data()], parameters)
       
       if inplace and output_class == ppg_signal.__class__:
           # Update existing signal for in-place operations
           ppg_signal.data = result_data
           ppg_signal.metadata.operations.append({"function_id": operation_name, "parameters": parameters})
           return ppg_signal
       else:
           # Create new signal for regular operations
           output_signal = output_class(data=result_data, metadata={...})
           return output_signal
   ```

4. **Base Name and Indexed Signal Support**: The system supports referencing signals by base names or specific indexed names:
   ```python
   # Reference all signals with base name "ppg"
   ppg_signals = workflow_executor.get_signals_by_input_specifier("ppg")
   # [ppg_0, ppg_1, ppg_2, ...]
   
   # Reference a specific indexed signal
   ppg_signal = workflow_executor.get_signals_by_input_specifier("ppg_0")
   # [ppg_0]
   ```
   
5. **Metadata Filtering Support**: The system supports selecting signals by combining base name with metadata criteria:
   ```python
   # Get all PPG signals from the chest position
   input_spec = {
       "base_name": "ppg",
       "criteria": {
           "sensor_type": "PPG",
           "body_position": "chest"
       }
   }
   chest_ppg_signals = workflow_executor.get_signals_by_criteria(input_spec)
   ```

6. **Validation for Input/Output Lists**: When using lists of inputs and outputs, the framework validates that they have equal length:
   ```python
   # Valid: matching input and output lists
   step = {
       "operation": "filter_lowpass",
       "input": ["ppg_left", "ppg_right"],
       "output": ["filtered_left", "filtered_right"],
       "parameters": {"cutoff": 5}
   }
   
   # Invalid: mismatched lengths
   step = {
       "operation": "filter_lowpass",
       "input": ["ppg_left", "ppg_right"], 
       "output": ["filtered"],  # Error: output list must have same length as input
       "parameters": {"cutoff": 5}
   }
   ```

7. **Default Base Name Generation**: If a base name is not specified in the workflow, it defaults to the lowercase signal type:
   ```python
   # Import section without base_name
   import:
     - signal_type: "PPG"
       # No base_name specified
   
   # Signals will use "ppg" as base name: ppg_0, ppg_1, etc.
   ```

#### 8.4.6 Registry Access in Workflows
The `WorkflowExecutor` uses the hybrid operation application for single-signal operations and the multi-signal registry for operations across multiple signals:

```python
# Single-signal operation with base name reference
signals = container.get_signals_by_input_specifier("ppg")
for signal in signals:
    result = signal.apply_operation("filter_lowpass", cutoff=5)
    container.add_signal(f"filtered_ppg_{i}", result)

# Single-signal operation with metadata filtering
input_spec = {
    "base_name": "ppg",
    "criteria": {
        "body_position": "chest"
    }
}
signals = container.get_signals_by_criteria(input_spec)
for signal in signals:
    result = signal.apply_operation("filter_lowpass", cutoff=5)
    container.add_signal(f"filtered_chest_ppg_{i}", result)

# Multi-signal operation
if operation_name in container.multi_signal_registry:
    func, output_class = container.multi_signal_registry[operation_name]
    signals = [container.get_signal(s) for s in step["inputs"]]
    result_data = func([s.get_data() for s in signals], parameters)
    output_signal = output_class(data=result_data, metadata={...})
```

This approach ensures:
- The registry is stored once per class, not per instance
- All instances of a class share the same registry object in memory
- Operations appropriate for a class are available to all its instances
- The `WorkflowExecutor` can handle both single-signal and multi-signal operations
- Signals can be selected by base name, indexed name, or metadata criteria

### 8.5 Applying Operations with Type Hints and In-Place Support
- **Direct Method** (preferred when available):
  ```python
  heart_rate = ppg_signal.compute_heart_rate(window_size=30)
  
  # In-place operation
  ppg_signal.filter_lowpass(cutoff=5.0, inplace=True)
  ```
- **Hybrid Method/Registry** (apply_operation):
  ```python
  heart_rate = ppg_signal.apply_operation("compute_heart_rate", window_size=30)
  
  # In-place operation
  ppg_signal.apply_operation("filter_lowpass", cutoff=5.0, inplace=True)
  ```
- **Multi-Signal Operations** (via workflow):
  ```yaml
  steps:
    - operation: "compute_correlation"
      inputs: ["ppg_0", "accelerometer_0"]
      output: "correlation_result"
      parameters: {"method": "pearson"}
  ```

Operations are defined with appropriate type hints to ensure type safety:

```python
# Operation that preserves signal type (supports in-place)
def filter_lowpass(signal: PPGSignal, cutoff: float = 5.0, inplace: bool = False) -> PPGSignal:
    """
    Apply a low-pass filter to a PPG signal.
    
    Args:
        signal: The PPG signal to filter.
        cutoff: Cutoff frequency in Hz.
        inplace: If True, modify the signal in place. If False, return a new signal.
        
    Returns:
        Filtered PPG signal (same instance if inplace=True, new instance otherwise).
        
    Notes:
        - Preserves signal type (PPGSignal -> PPGSignal).
        - Supports in-place operation.
    """
    # Implementation
    
# Operation that changes signal type (cannot be in-place)
def compute_heart_rate(signal: PPGSignal, window_size: int = 30) -> HeartRateSignal:
    """
    Compute heart rate from a PPG signal.
    
    Args:
        signal: The PPG signal to process.
        window_size: Window size in seconds for heart rate computation.
        
    Returns:
        A new HeartRateSignal instance containing the computed heart rate.
        
    Notes:
        - Changes signal type (PPGSignal -> HeartRateSignal).
        - In-place operation not supported.
    """
    # Implementation
```

### 8.6 Memory Optimization
- Mark a signal as temporary and clear its data:
  ```python
  filtered_ppg.clear_data()  # Data is cleared, but can be regenerated
  ```
- Regeneration is automatic when accessing `filtered_ppg.data`.

---

### 8.7 Type Safety with Enums

To enhance type safety and reduce errors, the framework uses enums instead of string literals for key concepts:

```python
from enum import Enum

# Using the previously defined enums for type safety
class OperationName(Enum):
    FILTER_LOWPASS = "filter_lowpass"
    COMPUTE_HEART_RATE = "compute_heart_rate"

# Example usage of enums
metadata = {
    "signal_type": SignalType.PPG,
    "units": Unit.BPM,
    "sensor_type": SensorType.PPG,
    "sensor_model": SensorModel.POLAR_H10,
    "body_position": BodyPosition.LEFT_WRIST
}

# Type-safe checks in code
if signal.metadata.signal_type == SignalType.PPG:
    # PPG-specific logic
    pass
```

These enums are used in metadata, workflow configurations, and operation lookups to ensure consistency and type safety. For example:

```python
# In metadata
metadata = {
    "signal_id": "sig1",
    "signal_type": SignalType.PPG,
    "units": Unit.BPM,
    "sensor_type": SensorType.PPG,
    "sensor_model": SensorModel.POLAR_H10,
    "body_position": BodyPosition.LEFT_WRIST
}

# In code
if signal.metadata.signal_type == SignalType.PPG.value:
    # PPG-specific logic

# In workflow configuration
step = {
    "operation": OperationName.FILTER_LOWPASS.value,
    "parameters": {"cutoff": 5}
}
```

Using enums provides:
- Compile-time checking for valid values
- Auto-completion in IDEs
- Self-documentation of valid options
- Prevention of typos and string formatting errors

### 8.8 Feature Extraction Implementation

The framework includes a dedicated module for feature extraction operations, which transform continuous signals into epoch-based features:

```python
# feature_extraction.py
from signal_collection import SignalCollection
from signals import FeatureSignal

def compute_mean(signal_data_list, parameters):
    """
    Compute mean values over epochs for one or more signals.
    
    Args:
        signal_data_list: List of signal data to process
        parameters: Dict containing window_length and step_size
        
    Returns:
        DataFrame with mean values for each epoch
    """
    window_length = parameters.get("window_length", 30)  # Default: 30 seconds
    step_size = parameters.get("step_size", window_length)  # Default: non-overlapping
    
    # Implementation to segment signals and compute means
    # Return a DataFrame with epoch timestamps as index and feature columns
    
    # Example return format:
    # timestamp | signal1_mean | signal2_mean
    # ---------------------------------------
    # 2023-01-01 00:00:00 | 72.5 | 0.85
    # 2023-01-01 00:00:30 | 73.1 | 0.87
    
    # Register the operation in the multi_signal_registry
    SignalCollection.multi_signal_registry["feature_mean"] = (compute_mean, FeatureSignal)

def compute_correlation(signal_data_list, parameters):
    """
    Compute correlation between two signals over epochs.
    
    Args:
        signal_data_list: List containing two signal data objects
        parameters: Dict containing window_length and step_size
        
    Returns:
        DataFrame with correlation values for each epoch
    """
    if len(signal_data_list) != 2:
        raise ValueError("compute_correlation requires exactly two input signals")
        
    window_length = parameters.get("window_length", 30)
    step_size = parameters.get("step_size", window_length)
    method = parameters.get("method", "pearson")  # Correlation method
    
    # Implementation to segment signals and compute correlation
    # Return a DataFrame with epoch timestamps as index and correlation values
    
    # Register the operation
    SignalCollection.multi_signal_registry["feature_correlation"] = (compute_correlation, FeatureSignal)
```

#### FeatureSignal Class Implementation

The `FeatureSignal` class represents epoch-based feature data extracted from one or more signals:

```python
class FeatureSignal(SignalData):
    signal_type = SignalType.FEATURES
    
    def __init__(self, data, metadata):
        """
        Initialize a FeatureSignal instance.
        
        Args:
            data: DataFrame with epochs as rows and features as columns
            metadata: Dict containing metadata for the feature signal
        """
        # Ensure required metadata fields exist
        metadata.setdefault("source_signals", [])  # List of source signal IDs
        metadata.setdefault("window_length", 0)    # Length of each epoch in seconds
        metadata.setdefault("step_size", 0)        # Step size between epochs in seconds
        metadata.setdefault("features", [])        # List of feature names computed
        super().__init__(data, metadata)
    
    def get_feature_names(self):
        """Return the list of feature names in this signal."""
        return self.metadata.get("features", [])
    
    def get_epoch_count(self):
        """Return the number of epochs in this signal."""
        return len(self.data)
```

#### Applying Feature Extraction in Workflows

Feature extraction can be specified in YAML workflows:

```yaml
steps:
  # Extract mean heart rate features in 30-second epochs
  - operation: "feature_mean"
    inputs: ["heart_rate"]
    parameters:
      window_length: 30
      step_size: 10  # 10-second sliding window (overlapping)
    output: "hr_mean_features"
    
  # Extract correlation between heart rate and respiratory rate
  - operation: "feature_correlation"
    inputs: ["heart_rate", "respiratory_rate"]
    parameters:
      window_length: 30
      step_size: 30
      method: "pearson"
    output: "hr_resp_correlation"
```

Or applied programmatically:

```python
# Extract mean features from heart rate signal
hr_features = signal_collection.apply_multi_signal_operation(
    "feature_mean",
    inputs=["heart_rate"],
    parameters={"window_length": 30, "step_size": 10}
)
signal_collection.add_signal("hr_features", hr_features)

# Extract correlation between heart rate and respiratory rate
correlation = signal_collection.apply_multi_signal_operation(
    "feature_correlation",
    inputs=["heart_rate", "respiratory_rate"],
    parameters={"window_length": 30, "step_size": 30, "method": "pearson"}
)
signal_collection.add_signal("hr_resp_correlation", correlation)
```

### 8.9 Framework Versioning

The framework's version will be stored in the metadata using Python's standard `__version__` attribute for simplicity and maintainability.

- **Version Definition**: The version is defined as `__version__` in the framework's main module (e.g., `__init__.py` or a dedicated `version.py` module). Example:
  ```python
  __version__ = "1.2.3"
  ```

- **Metadata Integration**: Both `SignalMetadata` and `CollectionMetadata` will include a `framework_version` field, which defaults to the current `__version__` of the framework. Example:
  ```python
  from your_framework import __version__ as FRAMEWORK_VERSION

  @dataclass
  class SignalMetadata:
      # Existing fields...
      framework_version: str = FRAMEWORK_VERSION

  @dataclass
  class CollectionMetadata:
      # Existing fields...
      framework_version: str = FRAMEWORK_VERSION
  ```

- **Exporting Version Information**: The `framework_version` field must be included when exporting metadata to files (e.g., JSON, CSV) to ensure the version is preserved with the data.

- **Version Maintenance**: The `__version__` attribute must be updated with each release as part of the release process to reflect the correct version of the framework.

- **Impact on Existing Functionality**: Adding the `framework_version` field to metadata does not affect existing workflows or data processing. It is a non-breaking change that enhances traceability.
- **Backward Compatibility**: Older metadata without `framework_version` will remain compatible, but new exports will include the version for improved auditing and debugging.

## 9. Conclusion

This design provides a comprehensive framework for signal processing in sleep analysis, balancing flexibility, traceability, and efficiency. It supports both structured workflows and ad-hoc processing, with a modular architecture that facilitates extension and maintenance.

Key design decisions include:
- Abstract base classes (`SignalData`, `SignalContainer`, `SignalImporter`) for consistent interfaces
- The `SignalCollection` as the central hub and single source of truth for all signals (imported, intermediate, and derived)
- A hybrid operation application approach that combines direct method calls with registry-based operations
- Global registry for multi-signal operations in `SignalCollection`
- Processing methods defined at the lowest appropriate level in the class hierarchy to maximize code reuse
- Registries implemented as class-level attributes for memory efficiency (one per class, not per instance)
- The `WorkflowExecutor` operating directly on any `SignalContainer` implementation to maintain data coherence
- Enums for type safety in signal types and operations
- Robust export functionality with multiple format support and combined dataframe generation
- Comprehensive testing strategy using pytest for both unit and integration tests

These design principles create a framework that is:
- **Memory efficient**: By using class-level registries with proper inheritance and strategic data clearing
- **Maintainable**: Through properly placed methods and clear separation of concerns 
- **Extensible**: Making it easy to add new signal types, operations, and importers
- **Type-safe**: Using enums for signal types and ensuring operations only apply to appropriate signal types
- **Flexible**: Supporting both direct method calls and registry-based operations, as well as generic signal retrieval
- **Traceable**: With precise metadata about all operations performed, including the exact state of input signals used for derivation
- **Exportable**: With robust support for exporting signals and metadata in various formats
- **Testable**: With a comprehensive testing strategy to ensure reliability and correctness

## 10. Documentation

- **User Documentation**:
  - Include instructions for using the export module, specifying supported formats, and configuring output options.
  - Provide guidance on writing unit and integration tests with pytest, including examples of fixtures and assertions.
  - Document all operations with their input types, output types, and in-place support.

Workflow Documentation
----------------------
The user documentation for workflows must include information about how to reference signals in workflow steps and how to use the command-line interface:

1. **By Base Name**: Use a string to reference all signals with the same base name
   ```yaml
   input: "ppg"  # References all signals with base name "ppg" (ppg_0, ppg_1, etc.)
   ```

2. **By Indexed Name**: Use a string with an index to reference a specific signal
   ```yaml
   input: "ppg_0"  # References only the specific signal ppg_0
   ```

3. **By Metadata Criteria**: Use a dictionary with base_name and criteria to filter signals
   ```yaml
   input:
     base_name: "ppg"  # Optional, if provided will only look at signals with this base name
     criteria:
       sensor_type: "PPG"
       body_position: "chest"
   ```

The system supports case-insensitive matching for metadata criteria values. For example, "ppg", "PPG", and "Ppg" are all equivalent when matching enum values.

Operation Reference
------------------
The framework includes a comprehensive reference of all operations, their input/output signal types, and whether they support in-place execution:

| Operation | Input Type | Output Type | Preserves Type | In-Place Support | Description |
|-----------|------------|-------------|---------------|-----------------|-------------|
| filter_lowpass | PPGSignal | PPGSignal | Yes | Yes | Apply a low-pass filter to PPG signals |
| filter_highpass | PPGSignal | PPGSignal | Yes | Yes | Apply a high-pass filter to PPG signals |
| filter_bandpass | PPGSignal | PPGSignal | Yes | Yes | Apply a band-pass filter to PPG signals |
| compute_heart_rate | PPGSignal | HeartRateSignal | No | No | Extract heart rate from PPG signal |
| filter_lowpass | AccelerometerSignal | AccelerometerSignal | Yes | Yes | Apply a low-pass filter to accelerometer signals |
| compute_activity | AccelerometerSignal | ActivitySignal | No | No | Compute activity levels from accelerometer data |
| feature_mean | Any | FeatureSignal | No | No | Compute mean values over epochs |
| feature_std | Any | FeatureSignal | No | No | Compute standard deviation over epochs |
| feature_correlation | Any (two signals) | FeatureSignal | No | No | Compute correlation between signals over epochs |

Operations must include type hints and detailed docstrings:

```python
def filter_lowpass(signal: PPGSignal, cutoff: float = 5.0, inplace: bool = False) -> PPGSignal:
    """
    Apply a low-pass filter to a PPG signal.
    
    Args:
        signal: The PPG signal to filter.
        cutoff: Cutoff frequency in Hz.
        inplace: If True, modify the signal in place. If False, return a new signal.
        
    Returns:
        Filtered PPG signal (same instance if inplace=True, new instance otherwise).
        
    Notes:
        - Preserves signal type (PPGSignal -> PPGSignal).
        - Supports in-place operation.
    """
    # Implementation
```

Command-Line Interface Usage
---------------------------
The framework provides a command-line interface for executing workflows:

1. **Basic Usage**:
   ```bash
   python -m sleep_analysis.cli.run_workflow --workflow workflow.yaml --data-dir /path/to/data
   ```

2. **With Optional Output Directory**:
   ```bash
   python -m sleep_analysis.cli.run_workflow --workflow workflow.yaml --data-dir /path/to/data --output-dir /path/to/output
   ```

3. **Getting Help**:
   ```bash
   python -m sleep_analysis.cli.run_workflow --help
   ```

4. **File Path Resolution**:
   - Paths in the `source` field of import specifications are resolved relative to the `data_dir`.
   - For example, if `data_dir` is `/experiment/data` and `source` is `polar/hr_data.csv`, the full path is `/experiment/data/polar/hr_data.csv`.
   - If `file_pattern` is specified, all matching files in the source directory are imported.

5. **Using File Patterns**:
   - Specify source directories with wildcards to match multiple files:
     ```yaml
     import:
       - signal_type: "heart_rate"
         source: "polar_data"
         file_pattern: "*_HR.txt"
     ```
   - This imports all files ending with `_HR.txt` from the `polar_data` directory within the `data_dir`.

6. **Reusing Workflows**:
   - The same workflow YAML can be used with different datasets by changing the `--data-dir` argument.
   - Example:
     ```bash
     # Process data from subject 1
     python -m sleep_analysis.cli.run_workflow --workflow standard_analysis.yaml --data-dir /subjects/subject1
     
     # Process data from subject 2 using the same workflow
     python -m sleep_analysis.cli.run_workflow --workflow standard_analysis.yaml --data-dir /subjects/subject2
     ```

Base Names and Indexing
----------------------
The framework supports referencing signals by base names or indexed names:

- **Base Names**: Used as prefixes for signal keys (e.g., "ppg", "filtered_ppg").
  - Specified in the `base_name` field in import configurations.
  - Default to lowercase signal type if not specified.
  - Multiple signals with the same base name are automatically indexed.

- **Indexed Names**: Specific signal instances (e.g., "ppg_0", "filtered_ppg_1").
  - Created automatically by appending index to base name.
  - Used to reference specific signals in workflows.

Example usage in workflows:
```yaml
# Reference by base name (applies to all matching signals)
input: "ppg"  # Applies to ppg_0, ppg_1, etc.

# Reference by indexed name (applies to specific signal)
input: "ppg_0"  # Applies only to ppg_0
```

In-Place Operations
------------------
Operations that preserve signal type support in-place execution:

- **In-place Mode**: Modifies existing signal instance, updates metadata.
  ```yaml
  operation: "filter_lowpass"
  input: "ppg_0"
  inplace: true  # Modifies ppg_0 directly, no output needed
  parameters:
    cutoff: 5
  ```

- **Standard Mode**: Creates new signal instance with updated data.
  ```yaml
  operation: "filter_lowpass"
  input: "ppg_0"
  output: "filtered_ppg"  # Creates new signal
  parameters:
    cutoff: 5
  ```

- **Type Restrictions**: Only operations that preserve signal type support in-place mode.
  - Operations that change type (e.g., PPGSignal -> HeartRateSignal) always create new signals.

This document serves as a comprehensive guide for implementation, covering all necessary aspects from requirements to detailed design and examples.

