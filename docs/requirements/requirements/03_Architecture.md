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