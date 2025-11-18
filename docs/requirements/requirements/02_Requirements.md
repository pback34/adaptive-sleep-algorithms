# 2. Requirements

## 2.1 Functional Requirements

### 2.1.1 Signal Representation and Management
- **FR1: Signal Representation**  
  Represent various signal types (e.g., PPG, accelerometer) with appropriate metadata.

- **FR2: Signal Import and Merging**  
  The framework shall support importing multiple fragmented data files from the same signal source and merging them into a single unified signal.

- **FR3: Merging Process Ordering**  
  The merging process shall order data chronologically based on a specified timestamp column.

- **FR4: Fragmented File Identification**  
  The system shall automatically identify related fragmented files using filename patterns (e.g., `ppg_001.csv`, `ppg_002.csv`) and timestamp continuity.

- **FR5: Merged Signal Metadata**  
  Merged signals shall include metadata listing all source files used in the merge.

- **FR6: Import Support**  
  Handle signals from various manufacturers and formats, converting them to a standard format.

### 2.1.2 Dataframe Export Configuration
- **FR7: Hierarchical Index Configuration**  
  The framework shall allow users to configure hierarchical (multi-index) structures for exported dataframes.

- **FR8: Index Field Support**  
  The index configuration shall support any combination of fields from `SignalMetadata`.

- **FR9: Configuration Methods**  
  Index configuration shall be specifiable via both programmatic calls and workflow YAML files.

- **FR10: Default Index**  
  The system shall default to a single-level index using `signal_id` if no custom configuration is provided.

- **FR11: Column Level in Multi-Index**  
  The system shall automatically add an additional level called 'column' to the multi-index to distinguish between different data columns from the same signal.

### 2.1.3 Processing and Workflow Management
- **FR12: Processing Traceability**  
  Track all operations applied to signals, including operation names, parameters, and the state of input signals at the time of derivation, sufficient to regenerate derived signals. This includes data provenance (e.g., source file details), full operation history, and the exact state of input signals used to create each derived signal.

- **FR13: Memory Optimization**  
  Support clearing data from intermediate signals while preserving the ability to regenerate them.

- **FR14: Type Safety**  
  Ensure processing operations are appropriate for specific signal types.

- **FR15: Flexibility**  
  Support both structured workflows (via YAML/JSON) and ad-hoc processing in scripts or notebooks.

- **FR16: Extensibility**  
  Easily add new signal types and processing functions.

- **FR17: Workflow Support**  
  Enable both sensor-agnostic (by signal type) and non-agnostic (by specific signal) workflows.

- **FR18: Framework Versioning**  
  The framework must store its version in the metadata of processed signals and collections to ensure traceability, compatibility, and support for auditing.

- **FR19: Epoch-Based Feature Extraction**  
  Transform continuous derived signals (e.g., heart rate, respiratory rate) into structured datasets of features computed over specified time windows (epochs), enabling tasks such as sleep stage classification and event detection.

- **FR20: Command-Line Interface**  
  The framework must provide a command-line interface that allows users to specify workflow files and data directories at runtime, enabling the same workflow to be applied to different datasets without modification.

### 2.1.4 Export Functionality
- **FR21: Export Formats**  
  The framework must support exporting signals and their associated metadata to the following file formats: Excel (.xlsx), CSV (.csv), Pickle (.pkl), HDF5 (.h5).

- **FR22: Metadata Inclusion**  
  Export collection-level metadata (`CollectionMetadata`) with all exported signals. Export individual signal metadata (`SignalMetadata`) for each signal, including operation history and framework version.

- **FR23: Combined Signal Dataframe Export**  
  Provide an option to export a combined dataframe containing all non-temporary signals, aligned by timestamps.

### 2.1.5 Design and Implementation Principles
- **FR31: Abstract Interfaces for Operations**  
  The framework shall define abstract interfaces (e.g., base classes or protocols) for signal operations, ensuring that core and ad-hoc operations adhere to a consistent contract, promoting modularity and scalability by decoupling implementation from usage.

- **FR32: Configuration-Driven Operation Execution**  
  The framework shall support executing operations via configuration objects (e.g., dictionaries or data classes) that declaratively specify "what" to do, rather than imperatively coding "how" it should be done, enhancing maintainability and flexibility.

### 2.1.6 Metadata and Operation Management
- **FR33: Pure Function Enforcement for Operations**
  The framework shall ensure that operations implemented as pure functions (no side effects outside their scope) append their metadata to `metadata.operations` without modifying external state, reinforcing declarative design and testability.
  **Status: IMPLEMENTED** - Operations record their execution in OperationInfo structures appended to metadata.operations.

- **FR34: Centralized Metadata Management**
  The framework shall provide a centralized mechanism (e.g., `MetadataHandler`) for managing metadata across all signals, ensuring consistent initialization, updates, and defaults for both standalone signals and those within collections.
  **Status: IMPLEMENTED** - MetadataHandler provides separate methods for TimeSeriesMetadata and FeatureMetadata initialization and updates.

- **FR35: Default Metadata Assignment**
  The framework shall automatically assign default values to metadata fields (e.g., `name` set to `signal_<signal_id>` for standalone signals or to the collection `key` for signals in a collection), unless overridden by the user.
  **Status: IMPLEMENTED** - MetadataHandler.set_name() implements fallback strategy: key > name > auto-generated from ID.

- **FR36: Declarative Metadata Configuration**
  The framework shall allow users to configure metadata declaratively, such as through configuration files or dictionaries, to specify custom defaults or overrides for metadata fields.
  **Status: IMPLEMENTED** - Workflow YAML supports collection_settings section for index_config, feature_index_config, and epoch_grid_config.

### 2.1.7 Metadata Structure Updates
- **FR37: Dual Metadata Classes**
  The framework implements separate metadata classes for time-series signals (`TimeSeriesMetadata`) and feature sets (`FeatureMetadata`), each optimized for their specific use case while sharing common fields where appropriate.
  **Status: IMPLEMENTED** - TimeSeriesMetadata and FeatureMetadata defined in core/metadata.py.

- **FR38: Feature Metadata Enhancements**
  FeatureMetadata shall include feature-specific fields including `epoch_window_length`, `epoch_step_size`, `feature_names`, `source_signal_keys`, `source_signal_ids`, and `feature_type` enum.
  **Status: IMPLEMENTED** - All feature-specific fields are defined in FeatureMetadata class.

## 2.2 Non-Functional Requirements
- **NFR1: Usability**
  Provide an intuitive API for users in scripts, notebooks, or workflows.
  **Status: IMPLEMENTED** - CLI with argparse, declarative YAML workflows, programmatic Python API.

- **NFR2: Performance**
  Ensure efficient memory usage and processing speed for large datasets.
  **Status: IMPLEMENTED** - Lazy loading, temporary signal data clearing, efficient pandas operations.

- **NFR3: Scalability**
  Handle multiple signals and complex workflows.
  **Status: IMPLEMENTED** - Repository pattern, service-based architecture, batch operations.

- **NFR4: Maintainability**
  Use a modular design for easy updates, debugging, and extension.
  **Status: IMPLEMENTED** - Clear separation of concerns with services, repositories, and models.

- **NFR5: Merging Process Time Complexity**  
  The merging process shall complete with a time complexity of O(n), where n is the total number of data points across all files.

- **NFR6: Merging Process Overlap Handling**  
  The merging process shall support files with both overlapping and non-overlapping timestamps, defaulting to retaining the earliest occurrence in overlaps.

- **NFR7: Multi-Index Configuration Performance**  
  Configuring and applying a multi-index shall not increase export processing time by more than 5%.

- **NFR8: Multi-Index Level Support**  
  The system shall support up to four levels of indexing without performance degradation.

- **NFR9: Interface Abstraction for Extensibility**  
  The framework shall leverage abstract interfaces to allow new signal types and operations to be integrated seamlessly, ensuring extensibility and adherence to the Open-Closed Principle.

- **NFR10: Maintainability through Declarative Design**  
  The framework shall prioritize declarative design to simplify understanding and modification of the codebase, reducing the cognitive load on developers and minimizing maintenance effort over time.

- **NFR11: Testability of Operations**  
  The framework shall ensure that operations, especially those following declarative principles and pure function guidelines, are easily testable by isolating their logic from external dependencies, improving reliability.

- **NFR12: Centralized Metadata Consistency**  
  The framework shall ensure consistent metadata handling across all signals, whether standalone or part of a collection, through the use of a centralized metadata handler.

## 2.3 Testing Requirements
- **Testing Framework**: Use pytest as the primary testing framework for both unit and integration tests.
- **Unit Testing**:
  - **Scope**: Test individual components of the framework, including:
    - Signal classes (e.g., `PPGSignal`, `AccelerometerSignal`)
    - Importers (e.g., `ManufacturerAImporter`, `CSVImporter`)
    - Operations (e.g., `PPGOperations`, `AccelerometerOperations`)
    - Workflow executor (`WorkflowExecutor`)
    - Export module (`ExportModule`)
    - Metadata handler (`MetadataHandler`)
  - **Objectives**:
    - Test that importers correctly map raw data columns to the expected columns using the declarative configuration and drop any extra columns not specified in `column_mapping`.
    - Verify that importers convert timestamps to the standard format specified in `CollectionMetadata` and handle various input formats correctly using the `convert_timestamp_format` utility.
    - Ensure that signal classes raise a `ValueError` when initialized with data missing any `required_columns`.
    - Verify type safety by ensuring operations are only applied to appropriate signal types.
    - Test metadata handling, including operation history, `derived_from` fields, framework version, and default assignments via the metadata handler.
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
    - Verify that the metadata handler consistently manages metadata for both standalone and collection signals.
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
        - test_metadata_handler.py
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