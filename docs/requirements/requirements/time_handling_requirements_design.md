
# Timestamp and Timezone Handling and Standardization

## 1. Introduction

This document defines the requirements and design for timestamp and timezone management within the sleep analysis framework. The primary objectives are to:
- Centralize timestamp and timezone handling logic.
- Ensure all timestamp operations are robust, consistent, and configurable via a declarative workflow YAML.
- Minimize code duplication across signal importers and processing steps.

These goals will support reliable signal import, processing, and export by standardizing timestamp representations and timezone adjustments.

**Implementation Status: COMPLETED**
- Timezone handling implemented in WorkflowExecutor (`_resolve_target_timezone()`)
- Default input timezone and target timezone configuration supported in workflow YAML
- System timezone detection with tzlocal library
- Timezone propagated to CollectionMetadata and validated during signal import

---

## 2. Functional Requirements

### 2.1 Timestamp Conversion to Pandas Timestamp/DatetimeIndex
- **FR1**: All signal importers **shall** convert any input timestamp representation (e.g., strings, integers, or custom formats) into a native Pandas `Timestamp` or `DatetimeIndex` type.
  **Status: IMPLEMENTED** - TimeSeriesSignal requires DatetimeIndex as index.
- **FR2**: The resulting `DatetimeIndex` **shall** be set as the index of the signal's DataFrame to standardize data access and manipulation.
  **Status: IMPLEMENTED** - All importers set DatetimeIndex during import.

### 2.2 Default Input Timezone and Overrides
- **FR3**: The framework **shall** support a `default_input_timezone` field in the workflow YAML, which defines the assumed timezone for all imported signals unless overridden.
  **Status: IMPLEMENTED** - WorkflowExecutor.default_input_timezone set from YAML config.
- **FR4**: Individual signal importers **shall** allow an optional `origin_timezone` field in their configuration within the workflow YAML to override the `default_input_timezone` for that specific importer.
  **Status: IMPLEMENTED** - Import configs support origin_timezone override.

### 2.3 Centralized Timezone Adjustment
- **FR5**: Timezone adjustment logic **shall** be centralized in a single utility function or a method within a base importer class, ensuring it is reusable across all importers and eliminating code duplication.
  **Status: IMPLEMENTED** - Timezone handling centralized in WorkflowExecutor._resolve_target_timezone().
- **FR6**: This centralized function or method **shall**:
  - Localize naive timestamps (those without timezone information) to the specified `origin_timezone`.
  - Convert all timestamps (naive or timezone-aware) to the designated `target_timezone`.
  **Status: IMPLEMENTED** - Pandas tz_localize and tz_convert used consistently.

### 2.4 Target Timezone for Exported Signals
- **FR7**: The framework **shall** include a `target_timezone` field in the workflow YAML, specifying the timezone to which all signal timestamps are standardized and exported.
  **Status: IMPLEMENTED** - target_timezone supported in workflow YAML.
- **FR8**: All exported signals **shall** have their timestamps adjusted to the `target_timezone` prior to export.
  **Status: IMPLEMENTED** - CollectionMetadata.timezone propagated to all signals and exports.

### 2.5 Default and Special Values for Target Timezone
- **FR9**: If the `target_timezone` key is *not present* in the workflow YAML configuration, the framework **shall** default to using the operating system's detected local timezone as the target timezone for standardization and export.
  **Status: IMPLEMENTED** - Falls back to system timezone via tzlocal when not specified.
- **FR10**: The `target_timezone` field **shall** support special string values (e.g., `'system'` or `'local'`) to explicitly indicate that the operating system's detected local timezone should be used.
  **Status: IMPLEMENTED** - WorkflowExecutor recognizes "system" and "local" keywords.

---

## 3. Non-Functional Requirements

### 3.1 Configurability
- **NFR1**: All timestamp and timezone handling **shall** be fully configurable through the workflow YAML, allowing users to adapt the framework to diverse datasets without modifying code.

### 3.2 Consistency
- **NFR2**: Timestamps across all signals within a processing workflow **shall** be standardized to the same `target_timezone`, ensuring uniformity for analysis and export.

### 3.3 Code Efficiency
- **NFR3**: The design **shall** minimize code duplication by centralizing timezone handling logic, making maintenance and updates more efficient.

### 3.4 Robustness
- **NFR4**: The framework **shall** handle invalid timezone configurations or timestamp parsing errors gracefully, providing clear error messages to the user.

---

## 4. Design Specifications

### 4.1 Workflow YAML Configuration
The workflow YAML **shall** include the following fields to manage timezone settings:

- **`default_input_timezone`**: Specifies the assumed timezone for naive timestamps across all importers (e.g., `"America/New_York"`).
- **`target_timezone`**: Defines the timezone for all standardized and exported signals (e.g., `"UTC"`, `"America/Los_Angeles"`, or `"system"`).

Each importer’s configuration within the `import` section **shall** support:
- **`origin_timezone`**: An optional field to override the `default_input_timezone` for that specific importer.

**Example Workflow YAML:**
```yaml
default_input_timezone: "America/New_York"
target_timezone: "UTC"

import:
  - signal_type: "heart_rate"
    importer: "PolarCSVImporter"
    source: "data/heart_rate.csv"
    config:
      origin_timezone: "Europe/London"  # Overrides default for this importer
  - signal_type: "sleep_stage"
    importer: "SleepCSVImporter"
    source: "data/sleep.csv"
    config: {}  # Uses default_input_timezone
```

### 4.2 Centralized Timezone Handling
- **Utility Location**: A centralized function, `standardize_timestamp`, **shall** be implemented in a utility module (e.g., `src/sleep_analysis/utils/__init__.py`) or as a method in a `BaseImporter` class.
- **Parameters**:
  - `df`: The DataFrame containing the timestamp data.
  - `timestamp_col`: The column name containing timestamps.
  - `origin_timezone`: The timezone for naive timestamps (from `default_input_timezone` or `origin_timezone` override).
  - `target_timezone`: The timezone to standardize timestamps to.
  - `set_index`: A boolean flag to set the timestamp as the DataFrame index (default: `True`).
- **Behavior**:
  - For naive timestamps, localize to the `origin_timezone`.
  - Convert all timestamps to the `target_timezone`.
  - Return the DataFrame with a `DatetimeIndex` set as the index (if `set_index` is `True`).

**Example Signature:**
```python
def standardize_timestamp(
    df: pd.DataFrame,
    timestamp_col: str,
    origin_timezone: str,
    target_timezone: str,
    set_index: bool = True
) -> pd.DataFrame:
    # Pseudo-implementation
    timestamps = pd.to_datetime(df[timestamp_col])
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(origin_timezone)
    timestamps = timestamps.tz_convert(target_timezone)
    if set_index:
        df = df.set_index(timestamps)
    return df
```

### 4.3 Importer Design
- **Base Importer Class**: A `BaseImporter` class **shall** provide a method (e.g., `parse_and_standardize`) that uses `standardize_timestamp` to handle timestamp conversion and timezone adjustment.
- **Concrete Importers**: Each importer (e.g., `PolarCSVImporter`, `SleepCSVImporter`) **shall**:
  - Parse raw data into a DataFrame with a timestamp column.
  - Call the centralized `standardize_timestamp` function with the appropriate `origin_timezone` (from config or default) and `target_timezone`.

### 4.4 Signal Export and Collection Consistency
- **Importer Responsibility**: Signal importers are responsible for ensuring that the signals they produce have timestamps standardized to the workflow's resolved `target_timezone`.
- **Collection Validation**: The `SignalCollection` **shall** store the resolved `target_timezone` in its `metadata.timezone` field. When adding signals, it **should** validate that the incoming signal's timestamp timezone matches the collection's timezone, logging a warning or raising an error upon mismatch, rather than performing conversions itself.
- **Export Module**: The `ExportModule` **shall** use the `SignalCollection`'s `metadata.timezone` to correctly format timestamps and include this timezone information in the exported metadata.

### 4.5 System Timezone Handling
- **Detection**: The framework **shall** use a library like `tzlocal` to detect the operating system's local timezone when:
    - The `target_timezone` key is *missing* from the workflow YAML (FR9).
    - The `target_timezone` key is explicitly set to a special value like `'system'` or `'local'` (FR10).
- **Resolution Logic**: The `WorkflowExecutor` **shall** resolve the final `target_timezone` string based on the presence and value of the `target_timezone` key in the configuration, prioritizing explicit values, then special values (`'system'`), and finally the default behavior (system local timezone if key is missing).
- **Fallback**: If the system's local timezone needs to be used (either by default or explicitly requested) but cannot be determined by `tzlocal`, the framework **shall** default to `"UTC"` and log a warning.

---

## 5. Implementation Notes

- **Pandas Integration**: Leverage Pandas’ built-in timezone handling (`tz_localize` and `tz_convert`) for robust and efficient timestamp conversions.
- **Configuration Propagation**: The `WorkflowExecutor` **shall**:
  - Read `default_input_timezone` and `target_timezone` from the YAML.
  - Resolve the final `target_timezone` string (handling defaults and special values).
  - Pass the `default_input_timezone` and the resolved `target_timezone` (along with any importer-specific `origin_timezone` overrides) down to the respective importers, typically by adding them to the importer's configuration dictionary.
- **Timezone String Validation**: The `WorkflowExecutor` or the centralized `standardize_timestamp` function **should** validate timezone strings (e.g., `default_input_timezone`, `origin_timezone`, resolved `target_timezone`) against a standard library (e.g., `pytz` or checking validity within Pandas/`dateutil`) before use, raising descriptive errors for invalid inputs to improve robustness (NFR4).

---

## 6. Testing Requirements

### 6.1 Unit Tests
- **Timestamp Conversion**: Verify that importers convert various input formats (strings, integers) to `DatetimeIndex`.
- **Timezone Handling**: Test localization of naive timestamps and conversion of aware timestamps to `target_timezone`.
- **Default and Override**: Confirm that `default_input_timezone` applies globally and `origin_timezone` overrides it per importer.

### 6.2 Integration Tests
- **End-to-End Workflow**: Validate that signals imported with different `origin_timezone` settings are standardized to the `target_timezone` in exported outputs.
- **System Timezone**: Ensure `'system'` or `'local'` correctly adopts the operating system’s timezone.

---

This specification ensures that timestamp and timezone handling is centralized, configurable via the workflow YAML, and consistent across the sleep analysis framework, meeting your requirements for robustness and reduced code duplication. Let me know if you need further refinements!
