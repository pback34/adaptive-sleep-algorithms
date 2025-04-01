Below is the **revised** implementation plan for the timestamp and timezone refactoring, incorporating feedback and clarifications. It focuses on Requirement 2.3 (centralized timezone adjustment) and aligns with the updated requirements in `time_handling_requirements_design.md`.

---

## Revised Implementation Plan for Timestamp and Timezone Refactoring

### Step 1: Centralize Timezone Adjustment Logic (No Change)
**Objective**: Create a single, reusable function to handle timestamp standardization and timezone adjustments.

- **Placement**:
  - **Location**: Implement the `standardize_timestamp` function in `src/sleep_analysis/utils/__init__.py`.
  - **Rationale**: Centralizes shared helper logic, accessible to all components, avoiding duplication and dependency issues. Aligns with Requirement 2.3.

- **Function Definition**:
  ```python
  import pandas as pd
  import logging
  # Consider adding pytz or similar for validation if needed
  # import pytz

  def standardize_timestamp(
      df: pd.DataFrame,
      timestamp_col: str,
      origin_timezone: Optional[str], # Can be None if source is already aware
      target_timezone: str, # Resolved target timezone string
      set_index: bool = True
  ) -> pd.DataFrame:
      logger = logging.getLogger(__name__)
      df = df.copy() # Work on a copy

      # Optional: Validate timezone strings early
      # try:
      #     if origin_timezone: pytz.timezone(origin_timezone)
      #     pytz.timezone(target_timezone)
      # except pytz.UnknownTimeZoneError as e:
      #     logger.error(f"Invalid timezone provided: {e}")
      #     raise ValueError(f"Invalid timezone provided: {e}") from e

      # Parse timestamp column to datetime objects
      df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
      # Handle potential parsing errors if needed

      # Localize naive timestamps to origin_timezone if provided
      if df[timestamp_col].dt.tz is None:
          if origin_timezone:
              try:
                  logger.debug(f"Localizing naive timestamp to origin timezone: {origin_timezone}")
                  df[timestamp_col] = df[timestamp_col].dt.tz_localize(origin_timezone, ambiguous='infer', nonexistent='raise')
              except Exception as tz_err:
                  logger.warning(f"Could not localize timestamp to {origin_timezone}: {tz_err}. Proceeding as naive (will likely convert to target assuming UTC).")
          else:
              # If origin_timezone is None and timestamp is naive, tz_convert might assume UTC or fail.
              # Consider adding a default assumption (e.g., UTC) or requiring origin_timezone for naive data.
              logger.warning(f"Timestamp column '{timestamp_col}' is naive but no origin_timezone was specified. Timezone conversion behavior may be ambiguous.")

      # Convert to target_timezone (handles both originally naive localized and already aware timestamps)
      try:
          logger.debug(f"Converting timestamp to target timezone: {target_timezone}")
          df[timestamp_col] = df[timestamp_col].dt.tz_convert(target_timezone)
      except Exception as tz_err:
          logger.error(f"Could not convert timestamp timezone to {target_timezone}: {tz_err}. Returning with original/localized timezone.")
          # Decide on error handling: raise error or return partially converted? Raising is safer.
          raise ValueError(f"Failed to convert timestamp to target timezone {target_timezone}: {tz_err}") from tz_err

      # Set as index if requested
      if set_index:
          df.set_index(df[timestamp_col], inplace=True)
          df.index.name = 'timestamp' # Ensure index is named
          df = df.drop(columns=[timestamp_col]) # Drop original column after setting index

      return df
  ```

- **Key Features**:
  - Parses timestamps.
  - Localizes naive timestamps using `origin_timezone`.
  - Converts *all* timestamps to the `target_timezone`.
  - Optionally sets the timestamp column as the DataFrame index.
  - Includes logging and basic error handling (consider adding explicit timezone validation).

---

### Step 2: Update the Base Importer Class (No Change)
**Objective**: Integrate the centralized timezone logic into the `SignalImporter` base class.

- **Placement**:
  - **Location**: Modify `src/sleep_analysis/importers/base.py`.
  - **Rationale**: Ensures all derived importers inherit standardized timestamp handling.

- **Method Addition**:
  ```python
  class SignalImporter(ABC):
      # Existing code...

      def _standardize_timestamp(self, df: pd.DataFrame, timestamp_col: str,
                                origin_timezone: Optional[str], target_timezone: str,
                                set_index: bool = True) -> pd.DataFrame:
          """Helper method to call the centralized timestamp standardization utility."""
          from ...utils import standardize_timestamp # Local import
          return standardize_timestamp(df, timestamp_col, origin_timezone, target_timezone, set_index)
  ```

- **Usage**: Concrete importers will call this method within their parsing logic.

---

### Step 3: Modify Concrete Importers (Minor Change)
**Objective**: Update each importer to leverage the centralized timezone adjustment logic.

- **Target Importers**:
  - `PolarCSVImporter` (`src/sleep_analysis/importers/sensors/polar.py`)
  - `EnchantedWaveImporter` (`src/sleep_analysis/importers/sensors/enchanted_wave.py`)
  - `MergingImporter` (`src/sleep_analysis/importers/merging.py`) - *Note: If MergingImporter wraps another importer, it should pass the timezones down to the underlying importer.*

- **Changes**:
  - In each importer’s `_parse_csv` (or equivalent parsing method):
    - Retrieve `origin_timezone` from `self.config.get("origin_timezone")` (this might be `None`).
    - Retrieve `target_timezone` from `self.config["target_timezone"]` (this will be *added* by the `WorkflowExecutor` and should always be present).
    - Invoke `self._standardize_timestamp` with the DataFrame, timestamp column name, `origin_timezone`, and `target_timezone`.

- **Example (PolarCSVImporter `_parse_csv`)**:
  ```python
  def _parse_csv(self, source: str) -> pd.DataFrame:
      # ... (Initial CSV parsing into df) ...
      df = pd.read_csv(source, ...) # Existing logic

      # Apply column mapping (existing logic)
      # ... df = df.rename(...) ...
      # ... df = df[mapped_cols] ...

      # Standardize timestamp using the base class helper
      timestamp_col_name = "timestamp" # The standard name *after* mapping
      origin_timezone = self.config.get("origin_timezone") # Might be None
      target_timezone = self.config["target_timezone"] # Should be injected by WorkflowExecutor

      if timestamp_col_name in df.columns:
           df = self._standardize_timestamp(df, timestamp_col_name, origin_timezone, target_timezone, set_index=True)
      else:
           self.logger.error(f"Timestamp column '{timestamp_col_name}' not found after mapping.")
           raise ValueError(f"Timestamp column '{timestamp_col_name}' not found after mapping.")

      return df
  ```

---

### Step 4: Update WorkflowExecutor for Timezone Configuration (Revised Logic)
**Objective**: Ensure the workflow executor correctly resolves and propagates timezone settings.

- **Placement**:
  - **Location**: Modify `src/sleep_analysis/workflows/workflow_executor.py`.

- **Changes**:
  - In `execute_workflow`:
    1.  Read `default_input_timezone = workflow_config.get("default_input_timezone")`. Log a warning if missing and potentially default to `None` or a specific value like 'UTC' if desired for naive inputs without overrides.
    2.  Resolve `target_timezone`:
        *   Get `target_tz_config = workflow_config.get("target_timezone")`.
        *   Call a helper method `_resolve_target_timezone(target_tz_config)` to get the final timezone string. Store this in `self.target_timezone`.
    3.  Update the `SignalCollection`'s metadata: `self.container.metadata.timezone = self.target_timezone`.
  - Add helper method `_resolve_target_timezone`:
    *   Takes `target_tz_config` (which can be `None`, `'system'`, `'local'`, or a specific zone string).
    *   If `target_tz_config` is `None` (key missing) or `'system'` or `'local'`:
        *   Try importing `tzlocal`.
        *   Call `tzlocal.get_localzone_name()` or `tzlocal.get_localzone().zone`.
        *   If `tzlocal` fails or isn't installed, log a warning and return `"UTC"` as fallback.
        *   Return the detected system timezone string.
    *   Otherwise (if it's a specific zone string):
        *   *Optional but recommended:* Validate the string (e.g., using `pytz.timezone(target_tz_config)` inside a try-except). If invalid, raise `ValueError`.
        *   Return `target_tz_config`.
  - In `_process_import_section`:
    *   For each `spec`:
        *   Get `origin_timezone = spec.get("config", {}).get("origin_timezone", self.default_input_timezone)`. Note: `origin_timezone` can correctly be `None` if the source data is already timezone-aware or if `default_input_timezone` wasn't set.
        *   Create `importer_config = spec.get("config", {}).copy()`.
        *   **Inject the resolved target timezone**: `importer_config["target_timezone"] = self.target_timezone`.
        *   **Inject the origin timezone to be used**: `importer_config["origin_timezone"] = origin_timezone`.
        *   Instantiate the importer: `importer = self._get_importer_instance(spec["importer"], importer_config)`.

- **Example Snippets (`WorkflowExecutor`)**:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  try:
      import tzlocal
      import pytz # Optional: for validation
  except ImportError:
      tzlocal = None
      # pytz = None

  class WorkflowExecutor:
      # ... (init) ...

      def _resolve_target_timezone(self, target_tz_config: Optional[str]) -> str:
          """Resolves the target timezone based on config, system, or fallback."""
          if target_tz_config is None or target_tz_config.lower() in ["system", "local"]:
              if tzlocal:
                  try:
                      system_tz = tzlocal.get_localzone_name()
                      # Optional: Validate system_tz before returning
                      # pytz.timezone(system_tz)
                      logger.info(f"Using system local timezone: {system_tz}")
                      return system_tz
                  except Exception as e:
                      logger.warning(f"Failed to detect system timezone using tzlocal: {e}. Falling back to UTC.")
                      return "UTC"
              else:
                  logger.warning("tzlocal library not found. Cannot detect system timezone. Falling back to UTC.")
                  return "UTC"
          else:
              # Optional: Validate the provided timezone string
              # try:
              #     pytz.timezone(target_tz_config)
              # except pytz.UnknownTimeZoneError:
              #     raise ValueError(f"Invalid target_timezone specified: '{target_tz_config}'")
              logger.info(f"Using specified target timezone: {target_tz_config}")
              return target_tz_config

      def execute_workflow(self, workflow_config: Dict[str, Any]):
          self.default_input_timezone = workflow_config.get("default_input_timezone") # Can be None
          if not self.default_input_timezone:
               logger.warning("No 'default_input_timezone' specified in workflow. Naive timestamps without an 'origin_timezone' override may be handled ambiguously.")

          target_tz_config = workflow_config.get("target_timezone")
          self.target_timezone = self._resolve_target_timezone(target_tz_config)

          # Update collection metadata
          if self.container:
              self.container.metadata.timezone = self.target_timezone
              logger.debug(f"Set SignalCollection timezone to: {self.target_timezone}")

          # Existing workflow execution logic...
          if "import" in workflow_config:
              self._process_import_section(workflow_config["import"])
          # ... (steps, visualization, export)

      def _process_import_section(self, import_specs: List[Dict[str, Any]]):
          for spec in import_specs:
              importer_config = spec.get("config", {}).copy()
              # Determine the origin timezone to use for this importer
              origin_timezone = importer_config.get("origin_timezone", self.default_input_timezone)

              # Inject resolved target and origin timezones into the config passed to the importer
              importer_config["target_timezone"] = self.target_timezone
              importer_config["origin_timezone"] = origin_timezone # Pass the determined origin_timezone

              importer = self._get_importer_instance(spec["importer"], importer_config)
              # ... (rest of import processing) ...
  ```

---

### Step 5: Support System Timezone as Target (Integrated into Step 4)
**Objective**: Allow `target_timezone` to use the system’s local timezone.
- **Placement**: Logic integrated into `WorkflowExecutor._resolve_target_timezone` as detailed in Step 4.
- **Implementation**: Uses `tzlocal` library within the helper method.

---

### Step 6: Update SignalCollection and Export Logic (Revised Focus)
**Objective**: Ensure `SignalCollection` metadata is consistent and `ExportModule` uses the correct timezone.

- **Placement**:
  - `src/sleep_analysis/core/signal_collection.py`
  - `src/sleep_analysis/export/export_module.py`

- **Changes**:
  - **`SignalCollection`**:
    - In `__init__` or via `WorkflowExecutor`, set `self.metadata.timezone` based on the *resolved* `target_timezone` from the workflow.
    - In `add_signal`:
        - **Remove** the automatic timezone conversion logic.
        - **Add validation (optional but recommended)**: Check if the incoming `signal.get_data().index.tz` matches `self.metadata.timezone`. Log a warning or raise an error if they don't match, as this indicates a problem earlier in the pipeline (importer).
        ```python
        # Example validation in SignalCollection.add_signal
        if isinstance(signal, TimeSeriesSignal):
            signal_tz = signal.get_data().index.tz
            collection_tz_str = self.metadata.timezone # The string name
            # Basic string comparison might suffice if zones are consistently named
            # More robust comparison might involve converting collection_tz_str to tzinfo object
            if signal_tz is None or str(signal_tz) != collection_tz_str:
                 logger.warning(f"Signal '{key}' timezone ({signal_tz}) does not match collection timezone ({collection_tz_str}). Potential inconsistency.")
                 # Optionally raise ValueError("Signal timezone mismatch")
        ```
  - **`ExportModule`**:
    - In `_serialize_metadata`, ensure `collection.metadata.timezone` is included.
    - In export methods (e.g., `_export_csv`, `_export_excel`), ensure timestamps are formatted correctly *respecting* the timezone information already present on the `DatetimeIndex` (which should be the `target_timezone`). Pandas' `to_csv` and `to_excel` generally handle timezone-aware indexes correctly. Explicit formatting might be needed for specific string representations.

---

### Step 7: Testing and Validation (No Change)
**Objective**: Verify the refactoring meets requirements.

- **Approach**:
  - **Unit Tests**: Test `standardize_timestamp` with naive/aware inputs, different timezones, and edge cases. Test `WorkflowExecutor._resolve_target_timezone`.
  - **Integration Tests**: Run workflows with various `default_input_timezone`, `origin_timezone`, and `target_timezone` settings (including `None`, specific zones, `'system'`). Verify timestamps in exported files and metadata.
  - **System Timezone Test**: Validate `'system'`/`'local'` and default (missing key) `target_timezone` functionality correctly uses the detected system timezone or falls back to UTC if detection fails. Test on systems with different local timezones if possible.

---

This revised plan clarifies the timezone resolution logic, reinforces the importer's responsibility for standardization, and adjusts the `SignalCollection`'s role to validation rather than conversion, providing a more robust and maintainable implementation.
