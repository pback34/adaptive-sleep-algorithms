# Design Specification: Sample Rate Calculation and Metadata Update

**Version:** 1.0
**Date:** 2025-03-31
**Status:** ✅ IMPLEMENTED

## 1. Overview

This document outlines the design for implementing a robust and automated mechanism to calculate and store the sampling rate (`sample_rate`) in the `TimeSeriesMetadata` for all `TimeSeriesSignal` instances and their subclasses within the `sleep_analysis` framework. The goal is to ensure the `sample_rate` metadata field accurately reflects the signal's data at key points in its lifecycle, minimizing code duplication and relying on a centralized calculation logic.

**Implementation Status:**
- `TimeSeriesSignal.get_sampling_rate()` implemented - calculates rate from timestamp differences
- `TimeSeriesSignal._update_sample_rate_metadata()` implemented - formats and stores rate in metadata
- Automatic metadata updates in `__init__` and `apply_operation`
- Importers refactored to not manually set sample_rate (handled automatically)
- Format: "X.XXXXHz", "Variable" (high variability), or "Unknown" (insufficient data)

## 2. Goals

-   **Accuracy:** The stored `sample_rate` in metadata should accurately represent the calculated sampling rate of the signal's data.
-   **Automation:** Calculation and metadata updates should occur automatically during signal initialization and after relevant operations.
-   **Centralization:** The core calculation logic should reside in the `TimeSeriesSignal` base class.
-   **Consistency:** The format of the stored `sample_rate` (string with "Hz", "Variable", or "Unknown") should be consistent.
-   **Robustness:** The calculation should handle edge cases like empty data, signals with fewer than two points, and irregular sampling intervals.
-   **Minimal Duplication:** Avoid redundant sample rate calculation code in subclasses or importers.

## 3. Design

### 3.1. Centralized Calculation (`TimeSeriesSignal.get_sampling_rate`)

-   The existing `TimeSeriesSignal.get_sampling_rate` method will remain the primary location for the *calculation* logic.
-   **Input:** Relies on `self.get_data()` to access the signal's `pd.DataFrame`.
-   **Logic:**
    -   Check if data exists and has at least two points. Return `None` otherwise.
    -   Calculate the differences between consecutive timestamps in the `DatetimeIndex`.
    -   Calculate the median of these differences in seconds (`median_diff_seconds`).
    -   If `median_diff_seconds` is positive:
        -   Calculate `rate = 1.0 / median_diff_seconds`.
        -   Return the calculated `rate` as a `float`.
    -   If `median_diff_seconds` is zero or negative, or if differences are highly variable (e.g., std dev > 10-50% of median), return `None` (indicating inability to determine a single rate). Consider logging variability.
    -   Return `None` if calculation fails for any other reason.
-   **Output:** Returns the calculated sampling rate as a `float` or `None`.
-   **Side Effects:** This method **must not** modify `self.metadata`. It is purely a calculator/getter for the *current* rate based on the data.

### 3.2. Metadata Update Mechanism (`TimeSeriesSignal._update_sample_rate_metadata`)

-   A new private helper method, `_update_sample_rate_metadata`, will be added to `TimeSeriesSignal`.
-   **Responsibility:** This method is responsible for calling `get_sampling_rate`, formatting the result, and updating the `SignalMetadata`.
-   **Logic:**
    1.  Call `calculated_rate = self.get_sampling_rate()`.
    2.  Determine the string representation:
        -   If `calculated_rate` is a positive float: `formatted_rate = f"{calculated_rate:.4f}Hz"` (using sufficient precision).
        -   If `calculated_rate` is `None` due to variability or zero/negative diff: `formatted_rate = "Variable"`.
        -   If `calculated_rate` is `None` due to insufficient data (< 2 points): `formatted_rate = "Unknown"`.
    3.  Use the `MetadataHandler` to update the metadata: `self.handler.update_metadata(self.metadata, sample_rate=formatted_rate)`.

### 3.3. Trigger Points for Metadata Update

The `_update_sample_rate_metadata` method will be called automatically at these points:

1.  **`TimeSeriesSignal.__init__`:** Called immediately after `super().__init__` and `self._data` has been assigned. This ensures the metadata reflects the initial state of the data.
2.  **`TimeSeriesSignal.apply_operation`:** Called *after* the operation logic (whether instance method or registry function) has been executed and the resulting data (`result_data` or modified `self._data`) is available, but *before* the resulting signal is returned.
    -   If `inplace=True`, call `self._update_sample_rate_metadata()`.
    -   If `inplace=False`, call `new_signal._update_sample_rate_metadata()` on the newly created signal instance.

### 3.4. Importer Responsibility

-   Importers (`CSVImporterBase`, `PolarCSVImporter`, `EnchantedWaveImporter`, `MergingImporter`, etc.) **should not** manually calculate or set the `sample_rate` in the metadata dictionary they pass to the `SignalData` constructor.
-   The responsibility for setting the initial `sample_rate` metadata now lies solely with the `TimeSeriesSignal` constructor via the `_update_sample_rate_metadata` call.
-   Importers *may* still calculate the rate internally for validation or logging purposes if needed (like `EnchantedWaveImporter` currently does), but the value stored in the final `SignalMetadata` will come from the `TimeSeriesSignal` logic.

### 3.5. Metadata Field

-   The `SignalMetadata.sample_rate` field remains `Optional[str]`.

### 3.6. Accessing Sample Rate

-   To get the **currently stored metadata value**: Access `signal.metadata.sample_rate`.
-   To get the **real-time calculated rate based on current data**: Call `signal.get_sampling_rate()` (returns `float` or `None`). Code performing calculations or requiring the numerical rate (like alignment logic) should use `get_sampling_rate()`.

## 4. Implementation Plan

1.  **Refine `TimeSeriesSignal.get_sampling_rate`:**
    -   Modify the method to return `float` or `None` as described in Design 3.1.
    -   Ensure it does not modify `self.metadata`.
    -   Keep existing logging for debugging time difference statistics.
2.  **Implement `TimeSeriesSignal._update_sample_rate_metadata`:**
    -   Create the private method as described in Design 3.2.
    -   Ensure it correctly formats the rate ("X.XXXXHz", "Variable", "Unknown").
    -   Ensure it uses `self.handler.update_metadata`.
3.  **Integrate Metadata Update Calls:**
    -   Add a call to `self._update_sample_rate_metadata()` at the end of `TimeSeriesSignal.__init__`.
    -   Add calls to `_update_sample_rate_metadata()` within `TimeSeriesSignal.apply_operation` for both the `inplace=True` and `inplace=False` paths, applying it to the correct signal object (self or new_signal) *after* data modification.
4.  **Modify Importers:**
    -   Remove the line setting `"sample_rate": "..."` from the `metadata` dictionary creation in `CSVImporterBase._extract_metadata`.
    -   Review `PolarCSVImporter._extract_metadata`, `EnchantedWaveImporter._extract_metadata`, and `MergingImporter.import_signal` to ensure they no longer explicitly set `metadata["sample_rate"]`. The calculation in `EnchantedWaveImporter` can remain for logging/validation if desired, but it shouldn't assign to the final metadata dict.
5.  **Review `SignalCollection`:**
    -   Modify `SignalCollection.get_target_sample_rate` to use `signal.get_sampling_rate()` (float) for finding the maximum rate, instead of relying on potentially stale metadata strings.
    -   Modify `SignalCollection.align_signals` to use `signal.get_sampling_rate()` when determining how to handle each signal based on its rate.
6.  **Testing:**
    -   Write unit tests for `TimeSeriesSignal._update_sample_rate_metadata` to verify correct formatting for different `get_sampling_rate` return values.
    -   Update tests for `TimeSeriesSignal.__init__` to assert that `metadata.sample_rate` is set correctly upon instantiation.
    -   Update tests for `TimeSeriesSignal.apply_operation` (especially for operations like resampling) to assert that `metadata.sample_rate` is updated correctly in the resulting signal.
    -   Verify importer tests still pass, ensuring signals created via importers have the correct `sample_rate` metadata set by the constructor.
    -   Add tests for edge cases (empty data, < 2 points, constant timestamps, highly irregular timestamps) to ensure "Unknown" or "Variable" is set appropriately.

## 5. Future Considerations

-   **Performance:** For very large signals, calculating the sample rate on every operation might add overhead. Caching could be considered if this becomes an issue, but the current median calculation should be reasonably efficient.
-   **User Override:** Consider if a mechanism is needed for users to manually override the calculated `sample_rate` metadata if the automatic calculation is deemed incorrect for a specific edge case. This could be done via `MetadataHandler.update_metadata`.
