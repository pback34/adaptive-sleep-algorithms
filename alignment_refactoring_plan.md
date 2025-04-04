# Refactoring Plan: SignalCollection Alignment and Combination

This plan outlines the steps to refactor the alignment and combination logic within the `SignalCollection` class and related components, based on the agreed-upon proposal for clarity, robustness, and simplicity.

**Goal:** Establish a clear, unambiguous workflow for generating aligned combined dataframes, offering two distinct paths: one that modifies original signals and uses efficient concatenation, and another that keeps originals intact using merge_asof alignment during combination. Both paths result in a stored combined dataframe.

**Core Methods in `SignalCollection`:**

1.  **`generate_alignment_grid()`** (Renamed from `align_signals`)
2.  **`apply_grid_alignment()`** (Modified to align all TS signals)
3.  **`combine_aligned_signals()`** (New - Concatenation path)
4.  **`align_and_combine_signals()`** (New - Merge_asof path)
5.  **`get_stored_combined_dataframe()`** (Renamed from `get_combined_dataframe`)

**Removed/Replaced Methods:**

*   `align_signals` (renamed)
*   `generate_and_store_aligned_dataframe` (replaced by the two new combination methods)
*   `get_combined_dataframe` (renamed)
*   `_calculate_combined_dataframe` / `_calculate_combined_dataframe_with_merge_asof` / `_calculate_combined_dataframe_with_concat` (Internal logic moved into the new public combination methods)

**Refactoring Steps:**

1.  **Rename `align_signals` to `generate_alignment_grid`:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:** Rename the method `align_signals` to `generate_alignment_grid`. Update its docstring to reflect that it only calculates and stores grid parameters (`grid_index`, `ref_time`, `target_rate`) and sets the `_alignment_params_calculated` flag. Ensure it raises an error if parameters cannot be calculated.

2.  **Modify `apply_grid_alignment`:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:**
        *   Remove the `signals_to_align` parameter or ensure its default behavior processes *all* `TimeSeriesSignal` instances in `self.signals`.
        *   Update the docstring to state it modifies all time-series signals in-place.
        *   Ensure it checks for `self._alignment_params_calculated == True` at the beginning and raises a `RuntimeError` if not.
        *   Consider error handling: If alignment fails for one signal, should it stop, or continue and report errors? (Current proposal implies it should raise an error, which `combine_aligned_signals` would later detect if it continued). Raising immediately might be cleaner.

3.  **Implement `combine_aligned_signals`:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:**
        *   Create the new method `combine_aligned_signals()`.
        *   Add docstring explaining it requires `generate_alignment_grid` and `apply_grid_alignment` to have run, uses concatenation, and stores the result.
        *   Implement prerequisite check: `if not self._alignment_params_calculated: raise RuntimeError(...)`.
        *   Iterate through all `TimeSeriesSignal` objects in `self.signals`.
        *   **Verification:** For each signal, check `signal.get_data().index.equals(self.grid_index)`. If *any* signal fails this check, raise a `RuntimeError` indicating `apply_grid_alignment` was not run successfully or failed.
        *   Collect the data from verified signals.
        *   Implement the concatenation logic (using MultiIndex or simple columns based on `self.metadata.index_config`).
        *   Store the resulting DataFrame in `self._aligned_dataframe`.
        *   Store the current alignment parameters in `self._aligned_dataframe_params`.
        *   Log success and shape.

4.  **Implement `align_and_combine_signals`:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:**
        *   Create the new method `align_and_combine_signals()`.
        *   Add docstring explaining it requires `generate_alignment_grid`, uses `merge_asof` on original data, and stores the result.
        *   Implement prerequisite check: `if not self._alignment_params_calculated: raise RuntimeError(...)`.
        *   Implement the `merge_asof` logic (adapt from previous `_calculate_combined_dataframe_with_merge_asof` draft):
            *   Create `target_df` from `self.grid_index`.
            *   Calculate `tolerance`.
            *   Iterate through `TimeSeriesSignal` objects, get *original* data.
            *   Perform `pd.merge_asof` for each signal against `target_df`.
            *   Collect aligned dataframes.
        *   Implement the combination logic (MultiIndex or simple columns) on the collected aligned dataframes.
        *   Store the resulting DataFrame in `self._aligned_dataframe`.
        *   Store the current alignment parameters in `self._aligned_dataframe_params`.
        *   Log success and shape.

5.  **Rename `get_combined_dataframe` to `get_stored_combined_dataframe`:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:** Rename the method. Update its docstring to clarify it only returns the stored dataframe (or `None`). Ensure it simply returns `self._aligned_dataframe`.

6.  **Update `ExportModule`:**
    *   **File:** `src/sleep_analysis/export/export_module.py`
    *   **Action:**
        *   Modify the logic within `_export_excel`, `_export_csv`, etc., where the combined dataframe is retrieved.
        *   Replace calls to `self.collection.get_combined_dataframe()` or `self._get_combined_dataframe_for_export()` with `self.collection.get_stored_combined_dataframe()`.
        *   If `get_stored_combined_dataframe()` returns `None`, log a warning and skip the export of the combined file for that format. Remove any logic for ephemeral generation.

7.  **Update `WorkflowExecutor`:**
    *   **File:** `src/sleep_analysis/workflows/workflow_executor.py`
    *   **Action:**
        *   In `execute_step`, modify the handling of `type: collection` operations.
        *   Map the new workflow operation names to the corresponding `SignalCollection` methods:
            *   `generate_alignment_grid` -> `collection.generate_alignment_grid()`
            *   `apply_grid_alignment` -> `collection.apply_grid_alignment()`
            *   `combine_aligned_signals` -> `collection.combine_aligned_signals()`
            *   `align_and_combine_signals` -> `collection.align_and_combine_signals()`
        *   Remove cases for old/replaced operation names (`align_signals`, `generate_and_store_aligned_dataframe`).

8.  **Update Documentation (README.md & Workflows):**
    *   **File:** `README.md`
    *   **Action:** Update the "Signal Alignment and Combined DataFrames" section and any workflow examples to reflect the new method names (`generate_alignment_grid`, `apply_grid_alignment`, `combine_aligned_signals`, `align_and_combine_signals`) and the two distinct workflow paths for generating the combined dataframe.
    *   **File:** `workflows/polar_workflow-dev.yaml` (and any other example workflows)
    *   **Action:** Update the `steps` section to use the new operation names. Choose one of the two valid combination steps (`combine_aligned_signals` or `align_and_combine_signals`) based on whether `apply_grid_alignment` is used.

9.  **Update Tests:**
    *   **File:** `tests/unit/test_signal_collection.py`
    *   **Action:**
        *   Rename test functions related to `align_signals` and `get_combined_dataframe`.
        *   Adapt tests for `apply_grid_alignment` to check the "all signals" behavior.
        *   Add new tests specifically for `generate_alignment_grid`.
        *   Add new tests for `combine_aligned_signals`:
            *   Test success case after `apply_grid_alignment`.
            *   Test failure case if `apply_grid_alignment` wasn't run.
            *   Test failure case if `generate_alignment_grid` wasn't run.
        *   Add new tests for `align_and_combine_signals`:
            *   Test success case (should work without `apply_grid_alignment`).
            *   Test failure case if `generate_alignment_grid` wasn't run.
        *   Test `get_stored_combined_dataframe` returns `None` initially and the correct dataframe after a combination step.
        *   Check that the correct combination logic (concat vs. merge_asof) is triggered based on the method called.

This detailed plan should provide a clear roadmap for implementing the refactoring.
