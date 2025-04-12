"""
Functions for epoch-based feature extraction from TimeSeriesSignals.
"""

import logging
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np

from ..core.signal_data import SignalData
from ..signals.time_series_signal import TimeSeriesSignal
from ..signals.feature_signal import FeatureSignal
from ..core.metadata import OperationInfo

logger = logging.getLogger(__name__)

# --- Core Feature Calculation Functions (Operating on single epoch data) ---

def _compute_basic_stats(segment: pd.DataFrame, aggregations: List[str]) -> Dict[str, float]:
    """Calculates basic statistics (mean, std, min, max, median, var) for numeric columns."""
    results = {}
    if segment.empty:
        for agg in aggregations:
            # Return NaN for all requested aggregations if segment is empty
            for col in segment.columns: # Use original columns even if empty
                 if pd.api.types.is_numeric_dtype(segment[col]): # Check original dtype
                      results[f"{col}_{agg}"] = np.nan
        return results

    numeric_segment = segment.select_dtypes(include=np.number)
    if numeric_segment.empty:
         # Return NaN for numeric aggregations if no numeric data
         for agg in aggregations:
              if agg in ['mean', 'std', 'min', 'max', 'median', 'var']:
                   for col in segment.columns: # Still iterate original columns
                        results[f"{col}_{agg}"] = np.nan
         return results # Return only NaNs for numeric stats

    # Calculate requested statistics only for numeric columns
    for col in numeric_segment.columns:
        if 'mean' in aggregations:
            results[f"{col}_mean"] = numeric_segment[col].mean()
        if 'std' in aggregations:
            results[f"{col}_std"] = numeric_segment[col].std()
        if 'min' in aggregations:
            results[f"{col}_min"] = numeric_segment[col].min()
        if 'max' in aggregations:
            results[f"{col}_max"] = numeric_segment[col].max()
        if 'median' in aggregations:
            results[f"{col}_median"] = numeric_segment[col].median()
        if 'var' in aggregations:
            results[f"{col}_var"] = numeric_segment[col].var()
            
    # Ensure all requested aggregations have NaN entries if calculation failed (e.g., all NaN input)
    for col in numeric_segment.columns:
         for agg in aggregations:
              if agg in ['mean', 'std', 'min', 'max', 'median', 'var']:
                   feature_name = f"{col}_{agg}"
                   if feature_name not in results:
                        results[feature_name] = np.nan

    return results

# --- Add other core feature functions here (e.g., _compute_correlation, _compute_hrv) ---


# --- Main Wrapper Function (Registered in SignalCollection) ---

def compute_feature_statistics(
    signals: List[SignalData],
    grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any]
) -> FeatureSignal:
    """
    Computes statistical features over epochs for one or more signals.

    Args:
        signals: List containing the input TimeSeriesSignal objects.
        grid_index: The pre-calculated DatetimeIndex defining the overall time range.
        parameters: Dictionary containing:
            - window_length (str): Duration of each epoch (e.g., "30s").
            - step_size (str): Time interval between epoch starts (e.g., "10s").
            - aggregations (List[str]): List of stats to compute (e.g., ["mean", "std"]).
            - Other parameters specific to feature functions.

    Returns:
        A FeatureSignal object containing the computed features.

    Raises:
        ValueError: If required parameters are missing or invalid, or if no
                    TimeSeriesSignals are provided, or if grid_index is missing.
        RuntimeError: If epoch generation fails.
    """
    if not signals:
        raise ValueError("No input signals provided for feature statistics.")
    if not all(isinstance(s, TimeSeriesSignal) for s in signals):
        raise ValueError("All input signals must be TimeSeriesSignal instances.")
    if grid_index is None or grid_index.empty:
         raise ValueError("A valid grid_index must be provided for epoch generation.")

    # --- Parameter Parsing ---
    try:
        window_length_str = parameters['window_length']
        step_size_str = parameters['step_size']
        aggregations = parameters.get('aggregations', ['mean', 'std']) # Default aggregations

        window_length = pd.Timedelta(window_length_str)
        step_size = pd.Timedelta(step_size_str)

        if window_length <= pd.Timedelta(0) or step_size <= pd.Timedelta(0):
             raise ValueError("window_length and step_size must be positive durations.")

    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid parameter format: {e}") from e

    logger.info(f"Computing feature statistics: window={window_length}, step={step_size}, aggs={aggregations}")

    # --- Epoch Generation ---
    try:
        # Use grid_index min/max for range
        start_time = grid_index.min()
        end_time = grid_index.max()

        # Generate epoch start times ensuring the *start* time is within the grid range
        # The window can extend beyond the end_time
        epoch_starts = pd.date_range(
            start=start_time,
            end=end_time, # Generate starts up to the grid end
            freq=step_size,
            inclusive='left' # Include start, exclude exact end if it falls on step
        )
        # Filter out any start times where the window would begin after the grid ends
        epoch_starts = epoch_starts[epoch_starts <= end_time]

        if epoch_starts.empty:
             logger.warning("No valid epoch start times generated based on grid_index and step_size.")
             # Return an empty FeatureSignal
             empty_data = pd.DataFrame(index=pd.DatetimeIndex([]))
             feature_names = [] # Will be populated later if needed
             metadata = {
                 "epoch_window_length": window_length,
                 "epoch_step_size": step_size,
                 "feature_names": feature_names,
                 "source_signal_keys": [s.metadata.name for s in signals], # Use name as key proxy
                 "derived_from": [(s.metadata.signal_id, len(s.metadata.operations)-1) for s in signals],
                 "operations": [OperationInfo("feature_statistics", parameters)]
             }
             # Need to determine columns for empty dataframe
             # Let's compute expected columns based on input signals and aggs
             expected_cols = []
             for signal in signals:
                  data_cols = signal.get_data().columns
                  numeric_cols = signal.get_data().select_dtypes(include=np.number).columns
                  for col in data_cols:
                       if col in numeric_cols:
                            for agg in aggregations:
                                 if agg in ['mean', 'std', 'min', 'max', 'median', 'var']:
                                      expected_cols.append(f"{signal.metadata.name}_{col}_{agg}") # Prefix with signal name
             empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_cols)
             metadata["feature_names"] = expected_cols
             return FeatureSignal(data=empty_data, metadata=metadata)


    except Exception as e:
        logger.error(f"Error generating epoch sequence: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate epoch sequence: {e}") from e

    # --- Feature Calculation Loop ---
    all_epoch_results = []
    processed_epochs = 0
    skipped_epochs = 0

    for epoch_start in epoch_starts:
        epoch_end = epoch_start + window_length
        epoch_features = {'epoch_start': epoch_start} # Store start time for index later

        # Get data segments for this epoch from all input signals
        segments = []
        signal_names = [] # To prefix feature names
        try:
            for signal in signals:
                # Slice data for the current epoch [start, end)
                segment = signal.get_data()[epoch_start:epoch_end]
                # Note: Pandas slicing [start:end] includes start, excludes end by default for DatetimeIndex
                # Adjust if strict inclusion/exclusion is needed, e.g., segment = signal.get_data().loc[epoch_start : epoch_end - pd.Timedelta(nanoseconds=1)]
                segments.append(segment)
                signal_names.append(signal.metadata.name or f"signal_{signal.metadata.signal_id[:4]}") # Use name or partial ID

            # --- Compute features for this epoch ---
            # For feature_statistics, we process each signal segment individually
            combined_epoch_stats = {}
            for i, segment in enumerate(segments):
                 signal_name_prefix = signal_names[i]
                 # Pass only the relevant aggregations for basic stats
                 basic_aggregations = [agg for agg in aggregations if agg in ['mean', 'std', 'min', 'max', 'median', 'var']]
                 if basic_aggregations:
                      stats = _compute_basic_stats(segment, basic_aggregations)
                      # Prefix feature names with signal name/key
                      prefixed_stats = {f"{signal_name_prefix}_{k}": v for k, v in stats.items()}
                      combined_epoch_stats.update(prefixed_stats)

                 # --- Add calls to other core feature functions here if needed ---
                 # Example: if 'hrv' in aggregations and signals[i].signal_type == SignalType.HEART_RATE:
                 #      hrv_features = _compute_hrv(segment, parameters.get('hrv_params', {}))
                 #      combined_epoch_stats.update(hrv_features)

            epoch_features.update(combined_epoch_stats)
            all_epoch_results.append(epoch_features)
            processed_epochs += 1

        except Exception as e:
            logger.warning(f"Skipping epoch {epoch_start} due to error: {e}", exc_info=False) # Log less verbosely for skipped epochs
            # Optionally append NaNs for skipped epochs? For now, we just skip.
            skipped_epochs += 1
            continue

    logger.info(f"Feature calculation complete. Processed epochs: {processed_epochs}, Skipped epochs: {skipped_epochs}")

    if not all_epoch_results:
        logger.warning("No features were successfully computed for any epoch.")
        # Return empty FeatureSignal (similar to above)
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]))
        feature_names = []
        metadata = {
            "epoch_window_length": window_length,
            "epoch_step_size": step_size,
            "feature_names": feature_names,
            "source_signal_keys": [s.metadata.name for s in signals],
            "derived_from": [(s.metadata.signal_id, len(s.metadata.operations)-1) for s in signals],
            "operations": [OperationInfo("feature_statistics", parameters)]
        }
        # Determine expected columns again
        expected_cols = []
        for signal in signals:
             data_cols = signal.get_data().columns
             numeric_cols = signal.get_data().select_dtypes(include=np.number).columns
             for col in data_cols:
                  if col in numeric_cols:
                       for agg in aggregations:
                            if agg in ['mean', 'std', 'min', 'max', 'median', 'var']:
                                 expected_cols.append(f"{signal.metadata.name}_{col}_{agg}")
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_cols)
        metadata["feature_names"] = expected_cols
        return FeatureSignal(data=empty_data, metadata=metadata)


    # --- Assemble Final DataFrame ---
    feature_df = pd.DataFrame(all_epoch_results)
    feature_df = feature_df.set_index('epoch_start')
    feature_df.index.name = 'timestamp' # Standard index name

    # --- Create FeatureSignal ---
    feature_names = list(feature_df.columns)
    metadata = {
        "epoch_window_length": window_length,
        "epoch_step_size": step_size,
        "feature_names": feature_names,
        "source_signal_keys": [s.metadata.name for s in signals], # Use name as key proxy
        # Link derivation to the state *before* this operation
        "derived_from": [(s.metadata.signal_id, len(s.metadata.operations)-1 if s.metadata.operations else -1) for s in signals],
        "operations": [OperationInfo("feature_statistics", parameters)] # Record this operation
    }

    # Pass the collection's handler if available (needed?) - FeatureSignal init handles it
    # handler = getattr(signals[0], 'handler', None) if signals else None

    return FeatureSignal(data=feature_df, metadata=metadata) #, handler=handler)
