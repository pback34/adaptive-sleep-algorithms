"""
Functions for epoch-based feature extraction from TimeSeriesSignals.
"""

import logging
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np

import itertools
import warnings # Added import

# Removed SignalData import
from ..signals.time_series_signal import TimeSeriesSignal
# Import Feature and FeatureMetadata from their correct locations
from ..features.feature import Feature
from ..core.metadata import FeatureMetadata, OperationInfo, FeatureType

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
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    # Add explicit arguments for global parameters
    global_window_length: pd.Timedelta,
    global_step_size: pd.Timedelta
) -> Feature:
    """
    Computes statistical features over epochs for one or more TimeSeriesSignals.

    Uses the provided global `epoch_grid_index` for epoch start times.
    The `window_length` can be specified in `parameters` to override the global
    setting, but `step_size` is determined solely by the `epoch_grid_index`.

    Args:
        signals: List containing the input TimeSeriesSignal objects.
        epoch_grid_index: The pre-calculated DatetimeIndex defining epoch start times.
        parameters: Dictionary containing operation parameters:
            - window_length (str, optional): Duration of each epoch (e.g., "30s").
                                             Overrides global if provided.
            - aggregations (List[str]): List of stats to compute (e.g., ["mean", "std"]).
            - Other parameters specific to feature functions.
        global_window_length: Global window length from collection settings.
        global_step_size: Global step size from collection settings.

    Returns:
        A Feature object containing the computed features, indexed by epoch_start_time.

    Raises:
        ValueError: If input signals are invalid, epoch_grid_index is missing/empty,
                    or required parameters are missing/invalid.
        RuntimeError: If feature calculation fails.
    """
    if not signals:
        raise ValueError("No input signals provided for feature statistics.")
    if not all(isinstance(s, TimeSeriesSignal) for s in signals):
        raise ValueError("All input signals must be TimeSeriesSignal instances.")
    if epoch_grid_index is None or epoch_grid_index.empty:
         raise ValueError("A valid epoch_grid_index must be provided.")

    # --- Parameter Parsing & Validation ---
    try:
        # REMOVE attempts to get global params from 'parameters' dict
        # global_window_length = parameters['global_epoch_window_length'] # REMOVE
        # global_step_size = parameters['global_epoch_step_size']         # REMOVE

        # Determine effective window length: override or global (using passed arg)
        window_length_str = parameters.get('window_length') # Optional override
        if window_length_str:
            effective_window_length = pd.Timedelta(window_length_str)
            logger.info(f"Using step-specific window_length override: {effective_window_length}")
        else:
            # Fallback to the explicitly passed global_window_length argument
            effective_window_length = global_window_length
            logger.info(f"Using global collection window_length: {effective_window_length}")

        if effective_window_length <= pd.Timedelta(0):
             raise ValueError("Effective window_length must be positive.")

        # Step size is implicitly defined by epoch_grid_index.freq or global_step_size
        # Use the explicitly passed global_step_size argument for metadata recording.
        epoch_step_size = global_step_size
        if epoch_grid_index.freq is not None and epoch_grid_index.freq != epoch_step_size:
             warnings.warn(f"Epoch grid index frequency ({epoch_grid_index.freq}) differs from global_epoch_step_size ({epoch_step_size}). Using global value for metadata.")
        # No need to infer from grid freq, as global_step_size is now guaranteed to be passed

        if epoch_step_size is None or epoch_step_size <= pd.Timedelta(0):
             # This check should ideally not fail if generate_epoch_grid worked
             raise ValueError("Could not determine a positive epoch_step_size from global parameters.")

        aggregations = parameters.get('aggregations', ['mean', 'std']) # Default aggregations

    # REMOVE KeyError handling for global params
    # except KeyError as e:
    #     raise ValueError(f"Missing required global parameter from executor: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid parameter format or value: {e}") from e

    logger.info(f"Computing feature statistics: effective_window={effective_window_length}, step={epoch_step_size} (from grid), aggs={aggregations}")

    # --- Handle Empty Epoch Grid ---
    if epoch_grid_index.empty:
         logger.warning("Provided epoch_grid_index is empty. Returning empty Feature object.")
         # Determine expected columns for the empty DataFrame
         expected_simple_cols = set()
         for signal in signals:
              try:
                   # Attempt to get data columns even if signal data might be empty/None
                   data_cols = signal.get_data().columns if signal.get_data() is not None else []
                   numeric_cols = signal.get_data().select_dtypes(include=np.number).columns if signal.get_data() is not None else []
                   for col in data_cols:
                        if col in numeric_cols:
                             for agg in aggregations:
                                  if agg in ['mean', 'std', 'min', 'max', 'median', 'var']:
                                       expected_simple_cols.add(f"{col}_{agg}")
              except Exception as e:
                   logger.warning(f"Could not determine columns for signal '{signal.metadata.name}' for empty feature output: {e}")

         # Create expected MultiIndex columns for the empty DataFrame
         expected_multiindex_cols = pd.MultiIndex.from_tuples(
             list(itertools.product([s.metadata.name for s in signals], sorted(list(expected_simple_cols)))),
             names=['signal_key', 'feature']
         )
         empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_multiindex_cols)
         empty_data.index.name = 'timestamp' # Set index name

         # Create metadata for the empty Feature object using passed global args
         metadata_dict = {
             "epoch_window_length": global_window_length, # Use passed global arg
             "epoch_step_size": global_step_size,       # Use passed global arg
             "feature_names": sorted(list(expected_simple_cols)),
             "feature_type": FeatureType.STATISTICAL,
             "source_signal_keys": [s.metadata.name for s in signals],
             "source_signal_ids": [s.metadata.signal_id for s in signals],
             "operations": [OperationInfo("feature_statistics", parameters)] # Store original params
         }
         # Note: FeatureMetadata requires specific fields, handled by Feature.__init__
         return Feature(data=empty_data, metadata=metadata_dict)


    # --- Feature Calculation Loop ---
    all_epoch_results = []
    processed_epochs = 0
    skipped_epochs = 0
    generated_feature_names = set() # Store the simple feature names generated

    # Iterate directly over the provided epoch grid index
    for epoch_start in epoch_grid_index:
        epoch_end = epoch_start + effective_window_length # Use effective window length
        epoch_features = {'epoch_start': epoch_start} # Use epoch_start from grid for index

        # Get data segments for this epoch from all input signals
        # Store segments along with their original signal key for context
        segments_with_keys = []
        try:
            segments_with_keys = []
            valid_epoch = True
            for signal in signals:
                try:
                    # Slice data for the current epoch [start, end)
                    # Ensure slicing handles potential timezone differences if necessary
                    # Assuming signal data index is compatible with epoch_start/epoch_end timezone
                    segment = signal.get_data()[epoch_start:epoch_end]
                    segments_with_keys.append((signal.metadata.name, segment)) # Store key and segment
                except Exception as slice_err:
                     logger.warning(f"Error slicing data for signal '{signal.metadata.name}' in epoch {epoch_start}: {slice_err}. Skipping epoch.")
                     valid_epoch = False
                     break # Stop processing this epoch if any signal fails slicing

            if not valid_epoch:
                 skipped_epochs += 1
                 continue

            # --- Compute features for this epoch ---
            # Process each signal segment individually (if epoch is valid)
            combined_epoch_stats = {}
            for signal_key, segment in segments_with_keys:
                 # Pass only the relevant aggregations for basic stats
                 basic_aggregations = [agg for agg in aggregations if agg in ['mean', 'std', 'min', 'max', 'median', 'var']]
                 if basic_aggregations:
                      # _compute_basic_stats returns simple names like 'X_mean', 'Y_std'
                      stats = _compute_basic_stats(segment, basic_aggregations)
                      # Store results keyed by (signal_key, feature_name) temporarily
                      for simple_feature_name, value in stats.items():
                           # Add the simple name to our set of generated features
                           generated_feature_names.add(simple_feature_name)
                           # Use a tuple key for the dictionary to avoid collisions
                           combined_epoch_stats[(signal_key, simple_feature_name)] = value

                 # --- Add calls to other core feature functions here if needed ---
                 # Example: if 'hrv' in aggregations and signal_type == SignalType.HEART_RATE:
                 #      hrv_features = _compute_hrv(segment, parameters.get('hrv_params', {}))
                 #      for simple_hrv_name, value in hrv_features.items():
                 #           generated_feature_names.add(simple_hrv_name)
                 #           combined_epoch_stats[(signal_key, simple_hrv_name)] = value

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
        # Return empty Feature object (using the same logic as above for empty grid)
        expected_simple_cols = set()
        for signal in signals:
             try:
                  data_cols = signal.get_data().columns if signal.get_data() is not None else []
                  numeric_cols = signal.get_data().select_dtypes(include=np.number).columns if signal.get_data() is not None else []
                  for col in data_cols:
                       if col in numeric_cols:
                            for agg in aggregations:
                                 if agg in ['mean', 'std', 'min', 'max', 'median', 'var']:
                                      expected_simple_cols.add(f"{col}_{agg}")
             except Exception as e:
                  logger.warning(f"Could not determine columns for signal '{signal.metadata.name}' for empty feature output: {e}")

        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            list(itertools.product([s.metadata.name for s in signals], sorted(list(expected_simple_cols)))),
            names=['signal_key', 'feature']
        )
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_multiindex_cols)
        empty_data.index.name = 'timestamp'

        metadata_dict = {
            "epoch_window_length": global_window_length,
            "epoch_step_size": global_step_size,
            "feature_names": sorted(list(expected_simple_cols)),
            "feature_type": FeatureType.STATISTICAL,
            "source_signal_keys": [s.metadata.name for s in signals],
            "source_signal_ids": [s.metadata.signal_id for s in signals],
            "operations": [OperationInfo("feature_statistics", parameters)]
        }
        return Feature(data=empty_data, metadata=metadata_dict)


    # --- Assemble Final DataFrame ---
    # Create DataFrame from results, using the tuple keys (signal_key, feature_name) as columns
    feature_df = pd.DataFrame(all_epoch_results)
    feature_df = feature_df.set_index('epoch_start') # Index is now epoch_start times from grid

    # Convert tuple columns to MultiIndex
    if not feature_df.empty:
         feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns, names=['signal_key', 'feature'])
    else:
         # Handle case where df is empty but columns might exist from skipped epochs
         # Recreate expected columns if df is empty after processing
         expected_simple_cols = generated_feature_names # Use names actually generated
         expected_multiindex_cols = pd.MultiIndex.from_tuples(
             list(itertools.product([s.metadata.name for s in signals], sorted(list(expected_simple_cols)))),
             names=['signal_key', 'feature']
         )
         feature_df = pd.DataFrame(index=feature_df.index, columns=expected_multiindex_cols)


    # Reindex to ensure the final DataFrame index *exactly* matches the epoch_grid_index
    # Fill missing epochs (e.g., skipped due to errors) with NaN
    feature_df = feature_df.reindex(epoch_grid_index)
    feature_df.index.name = 'timestamp' # Standard index name

    # --- Create Feature ---
    # Feature names are the *simple* names collected earlier
    simple_feature_names = sorted(list(generated_feature_names))

    # Create metadata dictionary using passed global args
    metadata_dict = {
        "epoch_window_length": global_window_length, # Use passed global arg
        "epoch_step_size": global_step_size,       # Use passed global arg
        "feature_names": simple_feature_names,
        "feature_type": FeatureType.STATISTICAL,
        "source_signal_keys": [s.metadata.name for s in signals],
        "source_signal_ids": [s.metadata.signal_id for s in signals], # Store source UUIDs
        # Record this operation, including the effective window length if overridden
        "operations": [OperationInfo("feature_statistics", parameters)] # Store original params passed
    }

    # Instantiate the Feature object
    # The Feature.__init__ handles metadata validation and creation
    return Feature(data=feature_df, metadata=metadata_dict)
