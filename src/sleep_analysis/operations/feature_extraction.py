"""
Functions for epoch-based feature extraction from TimeSeriesSignals.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
import numpy as np
import hashlib
import json
from functools import wraps
# Removed scipy.stats.mode import

import itertools
import warnings

# Removed SignalData import
from ..signals.time_series_signal import TimeSeriesSignal
# Import Feature and FeatureMetadata from their correct locations
from ..features.feature import Feature
from ..core.metadata import FeatureMetadata, OperationInfo, FeatureType # Import FeatureType
from ..core import validation  # Import validation utilities
from ..utils.thread_safety import ThreadSafeCache  # Thread-safe caching for parallel processing
from ..utils.parallel import parallel_map, batch_items, get_parallel_config  # Parallel processing utilities

logger = logging.getLogger(__name__)

# --- Constants ---

# Length of cache key hash (first N characters of MD5 hexdigest)
# Used to create a compact yet unique identifier for cached features
CACHE_HASH_LENGTH = 16

# NN50 threshold in milliseconds for HRV features
# NN50 is the number of pairs of successive RR intervals that differ by more than 50ms
NN50_THRESHOLD_MS = 50

# Activity threshold multiplier for movement detection
# Threshold = mean + (multiplier * std_dev) of acceleration magnitude
# Higher values make detection more selective (fewer movements detected)
ACTIVITY_THRESHOLD_MULTIPLIER = 0.5

# Default epoch window length for feature extraction (used in examples/docs)
# Standard sleep staging uses 30-second epochs
DEFAULT_EPOCH_WINDOW = "30s"

# Default aggregation methods for statistical features
DEFAULT_AGGREGATIONS = ['mean', 'std']

# Default HRV metrics for RR interval-based analysis
DEFAULT_HRV_METRICS_RR = ['sdnn', 'rmssd', 'pnn50']

# Default HRV metrics for heart rate-based approximation
DEFAULT_HRV_METRICS_HR = ['hr_mean', 'hr_std', 'hr_cv', 'hr_range']

# Default movement metrics for accelerometer analysis
DEFAULT_MOVEMENT_METRICS = ['magnitude_mean', 'magnitude_std', 'magnitude_max',
                           'activity_count', 'stillness_ratio', 'x_std', 'y_std', 'z_std']

# Thread-safe global cache for feature extraction results
_FEATURE_CACHE = ThreadSafeCache()
_CACHE_ENABLED = True  # Global flag to enable/disable caching


def enable_feature_cache(enabled: bool = True):
    """
    Enable or disable the global feature extraction cache.

    Args:
        enabled: If True, enables caching; if False, disables it.
    """
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled
    logger.info(f"Feature cache {'enabled' if enabled else 'disabled'}")


def clear_feature_cache():
    """Clear all cached feature extraction results."""
    global _FEATURE_CACHE
    cache_size = _FEATURE_CACHE.size()
    _FEATURE_CACHE.clear()
    logger.info(f"Cleared feature cache ({cache_size} entries removed)")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the feature cache.

    Returns:
        Dictionary containing cache statistics.
    """
    stats = _FEATURE_CACHE.get_stats()
    return {
        "enabled": _CACHE_ENABLED,
        "size": _FEATURE_CACHE.size(),
        "hit_rate": _FEATURE_CACHE.get_hit_rate(),
        "hits": stats['hits'],
        "misses": stats['misses'],
        "sets": stats['sets']
    }


def _compute_cache_key(
    signal_ids: List[str],
    operation_name: str,
    parameters: Dict[str, Any],
    epoch_grid_hash: str
) -> str:
    """
    Compute a unique cache key for feature extraction.

    Args:
        signal_ids: List of signal IDs being processed.
        operation_name: Name of the feature extraction operation.
        parameters: Operation parameters.
        epoch_grid_hash: Hash of the epoch grid index.

    Returns:
        Unique cache key string.
    """
    # Create a stable representation of the inputs
    key_components = {
        "signal_ids": sorted(signal_ids),  # Sort for consistency
        "operation": operation_name,
        "parameters": parameters,
        "epoch_grid": epoch_grid_hash
    }

    # Convert to JSON string (sorted for consistency)
    key_string = json.dumps(key_components, sort_keys=True)

    # Hash to create a fixed-length key
    return hashlib.md5(key_string.encode()).hexdigest()


def cache_features(func: Callable) -> Callable:
    """
    Decorator to cache feature extraction results.

    Caches Feature objects based on input signal IDs, operation name,
    parameters, and epoch grid. Cache is invalidated if any inputs change.

    Usage:
        @cache_features
        def compute_feature_statistics(...):
            ...

    Args:
        func: Feature extraction function to cache.

    Returns:
        Wrapped function with caching capability.
    """
    @wraps(func)
    def wrapper(
        signals: List[TimeSeriesSignal],
        epoch_grid_index: pd.DatetimeIndex,
        parameters: Dict[str, Any],
        global_window_length: pd.Timedelta,
        global_step_size: pd.Timedelta
    ) -> Feature:
        # Check if caching is enabled
        if not _CACHE_ENABLED:
            logger.debug(f"Cache disabled, computing {func.__name__} directly")
            return func(signals, epoch_grid_index, parameters,
                       global_window_length, global_step_size)

        # Extract signal IDs for cache key
        signal_ids = [s.metadata.signal_id for s in signals]

        # Create a hash of the epoch grid index
        # Using hash of index values for efficiency
        epoch_grid_hash = hashlib.md5(
            str(epoch_grid_index.values).encode()
        ).hexdigest()[:CACHE_HASH_LENGTH]

        # Compute cache key
        cache_key = _compute_cache_key(
            signal_ids=signal_ids,
            operation_name=func.__name__,
            parameters=parameters,
            epoch_grid_hash=epoch_grid_hash
        )

        # Use thread-safe get_or_compute for atomic cache check/set
        def compute_feature():
            logger.debug(f"Cache miss for {func.__name__} (key: {cache_key[:8]}...)")
            result = func(signals, epoch_grid_index, parameters,
                         global_window_length, global_step_size)
            logger.debug(f"Cached result for {func.__name__} (cache size: {_FEATURE_CACHE.size()})")
            return result

        # Thread-safe cache lookup or computation
        cached_value = _FEATURE_CACHE.get(cache_key)
        if cached_value is not None:
            logger.debug(f"Cache hit for {func.__name__} (key: {cache_key[:8]}...)")
            return cached_value

        # Use get_or_compute to handle race conditions
        return _FEATURE_CACHE.get_or_compute(cache_key, compute_feature)

    return wrapper


def _handle_empty_feature_data(
    signals: List[TimeSeriesSignal],
    feature_names: List[str],
    feature_type: FeatureType,
    operation_name: str,
    parameters: Dict[str, Any],
    global_window_length: str,
    global_step_size: str,
    column_index_name: str = 'signal_key'
) -> Feature:
    """
    Create an empty Feature object for when epoch_grid_index is empty.

    This utility function centralizes the empty DataFrame handling pattern
    that is duplicated across multiple feature extraction operations.

    Args:
        signals: List of input TimeSeriesSignal objects.
        feature_names: List of feature names to create columns for.
        feature_type: Type of feature being extracted (e.g., FeatureType.HRV).
        operation_name: Name of the operation (for metadata).
        parameters: Operation parameters (for metadata).
        global_window_length: Global window length setting.
        global_step_size: Global step size setting.
        column_index_name: Name for the first level of MultiIndex columns
                          (default 'signal_key', can be 'signal_pair' for correlation).

    Returns:
        Empty Feature object with appropriate metadata and column structure.
    """
    logger.warning(f"Empty epoch_grid_index in {operation_name}. Returning empty Feature object.")

    # Create expected MultiIndex columns
    if column_index_name == 'signal_pair':
        # Special case for correlation features
        signal_pair_name = f"{signals[0].metadata.name}_vs_{signals[1].metadata.name}"
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(signal_pair_name, feature) for feature in feature_names],
            names=[column_index_name, 'feature']
        )
    else:
        # Standard case for most features
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(s.metadata.name, feature) for s in signals for feature in feature_names],
            names=[column_index_name, 'feature']
        )

    # Create empty DataFrame with proper structure
    empty_data = pd.DataFrame(
        index=pd.DatetimeIndex([]),
        columns=expected_multiindex_cols
    )
    empty_data.index.name = 'timestamp'

    # Create metadata dictionary
    metadata_dict = {
        "epoch_window_length": global_window_length,
        "epoch_step_size": global_step_size,
        "feature_names": feature_names,
        "feature_type": feature_type,
        "source_signal_keys": [s.metadata.name for s in signals],
        "source_signal_ids": [s.metadata.signal_id for s in signals],
        "operations": [OperationInfo(operation_name, parameters)]
    }

    return Feature(data=empty_data, metadata=metadata_dict)


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

# --- HRV Feature Functions ---

def _compute_hrv_time_domain(rr_intervals: pd.Series) -> Dict[str, float]:
    """
    Computes time-domain HRV features from RR intervals.

    Args:
        rr_intervals: Series of RR intervals in milliseconds

    Returns:
        Dictionary of HRV features
    """
    results = {}

    if rr_intervals.empty or len(rr_intervals) < 2:
        # Return NaN for all features if insufficient data
        return {
            'rr_mean': np.nan,
            'rr_std': np.nan,
            'sdnn': np.nan,
            'rmssd': np.nan,
            'pnn50': np.nan,
            'sdsd': np.nan
        }

    # Basic RR statistics
    results['rr_mean'] = rr_intervals.mean()
    results['rr_std'] = rr_intervals.std()

    # SDNN: Standard deviation of NN (normal-to-normal) intervals
    results['sdnn'] = rr_intervals.std()

    # Calculate successive differences
    successive_diffs = np.diff(rr_intervals)

    if len(successive_diffs) > 0:
        # RMSSD: Root mean square of successive differences
        results['rmssd'] = np.sqrt(np.mean(successive_diffs ** 2))

        # SDSD: Standard deviation of successive differences
        results['sdsd'] = np.std(successive_diffs)

        # pNN50: Percentage of successive differences > NN50_THRESHOLD_MS
        nn50_count = np.sum(np.abs(successive_diffs) > NN50_THRESHOLD_MS)
        results['pnn50'] = (nn50_count / len(successive_diffs)) * 100
    else:
        results['rmssd'] = np.nan
        results['sdsd'] = np.nan
        results['pnn50'] = np.nan

    return results


def _compute_hrv_from_heart_rate(hr_data: pd.DataFrame) -> Dict[str, float]:
    """
    Computes HRV approximations from heart rate data.

    When RR intervals are not available, approximates HRV metrics from HR.
    Note: This is less accurate than true RR interval-based HRV.

    Args:
        hr_data: DataFrame with 'hr' column (heart rate in bpm)

    Returns:
        Dictionary of approximate HRV features
    """
    results = {}

    if hr_data.empty or 'hr' not in hr_data.columns:
        return {
            'hr_mean': np.nan,
            'hr_std': np.nan,
            'hr_cv': np.nan,
            'hr_range': np.nan
        }

    hr = hr_data['hr'].dropna()

    if len(hr) < 2:
        return {
            'hr_mean': np.nan,
            'hr_std': np.nan,
            'hr_cv': np.nan,
            'hr_range': np.nan
        }

    # Basic HR statistics
    results['hr_mean'] = hr.mean()
    results['hr_std'] = hr.std()

    # Coefficient of variation
    if results['hr_mean'] > 0:
        results['hr_cv'] = (results['hr_std'] / results['hr_mean']) * 100
    else:
        results['hr_cv'] = np.nan

    # Heart rate range
    results['hr_range'] = hr.max() - hr.min()

    return results


# --- Movement/Activity Feature Functions ---

def _compute_movement_features(accel_data: pd.DataFrame) -> Dict[str, float]:
    """
    Computes movement and activity features from accelerometer data.

    Args:
        accel_data: DataFrame with 'x', 'y', 'z' acceleration columns

    Returns:
        Dictionary of movement features
    """
    results = {}

    required_cols = ['x', 'y', 'z']
    if accel_data.empty or not all(col in accel_data.columns for col in required_cols):
        # Return NaN for all features if data is missing
        return {
            'magnitude_mean': np.nan,
            'magnitude_std': np.nan,
            'magnitude_max': np.nan,
            'activity_count': np.nan,
            'stillness_ratio': np.nan,
            'x_std': np.nan,
            'y_std': np.nan,
            'z_std': np.nan
        }

    # Drop NaN values
    accel_clean = accel_data[required_cols].dropna()

    if len(accel_clean) < 2:
        return {
            'magnitude_mean': np.nan,
            'magnitude_std': np.nan,
            'magnitude_max': np.nan,
            'activity_count': np.nan,
            'stillness_ratio': np.nan,
            'x_std': np.nan,
            'y_std': np.nan,
            'z_std': np.nan
        }

    # Calculate acceleration magnitude: sqrt(x^2 + y^2 + z^2)
    magnitude = np.sqrt(
        accel_clean['x']**2 +
        accel_clean['y']**2 +
        accel_clean['z']**2
    )

    # Magnitude statistics
    results['magnitude_mean'] = magnitude.mean()
    results['magnitude_std'] = magnitude.std()
    results['magnitude_max'] = magnitude.max()

    # Activity count: number of samples above threshold (indicating movement)
    # Threshold: mean + ACTIVITY_THRESHOLD_MULTIPLIER*std (adaptive to signal characteristics)
    threshold = results['magnitude_mean'] + ACTIVITY_THRESHOLD_MULTIPLIER * results['magnitude_std']
    activity_samples = (magnitude > threshold).sum()
    results['activity_count'] = activity_samples

    # Stillness ratio: percentage of samples below threshold
    results['stillness_ratio'] = ((len(magnitude) - activity_samples) / len(magnitude)) * 100

    # Individual axis variability (important for sleep posture detection)
    results['x_std'] = accel_clean['x'].std()
    results['y_std'] = accel_clean['y'].std()
    results['z_std'] = accel_clean['z'].std()

    return results


# --- Correlation Feature Functions ---

def _compute_correlation_features(
    signal1_data: pd.DataFrame,
    signal2_data: pd.DataFrame,
    signal1_col: str,
    signal2_col: str,
    method: str = 'pearson'
) -> Dict[str, float]:
    """
    Computes correlation between two signal columns.

    Args:
        signal1_data: DataFrame containing first signal
        signal2_data: DataFrame containing second signal
        signal1_col: Column name in signal1_data to correlate
        signal2_col: Column name in signal2_data to correlate
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Dictionary with correlation coefficient
    """
    results = {}

    # Check if columns exist
    if signal1_col not in signal1_data.columns or signal2_col not in signal2_data.columns:
        return {f'{method}_corr': np.nan}

    # Extract columns and align by index
    s1 = signal1_data[signal1_col]
    s2 = signal2_data[signal2_col]

    # Find common indices
    common_idx = s1.index.intersection(s2.index)

    if len(common_idx) < 3:  # Need at least 3 points for meaningful correlation
        return {f'{method}_corr': np.nan}

    # Align and drop NaNs
    s1_aligned = s1.loc[common_idx].dropna()
    s2_aligned = s2.loc[common_idx].dropna()

    # Further align after dropna
    final_common_idx = s1_aligned.index.intersection(s2_aligned.index)

    if len(final_common_idx) < 3:
        return {f'{method}_corr': np.nan}

    s1_final = s1_aligned.loc[final_common_idx]
    s2_final = s2_aligned.loc[final_common_idx]

    try:
        # Compute correlation
        corr_value = s1_final.corr(s2_final, method=method)
        results[f'{method}_corr'] = corr_value
    except Exception as e:
        logger.warning(f"Error computing {method} correlation: {e}")
        results[f'{method}_corr'] = np.nan

    return results


# --- Add other core feature functions here (e.g., _compute_correlation, _compute_hrv) ---


# --- Parallel Processing Helpers ---

def _process_epoch_batch(batch_data):
    """
    Process a batch of epochs for parallel execution.

    Args:
        batch_data: Tuple of (epoch_starts, signal_data_list, signal_keys, effective_window_length, aggregations)

    Returns:
        List of epoch feature dictionaries
    """
    epoch_starts, signal_data_list, signal_keys, effective_window_length, aggregations = batch_data

    batch_results = []
    for epoch_start in epoch_starts:
        epoch_end = epoch_start + effective_window_length
        epoch_features = {'epoch_start': epoch_start}

        try:
            # Get data segments for this epoch from all input signals
            segments_with_keys = []
            valid_epoch = True

            for signal_key, signal_data in zip(signal_keys, signal_data_list):
                try:
                    segment = signal_data[epoch_start:epoch_end]
                    segments_with_keys.append((signal_key, segment))
                except Exception as slice_err:
                    logger.warning(f"Error slicing data for signal '{signal_key}' in epoch {epoch_start}: {slice_err}")
                    valid_epoch = False
                    break

            if not valid_epoch:
                continue

            # Compute features for this epoch
            combined_epoch_stats = {}
            for signal_key, segment in segments_with_keys:
                basic_aggregations = [agg for agg in aggregations if agg in ['mean', 'std', 'min', 'max', 'median', 'var']]
                if basic_aggregations:
                    stats = _compute_basic_stats(segment, basic_aggregations)
                    for simple_feature_name, value in stats.items():
                        combined_epoch_stats[(signal_key, simple_feature_name)] = value

            epoch_features.update(combined_epoch_stats)
            batch_results.append(epoch_features)

        except Exception as e:
            logger.debug(f"Skipping epoch {epoch_start} in batch due to error: {e}")
            continue

    return batch_results


# --- Main Wrapper Function (Registered in SignalCollection) ---

@cache_features
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

    Example (Workflow YAML):
        ```yaml
        # Extract basic statistics from heart rate signal
        steps:
          - type: collection
            operation: "generate_epoch_grid"
            parameters: {}

          - type: multi_signal
            operation: "feature_statistics"
            inputs: ["hr"]  # Base name matches hr_0, hr_1, etc.
            parameters:
              aggregations: ["mean", "std", "min", "max"]
            output: "hr_stats"
        ```

    Example (Python API):
        ```python
        from sleep_analysis.core.signal_collection import SignalCollection
        from sleep_analysis.operations.feature_extraction import compute_feature_statistics
        import pandas as pd

        # Assuming collection has signals and epoch grid
        collection = SignalCollection()
        # ... import signals ...
        collection.generate_epoch_grid()

        # Get signals
        hr_signals = [collection.get_signal("hr_0")]

        # Compute features
        features = compute_feature_statistics(
            signals=hr_signals,
            epoch_grid_index=collection.epoch_grid_index,
            parameters={
                "aggregations": ["mean", "std", "min", "max"]
            },
            global_window_length=pd.Timedelta("30s"),
            global_step_size=pd.Timedelta("30s")
        )

        # Access feature data
        print(features.data.head())
        # Output: DataFrame with columns like hr_mean, hr_std, hr_min, hr_max
        ```
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

        aggregations = parameters.get('aggregations', DEFAULT_AGGREGATIONS)

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


    # --- Feature Calculation Loop (with parallel processing support) ---
    all_epoch_results = []
    processed_epochs = 0
    skipped_epochs = 0
    generated_feature_names = set() # Store the simple feature names generated

    # Check if parallel processing is enabled and we have enough epochs to benefit
    parallel_config = get_parallel_config()
    use_parallel = (
        parallel_config.enabled and
        len(epoch_grid_index) >= 20  # Only parallelize for sufficient epochs
    )

    if use_parallel:
        # Prepare data for parallel processing
        signal_data_list = [signal.get_data() for signal in signals]
        signal_keys = [signal.metadata.name for signal in signals]

        # Split epochs into batches for parallel processing
        # Use larger batches to reduce overhead
        batch_size = max(10, len(epoch_grid_index) // (parallel_config.max_workers_cpu * 2))
        epoch_batches = batch_items(list(epoch_grid_index), batch_size)

        logger.info(
            f"Processing {len(epoch_grid_index)} epochs in parallel "
            f"({len(epoch_batches)} batches, {parallel_config.max_workers_cpu} workers)"
        )

        # Create batch data for each batch
        batch_data_list = [
            (batch, signal_data_list, signal_keys, effective_window_length, aggregations)
            for batch in epoch_batches
        ]

        # Process batches in parallel using processes (CPU-bound)
        batch_results = parallel_map(
            _process_epoch_batch,
            batch_data_list,
            use_processes=True,
            desc="Computing features"
        )

        # Flatten results and collect feature names
        for batch_result in batch_results:
            for epoch_features in batch_result:
                # Extract feature names from tuple keys
                for key in epoch_features.keys():
                    if isinstance(key, tuple) and len(key) == 2:
                        generated_feature_names.add(key[1])
                all_epoch_results.append(epoch_features)
                processed_epochs += 1

        skipped_epochs = len(epoch_grid_index) - processed_epochs

    else:
        # Sequential processing (original logic)
        for epoch_start in epoch_grid_index:
            epoch_end = epoch_start + effective_window_length
            epoch_features = {'epoch_start': epoch_start}

            try:
                segments_with_keys = []
                valid_epoch = True
                for signal in signals:
                    try:
                        segment = signal.get_data()[epoch_start:epoch_end]
                        segments_with_keys.append((signal.metadata.name, segment))
                    except Exception as slice_err:
                         logger.warning(f"Error slicing data for signal '{signal.metadata.name}' in epoch {epoch_start}: {slice_err}. Skipping epoch.")
                         valid_epoch = False
                         break

                if not valid_epoch:
                     skipped_epochs += 1
                     continue

                # Compute features for this epoch
                combined_epoch_stats = {}
                for signal_key, segment in segments_with_keys:
                     basic_aggregations = [agg for agg in aggregations if agg in ['mean', 'std', 'min', 'max', 'median', 'var']]
                     if basic_aggregations:
                          stats = _compute_basic_stats(segment, basic_aggregations)
                          for simple_feature_name, value in stats.items():
                               generated_feature_names.add(simple_feature_name)
                               combined_epoch_stats[(signal_key, simple_feature_name)] = value

                epoch_features.update(combined_epoch_stats)
                all_epoch_results.append(epoch_features)
                processed_epochs += 1

            except Exception as e:
                logger.warning(f"Skipping epoch {epoch_start} due to error: {e}", exc_info=False)
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


# --- Add the new function ---
def compute_sleep_stage_mode(
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    # Add explicit arguments for global parameters
    global_window_length: pd.Timedelta,
    global_step_size: pd.Timedelta
) -> Feature:
    """
    Computes the modal (most frequent) sleep stage over epochs for EEGSleepStageSignals.

    Uses the provided global `epoch_grid_index` for epoch start times.
    The `window_length` can be specified in `parameters` to override the global
    setting, but `step_size` is determined solely by the `epoch_grid_index`.

    Args:
        signals: List containing the input TimeSeriesSignal objects (expected EEGSleepStageSignal).
        epoch_grid_index: The pre-calculated DatetimeIndex defining epoch start times.
        parameters: Dictionary containing operation parameters (currently unused for mode).
        global_window_length: Global window length from collection settings.
        global_step_size: Global step size from collection settings.

    Returns:
        A Feature object containing the computed modal sleep stage, indexed by epoch_start_time.

    Raises:
        ValueError: If input signals are invalid, epoch_grid_index is missing/empty.
        TypeError: If input signals are not EEGSleepStageSignals or don't have 'sleep_stage'.
    """
    if not signals:
        raise ValueError("No input signals provided for sleep stage mode calculation.")
    if epoch_grid_index is None or epoch_grid_index.empty:
        raise ValueError("A valid epoch_grid_index must be provided.")

    # Determine effective window length (same logic as feature_statistics)
    window_length_str = parameters.get('window_length')
    if window_length_str:
        effective_window_length = pd.Timedelta(window_length_str)
        logger.info(f"Using step-specific window_length override: {effective_window_length}")
    else:
        effective_window_length = global_window_length
        logger.info(f"Using global collection window_length: {effective_window_length}")

    if effective_window_length <= pd.Timedelta(0):
        raise ValueError("Effective window_length must be positive.")

    epoch_step_size = global_step_size # Use passed global arg

    logger.info(f"Computing sleep stage mode: effective_window={effective_window_length}, step={epoch_step_size} (from grid)")

    # --- Handle Empty Epoch Grid ---
    if epoch_grid_index.empty:
        return _handle_empty_feature_data(
            signals=signals,
            feature_names=['sleep_stage_mode'],
            feature_type=FeatureType.CATEGORICAL_MODE,
            operation_name="compute_sleep_stage_mode",
            parameters=parameters,
            global_window_length=global_window_length,
            global_step_size=global_step_size
        )

    # --- Feature Calculation Loop ---
    all_epoch_results = []
    processed_epochs = 0
    skipped_epochs = 0

    for epoch_start in epoch_grid_index:
        epoch_end = epoch_start + effective_window_length
        epoch_features = {'epoch_start': epoch_start}
        valid_epoch = True

        for signal in signals:
            signal_key = signal.metadata.name
            try:
                segment = signal.get_data()[epoch_start:epoch_end]

                # Check if the required column exists
                if 'sleep_stage' not in segment.columns:
                     raise TypeError(f"Signal '{signal_key}' missing required 'sleep_stage' column.")

                # Drop NaNs before calculating mode
                stages_in_epoch = segment['sleep_stage'].dropna()

                if stages_in_epoch.empty:
                    modal_stage = np.nan # Or specific code for empty/all NaN
                else:
                    # Use pandas Series.mode() for categorical data
                    mode_result = stages_in_epoch.mode()
                    if not mode_result.empty:
                        # .mode() can return multiple values if ties exist; take the first one.
                        modal_stage = mode_result.iloc[0]
                    else: # Handle case where input was all NaN or empty after dropna
                        modal_stage = np.nan

                # Store result using tuple key (signal_key, feature_name)
                epoch_features[(signal_key, 'sleep_stage_mode')] = modal_stage

            except TypeError as e:
                 # Catch TypeErrors specifically, often related to unexpected data types
                 logger.warning(f"TypeError processing signal '{signal_key}' in epoch {epoch_start}: {e}. Ensure 'sleep_stage' column contains expected data. Skipping epoch.")
                 valid_epoch = False
                 break
            except Exception as e:
                # Log other exceptions with traceback for debugging
                logger.warning(f"Error processing signal '{signal_key}' in epoch {epoch_start}: {e}. Skipping epoch.", exc_info=True)
                valid_epoch = False
                break # Stop processing this epoch

        if valid_epoch:
            all_epoch_results.append(epoch_features)
            processed_epochs += 1
        else:
            skipped_epochs += 1

    logger.info(f"Sleep stage mode calculation complete. Processed epochs: {processed_epochs}, Skipped epochs: {skipped_epochs}")

    if not all_epoch_results:
        logger.warning("No sleep stage mode features were successfully computed for any epoch.")
        # Return empty Feature object (using the same logic as above for empty grid)
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(s.metadata.name, 'sleep_stage_mode') for s in signals],
            names=['signal_key', 'feature']
        )
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_multiindex_cols)
        empty_data.index.name = 'timestamp'
        metadata_dict = {
            "epoch_window_length": global_window_length,
            "epoch_step_size": global_step_size,
            "feature_names": ['sleep_stage_mode'],
            "feature_type": FeatureType.CATEGORICAL_MODE,
            "source_signal_keys": [s.metadata.name for s in signals],
            "source_signal_ids": [s.metadata.signal_id for s in signals],
            "operations": [OperationInfo("compute_sleep_stage_mode", parameters)]
        }
        return Feature(data=empty_data, metadata=metadata_dict)

    # --- Assemble Final DataFrame ---
    feature_df = pd.DataFrame(all_epoch_results)
    feature_df = feature_df.set_index('epoch_start')

    # Convert tuple columns to MultiIndex
    if not feature_df.empty:
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns, names=['signal_key', 'feature'])
    else:
        # Recreate expected columns if df is empty after processing
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(s.metadata.name, 'sleep_stage_mode') for s in signals],
            names=['signal_key', 'feature']
        )
        feature_df = pd.DataFrame(index=feature_df.index, columns=expected_multiindex_cols)

    # Reindex to match the epoch grid exactly, filling skips with NaN
    feature_df = feature_df.reindex(epoch_grid_index)
    feature_df.index.name = 'timestamp'

    # --- Create Feature ---
    metadata_dict = {
        "epoch_window_length": global_window_length,
        "epoch_step_size": global_step_size,
        "feature_names": ['sleep_stage_mode'], # Simple feature name
        "feature_type": FeatureType.CATEGORICAL_MODE, # Use new type
        "source_signal_keys": [s.metadata.name for s in signals],
        "source_signal_ids": [s.metadata.signal_id for s in signals],
        "operations": [OperationInfo("compute_sleep_stage_mode", parameters)]
    }

    return Feature(data=feature_df, metadata=metadata_dict)


# --- HRV Feature Extraction Wrapper ---

@cache_features
def compute_hrv_features(
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    global_window_length: pd.Timedelta,
    global_step_size: pd.Timedelta
) -> Feature:
    """
    Computes Heart Rate Variability (HRV) features over epochs.

    Supports both RR interval signals and heart rate signals (with approximations).

    Args:
        signals: List of TimeSeriesSignal objects (heart rate or RR interval signals)
        epoch_grid_index: Pre-calculated DatetimeIndex defining epoch start times
        parameters: Dictionary containing:
            - window_length (str, optional): Duration of each epoch
            - hrv_metrics (List[str], optional): Specific HRV metrics to compute.
              Options: ['sdnn', 'rmssd', 'pnn50', 'sdsd', 'hr_cv'] or 'all'
              Default: ['sdnn', 'rmssd', 'pnn50']
            - use_rr_intervals (bool, optional): If True, expects RR interval data.
              If False, uses heart rate. Default: False
        global_window_length: Global window length from collection settings
        global_step_size: Global step size from collection settings

    Returns:
        Feature object containing computed HRV features

    Raises:
        ValueError: If input signals are invalid or parameters are incorrect

    Example (Workflow YAML):
        ```yaml
        # Extract HRV features from heart rate signal
        steps:
          - type: multi_signal
            operation: "compute_hrv_features"
            inputs: ["hr_h10"]
            parameters:
              hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
              use_rr_intervals: false  # Using heart rate approximation
            output: "hrv_features"
        ```

    Example (Python API):
        ```python
        # Compute HRV features using RR intervals
        hrv_features = compute_hrv_features(
            signals=[rr_signal],
            epoch_grid_index=collection.epoch_grid_index,
            parameters={
                "hrv_metrics": "all",  # All RR-based metrics
                "use_rr_intervals": true
            },
            global_window_length=pd.Timedelta("30s"),
            global_step_size=pd.Timedelta("30s")
        )
        ```
    """
    # Validate inputs using centralized validation utilities
    validation.validate_not_empty(signals, "input signals for HRV feature extraction")
    validation.validate_all_types(signals, TimeSeriesSignal, "input signals")
    validation.validate_not_empty(epoch_grid_index, "epoch_grid_index")

    # Parse parameters
    window_length_str = parameters.get('window_length')
    if window_length_str:
        effective_window_length = pd.Timedelta(window_length_str)
        logger.info(f"Using step-specific window_length override: {effective_window_length}")
    else:
        effective_window_length = global_window_length
        logger.info(f"Using global collection window_length: {effective_window_length}")

    validation.validate_timedelta_positive(effective_window_length, "Effective window_length")

    use_rr_intervals = parameters.get('use_rr_intervals', False)
    hrv_metrics = parameters.get('hrv_metrics', DEFAULT_HRV_METRICS_RR)

    if hrv_metrics == 'all':
        if use_rr_intervals:
            hrv_metrics = ['sdnn', 'rmssd', 'pnn50', 'sdsd', 'rr_mean', 'rr_std']
        else:
            hrv_metrics = DEFAULT_HRV_METRICS_HR

    epoch_step_size = global_step_size

    logger.info(f"Computing HRV features: window={effective_window_length}, step={epoch_step_size}, metrics={hrv_metrics}")

    # Handle empty epoch grid
    if epoch_grid_index.empty:
        return _handle_empty_feature_data(
            signals=signals,
            feature_names=hrv_metrics,
            feature_type=FeatureType.HRV,
            operation_name="compute_hrv_features",
            parameters=parameters,
            global_window_length=global_window_length,
            global_step_size=global_step_size
        )

    # Feature calculation loop
    all_epoch_results = []
    processed_epochs = 0
    skipped_epochs = 0
    generated_feature_names = set()

    for epoch_start in epoch_grid_index:
        epoch_end = epoch_start + effective_window_length
        epoch_features = {'epoch_start': epoch_start}
        valid_epoch = True

        for signal in signals:
            signal_key = signal.metadata.name
            try:
                segment = signal.get_data()[epoch_start:epoch_end]

                # Compute HRV features based on signal type
                if use_rr_intervals:
                    # Expect 'rr_interval' column
                    if 'rr_interval' not in segment.columns:
                        raise ValueError(f"Signal '{signal_key}' missing 'rr_interval' column")
                    rr_data = segment['rr_interval'].dropna()
                    hrv_results = _compute_hrv_time_domain(rr_data)
                else:
                    # Expect 'hr' column
                    hrv_results = _compute_hrv_from_heart_rate(segment)

                # Store results
                for feature_name, value in hrv_results.items():
                    if feature_name in hrv_metrics or 'all' in parameters.get('hrv_metrics', []):
                        generated_feature_names.add(feature_name)
                        epoch_features[(signal_key, feature_name)] = value

            except Exception as e:
                logger.warning(f"Error processing signal '{signal_key}' in epoch {epoch_start}: {e}")
                valid_epoch = False
                break

        if valid_epoch:
            all_epoch_results.append(epoch_features)
            processed_epochs += 1
        else:
            skipped_epochs += 1

    logger.info(f"HRV calculation complete. Processed epochs: {processed_epochs}, Skipped: {skipped_epochs}")

    if not all_epoch_results:
        logger.warning("No HRV features were successfully computed for any epoch.")
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(s.metadata.name, metric) for s in signals for metric in hrv_metrics],
            names=['signal_key', 'feature']
        )
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_multiindex_cols)
        empty_data.index.name = 'timestamp'

        metadata_dict = {
            "epoch_window_length": global_window_length,
            "epoch_step_size": global_step_size,
            "feature_names": list(generated_feature_names) if generated_feature_names else hrv_metrics,
            "feature_type": FeatureType.HRV,
            "source_signal_keys": [s.metadata.name for s in signals],
            "source_signal_ids": [s.metadata.signal_id for s in signals],
            "operations": [OperationInfo("compute_hrv_features", parameters)]
        }
        return Feature(data=empty_data, metadata=metadata_dict)

    # Assemble DataFrame
    feature_df = pd.DataFrame(all_epoch_results)
    feature_df = feature_df.set_index('epoch_start')

    if not feature_df.empty:
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns, names=['signal_key', 'feature'])
    else:
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(s.metadata.name, metric) for s in signals for metric in sorted(generated_feature_names)],
            names=['signal_key', 'feature']
        )
        feature_df = pd.DataFrame(index=feature_df.index, columns=expected_multiindex_cols)

    # Reindex to match epoch grid
    feature_df = feature_df.reindex(epoch_grid_index)
    feature_df.index.name = 'timestamp'

    # Create metadata
    metadata_dict = {
        "epoch_window_length": global_window_length,
        "epoch_step_size": global_step_size,
        "feature_names": sorted(list(generated_feature_names)),
        "feature_type": FeatureType.HRV,
        "source_signal_keys": [s.metadata.name for s in signals],
        "source_signal_ids": [s.metadata.signal_id for s in signals],
        "operations": [OperationInfo("compute_hrv_features", parameters)]
    }

    return Feature(data=feature_df, metadata=metadata_dict)


# --- Movement Feature Extraction Wrapper ---

@cache_features
def compute_movement_features(
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    global_window_length: pd.Timedelta,
    global_step_size: pd.Timedelta
) -> Feature:
    """
    Computes movement and activity features from accelerometer data over epochs.

    Args:
        signals: List of TimeSeriesSignal objects (accelerometer signals with x, y, z)
        epoch_grid_index: Pre-calculated DatetimeIndex defining epoch start times
        parameters: Dictionary containing:
            - window_length (str, optional): Duration of each epoch
            - movement_metrics (List[str], optional): Specific metrics to compute
              Options: ['magnitude_mean', 'magnitude_std', 'magnitude_max',
                       'activity_count', 'stillness_ratio', 'x_std', 'y_std', 'z_std']
              or 'all'. Default: 'all'
        global_window_length: Global window length from collection settings
        global_step_size: Global step size from collection settings

    Returns:
        Feature object containing computed movement features

    Raises:
        ValueError: If input signals are invalid or parameters are incorrect
    """
    if not signals:
        raise ValueError("No input signals provided for movement feature extraction.")
    if not all(isinstance(s, TimeSeriesSignal) for s in signals):
        raise ValueError("All input signals must be TimeSeriesSignal instances.")
    if epoch_grid_index is None or epoch_grid_index.empty:
        raise ValueError("A valid epoch_grid_index must be provided.")

    # Parse parameters
    window_length_str = parameters.get('window_length')
    if window_length_str:
        effective_window_length = pd.Timedelta(window_length_str)
        logger.info(f"Using step-specific window_length override: {effective_window_length}")
    else:
        effective_window_length = global_window_length
        logger.info(f"Using global collection window_length: {effective_window_length}")

    if effective_window_length <= pd.Timedelta(0):
        raise ValueError("Effective window_length must be positive.")

    movement_metrics = parameters.get('movement_metrics', 'all')
    if movement_metrics == 'all':
        movement_metrics = DEFAULT_MOVEMENT_METRICS

    epoch_step_size = global_step_size

    logger.info(f"Computing movement features: window={effective_window_length}, step={epoch_step_size}, metrics={movement_metrics}")

    # Handle empty epoch grid
    if epoch_grid_index.empty:
        return _handle_empty_feature_data(
            signals=signals,
            feature_names=movement_metrics,
            feature_type=FeatureType.MOVEMENT,
            operation_name="compute_movement_features",
            parameters=parameters,
            global_window_length=global_window_length,
            global_step_size=global_step_size
        )

    # Feature calculation loop
    all_epoch_results = []
    processed_epochs = 0
    skipped_epochs = 0
    generated_feature_names = set()

    for epoch_start in epoch_grid_index:
        epoch_end = epoch_start + effective_window_length
        epoch_features = {'epoch_start': epoch_start}
        valid_epoch = True

        for signal in signals:
            signal_key = signal.metadata.name
            try:
                segment = signal.get_data()[epoch_start:epoch_end]

                # Compute movement features
                movement_results = _compute_movement_features(segment)

                # Store results
                for feature_name, value in movement_results.items():
                    if feature_name in movement_metrics:
                        generated_feature_names.add(feature_name)
                        epoch_features[(signal_key, feature_name)] = value

            except Exception as e:
                logger.warning(f"Error processing signal '{signal_key}' in epoch {epoch_start}: {e}")
                valid_epoch = False
                break

        if valid_epoch:
            all_epoch_results.append(epoch_features)
            processed_epochs += 1
        else:
            skipped_epochs += 1

    logger.info(f"Movement calculation complete. Processed epochs: {processed_epochs}, Skipped: {skipped_epochs}")

    if not all_epoch_results:
        logger.warning("No movement features were successfully computed for any epoch.")
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(s.metadata.name, metric) for s in signals for metric in movement_metrics],
            names=['signal_key', 'feature']
        )
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_multiindex_cols)
        empty_data.index.name = 'timestamp'

        metadata_dict = {
            "epoch_window_length": global_window_length,
            "epoch_step_size": global_step_size,
            "feature_names": list(generated_feature_names) if generated_feature_names else movement_metrics,
            "feature_type": FeatureType.MOVEMENT,
            "source_signal_keys": [s.metadata.name for s in signals],
            "source_signal_ids": [s.metadata.signal_id for s in signals],
            "operations": [OperationInfo("compute_movement_features", parameters)]
        }
        return Feature(data=empty_data, metadata=metadata_dict)

    # Assemble DataFrame
    feature_df = pd.DataFrame(all_epoch_results)
    feature_df = feature_df.set_index('epoch_start')

    if not feature_df.empty:
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns, names=['signal_key', 'feature'])
    else:
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(s.metadata.name, metric) for s in signals for metric in sorted(generated_feature_names)],
            names=['signal_key', 'feature']
        )
        feature_df = pd.DataFrame(index=feature_df.index, columns=expected_multiindex_cols)

    # Reindex to match epoch grid
    feature_df = feature_df.reindex(epoch_grid_index)
    feature_df.index.name = 'timestamp'

    # Create metadata
    metadata_dict = {
        "epoch_window_length": global_window_length,
        "epoch_step_size": global_step_size,
        "feature_names": sorted(list(generated_feature_names)),
        "feature_type": FeatureType.MOVEMENT,
        "source_signal_keys": [s.metadata.name for s in signals],
        "source_signal_ids": [s.metadata.signal_id for s in signals],
        "operations": [OperationInfo("compute_movement_features", parameters)]
    }

    return Feature(data=feature_df, metadata=metadata_dict)


# --- Correlation Feature Extraction Wrapper ---

@cache_features
def compute_correlation_features(
    signals: List[TimeSeriesSignal],
    epoch_grid_index: pd.DatetimeIndex,
    parameters: Dict[str, Any],
    global_window_length: pd.Timedelta,
    global_step_size: pd.Timedelta
) -> Feature:
    """
    Computes correlation between two signals over epochs.

    Args:
        signals: List of exactly 2 TimeSeriesSignal objects
        epoch_grid_index: Pre-calculated DatetimeIndex defining epoch start times
        parameters: Dictionary containing:
            - window_length (str, optional): Duration of each epoch
            - signal1_column (str): Column name from first signal to correlate
            - signal2_column (str): Column name from second signal to correlate
            - method (str, optional): Correlation method ('pearson', 'spearman', 'kendall')
              Default: 'pearson'
        global_window_length: Global window length from collection settings
        global_step_size: Global step size from collection settings

    Returns:
        Feature object containing computed correlation

    Raises:
        ValueError: If number of signals != 2 or parameters are invalid
    """
    if len(signals) != 2:
        raise ValueError(f"Correlation requires exactly 2 signals, got {len(signals)}")
    if not all(isinstance(s, TimeSeriesSignal) for s in signals):
        raise ValueError("All input signals must be TimeSeriesSignal instances.")
    if epoch_grid_index is None or epoch_grid_index.empty:
        raise ValueError("A valid epoch_grid_index must be provided.")

    # Parse parameters
    window_length_str = parameters.get('window_length')
    if window_length_str:
        effective_window_length = pd.Timedelta(window_length_str)
        logger.info(f"Using step-specific window_length override: {effective_window_length}")
    else:
        effective_window_length = global_window_length
        logger.info(f"Using global collection window_length: {effective_window_length}")

    if effective_window_length <= pd.Timedelta(0):
        raise ValueError("Effective window_length must be positive.")

    signal1_column = parameters.get('signal1_column')
    signal2_column = parameters.get('signal2_column')
    method = parameters.get('method', 'pearson')

    if not signal1_column or not signal2_column:
        raise ValueError("Both 'signal1_column' and 'signal2_column' must be specified")

    epoch_step_size = global_step_size

    logger.info(f"Computing correlation: {signal1_column} vs {signal2_column}, method={method}, window={effective_window_length}")

    # Handle empty epoch grid
    if epoch_grid_index.empty:
        feature_name = f'{method}_corr'
        return _handle_empty_feature_data(
            signals=signals,
            feature_names=[feature_name],
            feature_type=FeatureType.CORRELATION,
            operation_name="compute_correlation_features",
            parameters=parameters,
            global_window_length=global_window_length,
            global_step_size=global_step_size,
            column_index_name='signal_pair'
        )

    # Feature calculation loop
    all_epoch_results = []
    processed_epochs = 0
    skipped_epochs = 0

    signal1, signal2 = signals[0], signals[1]
    signal_pair_name = f"{signal1.metadata.name}_vs_{signal2.metadata.name}"

    for epoch_start in epoch_grid_index:
        epoch_end = epoch_start + effective_window_length
        epoch_features = {'epoch_start': epoch_start}

        try:
            segment1 = signal1.get_data()[epoch_start:epoch_end]
            segment2 = signal2.get_data()[epoch_start:epoch_end]

            # Compute correlation
            corr_results = _compute_correlation_features(
                segment1, segment2,
                signal1_column, signal2_column,
                method
            )

            # Store results
            for feature_name, value in corr_results.items():
                epoch_features[(signal_pair_name, feature_name)] = value

            all_epoch_results.append(epoch_features)
            processed_epochs += 1

        except Exception as e:
            logger.warning(f"Error in epoch {epoch_start}: {e}")
            skipped_epochs += 1

    logger.info(f"Correlation calculation complete. Processed epochs: {processed_epochs}, Skipped: {skipped_epochs}")

    feature_name = f'{method}_corr'

    if not all_epoch_results:
        logger.warning("No correlation features were successfully computed for any epoch.")
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(signal_pair_name, feature_name)],
            names=['signal_pair', 'feature']
        )
        empty_data = pd.DataFrame(index=pd.DatetimeIndex([]), columns=expected_multiindex_cols)
        empty_data.index.name = 'timestamp'

        metadata_dict = {
            "epoch_window_length": global_window_length,
            "epoch_step_size": global_step_size,
            "feature_names": [feature_name],
            "feature_type": FeatureType.CORRELATION,
            "source_signal_keys": [s.metadata.name for s in signals],
            "source_signal_ids": [s.metadata.signal_id for s in signals],
            "operations": [OperationInfo("compute_correlation_features", parameters)]
        }
        return Feature(data=empty_data, metadata=metadata_dict)

    # Assemble DataFrame
    feature_df = pd.DataFrame(all_epoch_results)
    feature_df = feature_df.set_index('epoch_start')

    if not feature_df.empty:
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns, names=['signal_pair', 'feature'])
    else:
        expected_multiindex_cols = pd.MultiIndex.from_tuples(
            [(signal_pair_name, feature_name)],
            names=['signal_pair', 'feature']
        )
        feature_df = pd.DataFrame(index=feature_df.index, columns=expected_multiindex_cols)

    # Reindex to match epoch grid
    feature_df = feature_df.reindex(epoch_grid_index)
    feature_df.index.name = 'timestamp'

    # Create metadata
    metadata_dict = {
        "epoch_window_length": global_window_length,
        "epoch_step_size": global_step_size,
        "feature_names": [feature_name],
        "feature_type": FeatureType.CORRELATION,
        "source_signal_keys": [s.metadata.name for s in signals],
        "source_signal_ids": [s.metadata.signal_id for s in signals],
        "operations": [OperationInfo("compute_correlation_features", parameters)]
    }

    return Feature(data=feature_df, metadata=metadata_dict)
