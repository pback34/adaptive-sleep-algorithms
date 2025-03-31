"""
EEG Sleep Stage signal class implementation.

This module defines the EEGSleepStageSignal class for EEG-based sleep staging data.
"""

import pandas as pd
from .time_series_signal import TimeSeriesSignal
from ..signal_types import SignalType

class EEGSleepStageSignal(TimeSeriesSignal):
    """
    Class for EEG-based sleep stage signals.

    These signals typically contain the predicted sleep stage (e.g., Awake, N1, N2, N3, REM)
    at specific time points, often derived from EEG power spectral analysis.
    They may also include related metrics like EEG quality or spectral power sums.
    """
    _is_abstract = False
    signal_type = SignalType.EEG_SLEEP_STAGE
    required_columns = ['sleep_stage']
    optional_columns = ['sum_power', 'eeg_quality']

    # Removed get_sampling_rate override - will use TimeSeriesSignal implementation

    def get_stage_distribution(self) -> pd.Series:
        """
        Calculate the distribution of sleep stages.

        Returns:
            A pandas Series with sleep stages as index and counts as values.
            Returns None if 'sleep_stage' column is not present or empty.
        """
        df = self.get_data()
        if 'sleep_stage' not in df.columns or df['sleep_stage'].dropna().empty:
            return None

        return df['sleep_stage'].value_counts()
