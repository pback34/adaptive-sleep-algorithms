"""Signal classes derived from the SignalData base class."""

from ..core.signal_data import SignalData
from .time_series_signal import TimeSeriesSignal
from .ppg_signal import PPGSignal
from .accelerometer_signal import AccelerometerSignal
from .heart_rate_signal import HeartRateSignal
from .magnitude_signal import MagnitudeSignal
from .angle_signal import AngleSignal

__all__ = ['SignalData', 'TimeSeriesSignal', 'PPGSignal', 'AccelerometerSignal', 'HeartRateSignal', 'MagnitudeSignal', 'AngleSignal']
