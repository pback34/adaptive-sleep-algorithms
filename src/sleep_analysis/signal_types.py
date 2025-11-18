"""
Enum definitions for signal types, sensor types, and related metadata categories.

These enums ensure type safety and consistency throughout the framework.
"""

from enum import Enum

class SignalType(Enum):
    """Enum for different types of signals."""
    PPG = "PPG"
    ACCELEROMETER = "Accelerometer"
    HEART_RATE = "Heart Rate"
    HR = "Heart Rate"  # Alias for HEART_RATE
    EEG_SLEEP_STAGE = "EEG Sleep Stage"
    MAGNITUDE = "Magnitude"
    ANGLE = "Angle"
    FEATURES = "Features" # Added for epoch-based features
    # Additional signal types can be added as needed

class SensorType(Enum):
    """Enum for different types of sensors."""
    PPG = "PPG"
    ACCEL = "Accel"
    EKG = "EKG"
    EEG = "EEG"
    # Additional sensor types can be added as needed

class SensorModel(Enum):
    """Enum for different sensor models."""
    POLAR_H10 = "PolarH10"
    POLAR_SENSE = "PolarSense"
    ENCHANTED_WAVE = "EnchantedWave"
    # Additional sensor models can be added as needed

class BodyPosition(Enum):
    """Enum for different body positions."""
    CHEST = "chest"
    HEAD = "head"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    # Additional positions can be added as needed

class Unit(Enum):
    """Enum for different units of measurement."""
    G = "g"
    MILLI_G = "mg" # For accelerometers (e.g., Polar)
    BPM = "bpm"
    MILLISECONDS = "ms" # For HRV
    HZ = "Hz"
    DEGREES = "degrees" # For AngleSignal
    ARBITRARY = "arbitrary" # For values without standard physical units (e.g., quality scores)
    NONE = "none" # For categorical data like sleep stages
    # Additional units can be added as needed

    def __str__(self):
        return self.value
