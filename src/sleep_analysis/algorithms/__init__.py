"""
Sleep staging algorithms module.

This module provides sleep staging algorithms and evaluation utilities
for classifying sleep stages from physiological features.
"""

from .base import SleepStagingAlgorithm, compute_classification_metrics
from .random_forest import RandomForestSleepStaging
from .evaluation import (
    compare_algorithms,
    cross_validate_algorithm,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_hypnogram,
    compute_sleep_statistics
)

__all__ = [
    # Base classes
    'SleepStagingAlgorithm',
    'compute_classification_metrics',

    # Algorithms
    'RandomForestSleepStaging',

    # Evaluation utilities
    'compare_algorithms',
    'cross_validate_algorithm',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_hypnogram',
    'compute_sleep_statistics',
]
