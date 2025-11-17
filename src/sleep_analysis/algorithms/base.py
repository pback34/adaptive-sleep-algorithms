"""
Base classes and interfaces for sleep staging algorithms.

This module defines the abstract base class for sleep staging algorithms,
providing a common interface for training, prediction, and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SleepStagingAlgorithm(ABC):
    """
    Abstract base class for sleep staging algorithms.

    This class defines the interface that all sleep staging algorithms must implement.
    It provides a standardized way to train, predict, and evaluate sleep stage
    classification models.

    Attributes:
        is_fitted: Whether the algorithm has been trained
        feature_names: List of feature names expected by the algorithm
        stage_labels: List of sleep stage labels the algorithm can predict
    """

    def __init__(self, name: str = "SleepStagingAlgorithm"):
        """
        Initialize the sleep staging algorithm.

        Args:
            name: Name of the algorithm
        """
        self.name = name
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.stage_labels: Optional[List[str]] = None
        self._model = None

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        validation_split: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the sleep staging algorithm.

        Args:
            features: DataFrame of features (epochs x features)
            labels: Series of sleep stage labels for each epoch
            validation_split: Optional fraction of data to use for validation (0.0-1.0)
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dictionary containing training metrics and metadata

        Raises:
            ValueError: If features or labels are invalid
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict sleep stages for the given features.

        Args:
            features: DataFrame of features (epochs x features)

        Returns:
            Series of predicted sleep stage labels with the same index as features

        Raises:
            RuntimeError: If the algorithm has not been fitted
            ValueError: If features are invalid or incompatible
        """
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict sleep stage probabilities for the given features.

        Args:
            features: DataFrame of features (epochs x features)

        Returns:
            DataFrame of probabilities (epochs x stages) with stage labels as columns

        Raises:
            RuntimeError: If the algorithm has not been fitted
            ValueError: If features are invalid or incompatible
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the algorithm's performance on labeled data.

        Args:
            features: DataFrame of features (epochs x features)
            labels: Series of true sleep stage labels
            metrics: Optional list of metrics to compute. If None, compute all available.
                    Common metrics: 'accuracy', 'cohen_kappa', 'f1_macro', 'f1_weighted',
                    'precision_macro', 'recall_macro', 'confusion_matrix'

        Returns:
            Dictionary mapping metric names to values

        Raises:
            RuntimeError: If the algorithm has not been fitted
            ValueError: If features or labels are invalid
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save the trained algorithm to disk.

        Args:
            path: Path to save the algorithm (will create a directory)

        Raises:
            RuntimeError: If the algorithm has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before saving")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {self.name} to {path}")
        self._save_impl(path)

    @abstractmethod
    def _save_impl(self, path: Path) -> None:
        """
        Implementation-specific save logic.

        Args:
            path: Directory path to save algorithm artifacts
        """
        pass

    def load(self, path: Path) -> None:
        """
        Load a trained algorithm from disk.

        Args:
            path: Path to the saved algorithm directory

        Raises:
            FileNotFoundError: If the path does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Algorithm path not found: {path}")

        logger.info(f"Loading {self.name} from {path}")
        self._load_impl(path)
        self.is_fitted = True

    @abstractmethod
    def _load_impl(self, path: Path) -> None:
        """
        Implementation-specific load logic.

        Args:
            path: Directory path containing saved algorithm artifacts
        """
        pass

    def _validate_features(self, features: pd.DataFrame) -> None:
        """
        Validate that features match expected format.

        Args:
            features: Features to validate

        Raises:
            ValueError: If features are invalid
        """
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame")

        if features.empty:
            raise ValueError("Features DataFrame is empty")

        if self.is_fitted and self.feature_names is not None:
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                raise ValueError(
                    f"Missing required features: {missing_features}. "
                    f"Expected features: {self.feature_names}"
                )

            extra_features = set(features.columns) - set(self.feature_names)
            if extra_features:
                logger.warning(
                    f"Extra features will be ignored: {extra_features}"
                )

    def _validate_labels(self, labels: pd.Series) -> None:
        """
        Validate that labels match expected format.

        Args:
            labels: Labels to validate

        Raises:
            ValueError: If labels are invalid
        """
        if not isinstance(labels, pd.Series):
            raise ValueError("Labels must be a pandas Series")

        if labels.empty:
            raise ValueError("Labels Series is empty")

        if labels.isna().any():
            raise ValueError("Labels contain NaN values")

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores if available.

        Returns:
            Series mapping feature names to importance scores, or None if not available
        """
        return None

    def __repr__(self) -> str:
        """String representation of the algorithm."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({fitted_str})"


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics for sleep staging evaluation.

    Args:
        y_true: True sleep stage labels
        y_pred: Predicted sleep stage labels
        metrics: List of metrics to compute. If None, compute all.
                Available: 'accuracy', 'cohen_kappa', 'f1_macro', 'f1_weighted',
                'precision_macro', 'recall_macro', 'confusion_matrix'

    Returns:
        Dictionary mapping metric names to values
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            cohen_kappa_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix
        )
    except ImportError:
        raise ImportError(
            "scikit-learn is required for metric computation. "
            "Install it with: pip install scikit-learn"
        )

    if metrics is None:
        metrics = [
            'accuracy', 'cohen_kappa', 'f1_macro', 'f1_weighted',
            'precision_macro', 'recall_macro', 'confusion_matrix'
        ]

    results = {}

    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = accuracy_score(y_true, y_pred)

        elif metric == 'cohen_kappa':
            results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        elif metric == 'f1_macro':
            results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        elif metric == 'f1_weighted':
            results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        elif metric == 'precision_macro':
            results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)

        elif metric == 'recall_macro':
            results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

        elif metric == 'confusion_matrix':
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        else:
            logger.warning(f"Unknown metric: {metric}")

    return results
