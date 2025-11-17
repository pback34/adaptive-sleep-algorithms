"""
Random Forest sleep staging algorithm.

This module implements a Random Forest-based sleep staging classifier,
based on the methodology from:
"Sleep stage classification using heart rate variability and accelerometer data"
Nature Scientific Reports (2020)
https://www.nature.com/articles/s41598-020-79217-x

The algorithm uses HRV and movement features extracted from wearable sensors
to classify sleep into stages (wake, light, deep, REM).
"""

from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import logging
import joblib

from .base import SleepStagingAlgorithm, compute_classification_metrics

logger = logging.getLogger(__name__)


class RandomForestSleepStaging(SleepStagingAlgorithm):
    """
    Random Forest-based sleep staging algorithm.

    This implementation uses a Random Forest classifier to predict sleep stages
    from physiological features (HRV, movement, correlations). It supports both
    4-stage classification (wake, light, deep, REM) and 2-stage classification
    (sleep, wake).

    Attributes:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        class_weight: How to weight classes ('balanced', 'balanced_subsample', or None)
        random_state: Random seed for reproducibility
    """

    # Standard sleep stage mappings
    STAGE_4_LABELS = ['wake', 'light', 'deep', 'rem']
    STAGE_2_LABELS = ['wake', 'sleep']

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        n_stages: int = 4,
        **kwargs
    ):
        """
        Initialize the Random Forest sleep staging algorithm.

        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples to split a node (default: 2)
            min_samples_leaf: Minimum samples at leaf (default: 1)
            class_weight: Class weighting strategy ('balanced', 'balanced_subsample', None)
            random_state: Random seed for reproducibility
            n_stages: Number of sleep stages (2 or 4)
            **kwargs: Additional parameters passed to RandomForestClassifier
        """
        super().__init__(name="RandomForestSleepStaging")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_stages = n_stages
        self.rf_kwargs = kwargs

        # Set stage labels based on number of stages
        if n_stages == 4:
            self.stage_labels = self.STAGE_4_LABELS.copy()
        elif n_stages == 2:
            self.stage_labels = self.STAGE_2_LABELS.copy()
        else:
            raise ValueError(f"n_stages must be 2 or 4, got {n_stages}")

        self._model = None
        self._scaler = None
        self._feature_importance = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        validation_split: Optional[float] = None,
        normalize_features: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the Random Forest sleep staging model.

        Args:
            features: DataFrame of features (epochs x features)
            labels: Series of sleep stage labels for each epoch
            validation_split: Optional fraction for validation (0.0-1.0)
            normalize_features: Whether to standardize features (recommended)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics:
                - 'train_accuracy': Training set accuracy
                - 'val_accuracy': Validation set accuracy (if validation_split provided)
                - 'train_kappa': Training Cohen's kappa
                - 'val_kappa': Validation Cohen's kappa (if validation_split provided)
                - 'n_epochs': Number of training epochs
                - 'n_features': Number of features

        Raises:
            ValueError: If features or labels are invalid
            ImportError: If scikit-learn is not installed
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError(
                "scikit-learn is required for Random Forest. "
                "Install it with: pip install scikit-learn"
            )

        # Validate inputs
        self._validate_features(features)
        self._validate_labels(labels)

        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        if len(common_idx) == 0:
            raise ValueError("No common indices between features and labels")

        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        # Check for valid labels
        unique_labels = set(labels.unique())
        valid_labels = set(self.stage_labels)
        invalid = unique_labels - valid_labels
        if invalid:
            logger.warning(
                f"Found invalid labels: {invalid}. "
                f"Valid labels for {self.n_stages}-stage classification: {self.stage_labels}"
            )
            # Filter to valid labels only
            labels = labels[labels.isin(self.stage_labels)]
            features = features.loc[labels.index]

        # Store feature names
        self.feature_names = list(features.columns)

        logger.info(
            f"Training Random Forest with {len(features)} epochs, "
            f"{len(self.feature_names)} features, {self.n_stages} stages"
        )

        # Handle NaN values
        if features.isna().any().any():
            logger.warning("Features contain NaN values. Filling with column means.")
            features = features.fillna(features.mean())

        # Split into train/validation if requested
        if validation_split is not None and 0 < validation_split < 1:
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels,
                test_size=validation_split,
                random_state=self.random_state,
                stratify=labels
            )
        else:
            X_train = features
            y_train = labels
            X_val = None
            y_val = None

        # Normalize features
        if normalize_features:
            self._scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self._scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            if X_val is not None:
                X_val_scaled = pd.DataFrame(
                    self._scaler.transform(X_val),
                    index=X_val.index,
                    columns=X_val.columns
                )
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            self._scaler = None

        # Create and train Random Forest
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            **self.rf_kwargs
        )

        logger.info("Training Random Forest classifier...")
        self._model.fit(X_train_scaled, y_train)

        # Store feature importance
        self._feature_importance = pd.Series(
            self._model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        self.is_fitted = True

        # Compute training metrics
        y_train_pred = self._model.predict(X_train_scaled)
        train_metrics = compute_classification_metrics(
            y_train, y_train_pred,
            metrics=['accuracy', 'cohen_kappa']
        )

        results = {
            'train_accuracy': train_metrics['accuracy'],
            'train_kappa': train_metrics['cohen_kappa'],
            'n_epochs': len(X_train),
            'n_features': len(self.feature_names),
            'feature_importance': self._feature_importance.to_dict()
        }

        # Compute validation metrics if available
        if X_val is not None:
            y_val_pred = self._model.predict(X_val_scaled)
            val_metrics = compute_classification_metrics(
                y_val, y_val_pred,
                metrics=['accuracy', 'cohen_kappa']
            )
            results['val_accuracy'] = val_metrics['accuracy']
            results['val_kappa'] = val_metrics['cohen_kappa']
            results['n_val_epochs'] = len(X_val)

        logger.info(
            f"Training complete. Accuracy: {results['train_accuracy']:.3f}, "
            f"Kappa: {results['train_kappa']:.3f}"
        )
        if 'val_accuracy' in results:
            logger.info(
                f"Validation: Accuracy: {results['val_accuracy']:.3f}, "
                f"Kappa: {results['val_kappa']:.3f}"
            )

        return results

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict sleep stages for the given features.

        Args:
            features: DataFrame of features (epochs x features)

        Returns:
            Series of predicted sleep stage labels with the same index as features

        Raises:
            RuntimeError: If the algorithm has not been fitted
            ValueError: If features are invalid
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before prediction")

        self._validate_features(features)

        # Select only the features used during training
        features_subset = features[self.feature_names]

        # Handle NaN values
        if features_subset.isna().any().any():
            logger.warning("Features contain NaN values. Filling with column means.")
            features_subset = features_subset.fillna(features_subset.mean())

        # Normalize if scaler is available
        if self._scaler is not None:
            features_scaled = pd.DataFrame(
                self._scaler.transform(features_subset),
                index=features_subset.index,
                columns=features_subset.columns
            )
        else:
            features_scaled = features_subset

        # Predict
        predictions = self._model.predict(features_scaled)

        return pd.Series(predictions, index=features.index, name='predicted_stage')

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict sleep stage probabilities for the given features.

        Args:
            features: DataFrame of features (epochs x features)

        Returns:
            DataFrame of probabilities (epochs x stages) with stage labels as columns

        Raises:
            RuntimeError: If the algorithm has not been fitted
            ValueError: If features are invalid
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before prediction")

        self._validate_features(features)

        # Select only the features used during training
        features_subset = features[self.feature_names]

        # Handle NaN values
        if features_subset.isna().any().any():
            logger.warning("Features contain NaN values. Filling with column means.")
            features_subset = features_subset.fillna(features_subset.mean())

        # Normalize if scaler is available
        if self._scaler is not None:
            features_scaled = pd.DataFrame(
                self._scaler.transform(features_subset),
                index=features_subset.index,
                columns=features_subset.columns
            )
        else:
            features_scaled = features_subset

        # Predict probabilities
        probabilities = self._model.predict_proba(features_scaled)

        # Create DataFrame with stage labels as columns
        return pd.DataFrame(
            probabilities,
            index=features.index,
            columns=self._model.classes_
        )

    def evaluate(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the algorithm's performance on labeled data.

        Args:
            features: DataFrame of features (epochs x features)
            labels: Series of true sleep stage labels
            metrics: List of metrics to compute (None = all)

        Returns:
            Dictionary mapping metric names to values:
                - 'accuracy': Overall classification accuracy
                - 'cohen_kappa': Cohen's kappa agreement score
                - 'f1_macro': Macro-averaged F1 score
                - 'f1_weighted': Weighted F1 score
                - 'precision_macro': Macro-averaged precision
                - 'recall_macro': Macro-averaged recall
                - 'confusion_matrix': Confusion matrix
                - 'per_class_metrics': Per-class precision, recall, f1

        Raises:
            RuntimeError: If the algorithm has not been fitted
            ValueError: If features or labels are invalid
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before evaluation")

        self._validate_features(features)
        self._validate_labels(labels)

        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        if len(common_idx) == 0:
            raise ValueError("No common indices between features and labels")

        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        # Get predictions
        predictions = self.predict(features)

        # Compute metrics
        results = compute_classification_metrics(
            labels, predictions, metrics=metrics
        )

        # Add per-class metrics
        try:
            from sklearn.metrics import classification_report
            report = classification_report(
                labels, predictions,
                output_dict=True,
                zero_division=0
            )
            results['per_class_metrics'] = report
        except ImportError:
            logger.warning("scikit-learn not available for detailed metrics")

        return results

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores from the Random Forest.

        Returns:
            Series mapping feature names to importance scores (sorted descending),
            or None if not fitted
        """
        if not self.is_fitted or self._feature_importance is None:
            return None

        return self._feature_importance

    def _save_impl(self, path: Path) -> None:
        """
        Save the Random Forest model and scaler to disk.

        Args:
            path: Directory path to save model artifacts
        """
        # Save the Random Forest model
        model_path = path / 'random_forest_model.pkl'
        joblib.dump(self._model, model_path)
        logger.info(f"Saved Random Forest model to {model_path}")

        # Save the scaler if it exists
        if self._scaler is not None:
            scaler_path = path / 'scaler.pkl'
            joblib.dump(self._scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")

        # Save metadata
        metadata = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_stages': self.n_stages,
            'stage_labels': self.stage_labels,
            'feature_names': self.feature_names,
            'feature_importance': self._feature_importance.to_dict() if self._feature_importance is not None else None
        }

        metadata_path = path / 'metadata.pkl'
        joblib.dump(metadata, metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")

    def _load_impl(self, path: Path) -> None:
        """
        Load the Random Forest model and scaler from disk.

        Args:
            path: Directory path containing saved model artifacts
        """
        # Load metadata
        metadata_path = path / 'metadata.pkl'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata = joblib.load(metadata_path)
        self.n_estimators = metadata['n_estimators']
        self.max_depth = metadata['max_depth']
        self.min_samples_split = metadata['min_samples_split']
        self.min_samples_leaf = metadata['min_samples_leaf']
        self.class_weight = metadata['class_weight']
        self.random_state = metadata['random_state']
        self.n_stages = metadata['n_stages']
        self.stage_labels = metadata['stage_labels']
        self.feature_names = metadata['feature_names']

        if metadata['feature_importance'] is not None:
            self._feature_importance = pd.Series(metadata['feature_importance'])

        logger.info(f"Loaded metadata from {metadata_path}")

        # Load the Random Forest model
        model_path = path / 'random_forest_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = joblib.load(model_path)
        logger.info(f"Loaded Random Forest model from {model_path}")

        # Load the scaler if it exists
        scaler_path = path / 'scaler.pkl'
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            self._scaler = None

    def __repr__(self) -> str:
        """String representation of the algorithm."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"RandomForestSleepStaging("
            f"n_stages={self.n_stages}, "
            f"n_estimators={self.n_estimators}, "
            f"{fitted_str})"
        )
