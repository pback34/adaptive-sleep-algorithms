"""
Algorithm operations for workflow integration.

This module provides wrapper functions to integrate sleep staging algorithms
into the workflow system.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import logging
from pathlib import Path

from ..features.feature import Feature
from ..core.metadata import FeatureMetadata, FeatureType
from ..algorithms.random_forest import RandomForestSleepStaging

logger = logging.getLogger(__name__)


def random_forest_sleep_staging(
    signals: List,
    parameters: Dict[str, Any],
    epoch_grid: Optional[pd.DatetimeIndex] = None
) -> Feature:
    """
    Apply Random Forest sleep staging algorithm to feature matrix.

    This operation can either:
    1. Train a new Random Forest model (if train_mode=True and labels provided)
    2. Load a pre-trained model and predict (if model_path provided)
    3. Predict with a previously trained model in the same workflow

    Args:
        signals: List containing a single Feature object with the feature matrix
        parameters: Dictionary with algorithm parameters:
            - mode: 'train', 'predict', or 'train_predict' (default: 'predict')
            - model_path: Path to saved model (required for 'predict' mode)
            - labels_column: Column name in features containing ground truth labels
                           (required for 'train' and 'train_predict' modes)
            - n_estimators: Number of trees (default: 100)
            - max_depth: Maximum tree depth (default: None)
            - n_stages: Number of sleep stages - 2 or 4 (default: 4)
            - validation_split: Fraction for validation (default: 0.2)
            - normalize_features: Whether to standardize features (default: True)
            - save_model_path: Where to save trained model (optional)
            - class_weight: Class weighting strategy (default: 'balanced')
            - random_state: Random seed (default: 42)
        epoch_grid: Epoch grid (not used, for compatibility)

    Returns:
        Feature object containing:
            - predictions: Predicted sleep stages for each epoch
            - probabilities: Prediction probabilities (if available)
            - metadata: Training metrics and feature importance

    Raises:
        ValueError: If parameters are invalid or incompatible
        RuntimeError: If model not found or not trained

    Example workflow step:
        ```yaml
        - type: multi_signal
          operation: "random_forest_sleep_staging"
          inputs: ["combined_features"]
          parameters:
            mode: "train_predict"
            labels_column: "sleep_stage"
            n_estimators: 100
            n_stages: 4
            validation_split: 0.2
            save_model_path: "models/my_rf_model"
          output: "predicted_sleep_stages"
        ```
    """
    # Validate inputs
    if not signals or len(signals) != 1:
        raise ValueError(
            "random_forest_sleep_staging expects exactly 1 input signal "
            "(the combined feature matrix)"
        )

    feature_signal = signals[0]

    if not isinstance(feature_signal, Feature):
        raise ValueError(
            "Input signal must be a Feature object containing the feature matrix"
        )

    # Get the feature matrix
    features = feature_signal.compute()

    if features.empty:
        raise ValueError("Feature matrix is empty")

    logger.info(
        f"Random Forest sleep staging: {len(features)} epochs, "
        f"{len(features.columns)} features"
    )

    # Extract parameters
    mode = parameters.get('mode', 'predict')
    model_path = parameters.get('model_path')
    labels_column = parameters.get('labels_column')
    save_model_path = parameters.get('save_model_path')

    # Algorithm parameters
    n_estimators = parameters.get('n_estimators', 100)
    max_depth = parameters.get('max_depth', None)
    n_stages = parameters.get('n_stages', 4)
    validation_split = parameters.get('validation_split', 0.2)
    normalize_features = parameters.get('normalize_features', True)
    class_weight = parameters.get('class_weight', 'balanced')
    random_state = parameters.get('random_state', 42)

    # Validate mode
    if mode not in ['train', 'predict', 'train_predict']:
        raise ValueError(
            f"mode must be 'train', 'predict', or 'train_predict', got '{mode}'"
        )

    # Create algorithm instance
    algo = RandomForestSleepStaging(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_stages=n_stages
    )

    results_metadata = {}

    # TRAIN MODE
    if mode in ['train', 'train_predict']:
        if labels_column is None:
            raise ValueError(
                "labels_column parameter is required for training mode"
            )

        if labels_column not in features.columns:
            raise ValueError(
                f"labels_column '{labels_column}' not found in features. "
                f"Available columns: {list(features.columns)}"
            )

        # Extract labels and features
        labels = features[labels_column]
        feature_cols = [col for col in features.columns if col != labels_column]
        X = features[feature_cols]

        logger.info(f"Training Random Forest with {len(X)} epochs, {len(feature_cols)} features")

        # Train the model
        train_results = algo.fit(
            X, labels,
            validation_split=validation_split if mode == 'train' else None,
            normalize_features=normalize_features
        )

        results_metadata['train_results'] = train_results
        logger.info(
            f"Training complete. Accuracy: {train_results['train_accuracy']:.3f}, "
            f"Kappa: {train_results['train_kappa']:.3f}"
        )

        # Save model if requested
        if save_model_path:
            save_path = Path(save_model_path)
            algo.save(save_path)
            logger.info(f"Saved trained model to {save_path}")
            results_metadata['model_path'] = str(save_path)

        # If train-only mode, return training metrics
        if mode == 'train':
            # Create a dummy feature with just the training metrics
            metadata = FeatureMetadata(
                feature_names=['training_complete'],
                feature_type=FeatureType.CUSTOM,
                operation_name='random_forest_sleep_staging',
                parameters=parameters,
                additional_info=results_metadata
            )

            # Return a minimal DataFrame
            result_df = pd.DataFrame(
                {'training_complete': [1]},
                index=[features.index[0]]
            )

            return Feature(
                data=result_df,
                metadata=metadata,
                parent_signal_keys=feature_signal.signal_key,
                lazy=False
            )

    # PREDICT MODE
    elif mode == 'predict':
        if model_path is None:
            raise ValueError(
                "model_path parameter is required for predict mode"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}"
            )

        logger.info(f"Loading model from {model_path}")
        algo.load(model_path)

    # Generate predictions (for predict and train_predict modes)
    if mode in ['predict', 'train_predict']:
        # Separate features from labels if present
        if labels_column and labels_column in features.columns:
            feature_cols = [col for col in features.columns if col != labels_column]
            X = features[feature_cols]
        else:
            X = features

        logger.info(f"Generating predictions for {len(X)} epochs...")

        # Predict stages and probabilities
        predictions = algo.predict(X)
        probabilities = algo.predict_proba(X)

        # Get feature importance
        feature_importance = algo.get_feature_importance()
        if feature_importance is not None:
            results_metadata['feature_importance'] = feature_importance.to_dict()

        # Combine predictions and probabilities into result DataFrame
        result_df = pd.DataFrame(index=X.index)
        result_df['predicted_stage'] = predictions

        # Add probability columns
        for stage in probabilities.columns:
            result_df[f'prob_{stage}'] = probabilities[stage]

        # Create metadata
        feature_names = ['predicted_stage'] + [f'prob_{stage}' for stage in probabilities.columns]
        metadata = FeatureMetadata(
            feature_names=feature_names,
            feature_type=FeatureType.SLEEP_STAGE,
            operation_name='random_forest_sleep_staging',
            parameters=parameters,
            additional_info=results_metadata
        )

        logger.info(
            f"Sleep staging complete. Predictions: {len(predictions)} epochs, "
            f"{n_stages} stages"
        )

        # Log stage distribution
        stage_counts = predictions.value_counts()
        logger.info(f"Predicted stage distribution: {stage_counts.to_dict()}")

        return Feature(
            data=result_df,
            metadata=metadata,
            parent_signal_keys=feature_signal.signal_key,
            lazy=False
        )


def evaluate_sleep_staging(
    signals: List,
    parameters: Dict[str, Any],
    epoch_grid: Optional[pd.DatetimeIndex] = None
) -> Feature:
    """
    Evaluate sleep staging predictions against ground truth labels.

    Args:
        signals: List containing a single Feature object with predictions and labels
        parameters: Dictionary with evaluation parameters:
            - predictions_column: Column name with predictions (default: 'predicted_stage')
            - labels_column: Column name with ground truth labels (required)
            - metrics: List of metrics to compute (default: all)
            - confusion_matrix_path: Optional path to save confusion matrix plot
            - hypnogram_path: Optional path to save hypnogram plot
        epoch_grid: Epoch grid (not used, for compatibility)

    Returns:
        Feature object containing evaluation metrics

    Example workflow step:
        ```yaml
        - type: multi_signal
          operation: "evaluate_sleep_staging"
          inputs: ["predicted_sleep_stages"]
          parameters:
            predictions_column: "predicted_stage"
            labels_column: "true_stage"
            confusion_matrix_path: "results/confusion_matrix.png"
          output: "evaluation_metrics"
        ```
    """
    from ..algorithms.evaluation import plot_confusion_matrix, plot_hypnogram
    from ..algorithms.base import compute_classification_metrics

    # Validate inputs
    if not signals or len(signals) != 1:
        raise ValueError(
            "evaluate_sleep_staging expects exactly 1 input signal"
        )

    feature_signal = signals[0]
    if not isinstance(feature_signal, Feature):
        raise ValueError("Input signal must be a Feature object")

    data = feature_signal.compute()

    # Extract parameters
    predictions_column = parameters.get('predictions_column', 'predicted_stage')
    labels_column = parameters.get('labels_column')
    metrics = parameters.get('metrics', None)
    confusion_matrix_path = parameters.get('confusion_matrix_path')
    hypnogram_path = parameters.get('hypnogram_path')

    if labels_column is None:
        raise ValueError("labels_column parameter is required")

    if predictions_column not in data.columns:
        raise ValueError(
            f"predictions_column '{predictions_column}' not found in data"
        )

    if labels_column not in data.columns:
        raise ValueError(
            f"labels_column '{labels_column}' not found in data"
        )

    # Extract predictions and labels
    y_pred = data[predictions_column]
    y_true = data[labels_column]

    logger.info(f"Evaluating sleep staging: {len(y_pred)} epochs")

    # Compute metrics
    eval_results = compute_classification_metrics(y_true, y_pred, metrics=metrics)

    logger.info(
        f"Evaluation complete. Accuracy: {eval_results.get('accuracy', 0):.3f}, "
        f"Kappa: {eval_results.get('cohen_kappa', 0):.3f}"
    )

    # Plot confusion matrix if requested
    if confusion_matrix_path and 'confusion_matrix' in eval_results:
        stage_labels = sorted(y_true.unique())
        plot_confusion_matrix(
            eval_results['confusion_matrix'],
            stage_labels,
            output_path=confusion_matrix_path
        )

    # Plot hypnogram if requested
    if hypnogram_path:
        plot_hypnogram(
            y_pred, y_true,
            output_path=hypnogram_path
        )

    # Convert results to DataFrame
    result_data = {}
    for key, value in eval_results.items():
        if not isinstance(value, (list, dict, np.ndarray)):
            result_data[key] = [value]

    result_df = pd.DataFrame(result_data, index=[data.index[0]])

    # Create metadata
    metadata = FeatureMetadata(
        feature_names=list(result_data.keys()),
        feature_type=FeatureType.CUSTOM,
        operation_name='evaluate_sleep_staging',
        parameters=parameters,
        additional_info=eval_results
    )

    return Feature(
        data=result_df,
        metadata=metadata,
        parent_signal_keys=feature_signal.signal_key,
        lazy=False
    )
