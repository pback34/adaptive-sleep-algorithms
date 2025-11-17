"""
Evaluation utilities for sleep staging algorithms.

This module provides functions for evaluating and comparing sleep staging algorithms,
including metrics computation, visualization, and cross-validation.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from .base import SleepStagingAlgorithm, compute_classification_metrics

logger = logging.getLogger(__name__)


def compare_algorithms(
    algorithms: List[SleepStagingAlgorithm],
    features: pd.DataFrame,
    labels: pd.Series,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple sleep staging algorithms on the same dataset.

    Args:
        algorithms: List of fitted sleep staging algorithms to compare
        features: DataFrame of features (epochs x features)
        labels: Series of true sleep stage labels
        metrics: List of metrics to compute (None = default set)

    Returns:
        DataFrame with algorithms as rows and metrics as columns

    Raises:
        ValueError: If any algorithm is not fitted
    """
    if metrics is None:
        metrics = ['accuracy', 'cohen_kappa', 'f1_macro', 'f1_weighted']

    results = []

    for algo in algorithms:
        if not algo.is_fitted:
            raise ValueError(f"Algorithm {algo.name} is not fitted")

        logger.info(f"Evaluating {algo.name}...")
        algo_results = algo.evaluate(features, labels, metrics=metrics)

        # Extract numeric metrics only
        row = {'algorithm': algo.name}
        for metric in metrics:
            if metric in algo_results and not isinstance(algo_results[metric], np.ndarray):
                row[metric] = algo_results[metric]

        results.append(row)

    comparison_df = pd.DataFrame(results).set_index('algorithm')

    logger.info("Algorithm comparison complete")
    return comparison_df


def cross_validate_algorithm(
    algorithm_class: type,
    algorithm_params: Dict[str, Any],
    features: pd.DataFrame,
    labels: pd.Series,
    n_folds: int = 5,
    stratified: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for a sleep staging algorithm.

    Args:
        algorithm_class: Class of the algorithm to validate (not an instance)
        algorithm_params: Parameters to pass to algorithm constructor
        features: DataFrame of features (epochs x features)
        labels: Series of sleep stage labels
        n_folds: Number of cross-validation folds
        stratified: Whether to use stratified k-fold (recommended for imbalanced data)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - 'mean_accuracy': Mean accuracy across folds
            - 'std_accuracy': Standard deviation of accuracy
            - 'mean_kappa': Mean Cohen's kappa
            - 'std_kappa': Standard deviation of kappa
            - 'fold_results': List of per-fold results
            - 'confusion_matrices': List of confusion matrices per fold

    Raises:
        ImportError: If scikit-learn is not installed
    """
    try:
        from sklearn.model_selection import StratifiedKFold, KFold
    except ImportError:
        raise ImportError(
            "scikit-learn is required for cross-validation. "
            "Install it with: pip install scikit-learn"
        )

    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Create k-fold splitter
    if stratified:
        kfold = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )
    else:
        kfold = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )

    fold_results = []
    confusion_matrices = []

    logger.info(f"Starting {n_folds}-fold cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(kfold.split(features, labels)):
        logger.info(f"Fold {fold + 1}/{n_folds}")

        # Split data
        X_train = features.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]

        # Create and train algorithm
        algo = algorithm_class(**algorithm_params)
        algo.fit(X_train, y_train)

        # Evaluate
        metrics = algo.evaluate(
            X_test, y_test,
            metrics=['accuracy', 'cohen_kappa', 'f1_macro', 'confusion_matrix']
        )

        fold_results.append({
            'fold': fold + 1,
            'accuracy': metrics['accuracy'],
            'cohen_kappa': metrics['cohen_kappa'],
            'f1_macro': metrics['f1_macro'],
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        confusion_matrices.append(metrics['confusion_matrix'])

    # Aggregate results
    results_df = pd.DataFrame(fold_results)

    summary = {
        'mean_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'mean_kappa': results_df['cohen_kappa'].mean(),
        'std_kappa': results_df['cohen_kappa'].std(),
        'mean_f1_macro': results_df['f1_macro'].mean(),
        'std_f1_macro': results_df['f1_macro'].std(),
        'fold_results': fold_results,
        'confusion_matrices': confusion_matrices
    }

    logger.info(
        f"Cross-validation complete. "
        f"Accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}, "
        f"Kappa: {summary['mean_kappa']:.3f} ± {summary['std_kappa']:.3f}"
    )

    return summary


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    stage_labels: List[str],
    title: str = "Sleep Stage Confusion Matrix",
    output_path: Optional[Path] = None,
    normalize: bool = True
) -> Any:
    """
    Plot a confusion matrix for sleep stage predictions.

    Args:
        confusion_matrix: Confusion matrix (n_stages x n_stages)
        stage_labels: List of stage labels in order
        title: Plot title
        output_path: Optional path to save the plot
        normalize: Whether to normalize by true label counts

    Returns:
        Matplotlib figure object

    Raises:
        ImportError: If matplotlib is not installed
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )

    # Normalize if requested
    if normalize:
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = confusion_matrix
        fmt = 'd'

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=stage_labels,
        yticklabels=stage_labels,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Stage')
    ax.set_ylabel('True Stage')
    ax.set_title(title)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")

    return fig


def plot_feature_importance(
    feature_importance: pd.Series,
    title: str = "Feature Importance",
    top_n: int = 20,
    output_path: Optional[Path] = None
) -> Any:
    """
    Plot feature importance scores.

    Args:
        feature_importance: Series mapping feature names to importance scores
        title: Plot title
        top_n: Number of top features to display
        output_path: Optional path to save the plot

    Returns:
        Matplotlib figure object

    Raises:
        ImportError: If matplotlib is not installed
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    # Get top N features
    top_features = feature_importance.nlargest(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    # Plot horizontal bar chart
    top_features.plot(kind='barh', ax=ax, color='steelblue')

    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    ax.invert_yaxis()  # Highest importance at top

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")

    return fig


def plot_hypnogram(
    predicted_stages: pd.Series,
    true_stages: Optional[pd.Series] = None,
    title: str = "Sleep Hypnogram",
    output_path: Optional[Path] = None
) -> Any:
    """
    Plot a sleep hypnogram showing sleep stages over time.

    Args:
        predicted_stages: Series of predicted sleep stages with DatetimeIndex
        true_stages: Optional series of true sleep stages for comparison
        title: Plot title
        output_path: Optional path to save the plot

    Returns:
        Matplotlib figure object

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If predicted_stages doesn't have a DatetimeIndex
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    if not isinstance(predicted_stages.index, pd.DatetimeIndex):
        raise ValueError("predicted_stages must have a DatetimeIndex")

    # Define stage order and numeric mapping
    stage_order = ['deep', 'light', 'rem', 'wake']
    stage_to_num = {stage: i for i, stage in enumerate(stage_order)}

    # Convert stages to numeric
    pred_numeric = predicted_stages.map(stage_to_num)

    # Create figure
    if true_stages is not None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        true_numeric = true_stages.map(stage_to_num)

        # Plot true stages
        axes[0].plot(true_stages.index, true_numeric, linewidth=1, color='green')
        axes[0].set_ylabel('Sleep Stage')
        axes[0].set_yticks(range(len(stage_order)))
        axes[0].set_yticklabels(stage_order)
        axes[0].set_title(f"{title} - True Stages")
        axes[0].grid(True, alpha=0.3)

        # Plot predicted stages
        axes[1].plot(predicted_stages.index, pred_numeric, linewidth=1, color='blue')
        axes[1].set_ylabel('Sleep Stage')
        axes[1].set_xlabel('Time')
        axes[1].set_yticks(range(len(stage_order)))
        axes[1].set_yticklabels(stage_order)
        axes[1].set_title(f"{title} - Predicted Stages")
        axes[1].grid(True, alpha=0.3)

    else:
        fig, ax = plt.subplots(figsize=(14, 5))

        # Plot predicted stages
        ax.plot(predicted_stages.index, pred_numeric, linewidth=1, color='blue')
        ax.set_ylabel('Sleep Stage')
        ax.set_xlabel('Time')
        ax.set_yticks(range(len(stage_order)))
        ax.set_yticklabels(stage_order)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hypnogram to {output_path}")

    return fig


def compute_sleep_statistics(
    predicted_stages: pd.Series,
    epoch_duration_seconds: float = 30.0
) -> Dict[str, Any]:
    """
    Compute sleep statistics from predicted sleep stages.

    Args:
        predicted_stages: Series of predicted sleep stages
        epoch_duration_seconds: Duration of each epoch in seconds

    Returns:
        Dictionary containing:
            - 'total_sleep_time_min': Total sleep time in minutes
            - 'total_wake_time_min': Total wake time in minutes
            - 'sleep_efficiency': Sleep time / total time (0-1)
            - 'wake_pct': Percentage of time awake
            - 'light_pct': Percentage in light sleep
            - 'deep_pct': Percentage in deep sleep
            - 'rem_pct': Percentage in REM sleep
            - 'sleep_onset_latency_min': Time to first sleep epoch
            - 'waso_min': Wake after sleep onset (minutes)
            - 'n_awakenings': Number of wake periods after sleep onset
    """
    if predicted_stages.empty:
        return {}

    epoch_duration_min = epoch_duration_seconds / 60.0
    total_epochs = len(predicted_stages)

    # Count stages
    stage_counts = predicted_stages.value_counts()

    wake_epochs = stage_counts.get('wake', 0)
    sleep_epochs = total_epochs - wake_epochs

    # Time statistics
    total_time_min = total_epochs * epoch_duration_min
    total_sleep_time_min = sleep_epochs * epoch_duration_min
    total_wake_time_min = wake_epochs * epoch_duration_min

    sleep_efficiency = sleep_epochs / total_epochs if total_epochs > 0 else 0

    # Stage percentages
    wake_pct = (wake_epochs / total_epochs * 100) if total_epochs > 0 else 0
    light_pct = (stage_counts.get('light', 0) / total_epochs * 100) if total_epochs > 0 else 0
    deep_pct = (stage_counts.get('deep', 0) / total_epochs * 100) if total_epochs > 0 else 0
    rem_pct = (stage_counts.get('rem', 0) / total_epochs * 100) if total_epochs > 0 else 0

    # Sleep onset latency (time to first sleep epoch)
    first_sleep_idx = predicted_stages[predicted_stages != 'wake'].index.min() if any(predicted_stages != 'wake') else None
    if first_sleep_idx is not None:
        first_epoch_idx = predicted_stages.index.get_loc(first_sleep_idx)
        sleep_onset_latency_min = first_epoch_idx * epoch_duration_min
    else:
        sleep_onset_latency_min = None

    # Wake after sleep onset (WASO)
    if first_sleep_idx is not None:
        stages_after_onset = predicted_stages.loc[first_sleep_idx:]
        waso_epochs = (stages_after_onset == 'wake').sum()
        waso_min = waso_epochs * epoch_duration_min

        # Count awakenings (consecutive wake periods)
        is_wake = (stages_after_onset == 'wake').astype(int)
        wake_transitions = is_wake.diff()
        n_awakenings = (wake_transitions == 1).sum()
    else:
        waso_min = None
        n_awakenings = None

    return {
        'total_time_min': total_time_min,
        'total_sleep_time_min': total_sleep_time_min,
        'total_wake_time_min': total_wake_time_min,
        'sleep_efficiency': sleep_efficiency,
        'wake_pct': wake_pct,
        'light_pct': light_pct,
        'deep_pct': deep_pct,
        'rem_pct': rem_pct,
        'sleep_onset_latency_min': sleep_onset_latency_min,
        'waso_min': waso_min,
        'n_awakenings': n_awakenings
    }
