"""
Unit tests for sleep staging algorithms.

Tests the Random Forest sleep staging algorithm, base classes,
and evaluation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import algorithm classes
from src.sleep_analysis.algorithms.base import (
    SleepStagingAlgorithm,
    compute_classification_metrics
)
from src.sleep_analysis.algorithms.random_forest import RandomForestSleepStaging
from src.sleep_analysis.algorithms.evaluation import (
    compare_algorithms,
    cross_validate_algorithm,
    compute_sleep_statistics
)

# Skip all tests if scikit-learn is not installed
pytest.importorskip("sklearn", reason="scikit-learn required for algorithm tests")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_sleep_data_4stage():
    """
    Create synthetic sleep data with 4 stages for testing.

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    np.random.seed(42)
    n_epochs = 200

    # Create timestamp index (30-second epochs)
    timestamps = pd.date_range('2024-01-01 22:00:00', periods=n_epochs, freq='30s', tz='UTC')

    # Create features with patterns that correlate with sleep stages
    features = pd.DataFrame(index=timestamps)

    # HRV features (lower HR during deep sleep, higher during wake/REM)
    features['hr_mean'] = np.random.normal(65, 10, n_epochs)
    features['hr_std'] = np.random.normal(5, 2, n_epochs)
    features['hr_cv'] = features['hr_std'] / features['hr_mean']
    features['rmssd'] = np.random.normal(50, 15, n_epochs)

    # Movement features (higher during wake, lower during deep sleep)
    features['magnitude_mean'] = np.random.exponential(2, n_epochs)
    features['magnitude_std'] = np.random.exponential(1, n_epochs)
    features['stillness_ratio'] = np.random.uniform(0, 100, n_epochs)

    # Correlation features
    features['pearson_corr'] = np.random.uniform(-0.5, 0.5, n_epochs)

    # Create realistic sleep stage labels
    # Typical sleep cycle: wake -> light -> deep -> light -> REM
    labels = []

    # Initial wake period (first 10 epochs = 5 minutes)
    labels.extend(['wake'] * 10)

    # Sleep cycles (each cycle ~90 minutes = 180 epochs)
    remaining = n_epochs - 10
    cycle_length = 60  # epochs per cycle

    while len(labels) < n_epochs:
        if len(labels) + cycle_length > n_epochs:
            cycle_length = n_epochs - len(labels)

        # Light sleep (40% of cycle)
        labels.extend(['light'] * int(cycle_length * 0.4))

        # Deep sleep (20% of cycle)
        labels.extend(['deep'] * int(cycle_length * 0.2))

        # Light sleep again (20% of cycle)
        labels.extend(['light'] * int(cycle_length * 0.2))

        # REM sleep (20% of cycle)
        labels.extend(['rem'] * int(cycle_length * 0.2))

    labels = pd.Series(labels[:n_epochs], index=timestamps, name='sleep_stage')

    # Adjust features to match labels (make patterns more realistic)
    for i, stage in enumerate(labels):
        if stage == 'wake':
            features.loc[timestamps[i], 'hr_mean'] += 10
            features.loc[timestamps[i], 'magnitude_mean'] += 3
        elif stage == 'deep':
            features.loc[timestamps[i], 'hr_mean'] -= 10
            features.loc[timestamps[i], 'magnitude_mean'] -= 1
            features.loc[timestamps[i], 'stillness_ratio'] += 20
        elif stage == 'rem':
            features.loc[timestamps[i], 'hr_mean'] += 5
            features.loc[timestamps[i], 'hr_std'] += 2

    return features, labels


@pytest.fixture
def synthetic_sleep_data_2stage():
    """
    Create synthetic sleep data with 2 stages (wake/sleep) for testing.

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    features, labels_4stage = synthetic_sleep_data_4stage()

    # Convert 4-stage to 2-stage
    labels_2stage = labels_4stage.replace({
        'light': 'sleep',
        'deep': 'sleep',
        'rem': 'sleep'
    })

    return features, labels_2stage


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model saving/loading tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Test Base Classes
# ============================================================================

class TestSleepStagingAlgorithm:
    """Test the abstract base class for sleep staging algorithms."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that SleepStagingAlgorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SleepStagingAlgorithm()

    def test_compute_classification_metrics(self, synthetic_sleep_data_4stage):
        """Test the classification metrics computation utility."""
        features, labels = synthetic_sleep_data_4stage

        # Create simple predictions (just use labels with some noise)
        predictions = labels.copy()
        # Flip 10% of predictions to create errors
        flip_indices = np.random.choice(len(predictions), size=int(len(predictions) * 0.1), replace=False)
        stages = ['wake', 'light', 'deep', 'rem']
        for idx in flip_indices:
            current = predictions.iloc[idx]
            predictions.iloc[idx] = np.random.choice([s for s in stages if s != current])

        # Compute metrics
        metrics = compute_classification_metrics(labels, predictions)

        # Check that all expected metrics are present
        assert 'accuracy' in metrics
        assert 'cohen_kappa' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics

        # Check value ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert -1 <= metrics['cohen_kappa'] <= 1  # Kappa can be negative
        assert 0 <= metrics['f1_macro'] <= 1


# ============================================================================
# Test Random Forest Sleep Staging
# ============================================================================

class TestRandomForestSleepStaging:
    """Test the Random Forest sleep staging algorithm."""

    def test_initialization_4stage(self):
        """Test initialization with 4-stage classification."""
        algo = RandomForestSleepStaging(n_estimators=50, n_stages=4)

        assert algo.name == "RandomForestSleepStaging"
        assert algo.n_estimators == 50
        assert algo.n_stages == 4
        assert algo.stage_labels == ['wake', 'light', 'deep', 'rem']
        assert not algo.is_fitted

    def test_initialization_2stage(self):
        """Test initialization with 2-stage classification."""
        algo = RandomForestSleepStaging(n_stages=2)

        assert algo.n_stages == 2
        assert algo.stage_labels == ['wake', 'sleep']

    def test_initialization_invalid_stages(self):
        """Test that invalid n_stages raises error."""
        with pytest.raises(ValueError, match="n_stages must be 2 or 4"):
            RandomForestSleepStaging(n_stages=3)

    def test_fit_4stage(self, synthetic_sleep_data_4stage):
        """Test training on 4-stage data."""
        features, labels = synthetic_sleep_data_4stage

        algo = RandomForestSleepStaging(n_estimators=50, n_stages=4, random_state=42)
        results = algo.fit(features, labels)

        # Check that algorithm is now fitted
        assert algo.is_fitted

        # Check training results
        assert 'train_accuracy' in results
        assert 'train_kappa' in results
        assert 'n_epochs' in results
        assert 'n_features' in results

        # Check that accuracy is reasonable (should be high on training data)
        assert results['train_accuracy'] > 0.5  # At least better than random

        # Check feature names are stored
        assert algo.feature_names == list(features.columns)

    def test_fit_with_validation_split(self, synthetic_sleep_data_4stage):
        """Test training with validation split."""
        features, labels = synthetic_sleep_data_4stage

        algo = RandomForestSleepStaging(n_estimators=50, random_state=42)
        results = algo.fit(features, labels, validation_split=0.2)

        # Check validation results are present
        assert 'val_accuracy' in results
        assert 'val_kappa' in results
        assert 'n_val_epochs' in results

        # Validation accuracy should be positive
        assert results['val_accuracy'] > 0

    def test_fit_2stage(self, synthetic_sleep_data_4stage):
        """Test training on 2-stage data."""
        features, labels_4stage = synthetic_sleep_data_4stage

        # Convert to 2-stage labels
        labels = labels_4stage.replace({
            'light': 'sleep',
            'deep': 'sleep',
            'rem': 'sleep'
        })

        algo = RandomForestSleepStaging(n_estimators=50, n_stages=2, random_state=42)
        results = algo.fit(features, labels)

        assert algo.is_fitted
        assert results['train_accuracy'] > 0.5

    def test_fit_invalid_labels(self, synthetic_sleep_data_4stage):
        """Test that invalid labels are handled."""
        features, labels = synthetic_sleep_data_4stage

        # Add some invalid labels
        labels = labels.copy()
        labels.iloc[0] = 'invalid_stage'

        algo = RandomForestSleepStaging(n_stages=4)

        # Should log warning but continue (filters invalid labels)
        results = algo.fit(features, labels)
        assert algo.is_fitted

    def test_predict_before_fit_raises_error(self, synthetic_sleep_data_4stage):
        """Test that predicting before fitting raises error."""
        features, labels = synthetic_sleep_data_4stage

        algo = RandomForestSleepStaging()

        with pytest.raises(RuntimeError, match="must be fitted before prediction"):
            algo.predict(features)

    def test_predict_after_fit(self, synthetic_sleep_data_4stage):
        """Test prediction after fitting."""
        features, labels = synthetic_sleep_data_4stage

        # Split into train/test
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train = labels.iloc[:split_idx]

        # Train
        algo = RandomForestSleepStaging(n_estimators=50, random_state=42)
        algo.fit(X_train, y_train)

        # Predict
        predictions = algo.predict(X_test)

        # Check predictions
        assert len(predictions) == len(X_test)
        assert predictions.index.equals(X_test.index)
        assert all(stage in ['wake', 'light', 'deep', 'rem'] for stage in predictions)

    def test_predict_proba(self, synthetic_sleep_data_4stage):
        """Test probability prediction."""
        features, labels = synthetic_sleep_data_4stage

        # Train
        algo = RandomForestSleepStaging(n_estimators=50, random_state=42)
        algo.fit(features, labels)

        # Predict probabilities
        probabilities = algo.predict_proba(features)

        # Check probabilities
        assert len(probabilities) == len(features)
        # Check that all expected stages are in columns (order may vary)
        assert set(probabilities.columns) == {'wake', 'light', 'deep', 'rem'}

        # Each row should sum to 1
        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

        # All values should be between 0 and 1
        assert (probabilities >= 0).all().all()
        assert (probabilities <= 1).all().all()

    def test_evaluate(self, synthetic_sleep_data_4stage):
        """Test model evaluation."""
        features, labels = synthetic_sleep_data_4stage

        # Train
        algo = RandomForestSleepStaging(n_estimators=50, random_state=42)
        algo.fit(features, labels)

        # Evaluate
        metrics = algo.evaluate(features, labels)

        # Check metrics
        assert 'accuracy' in metrics
        assert 'cohen_kappa' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics

        # Training set accuracy should be high
        assert metrics['accuracy'] > 0.8

    def test_get_feature_importance(self, synthetic_sleep_data_4stage):
        """Test feature importance extraction."""
        features, labels = synthetic_sleep_data_4stage

        # Train
        algo = RandomForestSleepStaging(n_estimators=50, random_state=42)
        algo.fit(features, labels)

        # Get feature importance
        importance = algo.get_feature_importance()

        # Check importance
        assert importance is not None
        assert len(importance) == len(features.columns)
        assert all(imp >= 0 for imp in importance)

        # Sum of importance should be 1
        assert np.isclose(importance.sum(), 1.0)

    def test_save_and_load(self, synthetic_sleep_data_4stage, temp_model_dir):
        """Test model saving and loading."""
        features, labels = synthetic_sleep_data_4stage

        # Train model
        algo = RandomForestSleepStaging(n_estimators=50, n_stages=4, random_state=42)
        algo.fit(features, labels)

        # Get predictions before saving
        predictions_before = algo.predict(features)

        # Save model
        save_path = temp_model_dir / "test_model"
        algo.save(save_path)

        # Check that files were created
        assert save_path.exists()
        assert (save_path / "random_forest_model.pkl").exists()
        assert (save_path / "metadata.pkl").exists()

        # Load model
        algo_loaded = RandomForestSleepStaging()
        algo_loaded.load(save_path)

        # Check loaded model
        assert algo_loaded.is_fitted
        assert algo_loaded.n_stages == 4
        assert algo_loaded.feature_names == list(features.columns)

        # Get predictions after loading
        predictions_after = algo_loaded.predict(features)

        # Predictions should be identical
        assert predictions_before.equals(predictions_after)

    def test_save_before_fit_raises_error(self, temp_model_dir):
        """Test that saving before fitting raises error."""
        algo = RandomForestSleepStaging()

        with pytest.raises(RuntimeError, match="must be fitted before saving"):
            algo.save(temp_model_dir / "test")

    def test_load_nonexistent_model_raises_error(self, temp_model_dir):
        """Test that loading non-existent model raises error."""
        algo = RandomForestSleepStaging()

        with pytest.raises(FileNotFoundError):
            algo.load(temp_model_dir / "nonexistent")


# ============================================================================
# Test Evaluation Utilities
# ============================================================================

class TestEvaluationUtilities:
    """Test evaluation utility functions."""

    def test_compare_algorithms(self, synthetic_sleep_data_4stage):
        """Test algorithm comparison."""
        features, labels = synthetic_sleep_data_4stage

        # Create and train two algorithms with different parameters
        algo1 = RandomForestSleepStaging(n_estimators=50, random_state=42)
        algo1.fit(features, labels)

        algo2 = RandomForestSleepStaging(n_estimators=100, max_depth=10, random_state=42)
        algo2.fit(features, labels)

        # Compare algorithms
        comparison = compare_algorithms([algo1, algo2], features, labels)

        # Check comparison results
        assert len(comparison) == 2
        assert 'accuracy' in comparison.columns
        assert 'cohen_kappa' in comparison.columns

        # Both algorithms should have positive accuracy
        assert all(comparison['accuracy'] > 0)

    def test_cross_validate_algorithm(self, synthetic_sleep_data_4stage):
        """Test cross-validation."""
        features, labels = synthetic_sleep_data_4stage

        # Run cross-validation
        cv_results = cross_validate_algorithm(
            RandomForestSleepStaging,
            {'n_estimators': 30, 'n_stages': 4, 'random_state': 42},
            features,
            labels,
            n_folds=3,
            random_state=42
        )

        # Check results
        assert 'mean_accuracy' in cv_results
        assert 'std_accuracy' in cv_results
        assert 'mean_kappa' in cv_results
        assert 'fold_results' in cv_results

        assert len(cv_results['fold_results']) == 3

        # Mean accuracy should be positive
        assert cv_results['mean_accuracy'] > 0

    def test_compute_sleep_statistics(self):
        """Test sleep statistics computation."""
        # Create simple sleep stage series
        timestamps = pd.date_range('2024-01-01 22:00:00', periods=120, freq='30s')

        # 10 epochs wake, 100 epochs sleep (mixed), 10 epochs wake
        stages = ['wake'] * 10 + ['light'] * 40 + ['deep'] * 30 + ['rem'] * 30 + ['wake'] * 10
        predicted_stages = pd.Series(stages, index=timestamps)

        # Compute statistics
        stats = compute_sleep_statistics(predicted_stages, epoch_duration_seconds=30.0)

        # Check statistics
        assert 'total_sleep_time_min' in stats
        assert 'sleep_efficiency' in stats
        assert 'wake_pct' in stats
        assert 'light_pct' in stats
        assert 'deep_pct' in stats
        assert 'rem_pct' in stats

        # Check values
        assert stats['wake_pct'] == pytest.approx((20 / 120) * 100, rel=0.01)
        assert stats['sleep_efficiency'] == pytest.approx(100 / 120, rel=0.01)

        # Total time should be 120 epochs * 0.5 min/epoch = 60 minutes
        assert stats['total_time_min'] == 60.0

    def test_compute_sleep_statistics_empty_series(self):
        """Test sleep statistics with empty series."""
        predicted_stages = pd.Series([], dtype=str)

        stats = compute_sleep_statistics(predicted_stages)

        # Should return empty dict
        assert stats == {}


# ============================================================================
# Test Algorithm Operations (Workflow Integration)
# ============================================================================

class TestAlgorithmOperations:
    """Test the workflow operation wrappers for algorithms."""

    def test_random_forest_operation_import(self):
        """Test that algorithm operations can be imported."""
        from src.sleep_analysis.operations.algorithm_ops import (
            random_forest_sleep_staging,
            evaluate_sleep_staging
        )

        # Check functions exist
        assert callable(random_forest_sleep_staging)
        assert callable(evaluate_sleep_staging)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
