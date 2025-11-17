# Sleep Staging Algorithms

This module provides sleep staging algorithms for classifying sleep stages from physiological features extracted from wearable sensors.

## Overview

The algorithms module implements machine learning-based sleep staging classifiers that can predict sleep stages (wake, light, deep, REM) from features like:
- Heart Rate Variability (HRV)
- Movement/Activity patterns
- Correlation features between signals

## Architecture

The module follows a modular design with three main components:

### 1. Base Classes (`base.py`)

- **`SleepStagingAlgorithm`**: Abstract base class defining the interface all algorithms must implement
- **`compute_classification_metrics()`**: Utility function for computing evaluation metrics

### 2. Algorithms (`random_forest.py`)

- **`RandomForestSleepStaging`**: Random Forest-based classifier following the methodology from:
  - Paper: "Sleep stage classification using heart rate variability and accelerometer data"
  - Nature Scientific Reports (2020)
  - URL: https://www.nature.com/articles/s41598-020-79217-x

### 3. Evaluation Utilities (`evaluation.py`)

Functions for algorithm comparison and analysis:
- `compare_algorithms()`: Compare multiple algorithms on the same dataset
- `cross_validate_algorithm()`: Perform k-fold cross-validation
- `plot_confusion_matrix()`: Visualize classification performance
- `plot_feature_importance()`: Visualize feature importance scores
- `plot_hypnogram()`: Plot sleep stage predictions over time
- `compute_sleep_statistics()`: Extract sleep metrics (TST, efficiency, WASO, etc.)

## Quick Start

### Training a Model

```python
from sleep_analysis.algorithms import RandomForestSleepStaging
import pandas as pd

# Load your feature matrix and labels
features = pd.read_csv('features.csv', index_col=0, parse_dates=True)
labels = pd.read_csv('labels.csv', index_col=0, parse_dates=True)['sleep_stage']

# Create and train the algorithm
algo = RandomForestSleepStaging(
    n_estimators=100,
    n_stages=4,  # 4-stage: wake, light, deep, REM
    random_state=42
)

# Train with validation split
results = algo.fit(
    features,
    labels,
    validation_split=0.2,
    normalize_features=True
)

print(f"Training Accuracy: {results['train_accuracy']:.3f}")
print(f"Validation Accuracy: {results['val_accuracy']:.3f}")

# Save the trained model
algo.save('models/my_sleep_model')
```

### Making Predictions

```python
# Load a pre-trained model
algo = RandomForestSleepStaging()
algo.load('models/my_sleep_model')

# Load new data
new_features = pd.read_csv('new_features.csv', index_col=0, parse_dates=True)

# Predict sleep stages
predictions = algo.predict(new_features)
probabilities = algo.predict_proba(new_features)

print(predictions.value_counts())
```

### Evaluation

```python
from sleep_analysis.algorithms import (
    plot_confusion_matrix,
    plot_hypnogram,
    compute_sleep_statistics
)

# Evaluate on test set
metrics = algo.evaluate(test_features, test_labels)

print(f"Test Accuracy: {metrics['accuracy']:.3f}")
print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")

# Plot confusion matrix
plot_confusion_matrix(
    metrics['confusion_matrix'],
    stage_labels=['wake', 'light', 'deep', 'rem'],
    output_path='confusion_matrix.png'
)

# Plot hypnogram
plot_hypnogram(
    predictions,
    test_labels,
    output_path='hypnogram.png'
)

# Compute sleep statistics
stats = compute_sleep_statistics(predictions, epoch_duration_seconds=30)
print(f"Total Sleep Time: {stats['total_sleep_time_min']:.1f} minutes")
print(f"Sleep Efficiency: {stats['sleep_efficiency']:.2%}")
```

### Cross-Validation

```python
from sleep_analysis.algorithms import cross_validate_algorithm

# Perform 5-fold cross-validation
cv_results = cross_validate_algorithm(
    RandomForestSleepStaging,
    algorithm_params={'n_estimators': 100, 'n_stages': 4},
    features=features,
    labels=labels,
    n_folds=5,
    stratified=True
)

print(f"Mean Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
print(f"Mean Kappa: {cv_results['mean_kappa']:.3f} ± {cv_results['std_kappa']:.3f}")
```

## Workflow Integration

The algorithms can be used directly in YAML workflows:

### Training Workflow

```yaml
steps:
  # ... feature extraction steps ...

  # Train Random Forest model
  - type: multi_signal
    operation: "random_forest_sleep_staging"
    inputs: ["combined_features"]
    parameters:
      mode: "train_predict"
      labels_column: "sleep_stage"
      n_estimators: 100
      n_stages: 4
      validation_split: 0.2
      normalize_features: true
      save_model_path: "models/trained_model"
    output: "sleep_predictions"
```

### Prediction Workflow

```yaml
steps:
  # ... feature extraction steps ...

  # Load model and predict
  - type: multi_signal
    operation: "random_forest_sleep_staging"
    inputs: ["combined_features"]
    parameters:
      mode: "predict"
      model_path: "models/trained_model"
      n_stages: 4
    output: "sleep_predictions"

  # Evaluate (optional, if labels available)
  - type: multi_signal
    operation: "evaluate_sleep_staging"
    inputs: ["sleep_predictions"]
    parameters:
      predictions_column: "predicted_stage"
      labels_column: "sleep_stage"
      confusion_matrix_path: "results/confusion_matrix.png"
    output: "evaluation_metrics"
```

## Algorithm Parameters

### RandomForestSleepStaging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of trees in the forest |
| `max_depth` | int | None | Maximum tree depth (None = unlimited) |
| `min_samples_split` | int | 2 | Minimum samples to split a node |
| `min_samples_leaf` | int | 1 | Minimum samples at leaf |
| `class_weight` | str | 'balanced' | Class weighting strategy |
| `random_state` | int | 42 | Random seed for reproducibility |
| `n_stages` | int | 4 | Number of sleep stages (2 or 4) |

**4-Stage Classification**: wake, light, deep, REM
**2-Stage Classification**: wake, sleep

## Feature Requirements

The Random Forest algorithm expects features from the following operations:

### Required Features
- **HRV features**: `compute_hrv_features`
  - `hr_mean`, `hr_std`, `hr_cv`, `rmssd`, etc.
- **Movement features**: `compute_movement_features`
  - `magnitude_mean`, `stillness_ratio`, `activity_count`, etc.

### Recommended Features
- **Correlation features**: `compute_correlation_features`
  - `pearson_corr` between HR and movement

### Example Feature Matrix

```python
# Expected DataFrame structure
features = pd.DataFrame({
    'hr_mean': [...],
    'hr_std': [...],
    'hr_cv': [...],
    'rmssd': [...],
    'magnitude_mean': [...],
    'magnitude_std': [...],
    'stillness_ratio': [...],
    'pearson_corr': [...]
}, index=pd.DatetimeIndex([...]))  # 30-second epochs
```

## Performance Tips

1. **Use 30-second epochs**: Standard for sleep staging, matches the Nature paper
2. **Normalize features**: Always use `normalize_features=True` for better performance
3. **Class weighting**: Use `class_weight='balanced'` to handle imbalanced datasets
4. **Increase trees for production**: Use `n_estimators=200-500` for final models
5. **Limit depth to prevent overfitting**: Try `max_depth=15-20` if overfitting occurs
6. **Use validation split**: Always evaluate on held-out data

## Evaluation Metrics

The algorithms compute standard classification metrics:

- **Accuracy**: Overall correct predictions
- **Cohen's Kappa**: Agreement accounting for chance (more robust for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class performance breakdown
- **Per-Class Metrics**: Precision, recall, F1 for each sleep stage

## Sleep Statistics

The `compute_sleep_statistics()` function extracts clinical sleep metrics:

| Metric | Description |
|--------|-------------|
| `total_sleep_time_min` | Total time spent asleep (all non-wake stages) |
| `sleep_efficiency` | Sleep time / total time (0-1) |
| `sleep_onset_latency_min` | Time to first sleep epoch |
| `waso_min` | Wake After Sleep Onset (WASO) in minutes |
| `n_awakenings` | Number of wake periods after sleep onset |
| `wake_pct`, `light_pct`, `deep_pct`, `rem_pct` | Percentage in each stage |

## Dependencies

Required packages:
- `scikit-learn >= 1.3.0`: Random Forest implementation
- `pandas >= 1.3.0`: Data structures
- `numpy >= 1.19.0`: Numerical operations
- `joblib >= 1.0.0`: Model serialization

Optional packages:
- `scipy >= 1.11.0`: Advanced correlation methods
- `matplotlib >= 3.5.0`: Plotting utilities
- `seaborn >= 0.11.0`: Enhanced visualizations

Install all dependencies:
```bash
pip install scikit-learn pandas numpy joblib scipy matplotlib seaborn
```

Or install the package with optional dependencies:
```bash
pip install sleep-analysis[algorithms]
```

## Examples

See the `workflows/` directory for complete examples:
- `sleep_staging_with_rf.yaml`: End-to-end sleep staging workflow
- `train_sleep_staging_model.yaml`: Model training from labeled data

## References

1. **Nature Paper**: Fonseca et al. (2020). "Sleep stage classification using heart rate variability and accelerometer data." Scientific Reports, 10(1), 1-11.
   - URL: https://www.nature.com/articles/s41598-020-79217-x
   - Key insights: 30-second epochs, HRV + movement features, Random Forest classifier

2. **AASM Sleep Staging Manual**: Berry et al. (2012). "The AASM Manual for the Scoring of Sleep and Associated Events."
   - Defines standard sleep stage classifications

## Contributing

When adding new algorithms:

1. Inherit from `SleepStagingAlgorithm`
2. Implement all abstract methods (`fit`, `predict`, `predict_proba`, `evaluate`)
3. Add comprehensive tests in `tests/unit/test_algorithms.py`
4. Create a workflow operation wrapper in `operations/algorithm_ops.py`
5. Register the operation in `core/signal_collection.py`
6. Update this README with usage examples

## License

See the main project LICENSE file.
