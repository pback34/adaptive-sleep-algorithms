# Sleep Analysis Framework - Development TODO

## 🎯 HANDOFF NOTES - START HERE

### Current Status
We have completed **most of Phase 2 (Data Processing Infrastructure)** development. The framework now has:
- ✅ Core data structures and signal processing
- ✅ Feature extraction with epoch-based windowing
- ✅ Advanced visualization (Bokeh & Plotly backends)
- ✅ Flexible export system with combined features
- ✅ Comprehensive testing infrastructure
- ✅ Metadata handling (TimeSeriesMetadata & FeatureMetadata)
- ✅ Documentation and diagrams

### 🚀 Next Priority: Transition to Phase 3
**Current Focus**: Feature Extraction (Phase 3)

While the infrastructure for feature extraction exists, we need to expand the library of features available for sleep analysis algorithms. This prepares us for Phase 4 (Algorithm Exploration).

### Recommended Next Tasks (in order)
1. **Expand Feature Library** → See [Phase 3 - Feature Extraction](#phase-3-feature-extraction)
   - Add frequency-domain features (spectral power bands)
   - Implement HRV features (SDNN, RMSSD, pNN50, etc.)
   - Add derived accelerometer features (activity counts, magnitude metrics)

2. **Test with Real Data** → See [Phase 2 - Testing & Validation](#testing--validation)
   - Run polar_workflow with actual sensor data
   - Validate feature extraction accuracy
   - Test visualization outputs

3. **Performance Optimization** → See [Phase 2 - Performance](#performance-optimization)
   - Profile feature extraction on large datasets
   - Implement caching for repeated operations
   - Optimize memory usage for long recordings

### Reference Documentation
- **Project Overview**: `product-development/Project-Overview.md`
- **Algorithm Development Project**: `product-development/Development/AlgorithmDevelopment/Project-P02-AlgorithmDevelopment.md`
- **Feature Extraction Plan**: `docs/feature_extraction_plan.md`
- **Coding Guidelines**: `docs/coding_guidelines.md`

---

## 📋 DETAILED TASK BREAKDOWN

## Phase 2: Data Processing Infrastructure ✅ (MOSTLY COMPLETE)

### Core Infrastructure ✅ (DONE)
- [x] SignalData base class with type safety
- [x] TimeSeriesSignal implementation
- [x] SignalCollection with metadata handling
- [x] MetadataHandler for provenance tracking
- [x] Operation registry system
- [x] Workflow executor

### Metadata Refactoring ✅ (DONE)
- [x] Split SignalMetadata into TimeSeriesMetadata and FeatureMetadata
- [x] Create Feature class (distinct from TimeSeriesSignal)
- [x] Separate storage: time_series_signals and features dicts
- [x] Implement FeatureType enum (STATISTICAL, SPECTRAL, HRV, etc.)
- [x] Add epoch grid support (generate_epoch_grid operation)
- [x] Feature combination with MultiIndex columns

### Import System ✅ (DONE)
- [x] Base importer interface
- [x] CSV importer base class
- [x] Polar sensor importer
- [x] MergingImporter for fragmented files
- [x] Timezone handling (standardize_timestamp utility)
- [x] Metadata extraction from filenames

### Signal Processing ✅ (DONE)
- [x] Filtering operations (low-pass, high-pass, band-pass)
- [x] Resampling and alignment
- [x] Artifact removal utilities
- [x] Derived signal generation
- [x] Grid-based signal alignment (generate_alignment_grid, apply_grid_alignment)
- [x] Signal combination (combine_aligned_signals, align_and_combine_signals)

### Feature Extraction ✅ (BASIC DONE)
- [x] Epoch-based windowing system
- [x] Global epoch grid (epoch_grid_index)
- [x] Statistical features (mean, std, min, max)
- [x] Sleep stage mode computation (compute_sleep_stage_mode)
- [x] Feature metadata with traceability
- [x] Multi-signal feature operations
- [x] Feature combination (combine_features)

### Export System ✅ (DONE)
- [x] CSV export with metadata
- [x] Excel export with multiple sheets
- [x] Pickle export for Python objects
- [x] HDF5 export for large datasets
- [x] Flexible content selection ('all_ts', 'all_features', 'combined_ts', 'combined_features', 'summary')
- [x] MultiIndex column support
- [x] Timezone formatting for different formats

### Visualization ✅ (DONE)
- [x] Backend abstraction (BokehVisualizer, PlotlyVisualizer)
- [x] Time series plots with downsampling
- [x] Hypnogram plots for sleep stages
- [x] Sleep stage overlay on time series
- [x] Scatter plots
- [x] Multi-panel layouts (vertical, horizontal, grid)
- [x] Linked axes for synchronized viewing
- [x] Categorical data handling (sleep stages)

### Testing ✅ (DONE)
- [x] Unit tests for core classes
- [x] Tests for metadata handling
- [x] Tests for signal operations
- [x] Tests for feature extraction
- [x] Tests for signal collection
- [x] Integration tests for workflows

### Documentation ✅ (DONE)
- [x] Comprehensive README with examples
- [x] Class diagrams (Mermaid)
- [x] Data flow diagrams
- [x] Module dependency diagrams
- [x] Coding guidelines
- [x] Feature extraction plan
- [x] Workflow examples

### Testing & Validation ⚠️ (NEEDS ATTENTION)
- [ ] **Test with real Polar sensor data**
- [ ] Validate feature extraction accuracy
- [ ] Compare outputs with reference implementations
- [ ] Test edge cases (missing data, gaps, timezone issues)
- [ ] Validate visualization outputs
- [ ] Performance benchmarking with large datasets

### Performance Optimization ⚠️ (NEEDS ATTENTION)
- [ ] **Profile feature extraction pipeline**
- [ ] Implement feature caching decorator
- [ ] Optimize memory usage for long recordings
- [ ] Add lazy evaluation for features
- [ ] Implement chunk-based processing for export
- [ ] Optimize overlapping epoch computations

---

## Phase 3: Feature Extraction 🔄 (IN PROGRESS)

### Frequency-Domain Features
- [ ] **FFT-based spectral analysis**
- [ ] **Power spectral density (Welch method)**
- [ ] **Spectral bands for accelerometer** (e.g., 0.5-2 Hz for movement)
- [ ] **Spectral bands for heart rate** (LF, HF, VLF bands)
- [ ] Dominant frequency extraction
- [ ] Spectral entropy

### Heart Rate Variability (HRV) Features
- [ ] **Time-domain HRV metrics**
  - [ ] SDNN (standard deviation of NN intervals)
  - [ ] RMSSD (root mean square of successive differences)
  - [ ] pNN50 (percentage of successive NN intervals > 50ms)
  - [ ] SDSD (standard deviation of successive differences)
- [ ] **Frequency-domain HRV metrics**
  - [ ] LF power (0.04-0.15 Hz)
  - [ ] HF power (0.15-0.4 Hz)
  - [ ] LF/HF ratio
  - [ ] Total power
- [ ] **Non-linear HRV metrics**
  - [ ] Poincaré plot features (SD1, SD2)
  - [ ] Sample entropy
  - [ ] Detrended fluctuation analysis (DFA)

### Accelerometer-Derived Features
- [ ] **Activity counts** (Cole-Kripke algorithm)
- [ ] **Magnitude calculation** (sqrt(x² + y² + z²))
- [ ] **Angle/orientation features**
- [ ] **Movement variability**
- [ ] **Postural transitions** (sit-to-lie, etc.)
- [ ] Zero-crossing rate
- [ ] Signal magnitude area (SMA)

### Temperature Features
- [ ] Mean temperature per epoch
- [ ] Temperature variability
- [ ] Temperature trend (slope)
- [ ] Circadian temperature patterns

### Composite Features
- [ ] **Heart rate × activity interaction**
- [ ] **Correlation between signals** (HR vs. movement)
- [ ] **Conditional features** (e.g., HR only during low movement)
- [ ] **Ratio features** (e.g., HR/activity)

### Feature Engineering
- [ ] **Sliding window features** (overlapping epochs)
- [ ] **Multi-scale features** (different window sizes)
- [ ] **Temporal context features** (previous/next epoch)
- [ ] **Circadian alignment** (time-of-day features)

---

## Phase 4: Algorithm Exploration 🔜 (UPCOMING)

### Reference Algorithm Implementation
- [ ] Cole-Kripke actigraphy algorithm
- [ ] Sadeh actigraphy algorithm
- [ ] Simple threshold-based sleep/wake
- [ ] Rule-based sleep staging (heuristics)

### Machine Learning Algorithms
- [ ] Feature selection and importance analysis
- [ ] Random Forest classifier
- [ ] Gradient Boosting (XGBoost, LightGBM)
- [ ] Support Vector Machines (SVM)
- [ ] K-Nearest Neighbors (KNN)
- [ ] Logistic Regression baseline

### Deep Learning Approaches
- [ ] LSTM/GRU for sequence modeling
- [ ] CNN for feature learning
- [ ] Hybrid CNN-LSTM architectures
- [ ] Attention mechanisms
- [ ] Transfer learning from PSG data

### Algorithm Evaluation
- [ ] Accuracy metrics (overall, per-class)
- [ ] Confusion matrices
- [ ] Cohen's kappa
- [ ] Sensitivity/Specificity
- [ ] Epoch-by-epoch agreement
- [ ] Transition detection accuracy
- [ ] Latency measurement

### Algorithm Comparison Framework
- [ ] Standardized evaluation pipeline
- [ ] Cross-validation framework
- [ ] Hold-out test set evaluation
- [ ] Per-subject performance analysis
- [ ] Aggregate performance metrics
- [ ] Visualization of results

---

## Phase 5: Algorithm Refinement 🔜 (FUTURE)

### Optimization
- [ ] Hyperparameter tuning
- [ ] Feature subset optimization
- [ ] Model compression/quantization
- [ ] Ensemble methods
- [ ] Online learning approaches

### Real-Time Considerations
- [ ] Low-latency classification
- [ ] Incremental/streaming updates
- [ ] Change-point detection
- [ ] Short-window classification
- [ ] Edge deployment optimization

---

## Infrastructure Improvements 📝 (ONGOING)

### Error Handling
- [ ] Comprehensive error handling strategy
- [ ] Graceful degradation for missing data
- [ ] Workflow continuation on error (continue_on_error flag)
- [ ] Better error messages and logging

### Logging
- [ ] Structured logging framework
- [ ] Operation-level logging
- [ ] Performance metrics logging
- [ ] Debug mode with detailed traces

### Security & Privacy
- [ ] Data encryption options
- [ ] Secure credential handling
- [ ] Input validation (prevent injection)
- [ ] Anonymization utilities

### Usability
- [ ] Command-line interface improvements
- [ ] Interactive configuration wizard
- [ ] Pre-built workflow templates
- [ ] Example notebooks/tutorials
- [ ] Validation of workflow files

### Extensibility
- [ ] Plugin system for operations
- [ ] Custom feature registration API
- [ ] Algorithm registry system
- [ ] Importer plugin architecture

---

## Known Issues & Technical Debt 🐛

### High Priority
- [ ] **Validate polar_workflow.yaml with real data**
- [ ] Handle edge case: empty signals after filtering
- [ ] Memory optimization for very long recordings (>24 hours)
- [ ] Better error messages for workflow validation

### Medium Priority
- [ ] Registry name collision detection
- [ ] Simplify traceability implementation
- [ ] Add versioning strategy for workflows
- [ ] Thread safety for SignalCollection

### Low Priority
- [ ] Internationalization support
- [ ] Alternative backends for visualization
- [ ] Real-time streaming mode
- [ ] Database integration for large datasets

---

## Documentation Tasks 📚

### User Documentation
- [ ] Getting started tutorial
- [ ] Feature extraction guide
- [ ] Workflow configuration reference
- [ ] API reference (auto-generated)
- [ ] Common use cases / cookbook
- [ ] FAQ

### Developer Documentation
- [ ] Architecture decision records (ADR)
- [ ] Contribution guidelines
- [ ] Testing guidelines
- [ ] Release process
- [ ] Changelog

---

## Testing Requirements 🧪

### Unit Tests
- [x] Core data structures
- [x] Metadata handling
- [x] Signal operations
- [x] Feature extraction
- [ ] New feature types (spectral, HRV)
- [ ] Error handling paths

### Integration Tests
- [x] Basic workflow execution
- [ ] End-to-end workflow with real data
- [ ] Multi-importer workflows
- [ ] Export format validation
- [ ] Visualization generation

### Performance Tests
- [ ] Large dataset handling (multi-night)
- [ ] Memory profiling
- [ ] Feature extraction benchmarks
- [ ] Export speed tests

---

## References 📖

### Key Documents
- Feature Extraction Plan: `docs/feature_extraction_plan.md`
- Coding Guidelines: `docs/coding_guidelines.md`
- Requirements: `docs/requirements/requirements.md`
- Design: `docs/requirements/requirements/03_Architecture.md`

### Research Papers (in product-development/)
- Random Forest Algorithm: `product-development/Development/AlgorithmDevelopment/Resources/Random Forest Algorithm for HR and Accel.md`
- SKDH Technical Report: `product-development/Development/AlgorithmDevelopment/Resources/TechnicalReport-SKDH.md`
- Sleep Staging Methods: `product-development/Research/TechnicalReport-SleepStagingMethods.md`

### Example Workflows
- Polar workflow: `workflows/polar_workflow.yaml`
- Polar dev workflow: `workflows/polar_workflow-dev.yaml`

---

## Notes

### Recent Major Changes (commits d9bd9d1 - 938d4db)
1. **Feature System Refactor**: Split TimeSeriesSignal and Feature into separate entities
2. **Visualization**: Added hypnograms, sleep stage overlays, dual backend support
3. **Export**: Flexible content selection with 'all_ts', 'combined_features', etc.
4. **Metadata**: Split into TimeSeriesMetadata and FeatureMetadata
5. **Documentation**: Added class diagrams, data flow diagrams, Obsidian vault
6. **Testing**: Comprehensive test coverage for new features

### Development Principles
- Follow coding guidelines in `docs/coding_guidelines.md`
- Maintain traceability through metadata
- Write tests for new features
- Update documentation with changes
- Use type hints and docstrings
- Optimize for memory efficiency

### Useful Commands
```bash
# Run tests
pytest

# Run specific test file
pytest tests/test_feature_extraction.py -v

# Run workflow
python -m sleep_analysis.cli.run_workflow -w workflows/polar_workflow-dev.yaml -d data/ -o output/

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```
