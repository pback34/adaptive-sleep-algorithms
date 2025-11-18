# Technical Debt

**Last Updated**: 2025-11-17
**Status**: Active Tracking
**Priority System**: 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low

---

## Overview

This document tracks known technical debt in the Adaptive Sleep Algorithms Framework. Technical debt items are prioritized by impact and effort, helping guide future refactoring and improvement efforts.

**Current Status Summary**:
- 🔴 **Critical Items**: 3
- 🟠 **High Priority Items**: 6
- 🟡 **Medium Priority Items**: 8
- 🟢 **Low Priority Items**: 5

---

## Recently Resolved (2025-11-17)

### Quick Wins Session
- ✅ **Duplicate Imports**: Removed duplicate imports in signal_collection.py and workflow_executor.py
- ✅ **Magic Numbers**: Replaced hardcoded values with named constants in time_series_signal.py and feature_extraction.py
- ✅ **Dead Code**: Removed commented-out pytz validation code from workflow_executor.py
- ✅ **Missing Docstrings**: Enhanced documentation for _resolve_target_timezone()

---

## 🔴 Critical Technical Debt

### 1. God Object: SignalCollection (1,974 lines)
**Impact**: Very High | **Effort**: Very High (40 hours) | **Priority**: 🔴 Critical

**Description**:
The `SignalCollection` class violates the Single Responsibility Principle with 60+ methods and multiple responsibilities:
- Signal storage and retrieval
- Operation application (single-signal, multi-signal, collection-level)
- Alignment (grid calculation, application, combination)
- Epoch management (grid generation, feature extraction)
- Feature combination
- Signal summarization
- Metadata management

**Consequences**:
- Difficult to test in isolation
- Hard to understand and maintain
- Changes ripple through entire codebase
- New developers struggle to understand the class

**Recommended Solution**:
Break into focused classes:
1. `SignalCollection` - Core storage and retrieval (500 lines)
2. `SignalProcessor` - Operation application (400 lines)
3. `AlignmentManager` - Alignment operations (300 lines)
4. `EpochManager` - Epoch grid and feature extraction (400 lines)
5. `FeatureCombinator` - Feature combination (200 lines)
6. `CollectionSummarizer` - Summarization (150 lines)

**References**:
- File: `src/sleep_analysis/core/signal_collection.py`
- Architecture Evaluation: Section 4.3
- Lines: 1-1974

---

### 2. God Object: WorkflowExecutor (959 lines)
**Impact**: High | **Effort**: High (20 hours) | **Priority**: 🔴 Critical

**Description**:
The `WorkflowExecutor` class handles too many responsibilities:
- Step validation
- Operation execution
- Import coordination
- Export coordination
- Visualization coordination
- Error handling
- Timezone management

**Consequences**:
- Complex method (`execute_step`) with 230 lines
- Difficult to add new step types
- Hard to test individual concerns
- Validation logic tightly coupled

**Recommended Solution**:
Extract into focused classes:
1. `WorkflowStepValidator` - Step validation logic
2. `WorkflowStepExecutor` - Step execution strategies
3. `WorkflowImportCoordinator` - Import management
4. Keep orchestration in `WorkflowExecutor`

**References**:
- File: `src/sleep_analysis/workflows/workflow_executor.py`
- Architecture Evaluation: Section 4.3
- Lines: 1-959

---

### 3. Feature Extraction Code Duplication (~500 lines)
**Impact**: High | **Effort**: Medium (15 hours) | **Priority**: 🔴 Critical

**Description**:
Epoch-based feature extraction pattern is duplicated across all feature operations:
- Empty grid handling (~50 lines per function)
- Epoch loop logic (~100 lines per function)
- Result assembly (~50 lines per function)
- Total duplication: 500+ lines across 5+ operations

**Consequences**:
- Bug fixes must be applied to multiple locations
- Inconsistent error handling across operations
- High barrier to adding new features
- Difficult to test feature logic in isolation

**Recommended Solution**:
Create `BaseFeatureExtractor` abstract class with template method pattern:
```python
class BaseFeatureExtractor:
    def compute_epoch_feature(self, signals, epoch_data, parameters):
        # Override in subclass
        raise NotImplementedError

    def extract(self, signals, epoch_grid, parameters, ...):
        # Template method handles boilerplate
        # Calls compute_epoch_feature for each epoch
        ...
```

**References**:
- File: `src/sleep_analysis/operations/feature_extraction.py`
- Architecture Evaluation: Section 4.1
- Functions: compute_feature_statistics, compute_hrv_features, compute_movement_features, compute_correlation_features

---

## 🟠 High Priority Technical Debt

### 4. Validation Logic Duplication
**Impact**: Medium-High | **Effort**: Medium (8 hours) | **Priority**: 🟠 High

**Description**:
Parameter validation logic is scattered across multiple files with inconsistent patterns:
- Timezone validation repeated in 3+ places
- Index validation repeated in 5+ places
- Parameter type checking inconsistent

**Recommended Solution**:
Create `core/validation.py` module with reusable validators:
- `validate_dataframe_index(df, allow_naive=False)`
- `validate_signal_columns(signal, required_cols)`
- `validate_timezone_consistency(signals, target_tz)`
- `validate_parameter_types(params, schema)`

**References**:
- Files: signal_collection.py, workflow_executor.py, importers/
- Architecture Evaluation: Section 4.1

---

### 5. Empty DataFrame Handler Duplication (~250 lines)
**Impact**: Medium-High | **Effort**: Low (30 minutes) | **Priority**: 🟠 High

**Description**:
Empty DataFrame handling pattern is duplicated across 5+ feature extraction operations. Each function has ~50 lines of identical code to handle empty epoch grids.

**Status**: ✅ **PARTIALLY RESOLVED** - `_handle_empty_feature_data()` helper function created, but not yet used in all operations.

**Remaining Work**:
Replace direct empty handling in remaining operations with calls to `_handle_empty_feature_data()`.

**References**:
- File: `src/sleep_analysis/operations/feature_extraction.py`
- Function: `_handle_empty_feature_data()` (lines 180-240)

---

### 6. Large Module: Visualization Base (1,107 lines)
**Impact**: Medium | **Effort**: Medium (8 hours) | **Priority**: 🟠 High

**Description**:
Visualization modules exceed recommended size (500 lines):
- `visualization/base.py`: 1,107 lines
- `visualization/bokeh_visualizer.py`: 1,131 lines
- `visualization/plotly_visualizer.py`: 1,205 lines

**Recommended Solution**:
Split into focused modules:
- `visualization/base.py` - Core interfaces
- `visualization/time_series_plots.py` - Time series plotting
- `visualization/feature_plots.py` - Feature visualization
- `visualization/algorithm_plots.py` - Algorithm-specific plots

**References**:
- Directory: `src/sleep_analysis/visualization/`
- Architecture Evaluation: Section 1.1

---

### 7. God Object: TimeSeriesSignal (715 lines)
**Impact**: Medium | **Effort**: Medium (15 hours) | **Priority**: 🟠 High

**Description**:
`TimeSeriesSignal` class handles multiple responsibilities:
- Data storage and validation
- Metadata management
- Operation application (inplace and non-inplace)
- Sample rate calculation
- Resampling, filtering, slicing
- Type-specific operations

**Recommended Solution**:
Extract:
1. `SignalOperationRegistry` - Operation lookup
2. `SignalAnalyzer` - Sampling rate, statistics
3. Keep core signal data handling in `TimeSeriesSignal`

**References**:
- File: `src/sleep_analysis/signals/time_series_signal.py`
- Architecture Evaluation: Section 4.3

---

### 8. Inconsistent Error Handling
**Impact**: Medium | **Effort**: Medium (10 hours) | **Priority**: 🟠 High

**Description**:
Error handling patterns are inconsistent across the codebase:
- Some operations skip epochs (returning NaN)
- Some operations skip signals (returning empty)
- Some operations fail with exceptions
- Broad exception catching (`except Exception`) in multiple places
- Silent failures in non-strict workflow mode

**Recommended Solution**:
1. Define error handling strategy per operation type
2. Replace broad exception catches with specific types
3. Add error recovery documentation
4. Implement consistent logging for error cases

**References**:
- Architecture Evaluation: Section 7.1
- Files: workflow_executor.py, feature_extraction.py, operations/

---

### 9. Complex Method: execute_step (230 lines)
**Impact**: Medium-High | **Effort**: Low-Medium (8 hours) | **Priority**: 🟠 High

**Description**:
The `execute_step` method in WorkflowExecutor is overly complex:
- 230 lines in a single method
- Handles 4 different operation types
- Nested if/elif chains reach 5+ levels
- Difficult to add new step types

**Recommended Solution**:
Implement Strategy pattern:
```python
class StepExecutionStrategy:
    def execute(self, step, container): ...

class CollectionStepStrategy(StepExecutionStrategy): ...
class MultiSignalStepStrategy(StepExecutionStrategy): ...
class SingleSignalStepStrategy(StepExecutionStrategy): ...
```

**References**:
- File: `src/sleep_analysis/workflows/workflow_executor.py`
- Method: `execute_step` (lines 362-595)
- Architecture Evaluation: Section 4.2

---

## 🟡 Medium Priority Technical Debt

### 10. Missing Signal Quality Assessment
**Impact**: Medium | **Effort**: Medium (20 hours) | **Priority**: 🟡 Medium

**Description**:
No automatic detection or reporting of signal quality:
- Missing data segments not detected
- Outliers not flagged
- Data gaps not identified
- No quality metrics calculated

**Recommended Solution**:
Implement signal quality assessment module:
- `assess_signal_quality(signal) -> QualityMetrics`
- Calculate: data coverage, gap count, outlier ratio, SNR
- Provide quality flagging and filtering options

**References**:
- Architecture Evaluation: Section 10.1
- Use case: Research data validation

---

### 11. No Batch Processing Support
**Impact**: Medium | **Effort**: Medium (20 hours) | **Priority**: 🟡 Medium

**Description**:
Framework currently handles single subjects only:
- No multi-subject workflow support
- No batch processing infrastructure
- No aggregation of results across subjects

**Recommended Solution**:
1. Add `SubjectBatch` class for managing multiple subjects
2. Implement parallel processing for subjects
3. Add result aggregation utilities

**References**:
- Architecture Evaluation: Section 10.1
- Handoff Notes: Phase 3 recommendation

---

### 12. Missing Type Hints in Complex Functions
**Impact**: Medium | **Effort**: Low-Medium (8 hours) | **Priority**: 🟡 Medium

**Description**:
Some functions use overly generic types:
- `parameters: Dict[str, Any]` could be more specific
- Function callbacks not always fully typed
- Decorator signatures could be more explicit

**Recommended Solution**:
1. Create TypedDict classes for common parameter structures
2. Add proper typing to callback functions
3. Improve decorator type hints
4. Enable mypy strict mode

**References**:
- Documentation Evaluation: Section 1.3
- Architecture Evaluation: Section 11.2

---

### 13. Cache Management Issues
**Impact**: Medium | **Effort**: Low (10 hours) | **Priority**: 🟡 Medium

**Description**:
Feature cache has potential issues:
- Global cache without TTL or size limits (memory leak risk)
- Manual cache management via `clear_feature_cache()`
- Hash of epoch grid values may have collisions
- No cache statistics or monitoring

**Recommended Solution**:
1. Implement LRU cache with size limit
2. Add automatic cleanup on collection reset
3. Improve cache key computation
4. Add cache monitoring and statistics

**References**:
- File: `src/sleep_analysis/operations/feature_extraction.py`
- Architecture Evaluation: Section 6.1
- Functions: @cache_features decorator, _FEATURE_CACHE

---

### 14. Missing Performance Benchmarks
**Impact**: Low-Medium | **Effort**: Medium (10 hours) | **Priority**: 🟡 Medium

**Description**:
No performance benchmarking infrastructure:
- No baseline performance metrics
- No regression detection
- Unknown performance characteristics for large datasets

**Recommended Solution**:
1. Create benchmark suite for key operations
2. Track performance over commits
3. Add memory profiling
4. Document performance characteristics

**References**:
- Architecture Evaluation: Section 11.5
- Testing gaps identified

---

### 15. Incomplete Integration Tests
**Impact**: Medium | **Effort**: Medium (8 hours) | **Priority**: 🟡 Medium

**Description**:
Limited integration test coverage:
- Only 2 integration test files
- End-to-end workflows not fully tested
- Error scenarios under-tested
- Multi-signal workflows need more coverage

**Recommended Solution**:
Add integration tests for:
- Complete workflow scenarios
- Error handling and recovery
- Data preservation through pipeline
- Multi-signal feature extraction

**References**:
- Directory: `tests/integration/`
- Architecture Evaluation: Section 8.2

---

### 16. Scattered Design Decisions
**Impact**: Low-Medium | **Effort**: Low (4 hours) | **Priority**: 🟡 Medium

**Description**:
Design decisions are scattered across multiple documents:
- No centralized design decision log
- HANDOFF-NOTES.md partially serves this role (not ideal)
- Rationale for key decisions not always documented

**Recommended Solution**:
Create `docs/DESIGN_DECISIONS.md` with:
- Centralized decision log
- Rationale for key architectural choices
- Trade-offs considered
- Alternatives evaluated

**References**:
- Documentation Evaluation: Section 6.1
- Current: HANDOFF-NOTES.md (522 lines)

---

### 17. Missing Data Validation Rules
**Impact**: Medium | **Effort**: Medium (15 hours) | **Priority**: 🟡 Medium

**Description**:
No systematic data validation:
- No data quality rules
- No consistency checks
- No outlier detection
- Bad data not automatically flagged

**Recommended Solution**:
Implement data validation framework:
- `DataValidator` class with rule engine
- Common validation rules (range, consistency, completeness)
- Automatic validation in import pipeline
- Validation reporting

**References**:
- Architecture Evaluation: Section 10.1
- Use case: Data quality assurance

---

## 🟢 Low Priority Technical Debt

### 18. No Plugin Discovery System
**Impact**: Low | **Effort**: Medium (15 hours) | **Priority**: 🟢 Low

**Description**:
All extensions require code changes:
- No plugin discovery mechanism
- Signal types must be manually registered
- Operations must be manually added to registries

**Recommended Solution**:
Implement plugin system:
- Entry point-based plugin discovery
- Automatic registration of signal types
- Automatic registration of operations

**References**:
- Architecture Evaluation: Section 10.2
- Current: Manual registration required

---

### 19. Missing Environment Variable Support
**Impact**: Low | **Effort**: Low (5 hours) | **Priority**: 🟢 Low

**Description**:
No configuration via environment variables:
- Cannot externalize configuration
- Difficult to deploy in containerized environments
- No separation of code and config

**Recommended Solution**:
Add environment variable support:
- Support for data directories via env vars
- Configuration overrides
- Deployment-specific settings

**References**:
- Architecture Evaluation: Section 10.2

---

### 20. No Event System
**Impact**: Low | **Effort**: Medium (10 hours) | **Priority**: 🟢 Low

**Description**:
No event hooks or callbacks:
- Cannot integrate with external tools
- No progress callbacks
- No workflow lifecycle hooks

**Recommended Solution**:
Implement event system:
- Event hooks for workflow execution
- Progress callbacks
- Lifecycle events (start, complete, error)

**References**:
- Architecture Evaluation: Section 10.2

---

### 21. Missing Advanced Visualizations
**Impact**: Low | **Effort**: High (25 hours) | **Priority**: 🟢 Low

**Description**:
Limited visualization types:
- No spectrograms
- No scatter matrices
- No heatmaps
- No 3D plots

**Note**: Users can use external tools, so this is lower priority.

**References**:
- Architecture Evaluation: Section 10.1

---

### 22. No Hyperparameter Optimization
**Impact**: Low | **Effort**: High (20 hours) | **Priority**: 🟢 Low

**Description**:
No built-in HPO support:
- No grid search
- No random search
- Manual tuning required

**Note**: Specialized use case, can use external libraries.

**References**:
- Architecture Evaluation: Section 10.1

---

## Improvement Roadmap

### Phase 1: Quick Wins (Completed - 2 hours)
- ✅ Remove duplicate imports
- ✅ Fix magic numbers
- ✅ Remove dead code
- ✅ Add missing docstrings

### Phase 2: Critical Code Quality (4-6 weeks)
**Target**: Address Critical and High Priority items
- Decompose SignalCollection (40 hours)
- Simplify WorkflowExecutor (20 hours)
- Create Feature Extraction Base Class (15 hours)
- Centralize validation utilities (8 hours)
- Complete empty handler refactor (2 hours)

### Phase 3: Medium Priority Items (6-8 weeks)
**Target**: Improve robustness and testing
- Signal quality assessment (20 hours)
- Batch processing support (20 hours)
- Type hints improvement (8 hours)
- Cache management (10 hours)
- Integration tests (8 hours)

### Phase 4: Low Priority Items (Ongoing)
**Target**: Nice-to-have improvements
- Plugin system (15 hours)
- Event system (10 hours)
- Advanced visualizations (25 hours)

---

## Metrics & Tracking

### Current Codebase Statistics
- **Total Lines of Code**: ~13,900 Python
- **Largest Class**: SignalCollection (1,974 lines)
- **Test Coverage**: 98%
- **Test Pass Rate**: 211/211 (100%)

### Quality Goals
- **Average Module Size**: <250 lines (currently ~300)
- **Average Method Count per Class**: <10 (currently ~15)
- **Test Coverage**: Maintain >95%
- **God Objects**: Eliminate classes >500 lines with multiple responsibilities

### Progress Tracking
- **Technical Debt Items**: 22 total
- **Recently Resolved**: 4 items (2025-11-17)
- **Critical Items**: 3 (13.6%)
- **High Priority**: 6 (27.3%)
- **Medium Priority**: 8 (36.4%)
- **Low Priority**: 5 (22.7%)

---

## How to Use This Document

### For Developers
1. **Before Starting Work**: Review related technical debt items
2. **During Development**: Note if work addresses or adds debt
3. **After Completion**: Update status and metrics

### For Project Managers
1. **Sprint Planning**: Allocate time for high-priority debt
2. **Risk Assessment**: Monitor critical items for project health
3. **Resource Allocation**: Balance feature work with debt reduction

### For New Contributors
1. **Onboarding**: Understand known limitations
2. **Work Selection**: Choose items matching skill level
3. **Impact Assessment**: Focus on high-impact, low-effort items

---

## References

- **Architecture Evaluation**: `docs/evaluations/architecture-evaluation.md`
- **Documentation Evaluation**: `docs/evaluations/documentation-evaluation.md`
- **Handoff Notes**: `HANDOFF-NOTES.md`
- **Improvement Plan**: `CODE_QUALITY_IMPROVEMENT_PLAN.md`
- **Coding Guidelines**: `docs/coding_guidelines.md`

---

## Change Log

### 2025-11-17: Initial Version
- Created technical debt tracking document
- Documented 22 technical debt items
- Established priority system and roadmap
- Completed Phase 1 quick wins (4 items resolved)

---

**Last Review**: 2025-11-17
**Next Review**: 2025-12-01 (2 weeks)
**Owner**: Development Team
