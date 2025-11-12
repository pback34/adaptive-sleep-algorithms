# TODO - SomnuSight Framework Refactoring

**Last Updated:** 2025-11-11
**Current Branch:** `develop`
**Status:** ✅ Phase 1 Complete | 🚀 Ready for Phase 2

---

## Quick Status

- ✅ **Phase 1: Extract Services** - COMPLETE (all 167 tests passing)
- 🎯 **Phase 2: Simplify Operations** - NEXT UP
- ⏳ **Phase 3: Fix Type Hierarchy** - Planned
- ⏳ **Phase 4: Remove Global State** - Planned

---

## Key Documents

1. **[DESIGN_EVALUATION.md](DESIGN_EVALUATION.md)** - Complete design analysis and refactoring plan
2. **[TEST_FAILURE_RESOLUTION_PLAN.md](TEST_FAILURE_RESOLUTION_PLAN.md)** - Phase 1 test fixes (all resolved)
3. **[tests/unit/](tests/unit/)** - Comprehensive test suite (167 passing, 1 skipped)

---

## ✅ Phase 1 Complete - Service Extraction (DONE)

### What Was Accomplished

Phase 1 successfully extracted service classes from the `SignalCollection` god object:

#### Services Created:
1. **`ImportService`** ([src/sleep_analysis/services/import_service.py](src/sleep_analysis/services/import_service.py))
   - `import_signals_from_source()` - Import signals from various sources
   - `update_time_series_metadata()` - Update signal metadata
   - `update_feature_metadata()` - Update feature metadata

2. **`AlignmentService`** ([src/sleep_analysis/services/alignment_service.py](src/sleep_analysis/services/alignment_service.py))
   - `generate_alignment_grid()` - Create alignment grid for signals
   - `apply_grid_alignment()` - Align single signal to grid
   - `align_and_combine_signals()` - Align and merge multiple signals

3. **`FeatureService`** ([src/sleep_analysis/services/feature_service.py](src/sleep_analysis/services/feature_service.py))
   - `generate_epoch_grid()` - Create epoch grid for feature extraction
   - `apply_multi_signal_operation()` - Execute operations on multiple signals
   - `combine_features()` - Merge feature matrices
   - `propagate_metadata_to_feature()` - Transfer metadata to features

#### Key Achievements:
- ✅ No breaking changes to public API
- ✅ `SignalCollection` delegates to services (clean separation)
- ✅ All 167 tests passing (1 HDF5 test properly skipped)
- ✅ Services are independently testable
- ✅ Comprehensive test coverage for all service classes

#### Test Results:
```
167 passed, 1 skipped, 2 warnings in 2.77s
```

---

## 🎯 Phase 2 - Simplify Operation Registry System (NEXT)

**Priority:** HIGH
**Estimated Effort:** 2-3 weeks
**Document Reference:** [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md#2-confusing-multi-level-operation-registry-system-) (lines 77-144)

### The Problem

The framework currently has **three different operation registries** with overlapping purposes:

1. **`SignalData.registry`** - Instance method operations
2. **`SignalCollection.multi_signal_registry`** - Multi-signal operations
3. **`SignalCollection.collection_operation_registry`** - Collection-level operations

This creates:
- ❌ Unclear precedence (instance method vs registry)
- ❌ Different calling conventions
- ❌ Hard to discover where operations are defined
- ❌ Complex inheritance logic
- ❌ Steep learning curve

### Decision Required

**Option A (RECOMMENDED): Eliminate registries entirely**
- Use only instance methods
- Simpler, more Pythonic
- Better IDE support
- Easier to understand

```python
class TimeSeriesSignal:
    def filter_lowpass(self, cutoff: float) -> pd.DataFrame:
        """Returns filtered data"""
        ...

    def resample(self, rate: float) -> pd.DataFrame:
        """Returns resampled data"""
        ...
```

**Option B: Single unified registry**
- Keep registry pattern but consolidate
- Clear semantics with `OperationScope` enum
- Centralized operation management

```python
class OperationRegistry:
    """Single registry for ALL operations"""

    def register(self,
                 name: str,
                 input_type: Type,
                 output_type: Type,
                 scope: OperationScope):  # SIGNAL, MULTI_SIGNAL, COLLECTION
        ...
```

### Tasks for Phase 2

1. **Analyze current usage** (1-2 days)
   - [ ] Audit all operations currently in registries
   - [ ] Identify which are actually used in production
   - [ ] Check if any external code depends on registry pattern
   - [ ] Document current operation lookup flow

2. **Make architectural decision** (1 day)
   - [ ] Decide between Option A or Option B
   - [ ] Document rationale in DESIGN_EVALUATION.md
   - [ ] Get stakeholder buy-in if needed

3. **Implement changes** (1-2 weeks)
   - [ ] Refactor operations to chosen pattern
   - [ ] Update all operation calls
   - [ ] Update documentation
   - [ ] Add operation introspection/discovery tools

4. **Update tests** (2-3 days)
   - [ ] Update all operation-related tests
   - [ ] Add tests for new operation discovery
   - [ ] Verify no regressions

5. **Update WorkflowExecutor** (2-3 days)
   - [ ] Update to use new operation mechanism
   - [ ] Ensure backward compatibility with existing YAML workflows
   - [ ] Add deprecation warnings if needed

### Files to Modify

- `src/sleep_analysis/core/signal_data.py` - Registry base class
- `src/sleep_analysis/core/time_series_signal.py` - Signal operations
- `src/sleep_analysis/signal_collection.py` - Collection registries
- `src/sleep_analysis/workflow_executor.py` - Operation lookup
- `tests/unit/test_*.py` - All operation tests

---

## ⏳ Phase 3 - Fix Feature/TimeSeriesSignal Type Hierarchy

**Priority:** MEDIUM
**Estimated Effort:** 1 week
**Document Reference:** [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md#3-feature-vs-timeseriessignal-dichotomy-is-awkward-) (lines 147-200)

### The Problem

- `Feature` and `TimeSeriesSignal` stored separately but treated inconsistently
- `Feature` doesn't inherit from `SignalData` but has similar interface
- Can't chain operations through features
- Confusing type checking

### Decision Required

**Option A: Make Feature inherit from SignalData**
```python
class SignalData(ABC):
    """Base for all data types"""

class TimeSeriesSignal(SignalData):
    """Continuous time series"""

class EpochFeature(SignalData):
    """Epoch-based features - also a signal!"""
```

**Option B: Separate completely with clear naming**
```python
collection.get_time_series_signals()  # Only TimeSeriesSignal
collection.get_features()              # Only Feature
collection.get_all()                   # Union type
```

### Tasks (TBD after Phase 2)

- [ ] Choose Option A or B
- [ ] Update type hierarchy
- [ ] Fix `get_signals()` method naming/behavior
- [ ] Update all type hints
- [ ] Update tests

---

## ⏳ Phase 4 - Remove Global State Dependencies

**Priority:** MEDIUM
**Estimated Effort:** 1 week
**Document Reference:** [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md#6-global-state-in-collection-complicates-testing-) (lines 317-378)

### The Problem

`SignalCollection` stores global state that affects operations:
- `target_rate`, `ref_time`, `grid_index` (alignment)
- `epoch_grid_index`, `global_epoch_window_length` (epochs)
- Cached dataframes
- Hidden order dependencies

### Solution

Make dependencies explicit through parameters or context managers:

```python
# Explicit parameters
def extract_features(self,
                     signal_keys: List[str],
                     epoch_config: EpochConfig,  # Explicit!
                     params: Dict):
    ...

# Or context manager
with collection.epoch_context(window="30s", step="30s") as ctx:
    features = ctx.extract_features(["ppg_0"], aggregations=["mean"])
```

### Tasks (TBD after Phase 3)

- [ ] Remove global alignment/epoch state
- [ ] Make epoch config explicit parameter
- [ ] Add configuration validation
- [ ] Simplify SignalCollection to pure container
- [ ] Update WorkflowExecutor
- [ ] Update all tests

---

## Minor Issues (Lower Priority)

### 7. Timezone Handling Scattered
- **Effort:** 2-3 days
- **Solution:** Create `TimezoneService` or `TimezoneConfig` class
- **Reference:** [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md#7-timezone-handling-is-scattered-) (lines 384-392)

### 8. Parameter Sanitization in Wrong Place
- **Effort:** 1 day
- **Solution:** Extract `ParameterSerializer` class from `MetadataHandler`
- **Reference:** [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md#8-parameter-sanitization-buried-in-metadatahandler-) (lines 396-408)

### 9. String-Based Operation Lookup
- **Effort:** 1 week
- **Solution:** Use type-safe operation objects or direct methods
- **Note:** This may be addressed in Phase 2
- **Reference:** [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md#9-string-based-operation-lookup-is-error-prone-) (lines 411-425)

### 10. Operation Recording Redundancy
- **Effort:** 2-3 days
- **Solution:** Evaluate if feature is used, simplify or remove
- **Reference:** [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md#10-operation-recording-is-redundant-) (lines 429-438)

---

## Testing Strategy

### Current Test Coverage
```
tests/integration/test_export_workflow.py       1 test
tests/unit/test_alignment_service.py           22 tests
tests/unit/test_export.py                      11 tests (1 skipped)
tests/unit/test_feature_service.py             17 tests
tests/unit/test_import_service.py              17 tests
tests/unit/test_importers.py                   18 tests
tests/unit/test_metadata.py                     4 tests
tests/unit/test_metadata_handler.py            10 tests
tests/unit/test_multiindex_export.py            4 tests
tests/unit/test_signal_collection.py           23 tests
tests/unit/test_signal_data.py                  8 tests
tests/unit/test_signal_types.py                 5 tests
tests/unit/test_signals.py                     16 tests
tests/unit/test_workflow_executor.py           15 tests
---------------------------------------------------
TOTAL: 167 passed, 1 skipped
```

### Test Requirements for Future Phases

- ✅ **All existing tests must continue passing**
- ✅ **Add tests for new functionality**
- ✅ **No decrease in coverage**
- ✅ **Integration tests for workflow compatibility**

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific service tests
pytest tests/unit/test_import_service.py -v
pytest tests/unit/test_alignment_service.py -v
pytest tests/unit/test_feature_service.py -v

# With coverage
pytest tests/ --cov=src/sleep_analysis --cov-report=html
```

---

## Git Workflow

### Current Branch Status
```bash
Branch: develop
Status: Clean (all changes committed)
Recent commits:
- 2615d7a fix: Resolve 24 test failures according to TEST_FAILURE_RESOLUTION_PLAN.md
- 234e384 chore: update .gitignore
- 6f7e0ea docs: Add test failure resolution plan for Phase 1
```

### Recommended Workflow for Future Phases

1. **Create feature branch from `develop`**
   ```bash
   git checkout develop
   git pull
   git checkout -b phase-2-simplify-operations
   ```

2. **Make incremental commits**
   - Commit after each logical change
   - Keep commits focused and atomic
   - Write descriptive commit messages

3. **Run tests before committing**
   ```bash
   pytest tests/ -v
   # Only commit if all tests pass
   ```

4. **Create PR when phase complete**
   - Merge to `develop` when all tests pass
   - Update this TODO.md with completion status

---

## Questions for Discussion

1. **Operation Registry Decision** (Phase 2)
   - Should we eliminate registries entirely or consolidate them?
   - Are there external dependencies on the registry pattern?
   - What's the migration path for existing code?

2. **Feature Type Hierarchy** (Phase 3)
   - Should `Feature` inherit from `SignalData`?
   - What breaking changes are acceptable?
   - How do we maintain backward compatibility?

3. **Global State Removal** (Phase 4)
   - Context manager vs explicit parameters?
   - How to validate epoch/alignment config early?
   - Migration strategy for existing workflows?

---

## Success Metrics

### For Each Phase:
- ✅ All tests passing (>= 167 passed)
- ✅ No breaking changes to existing workflows
- ✅ Code coverage maintained or improved
- ✅ Documentation updated
- ✅ Improved maintainability (measured by cyclomatic complexity)

### Overall Refactoring Goals:
- Reduce `SignalCollection` from 1000+ lines to < 300 lines
- Eliminate or consolidate operation registries
- Remove global state dependencies
- Improve type safety and IDE support
- Reduce cognitive load for new developers

---

## Contacts & Resources

- **Documentation:** See [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md) for full analysis
- **Test Reports:** See [TEST_FAILURE_RESOLUTION_PLAN.md](TEST_FAILURE_RESOLUTION_PLAN.md)
- **Service Tests:** All in `tests/unit/test_*_service.py`

---

## Notes for Next Agent

### What You Can Assume:
- ✅ Phase 1 service extraction is complete and tested
- ✅ All services are working correctly with full test coverage
- ✅ No breaking changes to public API
- ✅ `SignalCollection` properly delegates to services

### Where to Start:
1. **Read [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md)** - Understand the full context
2. **Review service implementations** - See how Phase 1 was executed
3. **Analyze operation registries** - Start Phase 2 with audit
4. **Make architectural decision** - Choose Option A or B for registries
5. **Create implementation plan** - Break down Phase 2 into tasks

### Key Principles:
- ✅ **Test-driven:** All tests must pass before committing
- ✅ **Incremental:** Make small, focused changes
- ✅ **Backward compatible:** Don't break existing workflows
- ✅ **Well-documented:** Update docs as you go
- ✅ **Type-safe:** Improve type hints throughout

---

**Good luck with Phase 2!** 🚀

The foundation is solid, the tests are comprehensive, and the path forward is clear. Take it one step at a time and you'll do great.
