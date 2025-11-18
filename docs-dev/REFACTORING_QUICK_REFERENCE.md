# SignalCollection Refactoring - Quick Reference

## TL;DR: The Problem

**SignalCollection** is a God Object with:
- 1,971 lines of code
- 41 methods
- 13+ instance attributes
- 8+ distinct responsibilities

This violates Single Responsibility Principle and makes the code hard to test, understand, and maintain.

---

## Solution: Break It Into 13 Focused Services

### Proposed Service Classes

| Service | Methods | Lines | Responsibility |
|---------|---------|-------|---|
| **SignalRepository** | 6 | ~200 | CRUD for signals/features |
| **SignalQueryService** | 4 | ~150 | Filtering & querying |
| **MetadataManager** | 5 | ~200 | Metadata updates & validation |
| **AlignmentGridService** | 5 | ~250 | Compute alignment parameters |
| **EpochGridService** | 1 | ~150 | Compute epoch windows |
| **AlignmentExecutor** | 1 | ~100 | Apply alignment to signals |
| **SignalCombinationService** | 3 | ~300 | Combine signals into dataframes |
| **OperationExecutor** | 4 | ~300 | Execute multi-signal operations |
| **DataImportService** | 1 | ~80 | Import signals from sources |
| **SignalSummaryReporter** | 3 | ~150 | Generate reports & summaries |
| **SignalCollection** | 13 | ~300 | Orchestration & delegation |
| **AlignmentGridState** | - | ~10 | State data class |
| **EpochGridState** | - | ~10 | State data class |

---

## Current vs. Proposed Structure

### Current: One Large Class
```
SignalCollection (1,971 lines)
├── Signal management (4 methods)
├── Signal retrieval (4 methods)
├── Metadata management (6 methods)
├── Sample rate calculation (3 methods)
├── Alignment grid (2 methods)
├── Epoch grid (1 method)
├── Signal alignment (2 methods)
├── Signal combination (5 methods)
├── Feature combination (1 method)
├── Multi-signal operations (3 methods)
├── Single-signal operations (1 method)
├── Data import (1 method)
├── Data access (3 methods)
└── Reporting (2 methods)
```

### Proposed: Focused Services
```
SignalRepository (200 lines)
  ├── add_time_series_signal()
  ├── add_feature()
  ├── add_signal_with_base_name()
  ├── add_imported_signals()
  ├── get_time_series_signal()
  └── get_feature()

SignalQueryService (150 lines)
  ├── get_signals()
  └── _matches_criteria()

MetadataManager (200 lines)
  ├── update_time_series_metadata()
  ├── update_feature_metadata()
  └── set_index_config()

AlignmentGridService (250 lines)
  ├── generate_alignment_grid()
  └── _calculate_grid_index()

EpochGridService (150 lines)
  └── generate_epoch_grid()

AlignmentExecutor (100 lines)
  └── apply_grid_alignment()

SignalCombinationService (300 lines)
  ├── combine_aligned_signals()
  ├── combine_features()
  └── _perform_concatenation()

OperationExecutor (300 lines)
  ├── execute_multi_signal_operation()
  ├── execute_single_signal_operation()
  └── execute_batch_signal_operation()

DataImportService (80 lines)
  └── import_signals_from_source()

SignalSummaryReporter (150 lines)
  ├── summarize_signals()
  └── _format_summary_cell()

SignalCollection (300 lines) [Orchestrator]
  └── Delegates to all services
```

---

## Key Metrics Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| **Lines per class** | 1,971 | ~150 avg | -92% |
| **Methods per class** | 41 | 3-6 avg | -85% |
| **Testability** | Hard | Easy | +++++ |
| **Cognitive load** | Very High | Low | -80% |
| **Cyclomatic complexity** | 50+ per method | 5-10 avg | -80% |
| **Coupling** | High | Low | -75% |
| **Maintainability** | Low | High | +++++ |

---

## Implementation Timeline

### Phase 1: Preparation (Week 1)
- [ ] Review and approve proposal
- [ ] Design service interfaces
- [ ] Set up test framework

### Phase 2: Services (Weeks 2-4)
- [ ] Implement SignalRepository
- [ ] Implement SignalQueryService
- [ ] Implement MetadataManager
- [ ] Implement AlignmentGridService
- [ ] Implement EpochGridService
- [ ] Implement AlignmentExecutor

### Phase 3: Complex Services (Weeks 5-6)
- [ ] Implement SignalCombinationService
- [ ] Implement OperationExecutor
- [ ] Implement DataImportService
- [ ] Implement SignalSummaryReporter

### Phase 4: Integration (Week 7)
- [ ] Refactor SignalCollection as orchestrator
- [ ] Full integration testing
- [ ] Backward compatibility verification
- [ ] Documentation update

### Phase 5: Deployment (Week 8)
- [ ] Code review
- [ ] Performance testing
- [ ] Deploy
- [ ] Monitor

---

## Method Mapping: Old → New

### Signal Management
```
SignalCollection.add_time_series_signal()          → SignalRepository.add_time_series_signal()
SignalCollection.add_feature()                     → SignalRepository.add_feature()
SignalCollection.add_signal_with_base_name()      → SignalRepository.add_signal_with_base_name()
SignalCollection.add_imported_signals()            → SignalRepository.add_imported_signals()
SignalCollection.get_time_series_signal()          → SignalRepository.get_time_series_signal()
SignalCollection.get_feature()                     → SignalRepository.get_feature()
SignalCollection.get_signal()                      → SignalRepository.get_by_key()
SignalCollection.get_signals()                     → SignalQueryService.get_signals()
```

### Metadata Management
```
SignalCollection.update_time_series_metadata()     → MetadataManager.update_time_series_metadata()
SignalCollection.update_feature_metadata()         → MetadataManager.update_feature_metadata()
SignalCollection.set_index_config()                → MetadataManager.set_index_config()
SignalCollection.set_feature_index_config()        → MetadataManager.set_feature_index_config()
SignalCollection._validate_timestamp_index()       → MetadataManager._validate_timestamp_index()
```

### Alignment Grid
```
SignalCollection.generate_alignment_grid()         → AlignmentGridService.generate_alignment_grid()
SignalCollection.get_target_sample_rate()          → AlignmentGridService._get_target_sample_rate()
SignalCollection.get_nearest_standard_rate()       → AlignmentGridService._get_nearest_standard_rate()
SignalCollection.get_reference_time()              → AlignmentGridService._get_reference_time()
SignalCollection._calculate_grid_index()           → AlignmentGridService._calculate_grid_index()
```

### Epoch Grid
```
SignalCollection.generate_epoch_grid()             → EpochGridService.generate_epoch_grid()
```

### Alignment Execution
```
SignalCollection.apply_grid_alignment()            → AlignmentExecutor.apply_grid_alignment()
```

### Signal Combination
```
SignalCollection.combine_aligned_signals()         → SignalCombinationService.combine_aligned_signals()
SignalCollection.combine_features()                → SignalCombinationService.combine_features()
SignalCollection._perform_concatenation()          → SignalCombinationService._perform_concatenation()
SignalCollection.align_and_combine_signals()       → Orchestration in SignalCollection
```

### Operations
```
SignalCollection.apply_multi_signal_operation()    → OperationExecutor.execute_multi_signal_operation()
SignalCollection.apply_operation()                 → OperationExecutor.execute_collection_operation()
SignalCollection.apply_and_store_operation()       → OperationExecutor.execute_single_signal_operation()
SignalCollection.apply_operation_to_signals()      → OperationExecutor.execute_batch_signal_operation()
```

### Data Import
```
SignalCollection.import_signals_from_source()      → DataImportService.import_signals_from_source()
```

### Reporting
```
SignalCollection.summarize_signals()               → SignalSummaryReporter.summarize_signals()
SignalCollection._format_summary_cell()            → SignalSummaryReporter._format_summary_cell()
```

---

## Backward Compatibility

The refactored SignalCollection will **maintain full backward compatibility**:

```python
# Old code still works
collection = SignalCollection()
collection.add_time_series_signal(key, signal)
collection.generate_alignment_grid()
collection.apply_grid_alignment()
collection.combine_aligned_signals()

# New code can use services directly
service = collection.alignment_grid_service
state = service.generate_alignment_grid()
print(f"Grid index size: {len(state.grid_index)}")
```

---

## Testing Strategy

### Current: Integration Tests Only
```python
def test_signal_collection():
    collection = SignalCollection()
    # Must set up many dependencies
    # Hard to isolate failures
    # Slow tests
```

### Proposed: Unit Tests + Integration Tests
```python
# Unit test specific service
def test_alignment_grid_service():
    repo = MockRepository()
    service = AlignmentGridService(repo)
    state = service.generate_alignment_grid(target_sample_rate=100)
    assert state.is_calculated

# Integration test orchestration
def test_collection_alignment_workflow():
    collection = SignalCollection()
    collection.generate_alignment_grid()
    collection.apply_grid_alignment()
    # Test the orchestration
```

---

## File Structure (Post-Refactoring)

```
src/sleep_analysis/core/
├── signal_collection.py          # 300 lines - orchestrator
├── repositories/
│   └── signal_repository.py       # 200 lines
├── services/
│   ├── query_service.py          # 150 lines
│   ├── metadata_manager.py        # 200 lines
│   ├── alignment_grid_service.py  # 250 lines
│   ├── epoch_grid_service.py      # 150 lines
│   ├── alignment_executor.py      # 100 lines
│   ├── combination_service.py     # 300 lines
│   ├── operation_executor.py      # 300 lines
│   ├── import_service.py          # 80 lines
│   └── reporter.py               # 150 lines
├── models/
│   ├── alignment_state.py         # 20 lines
│   ├── epoch_state.py             # 20 lines
│   └── combination_result.py      # 15 lines
└── metadata_handler.py            # unchanged
```

---

## Benefits Summary

### For Developers
- Easier to understand (300 line class vs 1,971 line)
- Easier to test (unit test individual services)
- Easier to debug (issues isolated to specific service)
- Easier to extend (add new service without modifying existing ones)

### For Code Quality
- Follows SOLID principles
- Better encapsulation
- Lower coupling
- Higher cohesion
- Better code reusability

### For Maintenance
- Lower change risk (change one service)
- Easier onboarding (learn one service at a time)
- Better git history (changes scattered across services)
- Easier refactoring (small files = easier to refactor)

### For Performance
- No overhead (same operations, just organized)
- Better memory efficiency (state can be garbage collected)
- Same runtime performance

---

## Risk Assessment

### Low Risk Areas
- SignalRepository (straightforward CRUD)
- MetadataManager (clear delegation)
- DataImportService (thin wrapper)
- State classes (simple data holders)

### Medium Risk Areas
- SignalQueryService (complex filtering logic)
- AlignmentGridService (complex calculations)
- SignalCombinationService (complex pandas operations)

### High Risk Areas
- OperationExecutor (registry management, dynamic dispatch)
- Backward compatibility (API unchanged, but underlying structure different)

### Mitigation
- Comprehensive unit tests (90%+ coverage)
- Thorough integration tests
- Staged rollout with feature flags
- Keep old code as fallback initially

---

## Effort Estimate

- **Implementation**: 3-4 weeks
- **Testing**: 1-2 weeks
- **Documentation**: 1 week
- **Integration & deployment**: 1 week
- **Total**: ~6-8 weeks for experienced team

---

## Decision Points

### Should We Do This?

**YES, if:**
- Code is becoming hard to test
- Development is slowing down
- Multiple developers working on same file
- Onboarding new developers is slow
- Test coverage is hard to achieve

**MAYBE, if:**
- Code is stable and rarely changes
- Project has short lifespan
- Team is very small

**NO, if:**
- Code will be rewritten soon
- Project is in maintenance mode
- No budget for refactoring

---

## Next Steps

1. **Review** this proposal with the team
2. **Decide** whether to proceed
3. **Design** detailed interfaces for each service
4. **Implement** services one by one
5. **Test** comprehensively
6. **Deploy** with monitoring

---

## References

- Full analysis: `SIGNAL_COLLECTION_REFACTORING_ANALYSIS.md`
- God Object pattern: https://en.wikipedia.org/wiki/God_object
- SOLID principles: https://en.wikipedia.org/wiki/SOLID
- Single Responsibility: https://en.wikipedia.org/wiki/Single-responsibility_principle

