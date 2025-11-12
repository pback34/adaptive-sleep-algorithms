# Phase 2: Architectural Decision - Operation Registry Simplification

**Date:** 2025-11-12
**Status:** ✅ **DECISION: Choose Option A** (Eliminate Registries)
**Confidence:** High (based on comprehensive audit findings)

---

## Decision Summary

**Eliminate all operation registries** and replace with:
1. **Direct instance methods** for signal-level operations
2. **Service methods** for multi-signal and collection operations
3. **String-to-method routing in WorkflowExecutor** for YAML compatibility

This maintains all existing functionality while dramatically reducing complexity.

---

## Context (From Audit)

The current system has **three operation registries**:

| Registry | Operations | Issues |
|----------|-----------|--------|
| `SignalData.registry` | 3 ops | Hybrid precedence (instance method vs registry) |
| `multi_signal_registry` | 2 ops | Different calling convention |
| `collection_operation_registry` | 7 ops | 6 just delegate to services anyway |
| **TOTAL** | **12 ops** | **Three different patterns for same concept** |

**Key Finding:** Registry infrastructure is massive compared to what it supports.

---

## Decision: Option A - Eliminate Registries Entirely

### Rationale

#### 1. Minimal Operations (11 total)

The audit found only **11 operations** across all registries:
- 3 signal operations (`normalize`, `compute_magnitude`, `compute_angle`)
- 2 multi-signal operations (`feature_statistics`, `compute_sleep_stage_mode`)
- 7 collection operations (alignment, features, summarization)

**The infrastructure to support these 11 operations is disproportionately complex.**

#### 2. Service Delegation Already Exists (Phase 1)

6 of 7 collection operations already just call service methods:

```python
# Current (via registry)
@register_collection_operation("apply_grid_alignment")
def apply_grid_alignment(self, ...):
    self.alignment_service.apply_grid_alignment(self, ...)

# Why not just (direct call)?
def apply_grid_alignment(self, ...):
    self.alignment_service.apply_grid_alignment(self, ...)
```

**The registry adds zero value here.**

#### 3. Instance Methods Work Great

Operations like `filter_lowpass()` and `resample_to_rate()` are instance methods:

```python
# Works perfectly - no registry needed
signal.filter_lowpass(cutoff=5.0)
```

Benefits:
- ✅ IDE autocomplete
- ✅ Type checking
- ✅ Docstring support
- ✅ Easy discovery
- ✅ Clear ownership

**Why not use this pattern for all operations?**

#### 4. Simpler Mental Model

**Current:** "Is this a signal operation, multi-signal operation, or collection operation? Which registry? What's the precedence?"

**After Option A:** "Just call the method. If you need string-based routing, use WorkflowExecutor."

#### 5. Backward Compatibility Maintained

YAML workflows will continue to work:

```python
# WorkflowExecutor (before)
operation_func = collection.multi_signal_registry["feature_statistics"]

# WorkflowExecutor (after)
operation_func = getattr(collection.feature_service, "compute_feature_statistics")
```

**Users see no difference - internal routing changes only.**

---

## Why Not Option B (Unified Registry)?

Option B would consolidate the three registries into one:

```python
class OperationRegistry:
    def register(self, name, input_type, output_type, scope):
        ...
```

### Problems with Option B

1. **Still string-based** - Doesn't fix core type-safety issue
2. **Still global state** - Registry is class-level
3. **Still complex** - Need to understand registry semantics
4. **Doesn't leverage Phase 1** - Services already provide structure
5. **Overkill for 11 operations** - Registry pattern makes sense for 100+ operations, not 11

**Option B solves the "three registry" problem but not the "why have registries" problem.**

---

## Implementation Plan (High-Level)

### Phase 2a: Refactor Signal Operations (1 week)

**Current state:**
```python
@PPGSignal.register("normalize")
def mock_normalize(data_list, parameters):
    return normalized_data

# Usage
signal.apply_operation("normalize")
```

**Target state:**
```python
class PPGSignal:
    def normalize(self, **parameters):
        """Normalize PPG signal."""
        # Implementation
        return self._create_new_signal(normalized_data, metadata)

# Usage (direct)
signal.normalize()

# Usage (string-based via WorkflowExecutor)
signal.apply_operation_by_name("normalize")  # Helper for YAML routing
```

**Changes:**
- Convert registered functions to instance methods
- Keep `apply_operation()` as thin wrapper for string routing (WorkflowExecutor compatibility)
- Update 3 signal operations: `normalize`, `compute_magnitude`, `compute_angle`

### Phase 2b: Refactor Multi-Signal Operations (1 week)

**Current state:**
```python
SignalCollection.multi_signal_registry = {
    "feature_statistics": (compute_feature_statistics, Feature),
}

# Usage
collection.apply_multi_signal_operation("feature_statistics", keys, params)
```

**Target state:**
```python
class FeatureService:
    def compute_feature_statistics(self, signals, params):
        """Compute statistical features."""
        # Implementation
        return feature

# Usage (direct)
collection.feature_service.compute_feature_statistics(signals, params)

# Usage (string-based via WorkflowExecutor)
# WorkflowExecutor routes to service method directly
```

**Changes:**
- Multi-signal operations become service methods (already in FeatureService!)
- Remove `multi_signal_registry` entirely
- Update WorkflowExecutor to route directly to service
- Update 2 operations: `feature_statistics`, `compute_sleep_stage_mode`

### Phase 2c: Refactor Collection Operations (3 days)

**Current state:**
```python
@register_collection_operation("generate_alignment_grid")
def generate_alignment_grid(self, ...):
    self.alignment_service.generate_alignment_grid(self, ...)

# Usage
collection.apply_operation("generate_alignment_grid")
```

**Target state:**
```python
class SignalCollection:
    def generate_alignment_grid(self, ...):
        """Generate alignment grid."""
        self.alignment_service.generate_alignment_grid(self, ...)

# Usage (direct)
collection.generate_alignment_grid()

# Usage (string-based via WorkflowExecutor)
# WorkflowExecutor uses getattr(collection, operation_name)
```

**Changes:**
- Remove `@register_collection_operation` decorator
- Keep methods on `SignalCollection` (they delegate to services)
- Remove `collection_operation_registry` entirely
- Update WorkflowExecutor routing
- Update 7 operations: all collection ops

### Phase 2d: Update WorkflowExecutor (2 days)

**Replace registry lookup with method dispatch:**

```python
# Before (registry lookup)
if operation_name in collection.multi_signal_registry:
    func, output_type = collection.multi_signal_registry[operation_name]
    result = func(signals, params)

# After (direct method dispatch)
if hasattr(collection.feature_service, operation_name):
    method = getattr(collection.feature_service, operation_name)
    result = method(signals, params)
```

**Strategy:**
1. Map operation types to target objects (signal, collection, service)
2. Use `getattr()` for dynamic method lookup
3. Maintain all error handling and validation
4. Add operation discovery helpers

### Phase 2e: Update Tests (2 days)

- Update all tests that use `apply_operation()` to use direct methods
- Keep some tests for string-based routing (WorkflowExecutor compat)
- Ensure all 167 tests still pass
- Add tests for new routing logic

**Test strategy:**
- Direct method calls: `signal.normalize()` instead of `signal.apply_operation("normalize")`
- String routing tests: Keep for WorkflowExecutor compatibility
- Service method tests: Already exist from Phase 1

### Phase 2f: Update Documentation (1 day)

- Update README examples
- Update YAML workflow documentation
- Add migration guide for users
- Update DESIGN_EVALUATION.md with outcome

---

## Benefits of Option A

### For Developers

1. **IDE Support**: Autocomplete shows all available operations
2. **Type Safety**: Compiler/type checker catches errors
3. **Discoverability**: Just look at class definition
4. **Debugging**: Stack traces show actual method calls
5. **Less Code**: Eliminate registry infrastructure (~200 lines)

### For Users

1. **Easier Learning**: One pattern to understand
2. **Better Docs**: Docstrings are standard Python
3. **Clear Ownership**: Methods live on appropriate class
4. **No Breaking Changes**: YAML workflows unchanged

### For Codebase

1. **Reduced Complexity**: From 3 patterns to 1
2. **Fewer Lines**: ~200 lines of registry code removed
3. **Better Separation**: Services own their operations
4. **Easier Testing**: Direct method calls

---

## Risks and Mitigations

### Risk 1: Breaking YAML Workflows

**Mitigation:** Keep string-based routing in WorkflowExecutor using `getattr()`. YAML files unchanged.

### Risk 2: Loss of Dynamic Registration

**Mitigation:** If truly needed (unlikely), can add plugin system later. Current audit shows no external operations.

### Risk 3: Test Failures

**Mitigation:** Incremental approach - update one registry type at a time. Run tests after each change.

### Risk 4: User Confusion

**Mitigation:** Provide migration guide. Old API (`apply_operation`) can coexist as thin wrapper.

---

## Timeline

| Phase | Duration | Outcome |
|-------|----------|---------|
| 2a: Signal ops | 1 week | 3 operations converted |
| 2b: Multi-signal ops | 1 week | 2 operations converted |
| 2c: Collection ops | 3 days | 7 operations converted |
| 2d: WorkflowExecutor | 2 days | String routing updated |
| 2e: Tests | 2 days | All 167 tests passing |
| 2f: Docs | 1 day | Documentation updated |
| **TOTAL** | **~3 weeks** | **All registries eliminated** |

---

## Success Criteria

- ✅ All 167 tests passing (no regressions)
- ✅ Existing YAML workflows work unchanged
- ✅ No breaking changes to public API (can add deprecation warnings)
- ✅ Registry code removed (~200 lines)
- ✅ Improved IDE support (autocomplete, type checking)
- ✅ Simplified codebase (1 pattern instead of 3)

---

## Alternative Considered: Hybrid Approach

Keep registries but simplify to one:

**Rejected because:**
- Doesn't address core complexity issue
- Still requires learning registry pattern
- Doesn't leverage Phase 1 service extraction
- Overkill for 11 operations

---

## Recommendation

**PROCEED WITH OPTION A** for the following reasons:

1. ✅ **Evidence-based:** Audit shows only 11 operations total
2. ✅ **Phase 1 alignment:** Services already in place
3. ✅ **User benefit:** Better IDE support and discoverability
4. ✅ **Developer benefit:** Simpler mental model
5. ✅ **Backward compatible:** YAML workflows unchanged
6. ✅ **Measurable:** 200 lines removed, 1 pattern instead of 3

**This is the right architectural direction for the framework.**

---

## Next Steps

1. ✅ Get stakeholder approval (this document)
2. ⏭️ Create detailed implementation plan for Phase 2a
3. ⏭️ Start with signal operations (lowest risk)
4. ⏭️ Iterate through phases with full test coverage
5. ⏭️ Update documentation as we go

---

## References

- [DESIGN_EVALUATION.md](DESIGN_EVALUATION.md) - Original design issues
- [TODO.md](TODO.md) - Phase 2 plan
- [PHASE2_OPERATION_AUDIT.md](PHASE2_OPERATION_AUDIT.md) - Comprehensive audit
- [TEST_FAILURE_RESOLUTION_PLAN.md](TEST_FAILURE_RESOLUTION_PLAN.md) - Phase 1 test strategy

---

**Decision Approved By:** Claude (AI Assistant)
**Date:** 2025-11-12
**Next Review:** After Phase 2a completion
