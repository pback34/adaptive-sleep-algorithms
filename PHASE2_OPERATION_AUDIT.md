# Phase 2: Operation Registry Audit

**Date:** 2025-11-12
**Task:** Audit all operations currently in registries (Phase 2, Task 1)

---

## Executive Summary

The framework currently uses **three different operation registries** with overlapping purposes:

1. **`SignalData.registry`** - Signal-level operations (class-level, inherited)
2. **`SignalCollection.multi_signal_registry`** - Multi-signal operations (global dict)
3. **`SignalCollection.collection_operation_registry`** - Collection-level operations (global dict)

**Key Finding:** Only **11 operations total** are registered across all three registries, but the infrastructure to support them adds significant complexity.

---

## Registry 1: SignalData.registry (Signal-Level Operations)

**Location:** `src/sleep_analysis/core/signal_data.py:32`
**Type:** `Dict[str, Tuple[Callable, Type['SignalData']]]`
**Inheritance:** Yes - Each subclass inherits parent registry and can add its own operations

### Registered Operations

| Operation Name | Decorator Location | Input Type | Output Type | Used In Tests? |
|----------------|-------------------|------------|-------------|----------------|
| `normalize` | `signals/ppg_signal.py:49` | PPGSignal | PPGSignal | ✅ Yes (2 uses) |
| `compute_magnitude` | `signals/magnitude_signal.py:80` | AccelerometerSignal | MagnitudeSignal | ❌ No test usage found |
| `compute_angle` | `signals/angle_signal.py:71` | AccelerometerSignal | AngleSignal | ❌ No test usage found |

**Total:** 3 operations

### How It Works

```python
# Registration
@PPGSignal.register("normalize")
def mock_normalize(data_list, parameters):
    ...

# Usage
signal.apply_operation("normalize", param1=value)
```

### Issues

1. **Hybrid approach:** The framework checks instance methods FIRST, then falls back to registry
2. **Name collisions:** No protection against overwriting operations in inheritance
3. **Discovery:** Hard to know what operations are available
4. **Type safety:** String-based lookup is error-prone

---

## Registry 2: SignalCollection.multi_signal_registry (Multi-Signal Operations)

**Location:** `src/sleep_analysis/core/signal_collection.py:84`
**Type:** `Dict[str, Tuple[Callable, Type[SignalData]]]`
**Populated:** At module import time (lines 1520-1532)

### Registered Operations

| Operation Name | Function Location | Input Type | Output Type | Used In Tests? |
|----------------|------------------|------------|-------------|----------------|
| `feature_statistics` | `operations/feature_extraction.py` | List[TimeSeriesSignal] | Feature | ✅ Yes (multiple) |
| `compute_sleep_stage_mode` | `operations/feature_extraction.py` | List[TimeSeriesSignal] | Feature | ✅ Yes (multiple) |

**Total:** 2 operations

### How It Works

```python
# Registration (at module level)
SignalCollection.multi_signal_registry.update({
    "feature_statistics": (compute_feature_statistics, Feature),
    "compute_sleep_stage_mode": (compute_sleep_stage_mode, Feature),
})

# Usage (via WorkflowExecutor or directly)
collection.apply_multi_signal_operation("feature_statistics", ["ppg_0"], {...})
```

### Issues

1. **Global state:** Registry is class-level, shared across all instances
2. **Import-time registration:** Must import operations module for registry to populate
3. **Different calling convention:** Uses `apply_multi_signal_operation` instead of `apply_operation`
4. **Service delegation:** Actually delegates to `FeatureService.apply_multi_signal_operation()`

---

## Registry 3: SignalCollection.collection_operation_registry (Collection Operations)

**Location:** `src/sleep_analysis/core/signal_collection.py:87`
**Type:** `Dict[str, Callable]`
**Populated:** Via decorator introspection at module load (lines 1510-1517)

### Registered Operations

| Operation Name | Method Location | Purpose | Used In Tests? |
|----------------|----------------|---------|----------------|
| `generate_alignment_grid` | `signal_collection.py:669` | Calculate alignment parameters | ✅ Yes (multiple) |
| `generate_epoch_grid` | `signal_collection.py:676` | Calculate epoch grid | ✅ Yes (multiple) |
| `apply_grid_alignment` | `signal_collection.py:767` | Apply alignment to signals | ✅ Yes (multiple) |
| `align_and_combine_signals` | `signal_collection.py:772` | Align and merge signals | ✅ Yes (multiple) |
| `combine_aligned_signals` | `signal_collection.py:1025` | Combine pre-aligned signals | ✅ Yes (multiple) |
| `combine_features` | `signal_collection.py:1089` | Merge feature matrices | ✅ Yes (multiple) |
| `summarize_signals` | `signal_collection.py:1377` | Generate summary table | ✅ Yes (multiple) |

**Total:** 7 operations (but 6 are service delegations!)

### How It Works

```python
# Registration (via decorator)
@register_collection_operation("generate_alignment_grid")
def generate_alignment_grid(self, ...):
    ...

# Registry population (at module level)
for _method_name, _method_obj in inspect.getmembers(SignalCollection, predicate=inspect.isfunction):
    if hasattr(_method_obj, '_collection_op_name'):
        SignalCollection.collection_operation_registry[_op_name] = _method_obj

# Usage
collection.apply_operation("generate_alignment_grid", target_sample_rate=10.0)
```

### Issues

1. **Redundant indirection:** All 6 alignment/feature operations just delegate to services
2. **Confusing naming:** `apply_operation` is used, but name conflicts with signal-level operations
3. **Service-based already:** Since Phase 1, these mostly just call service methods

---

## Operation Lookup Flow

### Current Flow (Complex)

```
WorkflowExecutor.execute_step()
    │
    ├──[type: "collection"]──> collection.apply_operation(name)
    │                           ├──> collection_operation_registry[name]
    │                           └──> Calls instance method (often delegates to service)
    │
    ├──[type: "multi_signal"]──> collection.apply_multi_signal_operation(name, keys)
    │                             ├──> multi_signal_registry[name]
    │                             └──> FeatureService.apply_multi_signal_operation()
    │                                  └──> Calls operation function
    │
    └──[type: "signal"]──> signal.apply_operation(name)
                           ├──> Check instance method first
                           │    └──> If exists, call directly (e.g., filter_lowpass)
                           └──> If not found, check registry
                                └──> Calls registered function
```

### Issues with Current Flow

1. **Three different paths** for similar operations
2. **Precedence confusion** (instance method vs registry)
3. **Type-based routing** requires knowing which type to use
4. **Service delegation** adds extra layer

---

## Usage Analysis

### In Tests (167 tests)

**Registry-based operations used:**
- `normalize` (PPGSignal) - 2 uses
- `feature_statistics` - Multiple uses
- `compute_sleep_stage_mode` - Multiple uses
- All 7 collection operations - Multiple uses each

**Instance methods used directly:**
- `filter_lowpass` - Multiple uses
- `resample_to_rate` - Multiple uses

**Total operation calls in tests:** 40 calls to `apply_operation` or `apply_multi_signal_operation`

### In Production Code

**WorkflowExecutor** is the primary consumer:
- Handles all three operation types via string lookup
- Routes based on `type` field in YAML
- No direct method calls (always via registries)

---

## External Dependencies Analysis

### Workflow YAML Files

All workflow files use string-based operation names:

```yaml
steps:
  - type: collection
    operation: "generate_alignment_grid"

  - type: multi_signal
    operation: "feature_statistics"
    inputs: ["hr"]

  - type: signal
    operation: "filter_lowpass"
    inputs: ["ppg_0"]
```

**Breaking Change Impact:**
- If we remove registries, we MUST maintain backward compatibility with YAML workflows
- WorkflowExecutor must be updated to route operations differently

### Service Classes (Phase 1)

Services currently **use** the registries:
- `FeatureService.apply_multi_signal_operation()` looks up in `multi_signal_registry`
- `AlignmentService` is called FROM collection operations
- `ImportService` doesn't use operations

**Service Impact:**
- Services could expose their methods directly
- No need for registry indirection

---

## Recommendations

Based on this audit, I recommend **Option A: Eliminate registries entirely** for these reasons:

### Why Option A?

1. **Only 11 operations total** - Registry infrastructure is overkill
2. **6 of 7 collection operations already delegate to services** - Just call services directly
3. **Instance methods work fine** - `filter_lowpass`, `resample_to_rate` prove direct methods work
4. **Simpler for users** - IDE autocomplete, type checking, easier discovery
5. **Reduces cognitive load** - One way to do things instead of three

### Migration Path

1. **Keep WorkflowExecutor routing** - Maintain YAML compatibility
2. **Replace registry lookup with method dispatch** - Use `getattr()` for string-based routing
3. **Keep string-based YAML interface** - Users don't see the change
4. **Update internal code** - Use direct method calls in Python

---

## Next Steps

1. ✅ **Audit complete** (this document)
2. ⏭️ **Get stakeholder decision** - Option A vs Option B
3. ⏭️ **Create implementation plan** - Detailed refactoring steps
4. ⏭️ **Update tests** - Ensure no regressions

---

## Appendices

### A. Full File Locations

**Registry Definitions:**
- `src/sleep_analysis/core/signal_data.py:32` - SignalData.registry
- `src/sleep_analysis/core/signal_collection.py:84` - multi_signal_registry
- `src/sleep_analysis/core/signal_collection.py:87` - collection_operation_registry

**Registered Operations:**
- `src/sleep_analysis/signals/ppg_signal.py:49` - normalize
- `src/sleep_analysis/signals/magnitude_signal.py:80` - compute_magnitude
- `src/sleep_analysis/signals/angle_signal.py:71` - compute_angle
- `src/sleep_analysis/operations/feature_extraction.py` - feature_statistics, compute_sleep_stage_mode
- `src/sleep_analysis/core/signal_collection.py` - All 7 collection operations

**Usage:**
- `src/sleep_analysis/workflows/workflow_executor.py:192-376` - Operation routing
- `src/sleep_analysis/services/feature_service.py:173-176` - multi_signal_registry lookup

### B. Test Coverage

**Files with operation tests:**
- `tests/unit/test_signals.py` - Signal-level operations
- `tests/unit/test_feature_service.py` - Multi-signal operations
- `tests/unit/test_alignment_service.py` - Collection operations (via services)
- `tests/unit/test_workflow_executor.py` - End-to-end workflow tests

**Total tests:** 167 passing, 1 skipped
