# Testing Guide - Phase 1 Refactoring Verification

This guide helps you verify that the Phase 1 refactoring (service extraction) works correctly.

## Quick Start

### 1. Install Test Dependencies

```bash
# Option 1: Using pip
pip install pytest pytest-cov pytest-mock

# Option 2: If dev dependencies are in pyproject.toml
pip install -e ".[dev]"

# Option 3: Install from requirements file (if exists)
pip install -r requirements-dev.txt
```

### 2. Verify Installation

```bash
# Check pytest is installed
pytest --version

# Should see something like: pytest 7.4.3
```

### 3. Run Quick Verification

```bash
# Run syntax check on service files
python -m py_compile src/sleep_analysis/services/*.py

# Output: (no output means success)
```

### 4. Run Service Tests

```bash
# Run all Phase 1 service tests
pytest tests/unit/test_import_service.py \
       tests/unit/test_alignment_service.py \
       tests/unit/test_feature_service.py \
       -v

# Expected output: All tests should PASS
```

## Detailed Testing Instructions

### Test the ImportService

**What it tests:** Signal and metadata import functionality

```bash
# Run all import service tests
pytest tests/unit/test_import_service.py -v

# Run specific test categories
pytest tests/unit/test_import_service.py::TestImportSignalsFromSource -v
pytest tests/unit/test_import_service.py::TestUpdateTimeSeriesMetadata -v
pytest tests/unit/test_import_service.py::TestUpdateFeatureMetadata -v
```

**Expected results:**
- ✅ 18 tests should pass
- ✅ Tests verify importing from files, updating metadata, enum conversion
- ✅ Error handling for invalid inputs

### Test the AlignmentService

**What it tests:** Signal alignment and grid generation

```bash
# Run all alignment service tests
pytest tests/unit/test_alignment_service.py -v

# Run specific test categories
pytest tests/unit/test_alignment_service.py::TestGenerateAlignmentGrid -v
pytest tests/unit/test_alignment_service.py::TestApplyGridAlignment -v
pytest tests/unit/test_alignment_service.py::TestAlignAndCombineSignals -v
```

**Expected results:**
- ✅ 20 tests should pass
- ✅ Tests verify grid generation, alignment application, signal combination
- ✅ Validates sample rate detection and timezone handling

### Test the FeatureService

**What it tests:** Feature extraction and epoch grid management

```bash
# Run all feature service tests
pytest tests/unit/test_feature_service.py -v

# Run specific test categories
pytest tests/unit/test_feature_service.py::TestGenerateEpochGrid -v
pytest tests/unit/test_feature_service.py::TestApplyMultiSignalOperation -v
pytest tests/unit/test_feature_service.py::TestCombineFeatures -v
```

**Expected results:**
- ✅ 15 tests should pass
- ✅ Tests verify epoch grid generation, multi-signal operations, feature combination
- ✅ Validates metadata propagation

## Verify Backward Compatibility

**Important:** The refactoring should not break existing functionality.

```bash
# Test that SignalCollection still works (delegates to services)
pytest tests/unit/test_signal_collection.py -v

# Test workflow executor
pytest tests/unit/test_workflow_executor.py -v

# Run all integration tests
pytest tests/integration/ -v
```

**Expected results:**
- ✅ All existing tests should still pass
- ✅ No changes to public API
- ✅ SignalCollection methods delegate correctly to services

## Manual Testing

If you prefer manual testing or don't have pytest installed:

### Test 1: Import Service

```python
from sleep_analysis.services import ImportService
from sleep_analysis.core.metadata_handler import MetadataHandler

# Create service
service = ImportService()
print("✓ ImportService created successfully")

# Test with custom handler
handler = MetadataHandler(default_values={"test": "value"})
service = ImportService(handler)
print("✓ ImportService accepts custom MetadataHandler")
```

### Test 2: Alignment Service

```python
from sleep_analysis.services import AlignmentService
from sleep_analysis.core.signal_collection import SignalCollection
import pandas as pd
from sleep_analysis.signals.ppg_signal import PPGSignal
from sleep_analysis.signal_types import SignalType
import uuid

# Create service and collection
service = AlignmentService()
collection = SignalCollection(metadata={"timezone": "UTC"})

# Add a test signal
index = pd.date_range(start="2023-01-01", periods=100, freq="10ms", tz="UTC")
data = pd.DataFrame({"value": range(100)}, index=index)
data.index.name = 'timestamp'
signal = PPGSignal(
    data=data,
    metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
)
collection.add_time_series_signal("ppg_0", signal)

# Test alignment grid generation
service.generate_alignment_grid(collection, target_sample_rate=10.0)
assert collection._alignment_params_calculated is True
assert collection.target_rate == 10.0
print("✓ AlignmentService generates alignment grid successfully")
```

### Test 3: Feature Service

```python
from sleep_analysis.services import FeatureService
from sleep_analysis.core.signal_collection import SignalCollection
import pandas as pd
from sleep_analysis.signals.ppg_signal import PPGSignal
from sleep_analysis.signal_types import SignalType
import uuid

# Create service and collection with epoch config
service = FeatureService()
collection = SignalCollection(metadata={
    "timezone": "UTC",
    "epoch_grid_config": {
        "window_length": "10s",
        "step_size": "10s"
    }
})

# Add a test signal
index = pd.date_range(start="2023-01-01", periods=100, freq="1s", tz="UTC")
data = pd.DataFrame({"value": range(100)}, index=index)
data.index.name = 'timestamp'
signal = PPGSignal(
    data=data,
    metadata={"signal_id": str(uuid.uuid4()), "signal_type": SignalType.PPG}
)
collection.add_time_series_signal("ppg_0", signal)

# Test epoch grid generation
service.generate_epoch_grid(collection)
assert collection._epoch_grid_calculated is True
assert len(collection.epoch_grid_index) == 10  # 100s / 10s
print("✓ FeatureService generates epoch grid successfully")
```

## Test Coverage Report

Generate a detailed coverage report:

```bash
# Generate HTML coverage report
pytest tests/unit/test_*service.py --cov=sleep_analysis.services --cov-report=html

# Open in browser (macOS)
open htmlcov/index.html

# Open in browser (Linux)
xdg-open htmlcov/index.html

# View in terminal
pytest tests/unit/test_*service.py --cov=sleep_analysis.services --cov-report=term
```

**Target coverage for Phase 1:**
- ImportService: > 85%
- AlignmentService: > 85%
- FeatureService: > 85%

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'pytest'"

**Solution:**
```bash
pip install pytest pytest-cov pytest-mock
```

### Problem: "ModuleNotFoundError: No module named 'sleep_analysis'"

**Solution:**
```bash
# Install package in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Problem: "ImportError: cannot import name 'ImportService'"

**Solution:**
```bash
# Check that service files exist
ls -la src/sleep_analysis/services/

# Re-install package
pip install -e . --force-reinstall
```

### Problem: Tests hang or timeout

**Solution:**
```bash
# Add timeout (requires pytest-timeout)
pip install pytest-timeout
pytest tests/ --timeout=60
```

### Problem: Tests fail with "fixture not found"

**Solution:**
```bash
# Make sure conftest.py exists
ls -la tests/conftest.py

# Run from project root directory
cd /path/to/adaptive-sleep-algorithms
pytest tests/
```

## Success Criteria

Phase 1 refactoring is successful if:

✅ **All 53 new service tests pass**
- ImportService: 18 tests
- AlignmentService: 20 tests
- FeatureService: 15 tests

✅ **All existing tests still pass**
- SignalCollection tests pass (backward compatibility)
- Integration tests pass (end-to-end workflows work)

✅ **No regression in functionality**
- Public API unchanged
- Workflows execute correctly
- Results identical to pre-refactoring

## Next Steps

After verifying Phase 1:

1. **Review the design evaluation**: See `DESIGN_EVALUATION.md` for identified issues
2. **Plan Phase 2**: Simplify operation registry system
3. **Continue refactoring**: Address remaining design issues

## Getting Help

If tests fail or you encounter issues:

1. Check the error message carefully
2. Review the test code to understand what's being tested
3. Check the service implementation
4. Consult `tests/README.md` for detailed test documentation
5. Create an issue in the project repository

## Summary

```bash
# One command to verify everything
pytest tests/unit/test_import_service.py \
       tests/unit/test_alignment_service.py \
       tests/unit/test_feature_service.py \
       tests/unit/test_signal_collection.py \
       -v --cov=sleep_analysis.services --cov-report=term

# Expected: All tests pass, coverage > 85%
```

**Congratulations!** If all tests pass, Phase 1 refactoring is successfully verified. 🎉
