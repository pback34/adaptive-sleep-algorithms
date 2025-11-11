# SomnuSight Test Suite

This directory contains the test suite for the SomnuSight sleep analysis framework.

## Test Structure

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── test_signal_collection.py  # SignalCollection tests
│   ├── test_import_service.py     # ImportService tests (Phase 1)
│   ├── test_alignment_service.py  # AlignmentService tests (Phase 1)
│   ├── test_feature_service.py    # FeatureService tests (Phase 1)
│   ├── test_signals.py            # Signal class tests
│   ├── test_importers.py          # Importer tests
│   ├── test_metadata.py           # Metadata tests
│   ├── test_metadata_handler.py   # MetadataHandler tests
│   ├── test_export.py             # Export functionality tests
│   └── ...
├── integration/                   # Integration tests
│   └── test_export_workflow.py    # End-to-end workflow tests
├── conftest.py                    # Shared pytest fixtures
└── README.md                      # This file
```

## Prerequisites

### Install Dependencies

The test suite requires pytest and related testing tools. Install them using:

```bash
pip install pytest pytest-cov pytest-mock
```

Or install all development dependencies:

```bash
pip install -e ".[dev]"
```

If `pyproject.toml` doesn't include dev dependencies, you can install manually:

```bash
pip install pytest>=7.0.0 pytest-cov>=4.0.0 pytest-mock>=3.10.0
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/

# Or using python module syntax
python -m pytest tests/
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run tests for a specific module
pytest tests/unit/test_import_service.py
pytest tests/unit/test_alignment_service.py
pytest tests/unit/test_feature_service.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/unit/test_import_service.py::TestImportServiceInit

# Run a specific test function
pytest tests/unit/test_import_service.py::TestImportServiceInit::test_init_default_handler
```

### Verbose Output

```bash
# Show detailed test output
pytest tests/ -v

# Show even more detail including print statements
pytest tests/ -vv

# Show test names without running them
pytest tests/ --collect-only
```

### Test Coverage

```bash
# Run tests with coverage report
pytest tests/ --cov=sleep_analysis --cov-report=html

# View coverage in terminal
pytest tests/ --cov=sleep_analysis --cov-report=term

# Generate coverage report and open in browser
pytest tests/ --cov=sleep_analysis --cov-report=html && open htmlcov/index.html
```

### Run Tests in Parallel

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests using multiple cores
pytest tests/ -n auto
```

## Phase 1 Refactoring Tests

The Phase 1 refactoring extracted service classes from SignalCollection. New test files were added to verify this refactoring:

### Testing ImportService

```bash
# Run all ImportService tests
pytest tests/unit/test_import_service.py -v

# Test specific functionality
pytest tests/unit/test_import_service.py::TestImportSignalsFromSource -v
pytest tests/unit/test_import_service.py::TestUpdateTimeSeriesMetadata -v
```

**Key test areas:**
- Service initialization with custom/default handlers
- Importing from single files and file patterns
- Metadata updates for TimeSeriesSignals and Features
- Enum field conversion from strings
- Error handling for invalid inputs

### Testing AlignmentService

```bash
# Run all AlignmentService tests
pytest tests/unit/test_alignment_service.py -v

# Test specific functionality
pytest tests/unit/test_alignment_service.py::TestGenerateAlignmentGrid -v
pytest tests/unit/test_alignment_service.py::TestApplyGridAlignment -v
```

**Key test areas:**
- Target sample rate detection and selection
- Reference time calculation and alignment
- Grid index generation
- Applying alignment to signals
- Combining aligned signals with merge_asof

### Testing FeatureService

```bash
# Run all FeatureService tests
pytest tests/unit/test_feature_service.py -v

# Test specific functionality
pytest tests/unit/test_feature_service.py::TestGenerateEpochGrid -v
pytest tests/unit/test_feature_service.py::TestApplyMultiSignalOperation -v
```

**Key test areas:**
- Epoch grid generation from configuration
- Time range overrides
- Multi-signal operation application
- Metadata propagation to features
- Feature combination

### Running All Phase 1 Tests

```bash
# Run all service tests together
pytest tests/unit/test_import_service.py \
       tests/unit/test_alignment_service.py \
       tests/unit/test_feature_service.py \
       -v
```

## Verifying Backward Compatibility

The Phase 1 refactoring maintains backward compatibility. Verify this by running existing tests:

```bash
# Test that SignalCollection still works (now delegates to services)
pytest tests/unit/test_signal_collection.py -v

# Test that workflows still work end-to-end
pytest tests/integration/ -v
```

All existing tests should pass, confirming that the public API remains unchanged.

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_metadata()`: Sample metadata dictionary
- `sample_dataframe()`: Sample pandas DataFrame with timestamps

Additional fixtures are defined in individual test files for specific needs.

## Writing New Tests

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` or `Test<Functionality>`
- Test methods: `test_<what_is_being_tested>`

### Example Test Structure

```python
"""Tests for MyModule."""

import pytest
from sleep_analysis.my_module import MyClass


@pytest.fixture
def my_fixture():
    """Description of what this fixture provides."""
    return MyClass()


class TestMyClassInit:
    """Tests for MyClass initialization."""

    def test_basic_init(self, my_fixture):
        """Test basic initialization."""
        assert my_fixture is not None

    def test_init_with_params(self):
        """Test initialization with parameters."""
        obj = MyClass(param="value")
        assert obj.param == "value"


class TestMyClassMethod:
    """Tests for MyClass.method()."""

    def test_method_success(self, my_fixture):
        """Test successful method execution."""
        result = my_fixture.method()
        assert result is not None

    def test_method_error(self, my_fixture):
        """Test method error handling."""
        with pytest.raises(ValueError):
            my_fixture.method(invalid_param=True)
```

## Continuous Integration

Tests should be run automatically in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=sleep_analysis
```

## Troubleshooting

### Import Errors

If you get import errors when running tests:

```bash
# Make sure the package is installed in development mode
pip install -e .

# Or add the src directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Missing Dependencies

If tests fail due to missing dependencies:

```bash
# Check which packages are needed
grep "import" tests/unit/test_*.py | cut -d: -f2 | sort -u

# Install missing packages
pip install <missing_package>
```

### Fixture Not Found

If pytest can't find a fixture:

1. Check that `conftest.py` is in the correct directory
2. Ensure the fixture is defined in scope (file, class, or session)
3. Check fixture name spelling matches exactly

### Test Timeouts

If tests hang or timeout:

```bash
# Add timeout to prevent hanging
pytest tests/ --timeout=300

# Install pytest-timeout first
pip install pytest-timeout
```

## Best Practices

1. **Test one thing per test**: Each test should verify a single behavior
2. **Use descriptive names**: Test names should clearly describe what they test
3. **Arrange-Act-Assert**: Structure tests with setup, execution, and verification
4. **Use fixtures**: Avoid duplicating setup code across tests
5. **Mock external dependencies**: Use `unittest.mock` for external services
6. **Test edge cases**: Include tests for error conditions and boundary values
7. **Keep tests independent**: Tests should not depend on execution order
8. **Update tests with code**: When changing code, update related tests

## Getting Help

- Check pytest documentation: https://docs.pytest.org/
- Review existing tests for examples
- Ask questions in project issues or discussions

## Summary of Phase 1 Test Coverage

| Service Class      | Test File                    | Test Classes | Key Scenarios Covered |
|--------------------|------------------------------|--------------|----------------------|
| ImportService      | test_import_service.py       | 5            | 18 test methods      |
| AlignmentService   | test_alignment_service.py    | 6            | 20 test methods      |
| FeatureService     | test_feature_service.py      | 6            | 15 test methods      |

**Total new tests: 53 test methods across 3 new test files**

These tests verify:
✅ Service initialization
✅ Core functionality (import, alignment, features)
✅ Error handling and edge cases
✅ Integration with SignalCollection
✅ Backward compatibility
