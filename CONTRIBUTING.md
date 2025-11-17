# Contributing to Sleep Analysis Framework

Thank you for your interest in contributing to the Sleep Analysis Framework! This document provides guidelines for contributing code, documentation, and bug reports.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing Requirements](#testing-requirements)
8. [Documentation](#documentation)
9. [Reporting Bugs](#reporting-bugs)
10. [Suggesting Enhancements](#suggesting-enhancements)

---

## Code of Conduct

This project follows standard open-source community guidelines:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility and apologize for mistakes

---

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Git**: For version control
- **Virtual environment**: Recommended for isolated development

### Ways to Contribute

- **Code**: Bug fixes, new features, performance improvements
- **Documentation**: Tutorials, guides, API documentation, examples
- **Testing**: Add test cases, improve test coverage
- **Bug Reports**: Identify and report issues
- **Feature Requests**: Suggest new functionality
- **Code Reviews**: Review pull requests from other contributors

---

## Development Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub (use the "Fork" button)

# Clone your fork
git clone https://github.com/YOUR_USERNAME/adaptive-sleep-algorithms.git
cd adaptive-sleep-algorithms

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/adaptive-sleep-algorithms.git
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install package in editable mode with all dependencies
pip install -e ".[dev]"

# This includes:
# - Core dependencies (numpy, pandas, pyyaml, etc.)
# - Visualization (bokeh, plotly)
# - Algorithms (scikit-learn, scipy, etc.)
# - Export (h5py, tables, cairosvg, etc.)
# - Testing (pytest, pytest-cov)
# - Code quality (black, isort, mypy)
```

### 4. Verify Installation

```bash
# Run all tests
pytest tests/ -v

# Expected: 211 tests passing (1 may be skipped for pytables)

# Check code style
black --check src/
isort --check-only src/
```

### 5. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-new-feature

# Or for bug fixes
git checkout -b fix/bug-description
```

---

## Making Changes

### Branch Naming Convention

Use descriptive branch names:

- `feature/add-xyz` - For new features
- `fix/issue-123` - For bug fixes (reference issue number)
- `docs/improve-readme` - For documentation updates
- `test/add-coverage` - For test improvements
- `refactor/cleanup-xyz` - For code refactoring

### Commit Messages

Write clear, descriptive commit messages:

**Format**:
```
<type>: <short summary> (50 chars or less)

<optional detailed description>

<optional footer with issue references>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring without behavior changes
- `perf`: Performance improvements
- `style`: Code style changes (formatting, etc.)
- `chore`: Maintenance tasks

**Examples**:

```bash
# Good commit messages
git commit -m "feat: add Spearman correlation for feature extraction"

git commit -m "fix: handle missing HRV data in Polar imports

Previously, missing HRV values would cause import to fail.
Now gracefully handles NaN values as expected for Polar devices.

Fixes #123"

git commit -m "docs: add troubleshooting guide for common errors"

git commit -m "test: add unit tests for movement feature extraction"
```

### Code Changes

1. **Follow existing code style** (see [Coding Standards](#coding-standards))
2. **Write tests** for new functionality
3. **Update documentation** if changing public APIs
4. **Keep changes focused** - one feature/fix per PR

---

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code formatted: `black src/ tests/`
- [ ] Imports sorted: `isort src/ tests/`
- [ ] Type hints added for new code
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (if applicable)

### 2. Create Pull Request

```bash
# Push your branch to your fork
git push origin feature/my-new-feature
```

Then create a PR on GitHub from your branch to `main`.

### 3. PR Title and Description

**Title**: Clear, concise summary of changes
```
feat: Add gradient boosting sleep staging algorithm
fix: Resolve timezone handling for DST transitions
docs: Create quick start guide for new users
```

**Description template**:
```markdown
## Description
Brief description of what this PR does and why.

## Changes
- Bullet list of specific changes
- Made to the codebase

## Testing
How the changes were tested:
- [ ] Added new unit tests
- [ ] All existing tests pass
- [ ] Manually tested with sample data

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing change)

## Related Issues
Closes #123
Relates to #456
```

### 4. Code Review Process

1. **Automated checks** will run (tests, linting)
2. **Maintainers will review** your code
3. **Address feedback** by pushing new commits
4. **Once approved**, a maintainer will merge

**During review**:
- Respond to feedback promptly
- Ask questions if feedback is unclear
- Make requested changes or discuss alternatives
- Be open to suggestions

### 5. After Merge

```bash
# Update your local repository
git checkout main
git pull upstream main

# Delete feature branch
git branch -d feature/my-new-feature
git push origin --delete feature/my-new-feature
```

---

## Coding Standards

All code must adhere to the guidelines in [`docs/coding_guidelines.md`](docs/coding_guidelines.md).

### Key Principles

1. **Follow existing patterns** in the codebase
2. **Use type hints** for function parameters and returns
3. **Write docstrings** for all public functions and classes
4. **Declarative over imperative** - prefer configuration-driven approaches
5. **DRY principle** - extract common utilities to avoid duplication
6. **Signal encapsulation** - use `apply_operation()` instead of modifying `._data`

### Code Style

**Python version**: Python 3.8+ compatible

**Formatting**: Use Black with default settings
```bash
black src/ tests/
```

**Import sorting**: Use isort
```bash
isort src/ tests/
```

**Line length**: 88 characters (Black default)

**Type hints**: Required for new code
```python
def compute_feature(
    signal: TimeSeriesSignal,
    window_length: pd.Timedelta,
    parameters: Dict[str, Any]
) -> Feature:
    """Compute features from signal."""
    ...
```

### Docstring Style

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief one-line description.

    More detailed description if needed. Can span multiple lines
    and include usage examples.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param2 is negative.

    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    """
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` - `SignalCollection`, `TimeSeriesSignal`
- **Functions/Methods**: `snake_case` - `compute_hrv_features`, `apply_operation`
- **Constants**: `UPPER_SNAKE_CASE` - `DEFAULT_WINDOW_LENGTH`, `MAX_RETRIES`
- **Private members**: `_leading_underscore` - `_data`, `_validate_input`

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 90% for new code
- **All new features** must include tests
- **Bug fixes** should include regression tests

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src.sleep_analysis --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_sleep_features.py -v

# Run specific test
pytest tests/unit/test_sleep_features.py::test_hrv_features -v

# Run with verbose output
pytest tests/ -vv -s
```

### Test Structure

Place tests in appropriate directories:

```
tests/
├── unit/               # Unit tests for individual components
│   ├── test_signals.py
│   ├── test_features.py
│   └── ...
├── integration/        # Integration tests for workflows
│   ├── test_workflow_executor.py
│   └── ...
└── conftest.py        # Shared fixtures
```

### Writing Tests

**Use pytest fixtures** for setup:

```python
import pytest
import pandas as pd
from sleep_analysis.signals.heart_rate_signal import HeartRateSignal

@pytest.fixture
def sample_hr_signal():
    """Create a sample heart rate signal for testing."""
    data = pd.DataFrame({
        'hr': [60, 62, 64, 63, 61]
    }, index=pd.date_range('2024-01-01', periods=5, freq='1s'))

    return HeartRateSignal(
        data=data,
        metadata={
            'name': 'test_hr',
            'signal_id': 'test_id',
            'sampling_rate': 1.0
        }
    )

def test_hrv_computation(sample_hr_signal):
    """Test HRV feature computation."""
    # Test implementation
    ...
```

**Test naming**: `test_<functionality>_<scenario>`

```python
def test_hrv_features_basic_metrics():
    """Test computation of basic HRV metrics."""
    ...

def test_hrv_features_missing_data():
    """Test HRV handling with missing data."""
    ...

def test_hrv_features_invalid_window():
    """Test HRV with invalid window length."""
    ...
```

**Use descriptive assertions**:

```python
# Good
assert len(features.data) == 10, "Should generate 10 epochs"
assert 'hr_mean' in features.data.columns, "Should include hr_mean feature"

# Less helpful
assert len(features.data) == 10
```

### Test Types

1. **Unit tests**: Test individual functions/methods in isolation
2. **Integration tests**: Test workflows and component interactions
3. **Regression tests**: Ensure bugs don't reappear
4. **Edge case tests**: Test boundary conditions and error handling

---

## Documentation

### What to Document

1. **New features**: Add usage examples to relevant docs
2. **API changes**: Update API documentation
3. **Breaking changes**: Clearly document in CHANGELOG.md
4. **Complex logic**: Add inline comments explaining why (not what)

### Documentation Structure

```
docs/
├── quick-start.md           # Getting started guide
├── troubleshooting.md       # Common issues and solutions
├── data-preparation.md      # Data format requirements
├── feature_extraction_plan.md
├── coding_guidelines.md
└── requirements/
    └── requirements.md      # Detailed requirements
```

### Updating Documentation

- **README.md**: Update if adding major features
- **Docstrings**: Update for any API changes
- **Examples**: Add workflow examples for new features
- **CHANGELOG.md**: Document user-facing changes

### Example Workflow Documentation

When adding new operations, include workflow examples:

```yaml
# Example: Using the new gradient_boosting_sleep_staging operation
steps:
  - type: multi_signal
    operation: "gradient_boosting_sleep_staging"
    inputs: ["combined_features"]
    parameters:
      n_estimators: 200
      learning_rate: 0.1
      max_depth: 5
    output: "sleep_predictions"
```

---

## Reporting Bugs

### Before Reporting

1. **Check existing issues** to avoid duplicates
2. **Verify bug** with latest version
3. **Gather information** (see below)

### Bug Report Template

```markdown
**Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Create workflow with '...'
2. Run command '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Minimal Example**
Minimal workflow YAML and data that reproduces the issue.

**Environment**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g., 3.10.5]
- Package version: [output of `pip show sleep_analysis`]
- Installed packages: [output of `pip list`]

**Error Messages**
```
Full error traceback here
```

**Additional Context**
Any other relevant information.
```

### Critical vs. Non-Critical Bugs

**Critical** (report immediately):
- Data corruption or loss
- Security vulnerabilities
- Complete feature failure

**Non-critical** (can wait for triage):
- Minor UI issues
- Performance degradations
- Edge case failures

---

## Suggesting Enhancements

### Enhancement Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How would you implement this feature?

**Alternatives Considered**
What other approaches did you consider?

**Additional Context**
Mockups, examples, related issues, etc.
```

### Enhancement Criteria

Features should:
- Align with project goals (sleep analysis focus)
- Be broadly useful (not niche/single-user)
- Be maintainable (reasonable complexity)
- Have clear requirements
- Include test strategy

---

## Development Workflow Example

Here's a complete example of contributing a bug fix:

```bash
# 1. Set up environment
git clone https://github.com/YOUR_USERNAME/adaptive-sleep-algorithms.git
cd adaptive-sleep-algorithms
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 2. Create branch
git checkout -b fix/timezone-dst-handling

# 3. Make changes
# Edit files, add tests, update docs

# 4. Run tests
pytest tests/ -v
black src/ tests/
isort src/ tests/

# 5. Commit changes
git add src/sleep_analysis/utils/__init__.py tests/unit/test_timezone.py
git commit -m "fix: handle DST transitions in timezone conversion

Previously, timezone conversions during DST transitions would produce
incorrect timestamps. Now uses pytz localization to handle DST correctly.

Fixes #234"

# 6. Push and create PR
git push origin fix/timezone-dst-handling
# Create PR on GitHub

# 7. Address review feedback
# Make changes, commit, push again

# 8. After merge, clean up
git checkout main
git pull upstream main
git branch -d fix/timezone-dst-handling
```

---

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Feature ideas**: Open a GitHub Issue with [Enhancement] tag
- **Security issues**: Email maintainers directly (do not open public issue)

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes for significant contributions
- Acknowledged in relevant documentation

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

**Thank you for contributing to the Sleep Analysis Framework!**

Your contributions help make sleep research more accessible and reproducible.

---

**Last Updated**: 2025-11-17
