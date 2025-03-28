I'll add those new rules to the coding guidelines document. Here's the updated version:

# Coding Guidelines

This document outlines the rules for implementing the sleep analysis framework in `src/sleep_analysis/`. All code must adhere to these guidelines to ensure consistency with `docs/requirements/requirements.md`.

---

## Rule 1: Operation Application in `apply_operation`
- **Description**: The `apply_operation` method must use a hybrid approach to handle operations:
  - First, check if the operation exists as an instance method using `getattr(self, operation_name, None)`.
  - If a callable method is found, invoke it directly with the provided `inplace` flag and `**parameters`.
  - If no method is found, fall back to the class registry to locate and execute the operation.
- **Rationale**: This provides an intuitive method-based API (e.g., `signal.filter_lowpass()`) while retaining the flexibility of a registry system, per Section 8.2.5 of the requirements.
- **Example**:
  ```python
  def apply_operation(self, operation_name: str, inplace: bool = False, **parameters) -> 'SignalData':
      method = getattr(self, operation_name, None)
      if method and callable(method):
          return method(inplace=inplace, **parameters)
      # Fallback to registry
      registry = self.__class__.get_registry()
      if operation_name not in registry:
          raise ValueError(f"Operation '{operation_name}' not found")
      func, output_class = registry[operation_name]
      # Proceed with registry logic
  ```

## Rule 2: Placement of Operations
- **Description**: Processing methods must be defined in the **highest parent class** where they are universally applicable to all subclasses:
  - Generic operations (e.g., `filter_lowpass`, `downsample`) belong in `TimeSeriesSignal`.
  - Signal-specific operations (e.g., `compute_heart_rate`) belong in their respective classes (e.g., `PPGSignal`).
- **Rationale**: This minimizes code duplication and maximizes reusability across the signal hierarchy, aligning with Section 8.1 of the requirements.
- **Example**:
  - `filter_lowpass` is implemented in `TimeSeriesSignal` because it applies to all time-series signals.
  - `compute_heart_rate` is implemented in `PPGSignal` because it's specific to PPG signals.

## Rule 3: No Legacy Support Unless Specified
- **Description**: Do not add support for "legacy" functions, classes, or data structures unless explicitly instructed in the task or requirements.
- **Rationale**: For new codebases, backward compatibility is not necessary unless specified. Adding unnecessary legacy support complicates the code and maintenance.
- **Example**:
  - Do not create classes like `LegacyImporter` or functions like `old_import_method` without a clear requirement for backward compatibility.

## Rule 4: Declarative Design Principles
- **Description**: Always resort to declarative design principles whenever possible.
- **Rationale**: Declarative designs are more maintainable, easier to understand, and reduce side effects in the codebase.
- **Example**:
  - Prefer configuration-driven approaches over imperative code.
  - Use descriptive data structures that define "what" should happen rather than "how" it should happen.
  - Implement operations as pure functions where possible, avoiding side effects.

## Rule 5: Common Utility Functions
- **Description**: Develop common utility functions whenever possible to avoid code duplication.
- **Rationale**: Common utilities increase maintainability, ensure consistent behavior, and facilitate reuse across the codebase.
- **Example**:
  - Create a `utils.py` module with commonly used signal processing functions.
  - Implement shared validation logic in a central location.
  - Extract repeated operations into dedicated helper functions even if used in only a few places.

## Additional Guidelines
### Implementing Operations
- Prefer instance methods for core operations integral to a signal's interface.
- Use the registry for ad-hoc or plugin-like operations not central to the signal's core functionality.
- All methods must support `inplace` where applicable and update metadata.

### Metadata Management
- Append an `OperationInfo` object to `metadata.operations` with the operation name and parameters.
- For non-inplace operations, update `derived_from` to reference the source signal.

### Error Handling
- Raise `ValueError` if an operation isn't found or if `inplace=True` is unsupported.