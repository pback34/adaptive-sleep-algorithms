# SomnuSight Framework - Design Evaluation

## Executive Summary

This evaluation identifies fundamental design issues in the SomnuSight sleep analysis framework. While the framework demonstrates thoughtful architecture in many areas (strong use of enums, metadata traceability, separation of concerns), there are **significant opportunities for improvement** that would enhance maintainability, testability, and adherence to clean architecture principles.

**Overall Assessment**: The framework shows signs of organic growth without periodic refactoring, resulting in:
- Bloated central classes with too many responsibilities
- Confusing operation registration and lookup mechanisms
- Tight coupling between workflow execution and core domain logic
- Awkward separation between Features and TimeSeriesSignals
- Global state management that complicates testing

---

## Critical Issues (Priority 1)

### 1. SignalCollection is a "God Object" ⚠️ **HIGHEST PRIORITY**

**Problem**: `SignalCollection` has grown to over 1000+ lines and handles far too many responsibilities:

```
SignalCollection responsibilities:
├── Signal storage and retrieval (✓ appropriate)
├── Feature storage and retrieval (✓ appropriate)
├── Signal importing via importers (❌ business logic)
├── Metadata updates (❌ should delegate to handler)
├── Multi-signal operation execution (❌ orchestration)
├── Alignment grid generation (❌ domain service)
├── Epoch grid generation (❌ domain service)
├── Feature combination (❌ business logic)
├── Signal alignment and combination (❌ business logic)
├── Export configuration (❌ presentation concern)
└── Summary generation (❌ reporting logic)
```

**Evidence from code** (signal_collection.py:71-141):
- 9 different stored state attributes beyond the core dictionaries
- Methods for import, alignment, feature extraction, export, summarization
- Collection-level operation registry mixed with container responsibilities

**Impact**:
- Violates Single Responsibility Principle
- Extremely difficult to unit test (requires massive setup)
- Changes in any concern affect the entire class
- Hard to understand what SignalCollection's true purpose is

**Recommendation**:
Extract separate service classes:
```python
# Core - Keep in SignalCollection
- add_time_series_signal()
- add_feature()
- get_signal() / get_signals()

# Extract to SignalImportService
- import_signals_from_source()
- update_time_series_metadata()

# Extract to AlignmentService
- generate_alignment_grid()
- apply_grid_alignment()
- align_and_combine_signals()

# Extract to FeatureService
- generate_epoch_grid()
- apply_multi_signal_operation()
- combine_features()

# Extract to ExportService (already exists but collection has config logic)
- set_index_config()
- set_feature_index_config()
```

---

### 2. Confusing Multi-Level Operation Registry System ⚠️

**Problem**: The framework has **three different operation registries** with overlapping purposes and confusing lookup semantics:

1. **`SignalData.registry`** - Instance method operations on signals
2. **`SignalCollection.multi_signal_registry`** - Multi-signal operations (e.g., feature extraction)
3. **`SignalCollection.collection_operation_registry`** - Collection-level operations

**Evidence** from time_series_signal.py:206-260:
```python
def apply_operation(self, operation_name: str, ...):
    # First check instance method
    method = getattr(self, operation_name, None)
    if method is not None:
        if callable(method):
            core_logic_callable = method
            is_method = True

    # Then check registry
    if core_logic_callable is None:
        registry = self.__class__.get_registry()
        if operation_name in registry:
            func_registered, output_class_registered = registry[operation_name]
```

**Problems**:
1. **Unclear precedence**: Instance method vs registry - which should win?
2. **Different calling conventions**: Methods take `**parameters`, registry functions take `[dataframes], parameters`
3. **Output class confusion**: Registry stores output class, but methods use decorators
4. **Hard to discover**: Where is an operation defined? Method? Registry? Collection?
5. **Inheritance complexity**: `get_registry()` has complex MRO traversal logic (signal_data.py:128-167)

**Impact**:
- Steep learning curve for framework users
- Difficult to debug when operation not found
- Hard to extend with custom operations
- Registry pattern benefits unclear vs just using instance methods

**Recommendation**:
**Option A (Simpler)**: Eliminate registries entirely, use only instance methods:
```python
class TimeSeriesSignal:
    def filter_lowpass(self, cutoff: float) -> pd.DataFrame:
        """Returns filtered data"""
        ...

    def resample(self, rate: float) -> pd.DataFrame:
        """Returns resampled data"""
        ...
```

**Option B (If registry needed)**: Single registry with clear semantics:
```python
class OperationRegistry:
    """Single registry for ALL operations"""

    def register(self,
                 name: str,
                 input_type: Type,
                 output_type: Type,
                 scope: OperationScope):  # SIGNAL, MULTI_SIGNAL, COLLECTION
        ...

    def lookup(self, name: str, input_type: Type) -> Operation:
        """Unambiguous lookup"""
        ...
```

---

### 3. Feature vs TimeSeriesSignal Dichotomy is Awkward 🔧

**Problem**: Features and TimeSeriesSignals are stored separately but treated inconsistently:

**Evidence**:
- `SignalCollection` has two dictionaries: `time_series_signals` and `features`
- `get_signals()` searches both but method name implies time series
- `Feature` doesn't inherit from `SignalData` but has similar interface
- Operations like `apply_operation()` only work on TimeSeriesSignals
- Feature extraction returns `Feature` objects but they can't be further processed

**From signal_collection.py:319-410**:
```python
def get_signals(self, ...):
    """Retrieve TimeSeriesSignals and/or Features..."""
    search_space = {**self.time_series_signals, **self.features}  # Combine both!
```

**Inconsistencies**:
```python
# These work:
signal.apply_operation("filter_lowpass", cutoff=5.0)

# These don't:
feature.apply_operation("normalize")  # Feature has no apply_operation!
```

**Impact**:
- Users confused about when to use which
- Can't chain operations through features
- Awkward to work with feature extraction results
- Type checking becomes complicated

**Recommendation**:
**Option A**: Make Feature inherit from SignalData properly
```python
class SignalData(ABC):
    """Base for all data types"""

class TimeSeriesSignal(SignalData):
    """Continuous time series"""

class EpochFeature(SignalData):
    """Epoch-based features - also a signal!"""
    # Can use apply_operation, fits in same collection seamlessly
```

**Option B**: Separate completely and rename methods clearly
```python
collection.get_time_series_signals()  # Only returns TimeSeriesSignal
collection.get_features()              # Only returns Feature
collection.get_all()                   # Returns both with clear Union type
```

---

## Significant Issues (Priority 2)

### 4. WorkflowExecutor is Not a "Thin Coordinator" 🔧

**Problem**: Despite being documented as a "thin coordinator", `WorkflowExecutor` contains substantial business logic:

**From workflow_executor.py - 761 lines total**:
- Complex timezone resolution logic (lines 62-88)
- Import section processing with path resolution (lines 465-565)
- Export configuration translation (lines 567-683)
- Visualization processing with backend selection (lines 685-760)
- Special-case handling for deprecated operations (lines 194-205)
- Output key generation with regex matching (lines 345-358)

**Example of excessive logic** (lines 345-358):
```python
match = re.match(r"(.+)_(\d+)$", source_key)
if match:
    source_index = match.group(2)
    current_output_key = f"{output_key}_{source_index}"
else:
    current_output_key = output_key
```

**Impact**:
- Business logic mixed with orchestration
- Hard to test individual concerns
- Difficult to reuse logic outside workflows
- Framework tightly coupled to YAML format

**Recommendation**:
Extract helper services and reduce executor to pure orchestration:
```python
class WorkflowExecutor:
    def __init__(self,
                 collection: SignalCollection,
                 import_service: ImportService,
                 processing_service: ProcessingService,
                 export_service: ExportService):
        self.collection = collection
        self.import_service = import_service
        self.processing_service = processing_service
        self.export_service = export_service

    def execute_workflow(self, config: Dict):
        """Pure orchestration - delegates all logic"""
        if "import" in config:
            self.import_service.import_from_config(config["import"])
        if "steps" in config:
            self.processing_service.execute_steps(config["steps"])
        if "export" in config:
            self.export_service.export_from_config(config["export"])
```

---

### 5. Metadata Management is Overly Complex 🔧

**Problem**: Metadata handling is duplicated and inconsistent across the framework:

**Multiple metadata initialization paths**:
1. `MetadataHandler.initialize_time_series_metadata()` - For time series
2. `MetadataHandler.initialize_feature_metadata()` - For features
3. `SignalData.__init__()` - Sets defaults manually (lines 86-109)
4. `SignalCollection.update_time_series_metadata()` - Updates from workflow
5. `TimeSeriesSignal.__init__()` - Merges default units (lines 57-83)

**Evidence of duplication** (signal_data.py:86-113):
```python
# SignalData.__init__ manually initializes defaults
for field in ['derived_from', 'operations', 'source_files']:
    if field not in metadata_kwargs:
        metadata_kwargs[field] = []

for field in ['temporary', 'merged']:
    if field not in metadata_kwargs:
        metadata_kwargs[field] = False

# Feature-specific defaults (but this is SignalData, not Feature!)
for field in ['feature_names', 'source_signal_keys']:
    if field not in metadata_kwargs:
        metadata_kwargs[field] = []
```

**Problems**:
- Defaults scattered across multiple locations
- Not clear which method to use when
- Type checking complicated (Union[TimeSeriesMetadata, FeatureMetadata])
- Handler has separate required field lists for each type

**Recommendation**:
Use dataclass defaults and factory pattern:
```python
@dataclass
class TimeSeriesMetadata:
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operations: List[OperationInfo] = field(default_factory=list)
    derived_from: List[Tuple[str, int]] = field(default_factory=list)
    # ... all defaults in one place

class MetadataFactory:
    """Single responsibility: create metadata"""

    @staticmethod
    def create_time_series_metadata(**overrides) -> TimeSeriesMetadata:
        return TimeSeriesMetadata(**overrides)

    @staticmethod
    def create_feature_metadata(**overrides) -> FeatureMetadata:
        return FeatureMetadata(**overrides)
```

---

### 6. Global State in Collection Complicates Testing 🔧

**Problem**: SignalCollection stores global configuration state that affects all operations:

**From signal_collection.py:117-132**:
```python
# Alignment parameters
self.target_rate: Optional[float] = None
self.ref_time: Optional[pd.Timestamp] = None
self.grid_index: Optional[pd.DatetimeIndex] = None
self._alignment_params_calculated: bool = False

# Epoch grid parameters
self.epoch_grid_index: Optional[pd.DatetimeIndex] = None
self.global_epoch_window_length: Optional[pd.Timedelta] = None
self.global_epoch_step_size: Optional[pd.Timedelta] = None
self._epoch_grid_calculated: bool = False

# Cached dataframes
self._aligned_dataframe: Optional[pd.DataFrame] = None
self._combined_feature_matrix: Optional[pd.DataFrame] = None
```

**Problems**:
1. **Hidden state**: Operations behave differently based on whether grids are calculated
2. **Order dependencies**: Must call `generate_epoch_grid()` before feature extraction
3. **Testing complexity**: Must set up entire state machine for tests
4. **Thread safety**: Mutable global state is not thread-safe
5. **Unclear lifecycle**: When are grids invalidated? What if signals change?

**Impact**:
```python
# Example: This will fail silently or with confusing error
collection.add_time_series_signal("ppg_0", ppg)
collection.apply_multi_signal_operation("feature_statistics", ["ppg_0"], {})
# ERROR: No epoch grid calculated!

# Must do this first:
collection.generate_epoch_grid()  # Hidden prerequisite
collection.apply_multi_signal_operation("feature_statistics", ["ppg_0"], {})
```

**Recommendation**:
Make dependencies explicit through parameters:
```python
# Instead of implicit global state
def extract_features(self, signal_keys: List[str], params: Dict):
    # Uses self.epoch_grid_index implicitly
    ...

# Make it explicit
def extract_features(self,
                     signal_keys: List[str],
                     epoch_config: EpochConfig,  # Explicit!
                     params: Dict):
    grid = epoch_config.generate_grid(start, end, step)
    ...

# Or use a context manager
with collection.epoch_context(window="30s", step="30s") as ctx:
    features = ctx.extract_features(["ppg_0"], aggregations=["mean"])
```

---

## Minor Issues (Priority 3)

### 7. Timezone Handling is Scattered 🔧

Timezone logic appears in:
- `WorkflowExecutor._resolve_target_timezone()` (lines 62-88)
- Importer configurations (passed through dict updates)
- `SignalCollection.add_time_series_signal()` validation (lines 169-203)
- Various string comparisons instead of timezone-aware comparisons

**Recommendation**: Create a `TimezoneService` or `TimezoneConfig` class.

---

### 8. Parameter Sanitization Buried in MetadataHandler 🔧

**From metadata_handler.py:216-251**:
```python
def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize operation parameters for safe storage in metadata."""
    # Converts DataFrames to string representations
```

**Problem**: This is presentation logic, not metadata logic.

**Recommendation**: Move to a separate `ParameterSerializer` class.

---

### 9. String-Based Operation Lookup is Error-Prone 🔧

All operations are looked up by string names:
```python
signal.apply_operation("filter_lowpass", cutoff=5.0)  # Typo = runtime error
collection.apply_operation("generate_epoch_grid")      # No IDE support
```

**Recommendation**:
```python
# Type-safe operation objects
signal.apply_operation(Operations.FILTER_LOWPASS, cutoff=5.0)
# Or even better, just use methods:
signal.filter_lowpass(cutoff=5.0)
```

---

### 10. Operation Recording is Redundant 🔧

Operations are recorded in metadata for reproducibility, but:
- Parameters are sanitized (DataFrames → strings), losing information
- No actual replay mechanism implemented
- Adds overhead to every operation
- `derived_from` field duplicates some of this information

**Question**: Is this feature actually used? If not, consider removing or simplifying.

---

## Positive Aspects ✅

The framework does several things well:

1. **Strong Type System**: Excellent use of Enums for signal types, sensor types, etc.
2. **Metadata Traceability**: Comprehensive metadata for each signal
3. **Pluggable Visualization**: Clean abstraction over Bokeh/Plotly
4. **Importer Abstraction**: Good separation for different data formats
5. **Dataclass Usage**: Modern Python with dataclasses for metadata
6. **Logging**: Comprehensive logging throughout

---

## Recommended Refactoring Sequence

### Phase 1: Extract Services (Weeks 1-2)
1. Create `ImportService` and move import logic from `SignalCollection`
2. Create `AlignmentService` for alignment grid operations
3. Create `FeatureService` for feature extraction and epoch grids
4. Update `WorkflowExecutor` to use services

### Phase 2: Simplify Operations (Weeks 3-4)
1. Decide: Keep registries or use methods only?
2. If keeping registries, consolidate into single `OperationRegistry`
3. Document operation lookup semantics clearly
4. Add operation discovery/introspection tools

### Phase 3: Fix Type Hierarchy (Week 5)
1. Make `Feature` inherit from `SignalData` OR
2. Completely separate concerns and rename methods
3. Ensure consistent interface across all signal types

### Phase 4: Remove Global State (Week 6)
1. Make epoch/alignment config explicit parameters
2. Add configuration validation at workflow parse time
3. Simplify `SignalCollection` to pure container

---

## Conclusion

The SomnuSight framework has a solid foundation but suffers from **complexity creep** typical of evolving codebases. The most critical issue is the **SignalCollection god object** (Priority 1), which should be refactored into focused service classes.

The **registry pattern complexity** (Priority 1) provides unclear benefits over simple instance methods and should be simplified or removed.

These changes would:
- ✅ Improve testability dramatically
- ✅ Make the codebase more maintainable
- ✅ Reduce cognitive load for new developers
- ✅ Enable better IDE support and type checking
- ✅ Follow SOLID principles more closely

**Estimated effort**: 6-8 weeks for complete refactoring, but can be done incrementally without breaking existing workflows.
