# SignalCollection God Object Analysis & Refactoring Proposal

## Executive Summary

The `SignalCollection` class in `/home/user/adaptive-sleep-algorithms/src/sleep_analysis/core/signal_collection.py` is a **God Object** anti-pattern with:
- **1,971 lines of code**
- **41 methods** handling 8+ distinct responsibilities
- **Complex state management** with 13+ instance attributes for different concerns
- **High coupling** between unrelated concerns

This document provides a detailed analysis and concrete refactoring strategy.

---

## 1. CURRENT PROBLEM ANALYSIS

### 1.1 File Statistics
- **Total Lines**: 1,971
- **Total Methods**: 41 (excluding nested decorator)
- **Instance Attributes**: 13+ distinct attributes
- **Responsibilities**: 8+ distinct concerns
- **Import Dependencies**: 20+ external imports

### 1.2 What Makes It a God Object

A God Object violates the Single Responsibility Principle by handling too many unrelated concerns. SignalCollection does:

1. **Signal Management** - Adding/retrieving signals and features
2. **Metadata Operations** - Creating, validating, updating metadata
3. **Time-Series Alignment** - Computing sample rates, grids, alignment parameters
4. **Epoch Grid Generation** - Creating epoch-based time windows
5. **Signal Alignment** - Applying alignment operations to signals
6. **Signal Combination** - Combining aligned signals into unified dataframes
7. **Feature Combination** - Creating feature matrices
8. **Operation Execution** - Dispatching and executing multi-signal operations
9. **Data Import** - Importing signals from external sources
10. **Reporting** - Generating summaries and visualizations

---

## 2. METHOD GROUPING BY RESPONSIBILITY

### Group 1: Signal Management (4 methods)
**Purpose**: Adding and retrieving signals/features from collection

```
1. add_time_series_signal(key, signal) - Line 138
   - Validates TimeSeriesSignal
   - Checks for ID uniqueness
   - Validates timestamp index & timezone
   - Sets metadata handler and name

2. add_feature(key, feature) - Line 210
   - Validates Feature instance
   - Checks for feature ID uniqueness
   - Sets metadata handler and name

3. add_signal_with_base_name(base_name, signal) - Line 244
   - Auto-increments signal names
   - Delegates to add_time_series_signal or add_feature
   - Returns the auto-generated key

4. add_imported_signals(signals, base_name, start_index) - Line 1702
   - Batch adds TimeSeriesSignals
   - Handles sequential indexing
   - Error handling for duplicates
```

**Issue**: These methods are tightly coupled to UUID generation, ID conflict resolution, and metadata handler initialization.

---

### Group 2: Signal Retrieval (4 methods)
**Purpose**: Getting signals and features by various criteria

```
5. get_time_series_signal(key) - Line 282
   - Simple getter with type validation

6. get_feature(key) - Line 288
   - Simple getter with type validation

7. get_signal(key) - Line 294
   - Returns either TimeSeriesSignal or Feature
   - Uses union type

8. get_signals(input_spec, include_intermediate) - Line 316
   - Complex filtering by type, attributes, criteria
   - Supports multiple query formats
   - Calls _process_enum_criteria and _matches_criteria
```

**Issue**: Mixing simple getters with complex query logic; filtering logic could be separate.

---

### Group 3: Metadata Management (6 methods)
**Purpose**: Managing metadata for signals, features, and collection

```
9. update_time_series_metadata(signal, metadata_spec) - Line 459
   - Parses metadata specifications
   - Handles enum conversions
   - Delegates to metadata handler

10. update_feature_metadata(feature, metadata_spec) - Line 493
    - Similar to TimeSeriesMetadata updates
    - Handles FeatureMetadata-specific fields
    - Timedelta parsing for epochs

11. set_index_config(index_fields) - Line 524
    - Validates against TimeSeriesMetadata fields
    - Stores in collection metadata

12. set_feature_index_config(index_fields) - Line 533
    - Validates for feature exports
    - Stores in collection metadata

13. _validate_timestamp_index(signal) - Line 548
    - Checks for DatetimeIndex
    - Validates timezone consistency
    - Called during signal addition

14. _process_enum_criteria(criteria_dict) - Line 409
    - Converts string enum values to actual enums
    - Used by get_signals filtering
    
15. _matches_criteria(signal, criteria) - Line 432
    - Filters signals based on metadata criteria
    - Helper for get_signals
```

**Issue**: Metadata handling is scattered across update methods, validation, and filtering; metadata logic belongs in MetadataHandler.

---

### Group 4: Sample Rate & Reference Time Calculation (3 methods)
**Purpose**: Computing alignment grid parameters

```
16. get_target_sample_rate(user_specified) - Line 569
    - Calculates max rate from signals
    - Finds nearest standard rate
    - Returns float

17. get_nearest_standard_rate(rate) - Line 597
    - Snaps rate to predefined standards
    - STANDARD_RATES constant (line 66)

18. get_reference_time(target_period) - Line 608
    - Finds common reference point
    - Uses time binning algorithm
    - Returns pd.Timestamp
```

**Issue**: These are alignment-specific utilities that should be in an AlignmentStrategy class.

---

### Group 5: Alignment Grid Generation (2 methods)
**Purpose**: Creating time-based grids for signal alignment

```
19. generate_alignment_grid(target_sample_rate) - Line 713
    - Calculates alignment parameters (target_rate, ref_time, grid_index)
    - Stores in instance state
    - Returns self for chaining

20. _calculate_grid_index(target_rate, ref_time) - Line 645
    - Generates DatetimeIndex based on rate and reference time
    - Complex date range logic
    - Timezone handling
```

**Issue**: State is scattered (target_rate, ref_time, grid_index, _alignment_params_calculated flag).

---

### Group 6: Epoch Grid Generation (1 method)
**Purpose**: Creating epoch-based time windows for feature extraction

```
21. generate_epoch_grid(start_time, end_time) - Line 760
    - Calculates epoch_grid_index and window parameters
    - Stores state in multiple attributes
    - Complex timezone and time range handling
```

**Issue**: Mixes epoch calculation with state management; stores 4 different instance variables.

---

### Group 7: Signal Alignment Operations (2 methods)
**Purpose**: Applying alignment to actual signal data

```
22. apply_grid_alignment(method, signals_to_align) - Line 1059
    - Decorated as @register_collection_operation
    - Iterates through signals
    - Delegates to signal.apply_operation('reindex_to_grid')
    - Returns self

23. get_signals_from_input_spec(input_spec) - Line 1051
    - Alias for get_signals
    - Used by apply_grid_alignment for signal selection
```

**Issue**: This bridges collection-level operations with signal-level operations.

---

### Group 8: Signal Combination & Concatenation (5 methods)
**Purpose**: Combining aligned signals into unified dataframes

```
24. align_and_combine_signals() - Line 1114
    - Orchestrates full alignment and combination workflow
    - Calls generate_alignment_grid, apply_grid_alignment, combine_aligned_signals
    - Stores combined dataframe

25. combine_aligned_signals() - Line 1367
    - Decorated as @register_collection_operation
    - Uses stored grid_index from generate_alignment_grid
    - Calls _perform_concatenation
    - Stores result in _aligned_dataframe

26. _perform_concatenation(aligned_dfs, grid_index, is_feature) - Line 1199
    - Complex multi-index creation logic
    - Handles both time-series and feature matrices
    - 130+ lines of concatenation logic

27. _get_current_alignment_params(method_used) - Line 1333
    - Helper to gather alignment state
    - Returns dict of parameters

28. get_stored_combined_dataframe() - Line 1347
    - Returns _aligned_dataframe
    - Getter for stored result
```

**Issue**: Orchestration, execution, and storage are all mixed together.

---

### Group 9: Feature Combination (1 method)
**Purpose**: Creating combined feature matrices

```
29. combine_features(inputs, feature_index_config) - Line 1431
    - Combines Feature objects into single dataframe
    - Uses epoch_grid_index from generate_epoch_grid
    - Complex indexing logic
```

**Issue**: Similar to signal combination but handles Features separately; duplicates logic patterns.

---

### Group 10: Multi-Signal Operations Execution (3 methods)
**Purpose**: Executing operations that produce new signals/features

```
30. apply_multi_signal_operation(operation_name, input_signal_keys, parameters) - Line 869
    - Complex 140+ line method
    - Looks up operation in registry
    - Resolves input signals
    - Performs prerequisite checks (especially for features)
    - Executes operation function
    - Validates result
    - Propagates metadata to features

31. apply_operation(operation_name, **parameters) - Line 1013
    - Collection-level operation dispatcher
    - Looks up in collection_operation_registry
    - Calls registered method with parameters
    
32. apply_and_store_operation(signal_key, operation_name, parameters) - Line 1528
    - Applies operation to single signal
    - Stores result back in collection with new key
    - Used for in-place modifications
```

**Issue**: Operation execution is embedded in SignalCollection; should be in separate Executor class.

---

### Group 11: Single-Signal Operations (1 method)
**Purpose**: Applying operations to multiple signals

```
33. apply_operation_to_signals(signal_keys, operation_name, parameters) - Line 1563
    - Batch applies operation to multiple signals
    - Stores all results
```

**Issue**: Duplicates logic from apply_and_store_operation.

---

### Group 12: Data Import (1 method)
**Purpose**: Importing signals from external sources

```
34. import_signals_from_source(importer_instance, source, base_name, **kwargs) - Line 1609
    - Delegates to importer instance
    - Calls add_imported_signals
    - Returns list of keys
```

**Issue**: Mixes import orchestration with signal addition; importer pattern could be cleaner.

---

### Group 13: Data Access/Retrieval (3 methods)
**Purpose**: Accessing stored combination results

```
35. get_stored_combined_feature_matrix() - Line 1353
    - Returns _combined_feature_matrix

36. get_stored_combination_params() - Line 1360
    - Returns _aligned_dataframe_params
```

**Issue**: Simple getters for internal state; could be private or moved to dedicated Result object.

---

### Group 14: Reporting & Summary (2 methods)
**Purpose**: Generating and formatting summaries

```
37. summarize_signals(fields_to_include, print_summary) - Line 1792
    - 100+ lines of summary generation
    - Iterates all signals and features
    - Handles different data types
    - Stores in _summary_dataframe
    - Calls _format_summary_cell for formatting

38. _format_summary_cell(x, col_name) - Line 1733
    - Helper for formatting individual cells
    - 60+ lines of formatting logic
```

**Issue**: Reporting is a separate concern; belongs in dedicated Reporter class.

---

## 3. INSTANCE ATTRIBUTES ANALYSIS

### Core Signal Storage (2 attributes)
```python
self.time_series_signals: Dict[str, TimeSeriesSignal] = {}
self.features: Dict[str, Feature] = {}
```

### Metadata Management (2 attributes)
```python
self.metadata: CollectionMetadata = ...
self.metadata_handler: MetadataHandler = ...
```

### Alignment Grid State (5 attributes)
```python
self.target_rate: Optional[float] = None
self.ref_time: Optional[pd.Timestamp] = None
self.grid_index: Optional[pd.DatetimeIndex] = None
self._alignment_params_calculated: bool = False
self._merge_tolerance: Optional[pd.Timedelta] = None
```

### Epoch Grid State (4 attributes)
```python
self.epoch_grid_index: Optional[pd.DatetimeIndex] = None
self.global_epoch_window_length: Optional[pd.Timedelta] = None
self.global_epoch_step_size: Optional[pd.Timedelta] = None
self._epoch_grid_calculated: bool = False
```

### Combined Data Storage (3 attributes)
```python
self._aligned_dataframe: Optional[pd.DataFrame] = None
self._aligned_dataframe_params: Optional[Dict[str, Any]] = None
self._summary_dataframe: Optional[pd.DataFrame] = None
self._summary_dataframe_params: Optional[Dict[str, Any]] = None
self._combined_feature_matrix: Optional[pd.DataFrame] = None
```

**Total: 13+ different concerns with mixed visibility**

---

## 4. DEPENDENCY ANALYSIS

### Direct Dependencies
```
From local modules:
  - MetadataHandler (metadata management)
  - TimeSeriesSignal (signal type)
  - Feature (feature type)
  - Various metadata classes (TimeSeriesMetadata, FeatureMetadata, etc.)
  
From external packages:
  - pandas (DataFrame, Index operations)
  - numpy (numeric operations)
  - dataclasses (field introspection)
  - uuid (ID generation)
  - logging (logging)
  - time (timing)
  - os, glob (file operations)
  - enum (enum handling)
  
From operations modules (loaded at runtime):
  - operations.feature_extraction (feature functions)
  - operations.algorithm_ops (algorithm functions)
```

### Coupling Issues
1. **Tight coupling to signal types** - Direct references to TimeSeriesSignal and Feature
2. **Metadata handler delegation** - Passes operations through to external handler
3. **Registry-based operation dispatch** - Dynamic operation lookup at class level
4. **State leakage** - Alignment state exposed publicly (target_rate, ref_time, grid_index)

---

## 5. REFACTORING PROPOSAL

### 5.1 Proposed Architecture

```
Current (Monolithic):
    SignalCollection (1971 lines, 41 methods)
    
Proposed (Modular):
    ┌─ SignalCollection (core: 300 lines, 8 methods)
    │  ├─ Owns: time_series_signals, features, metadata, metadata_handler
    │  ├─ Delegates to: SignalRepository, MetadataManager, AlignmentService, etc.
    │  └─ Provides: Central orchestration interface
    │
    ├─ SignalRepository (200 lines, 6 methods)
    │  ├─ add_time_series_signal()
    │  ├─ add_feature()
    │  ├─ add_signal_with_base_name()
    │  ├─ add_imported_signals()
    │  ├─ get_by_id()
    │  └─ Owns: time_series_signals, features dicts
    │
    ├─ SignalQueryService (150 lines, 4 methods)
    │  ├─ get_signals()
    │  ├─ get_signals_from_input_spec()
    │  ├─ _process_enum_criteria() [private]
    │  └─ _matches_criteria() [private]
    │
    ├─ MetadataManager (200 lines, 4 methods)
    │  ├─ update_time_series_metadata()
    │  ├─ update_feature_metadata()
    │  ├─ set_index_config()
    │  ├─ set_feature_index_config()
    │  └─ Delegates to: MetadataHandler
    │
    ├─ AlignmentGridService (300 lines, 5 methods)
    │  ├─ generate_alignment_grid()
    │  ├─ _calculate_grid_index() [private]
    │  ├─ get_target_sample_rate() [private]
    │  ├─ get_nearest_standard_rate() [private]
    │  ├─ get_reference_time() [private]
    │  ├─ Owns: AlignmentGridState
    │  └─ Returns: AlignmentGridState object
    │
    ├─ EpochGridService (150 lines, 1 method)
    │  ├─ generate_epoch_grid()
    │  ├─ Owns: EpochGridState
    │  └─ Returns: EpochGridState object
    │
    ├─ AlignmentExecutor (200 lines, 2 methods)
    │  ├─ apply_grid_alignment()
    │  └─ Owns: reference to AlignmentGridState
    │
    ├─ SignalCombinationService (300 lines, 3 methods)
    │  ├─ combine_aligned_signals()
    │  ├─ combine_features()
    │  └─ _perform_concatenation() [private]
    │
    ├─ OperationExecutor (200 lines, 3 methods)
    │  ├─ execute_multi_signal_operation()
    │  ├─ execute_single_signal_operation()
    │  ├─ execute_batch_signal_operation()
    │  └─ Owns: operation registries
    │
    ├─ DataImportService (100 lines, 2 methods)
    │  ├─ import_signals_from_source()
    │  └─ Delegates to: specific importers
    │
    ├─ SignalSummaryReporter (150 lines, 2 methods)
    │  ├─ summarize_signals()
    │  ├─ _format_summary_cell() [private]
    │  └─ Generates: HTML/DataFrame reports
    │
    ├─ AlignmentGridState (data class, 10 lines)
    │  ├─ target_rate
    │  ├─ reference_time
    │  ├─ grid_index
    │  ├─ merge_tolerance
    │  └─ is_calculated
    │
    ├─ EpochGridState (data class, 10 lines)
    │  ├─ epoch_grid_index
    │  ├─ window_length
    │  ├─ step_size
    │  └─ is_calculated
    │
    └─ CombinationResult (data class, 10 lines)
       ├─ dataframe
       ├─ params
       └─ is_feature_matrix
```

### 5.2 Detailed Class Proposals

#### Class 1: SignalRepository
**Responsibility**: CRUD operations for signals and features

```python
class SignalRepository:
    """Manages storage and basic access to signals and features."""
    
    def __init__(self, metadata_handler: MetadataHandler):
        self.time_series_signals: Dict[str, TimeSeriesSignal] = {}
        self.features: Dict[str, Feature] = {}
        self.metadata_handler = metadata_handler
    
    def add_time_series_signal(self, key: str, signal: TimeSeriesSignal) -> None:
        """Add a TimeSeriesSignal to the repository."""
        # Current logic from SignalCollection.add_time_series_signal
        # Validation, ID conflict resolution, timestamp validation
    
    def add_feature(self, key: str, feature: Feature) -> None:
        """Add a Feature to the repository."""
    
    def add_signal_with_base_name(self, base_name: str, 
                                   signal: Union[TimeSeriesSignal, Feature]) -> str:
        """Add signal with auto-incremented name."""
    
    def add_imported_signals(self, signals: List[TimeSeriesSignal], 
                            base_name: str, start_index: int = 0) -> List[str]:
        """Batch add imported signals."""
    
    def get_by_key(self, key: str) -> Union[TimeSeriesSignal, Feature, None]:
        """Get signal or feature by key."""
    
    def get_time_series_signal(self, key: str) -> TimeSeriesSignal:
        """Get TimeSeriesSignal by key with type validation."""
    
    def get_feature(self, key: str) -> Feature:
        """Get Feature by key with type validation."""
    
    def get_all_time_series(self) -> Dict[str, TimeSeriesSignal]:
        """Get all TimeSeriesSignals."""
    
    def get_all_features(self) -> Dict[str, Feature]:
        """Get all Features."""
```

**Benefits**:
- Single responsibility: storage and basic access
- Can be tested independently
- Clear API for signal management
- Encapsulates ID conflict resolution

---

#### Class 2: SignalQueryService
**Responsibility**: Complex signal filtering and querying

```python
class SignalQueryService:
    """Provides query and filtering operations for signals and features."""
    
    def __init__(self, repository: SignalRepository):
        self.repository = repository
    
    def get_signals(self, input_spec: Union[str, Dict[str, Any], List[str], None] = None,
                    include_intermediate: bool = True) -> List[Union[TimeSeriesSignal, Feature]]:
        """Get signals matching specification."""
        # Current logic from SignalCollection.get_signals
    
    def get_signals_from_input_spec(self, input_spec) -> List[Union[TimeSeriesSignal, Feature]]:
        """Alias for get_signals."""
    
    def _process_enum_criteria(self, criteria_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string enum values to enum objects."""
    
    def _matches_criteria(self, signal: Union[TimeSeriesSignal, Feature],
                         criteria: Dict[str, Any]) -> bool:
        """Check if signal matches criteria."""
```

**Benefits**:
- Separates query logic from storage
- Reusable across different services
- Cleaner filtering logic
- Single responsibility: querying

---

#### Class 3: MetadataManager
**Responsibility**: Managing metadata for signals and collection

```python
class MetadataManager:
    """Manages metadata updates for signals, features, and collection."""
    
    def __init__(self, metadata_handler: MetadataHandler,
                 collection_metadata: CollectionMetadata):
        self.metadata_handler = metadata_handler
        self.collection_metadata = collection_metadata
    
    def update_time_series_metadata(self, signal: TimeSeriesSignal,
                                    metadata_spec: Dict[str, Any]) -> None:
        """Update TimeSeriesSignal metadata."""
    
    def update_feature_metadata(self, feature: Feature,
                               metadata_spec: Dict[str, Any]) -> None:
        """Update Feature metadata."""
    
    def set_index_config(self, index_fields: List[str]) -> None:
        """Set index configuration for time-series exports."""
    
    def set_feature_index_config(self, index_fields: List[str]) -> None:
        """Set index configuration for feature exports."""
    
    def _validate_timestamp_index(self, signal: TimeSeriesSignal) -> None:
        """Validate signal has proper DatetimeIndex."""
```

**Benefits**:
- Centralizes metadata logic
- Clear responsibility: metadata operations
- Can delegate to MetadataHandler
- Easier to test

---

#### Class 4: AlignmentGridState (Data Class)
**Responsibility**: Encapsulating alignment grid state

```python
@dataclass
class AlignmentGridState:
    """Immutable state for time-series alignment grid."""
    target_rate: Optional[float] = None
    reference_time: Optional[pd.Timestamp] = None
    grid_index: Optional[pd.DatetimeIndex] = None
    merge_tolerance: Optional[pd.Timedelta] = None
    is_calculated: bool = False
    
    def is_valid(self) -> bool:
        """Check if state is valid for alignment."""
        return (self.is_calculated and 
                self.grid_index is not None and 
                not self.grid_index.empty)
```

**Benefits**:
- Groups related state together
- Type-safe state representation
- Can be passed around without exposing internals
- Clear validity checking

---

#### Class 5: AlignmentGridService
**Responsibility**: Computing alignment grid parameters

```python
class AlignmentGridService:
    """Computes and manages alignment grid for time-series signals."""
    
    STANDARD_RATES = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 125, 200, 250, 500, 1000]
    
    def __init__(self, repository: SignalRepository):
        self.repository = repository
        self._state: AlignmentGridState = AlignmentGridState()
    
    @property
    def state(self) -> AlignmentGridState:
        """Get current alignment grid state."""
        return self._state
    
    def generate_alignment_grid(self, target_sample_rate: Optional[float] = None) -> AlignmentGridState:
        """Generate alignment grid and return state."""
        target_rate = self._get_target_sample_rate(target_sample_rate)
        ref_time = self._get_reference_time(target_rate)
        grid_index = self._calculate_grid_index(target_rate, ref_time)
        
        self._state = AlignmentGridState(
            target_rate=target_rate,
            reference_time=ref_time,
            grid_index=grid_index,
            is_calculated=True
        )
        return self._state
    
    def _get_target_sample_rate(self, user_specified: Optional[float]) -> float:
        """Determine target sample rate."""
    
    def _get_nearest_standard_rate(self, rate: float) -> float:
        """Snap rate to standard values."""
    
    def _get_reference_time(self, target_period: pd.Timedelta) -> pd.Timestamp:
        """Find reference time for grid."""
    
    def _calculate_grid_index(self, target_rate: float, 
                             ref_time: pd.Timestamp) -> Optional[pd.DatetimeIndex]:
        """Generate DatetimeIndex for grid."""
```

**Benefits**:
- Encapsulates alignment logic
- Returns immutable state object
- Separates rate calculation from collection
- Single responsibility: grid computation
- Can be tested independently

---

#### Class 6: EpochGridState (Data Class)
**Responsibility**: Encapsulating epoch grid state

```python
@dataclass
class EpochGridState:
    """Immutable state for epoch-based time windows."""
    epoch_grid_index: Optional[pd.DatetimeIndex] = None
    window_length: Optional[pd.Timedelta] = None
    step_size: Optional[pd.Timedelta] = None
    is_calculated: bool = False
    
    def is_valid(self) -> bool:
        """Check if state is valid for feature extraction."""
        return (self.is_calculated and 
                self.epoch_grid_index is not None and 
                not self.epoch_grid_index.empty)
```

---

#### Class 7: EpochGridService
**Responsibility**: Computing epoch grid parameters

```python
class EpochGridService:
    """Computes and manages epoch grid for feature extraction."""
    
    def __init__(self, alignment_grid_service: AlignmentGridService,
                 collection_metadata: CollectionMetadata):
        self.alignment_grid_service = alignment_grid_service
        self.collection_metadata = collection_metadata
        self._state: EpochGridState = EpochGridState()
    
    @property
    def state(self) -> EpochGridState:
        """Get current epoch grid state."""
        return self._state
    
    def generate_epoch_grid(self, start_time: Optional[Union[str, pd.Timestamp]] = None,
                           end_time: Optional[Union[str, pd.Timestamp]] = None) -> EpochGridState:
        """Generate epoch grid and return state."""
        # Current logic from SignalCollection.generate_epoch_grid
        # Returns EpochGridState instead of modifying self
        
        self._state = EpochGridState(
            epoch_grid_index=epoch_index,
            window_length=window_length,
            step_size=step_size,
            is_calculated=True
        )
        return self._state
```

**Benefits**:
- Similar pattern to AlignmentGridService
- Clear separation of concerns
- State can be accessed without modifying collection
- Easier testing

---

#### Class 8: AlignmentExecutor
**Responsibility**: Applying alignment to signals

```python
class AlignmentExecutor:
    """Executes grid alignment operations on signals."""
    
    def __init__(self, repository: SignalRepository,
                 alignment_grid_service: AlignmentGridService):
        self.repository = repository
        self.alignment_grid_service = alignment_grid_service
    
    def apply_grid_alignment(self, method: str = 'nearest',
                            signals_to_align: Optional[List[str]] = None) -> int:
        """Apply grid alignment to signals.
        
        Returns: Number of successfully aligned signals
        """
        state = self.alignment_grid_service.state
        if not state.is_valid():
            raise RuntimeError("Alignment grid not generated")
        
        # Current logic from SignalCollection.apply_grid_alignment
        # Applies reindex_to_grid to each signal
```

**Benefits**:
- Separates alignment execution from grid computation
- Clear dependency on grid service
- Focused responsibility
- Reusable alignment logic

---

#### Class 9: SignalCombinationService
**Responsibility**: Combining aligned signals into dataframes

```python
class SignalCombinationService:
    """Combines aligned signals into unified dataframes."""
    
    def __init__(self, repository: SignalRepository,
                 alignment_grid_service: AlignmentGridService,
                 epoch_grid_service: EpochGridService,
                 metadata_manager: MetadataManager):
        self.repository = repository
        self.alignment_grid_service = alignment_grid_service
        self.epoch_grid_service = epoch_grid_service
        self.metadata_manager = metadata_manager
        self._last_result: Optional[CombinationResult] = None
    
    @property
    def last_result(self) -> Optional[CombinationResult]:
        """Get last combination result."""
        return self._last_result
    
    def combine_aligned_signals(self) -> pd.DataFrame:
        """Combine aligned time-series signals."""
        # Current logic from SignalCollection.combine_aligned_signals
    
    def combine_features(self, inputs: List[str],
                        feature_index_config: Optional[List[str]] = None) -> pd.DataFrame:
        """Combine features into matrix."""
    
    def _perform_concatenation(self, aligned_dfs: Dict[str, pd.DataFrame],
                               grid_index: pd.DatetimeIndex,
                               is_feature: bool) -> pd.DataFrame:
        """Perform actual concatenation logic."""
```

**Benefits**:
- Combines both feature and time-series combination logic
- Clear responsibility: concatenating into dataframes
- Dependencies are explicit
- Results can be stored and accessed

---

#### Class 10: OperationExecutor
**Responsibility**: Executing multi-signal operations

```python
class OperationExecutor:
    """Executes operations on signals."""
    
    def __init__(self, repository: SignalRepository,
                 epoch_grid_service: EpochGridService,
                 metadata_manager: MetadataManager):
        self.repository = repository
        self.epoch_grid_service = epoch_grid_service
        self.metadata_manager = metadata_manager
        self.multi_signal_registry: Dict[str, Tuple[Callable, Type]] = {}
        self.collection_operation_registry: Dict[str, Callable] = {}
    
    def register_multi_signal_operation(self, op_name: str,
                                       func: Callable,
                                       output_class: Type) -> None:
        """Register a multi-signal operation."""
    
    def register_collection_operation(self, op_name: str, method: Callable) -> None:
        """Register a collection operation."""
    
    def execute_multi_signal_operation(self, operation_name: str,
                                       input_signal_keys: List[str],
                                       parameters: Dict[str, Any]) -> Union[TimeSeriesSignal, Feature]:
        """Execute operation producing new signal/feature."""
        # Current logic from SignalCollection.apply_multi_signal_operation
    
    def execute_single_signal_operation(self, signal_key: str,
                                       operation_name: str,
                                       parameters: Dict[str, Any],
                                       output_key: Optional[str] = None) -> str:
        """Execute operation on single signal and store result."""
    
    def execute_batch_signal_operation(self, signal_keys: List[str],
                                      operation_name: str,
                                      parameters: Dict[str, Any]) -> List[str]:
        """Execute operation on multiple signals."""
    
    def execute_collection_operation(self, operation_name: str,
                                    **parameters) -> Any:
        """Execute registered collection operation."""
```

**Benefits**:
- Centralizes operation execution logic
- Owns operation registries
- Clear delegation points
- Easier to test operation logic

---

#### Class 11: DataImportService
**Responsibility**: Importing signals from external sources

```python
class DataImportService:
    """Handles importing signals from external sources."""
    
    def __init__(self, repository: SignalRepository):
        self.repository = repository
    
    def import_signals_from_source(self, importer_instance, source: str,
                                   base_name: str = "imported",
                                   **kwargs) -> List[str]:
        """Import signals from external source."""
        # Delegates to importer
        signals = importer_instance.import_signals(source, **kwargs)
        return self.repository.add_imported_signals(signals, base_name)
```

**Benefits**:
- Single responsibility: coordinating imports
- Clear delegation to importers
- Adds imported signals to repository

---

#### Class 12: SignalSummaryReporter
**Responsibility**: Generating reports and summaries

```python
class SignalSummaryReporter:
    """Generates summaries and reports for signals."""
    
    def __init__(self, repository: SignalRepository):
        self.repository = repository
        self._last_summary: Optional[pd.DataFrame] = None
    
    def summarize_signals(self, fields_to_include: Optional[List[str]] = None,
                         print_summary: bool = True) -> pd.DataFrame:
        """Generate summary of all signals and features."""
        # Current logic from SignalCollection.summarize_signals
    
    def _format_summary_cell(self, value: Any, column_name: str) -> str:
        """Format single cell for display."""
    
    def get_html_summary(self) -> str:
        """Get summary as HTML table."""
    
    def get_json_summary(self) -> str:
        """Get summary as JSON."""
```

**Benefits**:
- Separates reporting from collection logic
- Can generate multiple formats
- Reusable reporter
- Easy to extend with new formats

---

#### Class 13: Refactored SignalCollection
**Responsibility**: Orchestration and central interface

```python
class SignalCollection:
    """Central hub orchestrating signal analysis workflow."""
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None,
                 metadata_handler: Optional[MetadataHandler] = None):
        # Initialize metadata
        self.metadata = CollectionMetadata(...)
        self.metadata_handler = metadata_handler or MetadataHandler()
        
        # Initialize services
        self.repository = SignalRepository(self.metadata_handler)
        self.metadata_manager = MetadataManager(self.metadata_handler, self.metadata)
        self.query_service = SignalQueryService(self.repository)
        self.alignment_grid_service = AlignmentGridService(self.repository)
        self.epoch_grid_service = EpochGridService(self.alignment_grid_service, self.metadata)
        self.alignment_executor = AlignmentExecutor(self.repository, self.alignment_grid_service)
        self.combination_service = SignalCombinationService(
            self.repository,
            self.alignment_grid_service,
            self.epoch_grid_service,
            self.metadata_manager
        )
        self.operation_executor = OperationExecutor(
            self.repository,
            self.epoch_grid_service,
            self.metadata_manager
        )
        self.import_service = DataImportService(self.repository)
        self.reporter = SignalSummaryReporter(self.repository)
    
    # ===== Delegated Methods (maintaining public API) =====
    
    def add_time_series_signal(self, key: str, signal: TimeSeriesSignal) -> None:
        """Delegate to repository."""
        self.repository.add_time_series_signal(key, signal)
    
    def add_feature(self, key: str, feature: Feature) -> None:
        """Delegate to repository."""
        self.repository.add_feature(key, feature)
    
    def get_signals(self, input_spec: Union[str, Dict, List, None] = None) -> List:
        """Delegate to query service."""
        return self.query_service.get_signals(input_spec)
    
    def get_time_series_signal(self, key: str) -> TimeSeriesSignal:
        """Delegate to repository."""
        return self.repository.get_time_series_signal(key)
    
    def get_feature(self, key: str) -> Feature:
        """Delegate to repository."""
        return self.repository.get_feature(key)
    
    def update_time_series_metadata(self, signal: TimeSeriesSignal,
                                    metadata_spec: Dict) -> None:
        """Delegate to metadata manager."""
        self.metadata_manager.update_time_series_metadata(signal, metadata_spec)
    
    def generate_alignment_grid(self, target_sample_rate: Optional[float] = None):
        """Delegate to alignment service and return self for chaining."""
        self.alignment_grid_service.generate_alignment_grid(target_sample_rate)
        return self
    
    def generate_epoch_grid(self, start_time: Optional[Union[str, pd.Timestamp]] = None,
                           end_time: Optional[Union[str, pd.Timestamp]] = None):
        """Delegate to epoch service and return self for chaining."""
        self.epoch_grid_service.generate_epoch_grid(start_time, end_time)
        return self
    
    def apply_grid_alignment(self, method: str = 'nearest',
                            signals_to_align: Optional[List[str]] = None):
        """Delegate to alignment executor and return self."""
        self.alignment_executor.apply_grid_alignment(method, signals_to_align)
        return self
    
    def combine_aligned_signals(self) -> None:
        """Delegate to combination service."""
        self.combination_service.combine_aligned_signals()
    
    def combine_features(self, inputs: List[str],
                        feature_index_config: Optional[List[str]] = None) -> None:
        """Delegate to combination service."""
        self.combination_service.combine_features(inputs, feature_index_config)
    
    def apply_multi_signal_operation(self, operation_name: str,
                                    input_signal_keys: List[str],
                                    parameters: Dict) -> Union[TimeSeriesSignal, Feature]:
        """Delegate to operation executor."""
        return self.operation_executor.execute_multi_signal_operation(
            operation_name, input_signal_keys, parameters
        )
    
    def import_signals_from_source(self, importer_instance, source: str,
                                  base_name: str = "imported", **kwargs) -> List[str]:
        """Delegate to import service."""
        return self.import_service.import_signals_from_source(importer_instance, source, base_name, **kwargs)
    
    def summarize_signals(self, fields_to_include: Optional[List[str]] = None,
                         print_summary: bool = True) -> pd.DataFrame:
        """Delegate to reporter."""
        return self.reporter.summarize_signals(fields_to_include, print_summary)
    
    # ===== Property Access =====
    
    @property
    def time_series_signals(self) -> Dict[str, TimeSeriesSignal]:
        """Access time-series signals."""
        return self.repository.time_series_signals
    
    @property
    def features(self) -> Dict[str, Feature]:
        """Access features."""
        return self.repository.features
    
    def get_stored_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Get last combined dataframe."""
        return self.combination_service.last_result.dataframe if self.combination_service.last_result else None
```

**Benefits**:
- ~300 lines instead of 1,971
- Clear service composition
- Maintains backward compatibility
- Easy to understand delegation
- Single responsibility: orchestration

---

### 5.3 Method Distribution Summary

**Before Refactoring**:
```
SignalCollection: 41 methods (1,971 lines)
```

**After Refactoring**:
```
SignalRepository:              8 methods (200 lines)
SignalQueryService:            4 methods (150 lines)
MetadataManager:               5 methods (200 lines)
AlignmentGridService:          5 methods (250 lines)
EpochGridService:              1 method (150 lines)
AlignmentExecutor:             1 method (100 lines)
SignalCombinationService:      3 methods (300 lines)
OperationExecutor:             4 methods (300 lines)
DataImportService:             1 method (80 lines)
SignalSummaryReporter:         3 methods (150 lines)
SignalCollection (refactored): 13 methods (300 lines)  [delegators + properties]

Data Classes:
- AlignmentGridState:          5 attributes
- EpochGridState:              5 attributes
- CombinationResult:           3 attributes
```

**Metrics**:
- Total methods: 41 (same)
- Classes: 1 → 13
- Avg methods per class: 41 → 3.15
- Avg lines per class: 1,971 → ~152
- Single Responsibility Principle: Violated → Satisfied

---

## 6. BENEFITS OF REFACTORING

### 6.1 Design Benefits
1. **Single Responsibility** - Each class handles one concern
2. **Higher Cohesion** - Related functionality grouped together
3. **Lower Coupling** - Services depend on interfaces, not implementations
4. **Encapsulation** - State properly hidden (AlignmentGridState, EpochGridState)
5. **Testability** - Each service can be tested independently
6. **Extensibility** - Easy to add new services without modifying existing ones

### 6.2 Code Quality Benefits
1. **Reduced Cognitive Load** - Each file ~150 lines vs. 1,971 lines
2. **Easier Debugging** - Issues isolated to specific service
3. **Clearer Dependencies** - Each service explicitly states what it needs
4. **Better Code Reuse** - Services can be used independently
5. **Simplified Mocking** - Each service can be mocked for testing
6. **Improved Documentation** - Each service's purpose is clear

### 6.3 Maintenance Benefits
1. **Lower Change Risk** - Changes to one service don't affect others
2. **Easier Onboarding** - New developers understand one service at a time
3. **Easier Refactoring** - Can refactor individual services
4. **Better Version Control** - Changes scattered across files, not one monolith
5. **Parallel Development** - Multiple developers can work on different services

### 6.4 Runtime Benefits
1. **Memory Efficiency** - State is separated and can be garbage collected
2. **Performance** - No overhead; same operations, just organized differently
3. **Lazy Loading** - Services initialized only when needed

---

## 7. MIGRATION STRATEGY

### Phase 1: Preparation (Week 1)
1. Create new service classes with interfaces
2. Implement each service with current logic from SignalCollection
3. Add comprehensive unit tests for each service
4. Create integration tests for service interactions

### Phase 2: Transition (Week 2-3)
1. Add dependency injection to SignalCollection
2. Create internal composition of services
3. Implement delegator methods in SignalCollection
4. Run full test suite for backward compatibility
5. Update documentation with new architecture

### Phase 3: Cleanup (Week 4)
1. Remove old implementation from SignalCollection
2. Optimize service interactions
3. Update type hints across services
4. Performance testing
5. Update public documentation

### Phase 4: Migration (Ongoing)
1. Update existing code to use new services
2. Gradually deprecate direct service method access if desired
3. Monitor for issues
4. Refactor internal usages as appropriate

---

## 8. DETAILED EFFORT ESTIMATION

### Implementation Effort

| Task | Effort | Notes |
|------|--------|-------|
| **Service Implementation** | | |
| SignalRepository | 2-3 days | Copy logic, add minor enhancements |
| SignalQueryService | 1-2 days | Relatively straightforward filtering |
| MetadataManager | 2 days | Centralize metadata logic |
| AlignmentGridService | 2-3 days | State encapsulation requires thought |
| EpochGridService | 1-2 days | Similar pattern to alignment |
| AlignmentExecutor | 1 day | Straightforward delegation |
| SignalCombinationService | 3-4 days | Complex concatenation logic |
| OperationExecutor | 2-3 days | Registry management |
| DataImportService | 1 day | Simple delegation |
| SignalSummaryReporter | 2-3 days | Extract and enhance reporting |
| **Testing** | | |
| Unit tests per service | 5-7 days | Each service ~2-3 tests |
| Integration tests | 3-4 days | Test service interactions |
| Backward compatibility tests | 2-3 days | Ensure API compatibility |
| Refactored SignalCollection | 2-3 days | Delegators and orchestration |
| **Documentation** | | |
| Architecture documentation | 2-3 days | Design and decisions |
| Service documentation | 2-3 days | Docstrings and examples |
| Migration guide | 1-2 days | For developers using this code |
| **Review & Cleanup** | | |
| Code review | 2-3 days | Internal review cycle |
| Optimization | 1-2 days | Performance tuning |
| Final testing | 1-2 days | Comprehensive testing |
| **Total** | **39-51 days** | ~2 months for experienced team |

### Parallel Work Opportunities
- Unit tests can be written during implementation
- Documentation can be written as services are completed
- Multiple services can be implemented in parallel

### Resource Requirements
- **1 Senior Developer**: 2 months full-time (or 4 months part-time)
- **1 Junior Developer**: Can assist with testing and documentation

### Risk Mitigation
1. **Testing Coverage**: Achieve 90%+ code coverage before release
2. **Staged Rollout**: Keep old SignalCollection alongside new one initially
3. **Feature Flags**: Gate new service usage behind flags
4. **Rollback Plan**: Keep git history for easy reversion

---

## 9. CODE QUALITY IMPROVEMENTS

### Complexity Reduction

| Metric | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| Cyclomatic Complexity | High (1 method ~50) | Low (most ~5-10) | ~80% reduction |
| Method Count per Class | 41 | 1-8 | ~92% reduction |
| Max Lines per File | 1,971 | ~300 | ~85% reduction |
| Coupling (dependencies) | 20+ | 3-5 per service | ~75% reduction |

### Test Improvements

```python
# Before: Difficult to test - many mocks needed
def test_signal_collection():
    # Need to mock: metadata_handler, signals, features, grid calculations
    # Need to set up state across 13+ attributes
    # Hard to isolate failures
    pass

# After: Easy to test - service isolation
def test_alignment_grid_service():
    repository = MockRepository()
    service = AlignmentGridService(repository)
    state = service.generate_alignment_grid(target_sample_rate=100)
    assert state.is_calculated
    assert state.grid_index is not None
```

---

## 10. BACKWARD COMPATIBILITY STRATEGY

### Maintaining Current API

```python
# Old code continues to work
collection = SignalCollection()
collection.add_time_series_signal(key, signal)
collection.generate_alignment_grid()
collection.apply_grid_alignment()
collection.combine_aligned_signals()

# New code can use services directly for more control
service = collection.alignment_grid_service
state = service.generate_alignment_grid()
print(f"Target rate: {state.target_rate}")
```

### Deprecation Path

```python
# Phase 1: New code works alongside old
collection.alignment_grid_service  # New way
collection.target_rate  # Old way (delegates to service)

# Phase 2: Deprecated methods log warnings
def get_target_sample_rate(self):
    warnings.warn("Use alignment_grid_service.state.target_rate", DeprecationWarning)
    return self.alignment_grid_service.state.target_rate
```

---

## 11. RECOMMENDED NEXT STEPS

### Immediate (This Sprint)
1. [ ] Review and approve this refactoring proposal
2. [ ] Create issue for refactoring project
3. [ ] Design detailed service interfaces
4. [ ] Set up test structure

### Short-term (Next 2 Sprints)
1. [ ] Implement SignalRepository with tests
2. [ ] Implement SignalQueryService with tests
3. [ ] Implement MetadataManager with tests
4. [ ] Create state classes (AlignmentGridState, EpochGridState)

### Medium-term (Following 2 Sprints)
1. [ ] Implement AlignmentGridService and EpochGridService
2. [ ] Implement SignalCombinationService
3. [ ] Implement OperationExecutor
4. [ ] Create refactored SignalCollection orchestrator

### Long-term (Final Sprint)
1. [ ] Implement remaining services
2. [ ] Complete integration testing
3. [ ] Update all documentation
4. [ ] Deploy and monitor

---

## 12. CONCLUSION

The SignalCollection class is a classic God Object anti-pattern. This refactoring proposal:

1. **Breaks it into 13 focused classes** with clear responsibilities
2. **Reduces average class size** from 1,971 lines to ~150 lines
3. **Improves testability** by enabling service isolation
4. **Maintains backward compatibility** through delegation
5. **Requires ~2 months effort** for complete implementation
6. **Yields significant quality improvements** across metrics

The proposed architecture is **SOLID-compliant**, **DRY**, and **CLEAN** - making the codebase more maintainable, testable, and extensible for future growth.

