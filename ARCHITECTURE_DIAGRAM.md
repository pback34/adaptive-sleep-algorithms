# SignalCollection Refactoring - Architecture Diagram

## Current Monolithic Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SignalCollection                                    │
│                         (1,971 lines, 41 methods)                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Signal Management                                                   │  │
│  │  • add_time_series_signal()                                         │  │
│  │  • add_feature()                                                    │  │
│  │  • get_time_series_signal()                                         │  │
│  │  • get_signals()                                                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Metadata Management                                                 │  │
│  │  • update_time_series_metadata()                                    │  │
│  │  • set_index_config()                                               │  │
│  │  • _validate_timestamp_index()                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Alignment & Grid Generation                                         │  │
│  │  • generate_alignment_grid()                                        │  │
│  │  • get_target_sample_rate()                                         │  │
│  │  • _calculate_grid_index()                                          │  │
│  │  • generate_epoch_grid()                                            │  │
│  │  • apply_grid_alignment()                                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Signal Combination                                                  │  │
│  │  • combine_aligned_signals()                                        │  │
│  │  • combine_features()                                               │  │
│  │  • _perform_concatenation()                                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Operation Execution                                                 │  │
│  │  • apply_multi_signal_operation()                                   │  │
│  │  • apply_operation()                                                │  │
│  │  • apply_and_store_operation()                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Data Import & Reporting                                             │  │
│  │  • import_signals_from_source()                                     │  │
│  │  • summarize_signals()                                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Instance Attributes: 13+                                            │  │
│  │  • time_series_signals, features                                    │  │
│  │  • metadata, metadata_handler                                       │  │
│  │  • target_rate, ref_time, grid_index (alignment state)             │  │
│  │  • epoch_grid_index, global_epoch_window_length (epoch state)      │  │
│  │  • _aligned_dataframe, _combined_feature_matrix (results)          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ✗ High Cohesion                                                           │
│  ✗ Low Coupling                                                            │
│  ✗ High Testability                                                        │
│  ✗ Single Responsibility                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Modular Architecture

```
                             Client Code
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  SignalCollection       │
                    │  (Orchestrator)         │
                    │  ~300 lines, 13 methods │
                    │                         │
                    │ Delegates to:           │
                    │  • repository           │
                    │  • query_service        │
                    │  • metadata_manager     │
                    │  • alignment services   │
                    │  • combination_service  │
                    │  • operation_executor   │
                    │  • import_service       │
                    │  • reporter             │
                    └──────────┬──────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼──────┐ ┌─────▼──────┐ ┌────▼─────────┐
        │  Repository  │ │   Query    │ │   Metadata  │
        │   Layer      │ │  Service   │ │   Manager   │
        │  ~200 lines  │ │ ~150 lines │ │ ~200 lines  │
        │              │ │            │ │             │
        │ • add_ts_sig │ │ • get_sigs │ │ • update_ts │
        │ • add_feat   │ │ • _matches │ │ • set_index │
        │ • get_ts_sig │ │            │ │ • validate  │
        │ • get_feat   │ │            │ │             │
        └──────────────┘ └────────────┘ └─────────────┘
                │              │              │
                ▼              ▼              ▼
        ┌──────────────┐ ┌────────────┐ ┌──────────────┐
        │ Signal Dict  │ │ Filtering  │ │ Enum & Type  │
        │ • time_      │ │ Logic      │ │ Conversion   │
        │   series     │ └────────────┘ └──────────────┘
        │ • features   │
        └──────────────┘


        ┌─────────────────────────────────────────────────────────┐
        │         Alignment & Grid Services                       │
        │                                                         │
        │ ┌───────────────────────────────────────────────────┐  │
        │ │  AlignmentGridService    EpochGridService        │  │
        │ │  ~250 lines, 5 methods   ~150 lines, 1 method   │  │
        │ │                                                   │  │
        │ │  • generate_alignment_   • generate_epoch_      │  │
        │ │    grid()                  grid()                │  │
        │ │  • _get_target_rate()    • Returns:             │  │
        │ │  • _calculate_grid_      EpochGridState         │  │
        │ │    index()                                       │  │
        │ │  • Returns: AlignmentGridState                  │  │
        │ └───────────────────────────────────────────────────┘  │
        │                                                         │
        │ ┌───────────────────────────────────────────────────┐  │
        │ │  AlignmentExecutor                               │  │
        │ │  ~100 lines, 1 method                            │  │
        │ │                                                   │  │
        │ │  • apply_grid_alignment()                        │  │
        │ │  • Uses: AlignmentGridState                      │  │
        │ └───────────────────────────────────────────────────┘  │
        │                                                         │
        │ State Objects (Data Classes):                           │
        │ • AlignmentGridState (~10 lines)                       │
        │   - target_rate, reference_time, grid_index, etc.      │
        │ • EpochGridState (~10 lines)                           │
        │   - epoch_grid_index, window_length, step_size         │
        │                                                         │
        └─────────────────────────────────────────────────────────┘


        ┌─────────────────────────────────────────────────────────┐
        │         Combination & Processing Services               │
        │                                                         │
        │ ┌───────────────────────────────────────────────────┐  │
        │ │  SignalCombinationService                         │  │
        │ │  ~300 lines, 3 methods                            │  │
        │ │                                                   │  │
        │ │  • combine_aligned_signals()                      │  │
        │ │  • combine_features()                             │  │
        │ │  • _perform_concatenation()                       │  │
        │ │  • Returns: DataFrame                             │  │
        │ │  • Stores: Last result                            │  │
        │ └───────────────────────────────────────────────────┘  │
        │                                                         │
        │ ┌───────────────────────────────────────────────────┐  │
        │ │  OperationExecutor                                │  │
        │ │  ~300 lines, 4 methods                            │  │
        │ │                                                   │  │
        │ │  • execute_multi_signal_operation()               │  │
        │ │  • execute_single_signal_operation()              │  │
        │ │  • execute_batch_signal_operation()               │  │
        │ │  • execute_collection_operation()                 │  │
        │ │  • Owns: Registries (multi, collection)           │  │
        │ └───────────────────────────────────────────────────┘  │
        │                                                         │
        │ ┌───────────────────────────────────────────────────┐  │
        │ │  DataImportService   SignalSummaryReporter       │  │
        │ │  ~80 lines           ~150 lines, 3 methods       │  │
        │ │                                                   │  │
        │ │  • import_signals_   • summarize_signals()       │  │
        │ │    from_source()     • _format_summary_cell()   │  │
        │ │                      • get_html_summary()        │  │
        │ └───────────────────────────────────────────────────┘  │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
```

---

## Dependency Flow Diagram

```
Client Code
    │
    ▼
SignalCollection (Orchestrator)
    │
    ├──────────────────────────────────┬──────────────────────────┬──────────────┐
    │                                  │                          │              │
    ▼                                  ▼                          ▼              ▼
SignalRepository              SignalQueryService           MetadataManager   Other Services
(Storage & Basic Access)      (Filtering & Querying)       (Metadata Ops)
    │                              │                            │
    ├─ time_series_signals         ├─ Uses:                     ├─ Uses:
    ├─ features                    │  Repository               │  MetadataHandler
    └─ metadata_handler             └─ Filtering logic          └─ CollectionMetadata
                                                                 

SignalCollection delegates to:

1. Repository Layer
   └─ SignalRepository
      ├─ Owns: signal & feature dicts
      └─ Provides: basic CRUD

2. Query Layer
   └─ SignalQueryService
      └─ Provides: complex filtering

3. Metadata Layer
   └─ MetadataManager
      └─ Manages: metadata updates & validation

4. Alignment Layers
   ├─ AlignmentGridService → AlignmentGridState
   ├─ EpochGridService → EpochGridState
   └─ AlignmentExecutor → uses grid state

5. Combination Layer
   └─ SignalCombinationService
      ├─ Uses: alignment state, epoch state
      └─ Returns: DataFrames

6. Operation Layer
   └─ OperationExecutor
      ├─ Owns: registries
      └─ Delegates: to operation functions

7. Import Layer
   └─ DataImportService
      └─ Delegates: to importers

8. Reporting Layer
   └─ SignalSummaryReporter
      ├─ Uses: Repository
      └─ Returns: Reports
```

---

## State Management Evolution

### Current (Scattered)
```
SignalCollection Instance
├── time_series_signals: Dict[str, TimeSeriesSignal]
├── features: Dict[str, Feature]
├── metadata: CollectionMetadata
├── metadata_handler: MetadataHandler
│
├── Alignment Grid State (scattered)
│  ├── target_rate: Optional[float]
│  ├── ref_time: Optional[pd.Timestamp]
│  ├── grid_index: Optional[pd.DatetimeIndex]
│  ├── _alignment_params_calculated: bool
│  └── _merge_tolerance: Optional[pd.Timedelta]
│
├── Epoch Grid State (scattered)
│  ├── epoch_grid_index: Optional[pd.DatetimeIndex]
│  ├── global_epoch_window_length: Optional[pd.Timedelta]
│  ├── global_epoch_step_size: Optional[pd.Timedelta]
│  └── _epoch_grid_calculated: bool
│
└── Results Storage (scattered)
   ├── _aligned_dataframe: Optional[pd.DataFrame]
   ├── _aligned_dataframe_params: Optional[Dict]
   ├── _summary_dataframe: Optional[pd.DataFrame]
   ├── _summary_dataframe_params: Optional[Dict]
   └── _combined_feature_matrix: Optional[pd.DataFrame]
```

### Proposed (Organized)
```
SignalCollection Instance
├── metadata: CollectionMetadata
├── metadata_handler: MetadataHandler
│
├── Services (Composition)
│  ├── repository: SignalRepository
│  │  ├── time_series_signals: Dict
│  │  └── features: Dict
│  │
│  ├── alignment_grid_service: AlignmentGridService
│  │  └── _state: AlignmentGridState
│  │     ├── target_rate
│  │     ├── reference_time
│  │     ├── grid_index
│  │     ├── merge_tolerance
│  │     └── is_calculated
│  │
│  ├── epoch_grid_service: EpochGridService
│  │  └── _state: EpochGridState
│  │     ├── epoch_grid_index
│  │     ├── window_length
│  │     ├── step_size
│  │     └── is_calculated
│  │
│  ├── combination_service: SignalCombinationService
│  │  └── _last_result: CombinationResult
│  │     ├── dataframe
│  │     ├── params
│  │     └── is_feature_matrix
│  │
│  └── Other services (minimal state)
```

---

## Data Flow: Signal Addition to Combination

### Current (All in SignalCollection)
```
SignalCollection.add_time_series_signal()
    ├─ Validate signal
    ├─ Check ID uniqueness
    ├─ Validate timestamp index
    ├─ Check timezone
    ├─ Set metadata handler
    ├─ Set signal name
    └─ Store in time_series_signals

SignalCollection.generate_alignment_grid()
    ├─ Calculate target rate
    ├─ Calculate reference time
    ├─ Generate grid index
    └─ Store in state attributes

SignalCollection.apply_grid_alignment()
    ├─ For each signal:
    │  └─ Call signal.apply_operation('reindex_to_grid')
    └─ Modify signal in place

SignalCollection.combine_aligned_signals()
    ├─ Prepare aligned dataframes
    ├─ Perform concatenation
    └─ Store result
```

### Proposed (Orchestrated Through Services)
```
SignalCollection.add_time_series_signal()
    └─ Delegate to repository
       └─ SignalRepository.add_time_series_signal()
           ├─ Validate signal
           ├─ Check ID uniqueness
           ├─ Validate timestamp index
           ├─ Check timezone
           ├─ Set metadata handler
           ├─ Set signal name
           └─ Store in time_series_signals

SignalCollection.generate_alignment_grid()
    └─ Delegate to service
       └─ AlignmentGridService.generate_alignment_grid()
           ├─ _get_target_sample_rate()
           ├─ _get_reference_time()
           ├─ _calculate_grid_index()
           └─ Return AlignmentGridState

SignalCollection.apply_grid_alignment()
    └─ Delegate to executor
       └─ AlignmentExecutor.apply_grid_alignment()
           ├─ Get state from alignment_grid_service
           ├─ For each signal:
           │  └─ Call signal.apply_operation('reindex_to_grid')
           └─ Modify signal in place

SignalCollection.combine_aligned_signals()
    └─ Delegate to service
       └─ SignalCombinationService.combine_aligned_signals()
           ├─ Prepare aligned dataframes
           ├─ Call _perform_concatenation()
           └─ Store in _last_result (CombinationResult)
```

---

## Testing Architecture

### Current Testing (Difficult)
```
Test Suite
    │
    └─ test_signal_collection.py
        ├─ test_add_signals() ─────────┐
        │  • Mock metadata_handler      │ All interdependent
        │  • Mock time_series_signals   │ One failure = many tests fail
        │  • Mock features              │ Hard to isolate issues
        │  • Mock alignment state       │ Slow test execution
        ├─ test_alignment_grid() ──────┐│
        │  • Set up all signal dicts    ││
        │  • Pre-populate state         ││
        ├─ test_combination() ─────────┐││
        │  • Pre-compute alignment     ││
        │  • Pre-setup states          │││
        └─ ...                        │││
                                       ▼▼▼
                            Hard to test in isolation
                            Test interdependencies
                            Brittle tests
```

### Proposed Testing (Easy)
```
Test Suite
    ├─ repositories/
    │  └─ test_signal_repository.py
    │     ├─ test_add_time_series_signal()
    │     ├─ test_add_feature()
    │     ├─ test_get_by_key()
    │     └─ ... (focused tests)
    │
    ├─ services/
    │  ├─ test_query_service.py
    │  │  ├─ test_get_signals_by_type()
    │  │  ├─ test_filter_by_criteria()
    │  │  └─ ...
    │  │
    │  ├─ test_alignment_grid_service.py
    │  │  ├─ test_generate_alignment_grid()
    │  │  ├─ test_calculate_grid_index()
    │  │  └─ ...
    │  │
    │  ├─ test_combination_service.py
    │  │  ├─ test_combine_aligned_signals()
    │  │  ├─ test_combine_features()
    │  │  └─ ...
    │  │
    │  └─ test_operation_executor.py
    │     ├─ test_execute_multi_signal_op()
    │     └─ ...
    │
    └─ integration/
       ├─ test_signal_collection_workflow.py
       │  ├─ test_add_to_combine_workflow()
       │  ├─ test_alignment_workflow()
       │  └─ ...
       │
       └─ test_backward_compatibility.py
          ├─ test_old_api_still_works()
          └─ ...

Benefits:
    ✓ Can test each service in isolation
    ✓ Mock only necessary dependencies
    ✓ Fast unit tests
    ✓ Clear failure points
    ✓ Better code coverage
    ✓ Easier debugging
```

---

## Migration Path

```
Week 1-2: Create Services
    └─ Create new service classes alongside current SignalCollection
       ├─ No changes to existing code yet
       └─ Services are tested independently

Week 3-4: Refactor SignalCollection
    └─ Add services as instance attributes
    └─ Create delegator methods
    └─ Old methods still work (backward compatibility)

Week 5-6: Gradual Migration
    └─ Update internal code to use new services
    └─ Add deprecation warnings to old methods
    └─ Monitor for issues

Week 7-8: Cleanup & Finalization
    └─ Option 1: Remove old implementation
    └─ Option 2: Keep old implementation as fallback
    └─ Full documentation update
```

---

## SOLID Principles Alignment

### Single Responsibility Principle (SRP)
```
Current: ✗ SignalCollection handles 10+ responsibilities
Proposed: ✓ Each service has one responsibility
```

### Open/Closed Principle (OCP)
```
Current: ✗ Must modify SignalCollection to add features
Proposed: ✓ Can add new services without modifying existing ones
```

### Liskov Substitution Principle (LSP)
```
Current: ✓ Not applicable (no inheritance)
Proposed: ✓ Services can be substituted via interfaces
```

### Interface Segregation Principle (ISP)
```
Current: ✗ Large interface with many methods
Proposed: ✓ Each service has focused interface
```

### Dependency Inversion Principle (DIP)
```
Current: ✗ Direct dependencies on concrete classes
Proposed: ✓ Can introduce interfaces for loose coupling
```

---

## Performance Considerations

### Memory
```
Current: All state in one instance
         Memory not freed until collection destroyed

Proposed: State in separate objects
          Can garbage collect intermediate results
          More efficient memory usage
```

### CPU
```
Current: No additional overhead
Proposed: Slight overhead from service method calls
          BUT: Offset by better code organization and caching
          NET: Negligible impact (~1-2%)
```

### I/O
```
Current: No change
Proposed: No change
         Same signal reading/writing operations
```

---

## Summary

The refactoring transforms:
- 1 large, complex class (1,971 lines)
- Into 13 focused, single-purpose classes (~150 lines each)

Benefits:
- Easier to understand
- Easier to test
- Easier to maintain
- Easier to extend
- Follows SOLID principles
- Maintains backward compatibility

Cost:
- ~6-8 weeks implementation
- Some overhead from service composition (negligible)
- Initial learning curve for developers

Result:
- **Higher code quality**
- **Better maintainability**
- **Easier development**
- **Reduced technical debt**

