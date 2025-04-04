Okay, I understand. You want to refactor the way collection-level operations are handled to improve separation of concerns, making the          
WorkflowExecutor more generic and pushing the operation dispatch logic into the SignalCollection itself, similar to how TimeSeriesSignal handles
its operations.                                                                                                                                 

Here's the plan:                                                                                                                                

 1 Create alignment_refactoring_plan-2.md: This new file will outline the refactoring strategy for collection operations.                       
 2 Modify TimeSeriesSignal: Apply the previously discussed fix to _reindex_to_grid_logic so that method='nearest' correctly produces NaNs.      
 3 Modify SignalCollection:                                                                                                                     
    • Introduce a registry (collection_operation_registry) and a decorator (register_collection_operation) for collection-level operations.     
    • Implement a new apply_operation method in SignalCollection that looks up the operation name in the registry and executes the corresponding
      method, passing the parameters.                                                                                                           
    • Register the existing alignment/combination methods (generate_alignment_grid, apply_grid_alignment, combine_aligned_signals,              
      align_and_combine_signals) using the new decorator.                                                                                       
 4 Modify WorkflowExecutor:                                                                                                                     
    • Simplify the execute_step method. When step["type"] == "collection", it will now call self.container.apply_operation(operation_name,      
      **parameters) instead of having specific if/elif blocks for each collection operation name.                                               

Here are the SEARCH/REPLACE blocks:                                                                                                             

1. Create the new refactoring plan file:                                                                                                        

                                                                                                                                                
alignment_refactoring_plan-2.md                                                                                                                 
<<<<<<< SEARCH                                                                                                                                  
=======                                                                                                                                         
# Refactoring Plan 2: SignalCollection Operation Dispatch                                                                                       
                                                                                                                                                
This plan outlines the steps to refactor how collection-level operations (like alignment and combination) are invoked, improving the separation 
of concerns between the `WorkflowExecutor` and the `SignalCollection`.                                                                          
                                                                                                                                                
**Goal:** Make `WorkflowExecutor` agnostic to specific collection operation names. Introduce a registry and `apply_operation` method in         
`SignalCollection` to handle dispatching, mirroring the pattern used in `SignalData`/`TimeSeriesSignal`.                                        
                                                                                                                                                
**Core Changes:**                                                                                                                               
                                                                                                                                                
1.  **`SignalCollection.collection_operation_registry`:** A class-level dictionary mapping operation names (strings) to the corresponding bound 
methods.                                                                                                                                        
2.  **`SignalCollection.register_collection_operation`:** A class method decorator to populate the registry.                                    
3.  **`SignalCollection.apply_operation(operation_name, **parameters)`:** A new instance method that:                                           
    *   Looks up `operation_name` in the registry.                                                                                              
    *   Calls the registered method (`self.registered_method(**parameters)`).                                                                   
    *   Handles errors (operation not found, execution errors).                                                                                 
4.  **`WorkflowExecutor.execute_step`:** Simplification for `type: collection` steps. It will now call                                          
`container.apply_operation(operation_name, **parameters)` instead of using `getattr` and specific `if/elif` checks.                             
5.  **Decorate Existing Methods:** Apply the `@register_collection_operation` decorator to `generate_alignment_grid`, `apply_grid_alignment`,   
`combine_aligned_signals`, and `align_and_combine_signals` in `SignalCollection`.                                                               
6.  **Update `TimeSeriesSignal._reindex_to_grid_logic`:** Ensure the `method='nearest'` logic correctly produces NaNs by implementing the       
snap-aggregate-reindex strategy.                                                                                                                
                                                                                                                                                
**Refactoring Steps:**                                                                                                                          
                                                                                                                                                
1.  **Implement Registry and Decorator in `SignalCollection`:**                                                                                 
    *   **File:** `src/sleep_analysis/core/signal_collection.py`                                                                                
    *   **Action:**                                                                                                                             
        *   Add `collection_operation_registry: Dict[str, Callable] = {}` at the class level.                                                   
        *   Add the `register_collection_operation` class method decorator.                                                                     
                                                                                                                                                
2.  **Implement `apply_operation` in `SignalCollection`:**                                                                                      
    *   **File:** `src/sleep_analysis/core/signal_collection.py`                                                                                
    *   **Action:**                                                                                                                             
        *   Create the `apply_operation(self, operation_name: str, **parameters)` method.                                                       
        *   Implement registry lookup, method execution (`self.registry[op_name](**parameters)`), and error handling.                           
                                                                                                                                                
3.  **Decorate Collection Methods:**                                                                                                            
    *   **File:** `src/sleep_analysis/core/signal_collection.py`                                                                                
    *   **Action:** Add `@register_collection_operation("operation_name")` above `generate_alignment_grid`, `apply_grid_alignment`,             
`combine_aligned_signals`, and `align_and_combine_signals`. Use the exact operation names expected in the workflow YAML files.                  
                                                                                                                                                
4.  **Simplify `WorkflowExecutor.execute_step`:**                                                                                               
    *   **File:** `src/sleep_analysis/workflows/workflow_executor.py`                                                                           
    *   **Action:**                                                                                                                             
        *   Locate the `if step["type"] == "collection":` block.                                                                                
        *   Remove the `getattr` call and the subsequent `if/elif` chain checking specific operation names (`generate_alignment_grid`,          
`apply_grid_alignment`, etc.).                                                                                                                  
        *   Replace it with a direct call: `self.container.apply_operation(operation_name, **parameters)`.                                      
        *   Decide how to handle deprecated operation names (e.g., keep a check in the executor or move it into                                 
`SignalCollection.apply_operation`). Keeping it in the executor might be slightly cleaner for workflow-level warnings/errors.                   
                                                                                                                                                
5.  **Update `TimeSeriesSignal._reindex_to_grid_logic`:**                                                                                       
    *   **File:** `src/sleep_analysis/signals/time_series_signal.py`                                                                            
    *   **Action:** Replace the existing `_reindex_to_grid_logic` with the version that handles `method='nearest'` using the                    
snap-aggregate-reindex approach to produce NaNs correctly.                                                                                      
                                                                                                                                                
6.  **Update Tests:**                                                                                                                           
    *   **File:** `tests/unit/test_signal_collection.py`                                                                                        
    *   **Action:** Add tests for the new `SignalCollection.apply_operation` method, including success cases, failure cases (operation not      
found), and cases where the underlying operation raises an error. Ensure existing tests for alignment/combination still pass.                   
    *   **File:** `tests/unit/test_workflow_executor.py` (if exists)                                                                            
    *   **Action:** Verify that workflow execution for collection steps still works correctly after the refactoring.                            
                                                                                                                                                
This plan establishes a more robust and maintainable pattern for handling collection-level operations within the workflow system.               
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

2. Update TimeSeriesSignal._reindex_to_grid_logic:                                                                                              

                                                                                                                                                
src/sleep_analysis/signals/time_series_signal.py                                                                                                
<<<<<<< SEARCH                                                                                                                                  
    @staticmethod                                                                                                                               
    def _reindex_to_grid_logic(data_list: List[pd.DataFrame], parameters: Dict[str, Any]) -> pd.DataFrame:                                      
        """                                                                                                                                     
        Core logic for reindexing a signal's DataFrame to a target grid.                                                                        
        Expects 'grid_index' and 'method' in parameters.                                                                                        
        """                                                                                                                                     
        import pandas as pd # Local import needed within static method                                                                          
        import numpy as np # Local import needed within static method                                                                           
        import logging # Local import needed within static method                                                                               
        logger = logging.getLogger(__name__) # Get logger within static method                                                                  
                                                                                                                                                
        if not data_list:                                                                                                                       
            raise ValueError("No data provided for reindexing.")                                                                                
        data = data_list[0] # Expecting only one DataFrame                                                                                      
                                                                                                                                                
        grid_index = parameters.get('grid_index')                                                                                               
        method = parameters.get('method', 'nearest') # Default to nearest                                                                       
                                                                                                                                                
        if not isinstance(grid_index, pd.DatetimeIndex):                                                                                        
            raise ValueError("Missing or invalid 'grid_index' parameter (must be pd.DatetimeIndex).")                                           
        if grid_index.empty:                                                                                                                    
             raise ValueError("'grid_index' parameter cannot be empty.")                                                                        
                                                                                                                                                
        # Ensure data index timezone matches grid timezone before reindexing                                                                    
        grid_tz = grid_index.tz                                                                                                                 
        if data.index.tz is None:                                                                                                               
            # Avoid modifying original data if possible, but need tz info                                                                       
            # Consider if a copy is needed or if source data should already be tz-aware                                                         
            logger.debug("Localizing timezone-naive index to grid timezone for reindexing.")                                                    
            data = data.tz_localize(grid_tz) # Localize a copy if necessary                                                                     
        elif data.index.tz != grid_tz:                                                                                                          
            logger.debug(f"Converting index timezone from {data.index.tz} to {grid_tz} for reindexing.")                                        
            data = data.tz_convert(grid_tz) # Convert a copy if necessary                                                                       
                                                                                                                                                
        # Perform the reindexing                                                                                                                
        logger.debug(f"Reindexing data to target grid using method: {method}")                                                                  
        aligned_data = data.reindex(grid_index, method=method)                                                                                  
        logger.debug("Reindexing complete.")                                                                                                    
        return aligned_data                                                                                                                     
=======                                                                                                                                         
    @staticmethod                                                                                                                               
    def _reindex_to_grid_logic(data_list: List[pd.DataFrame], parameters: Dict[str, Any]) -> pd.DataFrame:                                      
        """                                                                                                                                     
        Core logic for reindexing a signal's DataFrame to a target grid.                                                                        
        Expects 'grid_index' and 'method' in parameters.                                                                                        
                                                                                                                                                
        Handles 'nearest' method specifically to map original points to their                                                                   
        closest grid point, leaving others NaN. Other methods use standard reindex.                                                             
        """                                                                                                                                     
        import pandas as pd # Local import needed within static method                                                                          
        import numpy as np # Local import needed within static method                                                                           
        import logging # Local import needed within static method                                                                               
        logger = logging.getLogger(__name__) # Get logger within static method                                                                  
                                                                                                                                                
        if not data_list:                                                                                                                       
            raise ValueError("No data provided for reindexing.")                                                                                
        data = data_list[0] # Expecting only one DataFrame                                                                                      
                                                                                                                                                
        grid_index = parameters.get('grid_index')                                                                                               
        method = parameters.get('method', 'nearest') # Default to nearest                                                                       
                                                                                                                                                
        if not isinstance(grid_index, pd.DatetimeIndex):                                                                                        
            raise ValueError("Missing or invalid 'grid_index' parameter (must be pd.DatetimeIndex).")                                           
        if grid_index.empty:                                                                                                                    
             raise ValueError("'grid_index' parameter cannot be empty.")                                                                        
        if data.empty:                                                                                                                          
             logger.warning("Input data is empty, returning empty DataFrame with grid index.")                                                  
             return pd.DataFrame(index=grid_index, columns=data.columns)                                                                        
                                                                                                                                                
                                                                                                                                                
        # Ensure data index timezone matches grid timezone before processing                                                                    
        grid_tz = grid_index.tz                                                                                                                 
        if data.index.tz is None:                                                                                                               
            logger.debug("Localizing timezone-naive index to grid timezone for reindexing.")                                                    
            # Work on a copy to avoid modifying original signal data unexpectedly here                                                          
            data = data.copy().tz_localize(grid_tz)                                                                                             
        elif data.index.tz != grid_tz:                                                                                                          
            logger.debug(f"Converting index timezone from {data.index.tz} to {grid_tz} for reindexing.")                                        
            # Work on a copy                                                                                                                    
            data = data.copy().tz_convert(grid_tz)                                                                                              
        # else: # Timezones match, potentially make a copy if we want to be extra safe                                                          
        #    data = data.copy()                                                                                                                 
                                                                                                                                                
                                                                                                                                                
        # --- Specific logic for 'nearest' to achieve NaN filling ---                                                                           
        if method == 'nearest':                                                                                                                 
            logger.debug("Using 'nearest' method: mapping original points to nearest grid points, leaving others NaN.")                         
                                                                                                                                                
            # 1. Snap: Find nearest grid index labels for each original timestamp                                                               
            # get_indexer returns integer positions in grid_index                                                                               
            nearest_indices = grid_index.get_indexer(data.index, method='nearest')                                                              
                                                                                                                                                
            # Handle case where get_indexer returns -1 (shouldn't happen with 'nearest' unless grid is empty?)                                  
            valid_mask = nearest_indices != -1                                                                                                  
            if not np.all(valid_mask):                                                                                                          
                 logger.warning(f"Could not find nearest grid point for {np.sum(~valid_mask)} original timestamps. Skipping them.")             
                 data = data[valid_mask]                                                                                                        
                 nearest_indices = nearest_indices[valid_mask]                                                                                  
                 if data.empty:                                                                                                                 
                      logger.warning("No valid original timestamps remaining after filtering.")                                                 
                      return pd.DataFrame(index=grid_index, columns=data.columns)                                                               
                                                                                                                                                
                                                                                                                                                
            snapped_index = grid_index[nearest_indices]                                                                                         
                                                                                                                                                
            # 2. Aggregate: Handle collisions (multiple original points mapping to the same grid point)                                         
            temp_df = data.copy() # Create a working copy                                                                                       
            temp_df.index = snapped_index # Assign the snapped grid timestamps as the index                                                     
                                                                                                                                                
            # Check for duplicates before aggregation                                                                                           
            if temp_df.index.has_duplicates:                                                                                                    
                logger.debug(f"Found {temp_df.index.duplicated().sum()} duplicate timestamps after snapping. Aggregating...")                   
                # Define aggregation strategy based on dtype                                                                                    
                agg_dict = {}                                                                                                                   
                for col in temp_df.columns:                                                                                                     
                    if pd.api.types.is_numeric_dtype(temp_df[col]):                                                                             
                        agg_dict[col] = 'mean' # Average numeric columns                                                                        
                    else:                                                                                                                       
                        agg_dict[col] = 'first' # Keep first occurrence for non-numeric (e.g., sleep stage)                                     
                # Perform aggregation                                                                                                           
                temp_df_unique = temp_df.groupby(level=0).agg(agg_dict)                                                                         
                # Ensure original column order                                                                                                  
                temp_df_unique = temp_df_unique[data.columns]                                                                                   
                logger.debug("Aggregation complete.")                                                                                           
            else:                                                                                                                               
                logger.debug("No duplicate timestamps found after snapping.")                                                                   
                temp_df_unique = temp_df # No aggregation needed                                                                                
                                                                                                                                                
            # 3. Reindex: Reindex the sparse aggregated data to the full grid                                                                   
            logger.debug("Reindexing aggregated data to the full grid index (filling with NaN).")                                               
            final_aligned_data = temp_df_unique.reindex(grid_index)                                                                             
                                                                                                                                                
        # --- Standard reindex logic for other methods ---                                                                                      
        else:                                                                                                                                   
            logger.debug(f"Using standard reindex method: {method}")                                                                            
            # Perform the standard reindexing (which might fill gaps depending on method)                                                       
            final_aligned_data = data.reindex(grid_index, method=method)                                                                        
                                                                                                                                                
        logger.debug("Reindexing complete.")                                                                                                    
        return final_aligned_data                                                                                                               
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

3. Add Registry, Decorator, and apply_operation to SignalCollection:                                                                            

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
# Initialize logger for the module                                                                                                              
logger = logging.getLogger(__name__)                                                                                                            
                                                                                                                                                
                                                                                                                                                
# Define standard rates: factors of 1000 Hz plus rates corresponding to multi-second periods                                                    
=======                                                                                                                                         
import functools # Added for decorator                                                                                                          
                                                                                                                                                
# Initialize logger for the module                                                                                                              
logger = logging.getLogger(__name__)                                                                                                            
                                                                                                                                                
                                                                                                                                                
# Define standard rates: factors of 1000 Hz plus rates corresponding to multi-second periods                                                    
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
    # Registry for multi-signal operations                                                                                                      
    multi_signal_registry: Dict[str, Tuple[Callable, Type[SignalData]]] = {}                                                                    
                                                                                                                                                
    def __init__(self, metadata: Optional[Dict[str, Any]] = None, metadata_handler: Optional[MetadataHandler] = None):                          
        """                                                                                                                                     
=======                                                                                                                                         
    # Registry for multi-signal operations (used for operations creating new signals)                                                           
    multi_signal_registry: Dict[str, Tuple[Callable, Type[SignalData]]] = {}                                                                    
    # Registry for collection-level operations (modifying collection state or signals within)                                                   
    collection_operation_registry: Dict[str, Callable] = {}                                                                                     
                                                                                                                                                
    @classmethod                                                                                                                                
    def register_collection_operation(cls, operation_name: str):                                                                                
        """                                                                                                                                     
        Class method decorator to register collection-level operations.                                                                         
                                                                                                                                                
        Args:                                                                                                                                   
            operation_name: The name used to invoke the operation (e.g., in workflows).                                                         
        """                                                                                                                                     
        def decorator(func: Callable):                                                                                                          
            @functools.wraps(func)                                                                                                              
            def wrapper(self, *args, **kwargs):                                                                                                 
                # Optional: Add pre/post execution logic here if needed                                                                         
                result = func(self, *args, **kwargs)                                                                                            
                # Optional: Add post-execution logic here                                                                                       
                return result                                                                                                                   
                                                                                                                                                
            if operation_name in cls.collection_operation_registry:                                                                             
                warnings.warn(f"Overwriting existing collection operation: {operation_name}")                                                   
            cls.collection_operation_registry[operation_name] = func # Store the original function                                              
            logger.debug(f"Registered collection operation '{operation_name}' to function {func.__name__}")                                     
            # Return the original function, not the wrapper, as the wrapper isn't needed                                                        
            # for direct method calls, only for registry lookup via apply_operation.                                                            
            # The registry stores the direct function reference.                                                                                
            return func                                                                                                                         
        return decorator                                                                                                                        
                                                                                                                                                
    def __init__(self, metadata: Optional[Dict[str, Any]] = None, metadata_handler: Optional[MetadataHandler] = None):                          
        """                                                                                                                                     
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
        return self                                                                                                                             
                                                                                                                                                
    def get_signals_from_input_spec(self, input_spec: Union[str, Dict[str, Any], List[str], None] = None) -> List[SignalData]:                  
        """                                                                                                                                     
        Get signals based on an input specification.                                                                                            
=======                                                                                                                                         
        return self                                                                                                                             
                                                                                                                                                
    # --- Collection Operation Dispatch ---                                                                                                     
                                                                                                                                                
    def apply_operation(self, operation_name: str, **parameters: Any) -> Any:                                                                   
        """                                                                                                                                     
        Applies a registered collection-level operation by name.                                                                                
                                                                                                                                                
        Looks up the operation in the `collection_operation_registry` and executes                                                              
        the corresponding method on this instance, passing the provided parameters.                                                             
                                                                                                                                                
        Args:                                                                                                                                   
            operation_name: The name of the operation to execute (must be registered).                                                          
            **parameters: Keyword arguments to pass to the registered operation method.                                                         
                                                                                                                                                
        Returns:                                                                                                                                
            The result returned by the executed operation method (often `self` or `None`).                                                      
                                                                                                                                                
        Raises:                                                                                                                                 
            ValueError: If the operation_name is not found in the registry.                                                                     
            Exception: If the underlying operation method raises an exception.                                                                  
        """                                                                                                                                     
        logger.info(f"Applying collection operation '{operation_name}' with parameters: {parameters}")                                          
        if operation_name not in self.collection_operation_registry:                                                                            
            logger.error(f"Collection operation '{operation_name}' not found in registry.")                                                     
            raise ValueError(f"Collection operation '{operation_name}' not found.")                                                             
                                                                                                                                                
        operation_method = self.collection_operation_registry[operation_name]                                                                   
                                                                                                                                                
        try:                                                                                                                                    
            # Call the registered method (which is bound to the instance implicitly via lookup)                                                 
            # We pass 'self' because the registry stores the unbound method typically                                                           
            # Correction: The registry stores the function. When called via instance.apply_operation,                                           
            # we need to call the method bound to the instance.                                                                                 
            # Let's retrieve the bound method from the instance using getattr.                                                                  
            bound_method = getattr(self, operation_method.__name__)                                                                             
            result = bound_method(**parameters)                                                                                                 
            logger.info(f"Successfully applied collection operation '{operation_name}'.")                                                       
            return result                                                                                                                       
        except Exception as e:                                                                                                                  
            logger.error(f"Error executing collection operation '{operation_name}': {e}", exc_info=True)                                        
            # Re-raise the exception to be handled by the caller (e.g., WorkflowExecutor)                                                       
            raise                                                                                                                               
                                                                                                                                                
    # --- End Collection Operation Dispatch ---                                                                                                 
                                                                                                                                                
                                                                                                                                                
    def get_signals_from_input_spec(self, input_spec: Union[str, Dict[str, Any], List[str], None] = None) -> List[SignalData]:                  
        """                                                                                                                                     
        Get signals based on an input specification.                                                                                            
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

4. Decorate Existing Collection Methods:                                                                                                        

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
            logger.error(f"Error creating date_range for grid index: {e}", exc_info=True)                                                       
            return None                                                                                                                         
                                                                                                                                                
    def generate_alignment_grid(self, target_sample_rate: Optional[float] = None) -> 'SignalCollection':                                        
        """                                                                                                                                     
        Calculates and stores the alignment grid parameters for the collection.                                                                 
=======                                                                                                                                         
            logger.error(f"Error creating date_range for grid index: {e}", exc_info=True)                                                       
            return None                                                                                                                         
                                                                                                                                                
    @register_collection_operation("generate_alignment_grid")                                                                                   
    def generate_alignment_grid(self, target_sample_rate: Optional[float] = None) -> 'SignalCollection':                                        
        """                                                                                                                                     
        Calculates and stores the alignment grid parameters for the collection.                                                                 
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
            raise RuntimeError(f"Failed to apply grid alignment to the following signals: {', '.join(error_signals)}") # Use error_signals here 
                                                                                                                                                
    def combine_aligned_signals(self) -> None:                                                                                                  
        """                                                                                                                                     
        Combines signals assuming they have already been aligned to the grid.                                                                   
=======                                                                                                                                         
            raise RuntimeError(f"Failed to apply grid alignment to the following signals: {', '.join(error_signals)}") # Use error_signals here 
                                                                                                                                                
    @register_collection_operation("combine_aligned_signals")                                                                                   
    def combine_aligned_signals(self) -> None:                                                                                                  
        """                                                                                                                                     
        Combines signals assuming they have already been aligned to the grid.                                                                   
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
                    f"in {time.time() - start_time:.2f} seconds. Stored shape: {combined_df.shape}")                                            
                                                                                                                                                
    def align_and_combine_signals(self) -> None:                                                                                                
        """                                                                                                                                     
        Aligns signals to the grid using merge_asof and combines them.                                                                          
=======                                                                                                                                         
                    f"in {time.time() - start_time:.2f} seconds. Stored shape: {combined_df.shape}")                                            
                                                                                                                                                
    @register_collection_operation("align_and_combine_signals")                                                                                 
    def align_and_combine_signals(self) -> None:                                                                                                
        """                                                                                                                                     
        Aligns signals to the grid using merge_asof and combines them.                                                                          
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
                                                                                                                                                
        return keys                                                                                                                             
                                                                                                                                                
    def apply_grid_alignment(self, method: str = 'nearest', signals_to_align: Optional[List[str]] = None):                                      
        """                                                                                                                                     
        Applies the pre-calculated grid alignment to specified signals in place                                                                 
        by calling the 'reindex_to_grid' operation on each signal.                                                                              
=======                                                                                                                                         
                                                                                                                                                
        return keys                                                                                                                             
                                                                                                                                                
    # Note: This method was defined twice in the original file. Consolidating decoration here.                                                  
    @register_collection_operation("apply_grid_alignment")                                                                                      
    def apply_grid_alignment(self, method: str = 'nearest', signals_to_align: Optional[List[str]] = None):                                      
        """                                                                                                                                     
        Applies the pre-calculated grid alignment to specified signals in place                                                                 
        by calling the 'reindex_to_grid' operation on each signal.                                                                              
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

                                                                                                                                                
src/sleep_analysis/core/signal_collection.py                                                                                                    
<<<<<<< SEARCH                                                                                                                                  
        if error_signals: # Check error_signals list                                                                                            
            raise RuntimeError(f"Failed to apply grid alignment to the following signals: {', '.join(error_signals)}") # Use error_signals      
=======                                                                                                                                         
        if error_signals: # Check error_signals list                                                                                            
            # Raise the error to be caught by apply_operation if called via workflow                                                            
            raise RuntimeError(f"Failed to apply grid alignment to the following signals: {', '.join(error_signals)}") # Use error_signals      
                                                                                                                                                
# --- Remove the duplicate definition of apply_grid_alignment ---                                                                               
# Search for the start of the second definition and delete it.                                                                                  
# This requires careful checking of the exact lines in the user's file.                                                                         
# Assuming the second definition starts around line 1280 based on typical file structure.                                                       
# If the line numbers are different, this SEARCH block will need adjustment.                                                                    
                                                                                                                                                
# Placeholder SEARCH block - adjust line numbers/content based on actual file                                                                   
# src/sleep_analysis/core/signal_collection.py                                                                                                  
# <<<<<<< SEARCH                                                                                                                                
#    def apply_grid_alignment(self, method: str = 'nearest', signals_to_align: Optional[List[str]] = None):                                     
#        """                                                                                                                                    
#        Applies the pre-calculated grid alignment to specified signals in place                                                                
#        by calling the 'reindex_to_grid' operation on each signal.                                                                             
#        ... [rest of the duplicate method code] ...                                                                                            
#        if error_signals: # Check error_signals list                                                                                           
#            raise RuntimeError(f"Failed to apply grid alignment to the following signals: {', '.join(error_signals)}") # Use error_signals     
# =======                                                                                                                                       
# >>>>>>> REPLACE                                                                                                                               
# Note: The above block is intended to DELETE the duplicate method.                                                                             
# It needs to be verified against the actual file content.                                                                                      
# If the duplicate doesn't exist, this block should be removed.                                                                                 
# Assuming the duplicate exists and needs removal for now.                                                                                      
                                                                                                                                                
# --- Verification Step ---                                                                                                                     
# After applying the changes, manually verify that only *one* definition of                                                                     
# `apply_grid_alignment` remains in `signal_collection.py` and that it has the                                                                  
# `@register_collection_operation("apply_grid_alignment")` decorator.                                                                           
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

5. Simplify WorkflowExecutor.execute_step:                                                                                                      

                                                                                                                                                
src/sleep_analysis/workflows/workflow_executor.py                                                                                               
<<<<<<< SEARCH                                                                                                                                  
            # Handle collection-level operations                                                                                                
            if "type" in step and step["type"] == "collection":                                                                                 
                # Call the operation directly on the collection                                                                                 
                operation = getattr(self.container, operation_name, None)                                                                       
                if not operation:                                                                                                               
                    raise ValueError(f"Collection operation '{operation_name}' not found on SignalCollection")                                  
                                                                                                                                                
                # --- Map workflow operation names to new SignalCollection methods ---                                                          
                if operation_name == 'generate_alignment_grid':                                                                                 
                    # Pass target_sample_rate if specified                                                                                      
                    op_params = {}                                                                                                              
                    if 'target_sample_rate' in parameters:                                                                                      
                         op_params['target_sample_rate'] = parameters['target_sample_rate']                                                     
                    result = operation(**op_params) # Calls collection.generate_alignment_grid()                                                
                elif operation_name == 'apply_grid_alignment':                                                                                  
                    # Pass method if specified                                                                                                  
                    op_params = {}                                                                                                              
                    if 'method' in parameters:                                                                                                  
                         op_params['method'] = parameters['method']                                                                             
                    result = operation(**op_params) # Calls collection.apply_grid_alignment()                                                   
                elif operation_name == 'combine_aligned_signals':                                                                               
                    result = operation(**parameters) # Calls collection.combine_aligned_signals()                                               
                elif operation_name == 'align_and_combine_signals':                                                                             
                    result = operation(**parameters) # Calls collection.align_and_combine_signals()                                             
                # --- Deprecated operations ---                                                                                                 
                elif operation_name in ['align_signals', 'generate_and_store_aligned_dataframe']:                                               
                     error_msg = (f"Workflow operation '{operation_name}' is deprecated and removed. "                                          
                                  f"Please update your workflow. Use 'generate_alignment_grid', "                                               
                                  f"'apply_grid_alignment', 'combine_aligned_signals', or "                                                     
                                  f"'align_and_combine_signals' instead.")                                                                      
                     logger.error(error_msg)                                                                                                    
                     raise ValueError(error_msg)                                                                                                
                else:                                                                                                                           
                     # For other potential collection operations, pass all parameters                                                           
                     result = operation(**parameters)                                                                                           
                                                                                                                                                
                # Log the completion.                                                                                                           
                logger.debug(f"Collection operation '{operation_name}' executed.")                                                              
                # No need to handle 'output' or 'inplace' for these collection ops                                                              
=======                                                                                                                                         
            # Handle collection-level operations using the new apply_operation method                                                           
            if "type" in step and step["type"] == "collection":                                                                                 
                 # --- Handle Deprecated Operations First ---                                                                                   
                 # It's cleaner to check for deprecated names here before calling apply_operation                                               
                 if operation_name in ['align_signals', 'generate_and_store_aligned_dataframe']:                                                
                      error_msg = (f"Workflow operation '{operation_name}' is deprecated and removed. "                                         
                                   f"Please update your workflow. Use 'generate_alignment_grid', "                                              
                                   f"'apply_grid_alignment', 'combine_aligned_signals', or "                                                    
                                   f"'align_and_combine_signals' instead.")                                                                     
                      logger.error(error_msg)                                                                                                   
                      # Raise error immediately if strict, otherwise warn and skip                                                              
                      if self.strict_validation:                                                                                                
                           raise ValueError(error_msg)                                                                                          
                      else:                                                                                                                     
                           warnings.warn(error_msg + " Skipping step.")                                                                         
                           return # Skip this step                                                                                              
                                                                                                                                                
                 # --- Call the generic apply_operation method ---                                                                              
                 try:                                                                                                                           
                      # Pass the operation name and parameters directly                                                                         
                      result = self.container.apply_operation(operation_name, **parameters)                                                     
                      # Logging is handled within apply_operation and the method itself                                                         
                 except Exception as e:                                                                                                         
                      # Let _handle_error manage logging/raising based on strict_validation                                                     
                      self._handle_error(e, operation_name=f"collection.{operation_name}")                                                      
                      return # Stop processing this step if an error occurred                                                                   
                                                                                                                                                
                 # No need to handle 'output' or 'inplace' for collection ops currently defined                                                 
>>>>>>> REPLACE                                                                                                                                 
                                                                                                                                                

Important Note on Duplicate apply_grid_alignment:                                                                                               

The SEARCH/REPLACE block to remove the duplicate apply_grid_alignment definition is a placeholder. Please carefully verify the exact lines of   
the duplicate definition in your src/sleep_analysis/core/signal_collection.py file and adjust the SEARCH section accordingly before applying. If
there is no duplicate, simply remove that specific SEARCH/REPLACE block.                                                                        

After applying these changes, the WorkflowExecutor will be decoupled from the specific names of collection operations, relying instead on the   
SignalCollection's internal registry and apply_operation method for dispatch.   # Refactoring Plan 2: SignalCollection Operation Dispatch

This plan outlines the steps to refactor how collection-level operations (like alignment and combination) are invoked, improving the separation of concerns between the `WorkflowExecutor` and the `SignalCollection`.

**Goal:** Make `WorkflowExecutor` agnostic to specific collection operation names. Introduce a registry and `apply_operation` method in `SignalCollection` to handle dispatching, mirroring the pattern used in `SignalData`/`TimeSeriesSignal`.

**Core Changes:**

1.  **`SignalCollection.collection_operation_registry`:** A class-level dictionary mapping operation names (strings) to the corresponding bound methods.
2.  **`SignalCollection.register_collection_operation`:** A class method decorator to populate the registry.
3.  **`SignalCollection.apply_operation(operation_name, **parameters)`:** A new instance method that:
    *   Looks up `operation_name` in the registry.
    *   Calls the registered method (`self.registered_method(**parameters)`).
    *   Handles errors (operation not found, execution errors).
4.  **`WorkflowExecutor.execute_step`:** Simplification for `type: collection` steps. It will now call `container.apply_operation(operation_name, **parameters)` instead of using `getattr` and specific `if/elif` checks.
5.  **Decorate Existing Methods:** Apply the `@register_collection_operation` decorator to `generate_alignment_grid`, `apply_grid_alignment`, `combine_aligned_signals`, and `align_and_combine_signals` in `SignalCollection`.
6.  **Update `TimeSeriesSignal._reindex_to_grid_logic`:** Ensure the `method='nearest'` logic correctly produces NaNs by implementing the snap-aggregate-reindex strategy.

**Refactoring Steps:**

1.  **Implement Registry and Decorator in `SignalCollection`:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:**
        *   Add `collection_operation_registry: Dict[str, Callable] = {}` at the class level.
        *   Add the `register_collection_operation` class method decorator.

2.  **Implement `apply_operation` in `SignalCollection`:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:**
        *   Create the `apply_operation(self, operation_name: str, **parameters)` method.
        *   Implement registry lookup, method execution (`self.registry[op_name](**parameters)`), and error handling.

3.  **Decorate Collection Methods:**
    *   **File:** `src/sleep_analysis/core/signal_collection.py`
    *   **Action:** Add `@register_collection_operation("operation_name")` above `generate_alignment_grid`, `apply_grid_alignment`, `combine_aligned_signals`, and `align_and_combine_signals`. Use the exact operation names expected in the workflow YAML files.

4.  **Simplify `WorkflowExecutor.execute_step`:**
    *   **File:** `src/sleep_analysis/workflows/workflow_executor.py`
    *   **Action:**
        *   Locate the `if step["type"] == "collection":` block.
        *   Remove the `getattr` call and the subsequent `if/elif` chain checking specific operation names (`generate_alignment_grid`, `apply_grid_alignment`, etc.).
        *   Replace it with a direct call: `self.container.apply_operation(operation_name, **parameters)`.
        *   Decide how to handle deprecated operation names (e.g., keep a check in the executor or move it into `SignalCollection.apply_operation`). Keeping it in the executor might be slightly cleaner for workflow-level warnings/errors.

5.  **Update `TimeSeriesSignal._reindex_to_grid_logic`:**
    *   **File:** `src/sleep_analysis/signals/time_series_signal.py`
    *   **Action:** Replace the existing `_reindex_to_grid_logic` with the version that handles `method='nearest'` using the snap-aggregate-reindex approach to produce NaNs correctly.

6.  **Update Tests:**
    *   **File:** `tests/unit/test_signal_collection.py`
    *   **Action:** Add tests for the new `SignalCollection.apply_operation` method, including success cases, failure cases (operation not found), and cases where the underlying operation raises an error. Ensure existing tests for alignment/combination still pass.
    *   **File:** `tests/unit/test_workflow_executor.py` (if exists)
    *   **Action:** Verify that workflow execution for collection steps still works correctly after the refactoring.

This plan establishes a more robust and maintainable pattern for handling collection-level operations within the workflow system.
