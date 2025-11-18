"""
Parallel and concurrent execution utilities for the sleep analysis framework.

This module provides infrastructure for parallel processing of signal operations,
feature extraction, and workflow execution to leverage multi-core systems.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from typing import (
    Callable, List, Any, Dict, Optional, TypeVar, Iterator, Tuple
)
from dataclasses import dataclass
from functools import wraps
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def _detect_test_environment() -> bool:
    """
    Detect if we're running in a test environment.

    Returns:
        True if running under pytest or unittest, False otherwise
    """
    import sys
    # Check for pytest
    if 'pytest' in sys.modules:
        return True
    # Check for unittest
    if 'unittest' in sys.modules:
        return True
    # Check for PYTEST environment variable
    if os.environ.get('PYTEST_CURRENT_TEST'):
        return True
    return False


@dataclass
class ParallelConfig:
    """
    Configuration for parallel execution.

    Attributes:
        enabled: Whether parallel processing is enabled
        max_workers_cpu: Maximum worker processes for CPU-bound tasks
        max_workers_io: Maximum worker threads for I/O-bound tasks
        chunk_size: Default chunk size for batch processing
        enable_progress: Whether to log progress during parallel execution
    """
    enabled: bool = True
    max_workers_cpu: Optional[int] = None
    max_workers_io: Optional[int] = None
    chunk_size: int = 10
    enable_progress: bool = True

    def __post_init__(self):
        """Set default worker counts based on CPU count."""
        cpu_count = os.cpu_count() or 1

        if self.max_workers_cpu is None:
            # Use all CPUs for CPU-bound tasks, leaving one for system
            self.max_workers_cpu = max(1, cpu_count - 1)

        if self.max_workers_io is None:
            # Use more threads for I/O-bound tasks (2-4x CPU count)
            self.max_workers_io = min(32, cpu_count * 4)

        # Auto-disable in test environments to prevent hanging
        if _detect_test_environment() and self.enabled:
            logger.debug("Test environment detected - disabling parallel processing by default")
            self.enabled = False


# Global configuration instance
_global_config = ParallelConfig()


def get_parallel_config() -> ParallelConfig:
    """Get the global parallel processing configuration."""
    return _global_config


def set_parallel_config(config: ParallelConfig) -> None:
    """Set the global parallel processing configuration."""
    global _global_config
    _global_config = config
    logger.info(
        f"Parallel config updated: enabled={config.enabled}, "
        f"cpu_workers={config.max_workers_cpu}, io_workers={config.max_workers_io}"
    )


def disable_parallel_processing() -> None:
    """Disable all parallel processing (useful for debugging)."""
    global _global_config
    _global_config.enabled = False
    logger.info("Parallel processing disabled")


def enable_parallel_processing() -> None:
    """Enable parallel processing."""
    global _global_config
    _global_config.enabled = True
    logger.info("Parallel processing enabled")


class ParallelExecutor:
    """
    Manages parallel execution of tasks using thread or process pools.

    This class provides a high-level interface for parallel execution with
    automatic resource management and error handling.
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel executor.

        Args:
            config: Parallel configuration, uses global config if None
        """
        self.config = config or get_parallel_config()
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def shutdown(self) -> None:
        """Shutdown all executor pools."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None

    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers_io
            )
        return self._thread_pool

    def _get_process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool executor."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self.config.max_workers_cpu
            )
        return self._process_pool

    def map_threaded(
        self,
        func: Callable[[T], R],
        items: List[T],
        desc: str = "Processing"
    ) -> List[R]:
        """
        Execute function over items in parallel using threads (I/O-bound tasks).

        Args:
            func: Function to apply to each item
            items: Items to process
            desc: Description for logging

        Returns:
            List of results in the same order as items
        """
        if not self.config.enabled or len(items) <= 1:
            return [func(item) for item in items]

        start_time = time.time()
        if self.config.enable_progress:
            logger.info(f"{desc}: Processing {len(items)} items with threads")

        pool = self._get_thread_pool()
        results = list(pool.map(func, items))

        if self.config.enable_progress:
            elapsed = time.time() - start_time
            logger.info(f"{desc}: Completed in {elapsed:.2f}s")

        return results

    def map_processes(
        self,
        func: Callable[[T], R],
        items: List[T],
        desc: str = "Processing",
        chunksize: Optional[int] = None
    ) -> List[R]:
        """
        Execute function over items in parallel using processes (CPU-bound tasks).

        Args:
            func: Function to apply to each item
            items: Items to process
            desc: Description for logging
            chunksize: Chunk size for batching, uses config default if None

        Returns:
            List of results in the same order as items
        """
        if not self.config.enabled or len(items) <= 1:
            return [func(item) for item in items]

        start_time = time.time()
        if self.config.enable_progress:
            logger.info(
                f"{desc}: Processing {len(items)} items with "
                f"{self.config.max_workers_cpu} processes"
            )

        chunksize = chunksize or self.config.chunk_size
        pool = self._get_process_pool()
        results = list(pool.map(func, items, chunksize=chunksize))

        if self.config.enable_progress:
            elapsed = time.time() - start_time
            logger.info(f"{desc}: Completed in {elapsed:.2f}s")

        return results

    def submit_tasks(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        use_processes: bool = False,
        desc: str = "Tasks"
    ) -> List[Any]:
        """
        Submit multiple heterogeneous tasks for parallel execution.

        Args:
            tasks: List of (function, args, kwargs) tuples
            use_processes: Use processes instead of threads
            desc: Description for logging

        Returns:
            List of results in the same order as tasks
        """
        if not self.config.enabled or len(tasks) <= 1:
            return [func(*args, **kwargs) for func, args, kwargs in tasks]

        start_time = time.time()
        if self.config.enable_progress:
            worker_type = "processes" if use_processes else "threads"
            logger.info(f"{desc}: Submitting {len(tasks)} tasks to {worker_type}")

        pool = self._get_process_pool() if use_processes else self._get_thread_pool()

        # Submit all tasks
        futures = []
        for func, args, kwargs in tasks:
            future = pool.submit(func, *args, **kwargs)
            futures.append(future)

        # Collect results in order
        results = [future.result() for future in futures]

        if self.config.enable_progress:
            elapsed = time.time() - start_time
            logger.info(f"{desc}: Completed in {elapsed:.2f}s")

        return results

    def map_with_progress(
        self,
        func: Callable[[T], R],
        items: List[T],
        use_processes: bool = False,
        desc: str = "Processing"
    ) -> Iterator[Tuple[int, R]]:
        """
        Execute function over items with progress tracking.

        Yields results as they complete (not in input order).

        Args:
            func: Function to apply to each item
            items: Items to process
            use_processes: Use processes instead of threads
            desc: Description for logging

        Yields:
            Tuples of (index, result) as tasks complete
        """
        if not self.config.enabled or len(items) <= 1:
            for idx, item in enumerate(items):
                yield idx, func(item)
            return

        start_time = time.time()
        pool = self._get_process_pool() if use_processes else self._get_thread_pool()

        # Submit all tasks with their indices
        future_to_index = {
            pool.submit(func, item): idx
            for idx, item in enumerate(items)
        }

        # Yield results as they complete
        completed = 0
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()
            completed += 1

            if self.config.enable_progress and completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"{desc}: {completed}/{len(items)} complete "
                    f"({rate:.1f} items/sec)"
                )

            yield idx, result


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    use_processes: bool = False,
    max_workers: Optional[int] = None,
    desc: str = "Processing"
) -> List[R]:
    """
    Convenience function for parallel map operation.

    Args:
        func: Function to apply to each item
        items: Items to process
        use_processes: Use processes (CPU-bound) instead of threads (I/O-bound)
        max_workers: Override default worker count
        desc: Description for logging

    Returns:
        List of results in the same order as items

    Example:
        ```python
        # Parallel CPU-bound operation
        results = parallel_map(
            expensive_computation,
            data_items,
            use_processes=True,
            desc="Computing features"
        )

        # Parallel I/O-bound operation
        results = parallel_map(
            load_file,
            file_paths,
            use_processes=False,
            desc="Loading files"
        )
        ```
    """
    config = get_parallel_config()
    if max_workers is not None:
        config = ParallelConfig(
            enabled=config.enabled,
            max_workers_cpu=max_workers if use_processes else config.max_workers_cpu,
            max_workers_io=max_workers if not use_processes else config.max_workers_io,
            chunk_size=config.chunk_size,
            enable_progress=config.enable_progress
        )

    with ParallelExecutor(config) as executor:
        if use_processes:
            return executor.map_processes(func, items, desc=desc)
        else:
            return executor.map_threaded(func, items, desc=desc)


def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    """
    Split items into batches for parallel processing.

    Args:
        items: Items to batch
        batch_size: Size of each batch

    Returns:
        List of batches

    Example:
        ```python
        batches = batch_items(range(100), batch_size=10)
        # Returns [[0-9], [10-19], ..., [90-99]]
        ```
    """
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def chunk_for_workers(items: List[T], num_workers: Optional[int] = None) -> List[List[T]]:
    """
    Split items into chunks optimized for number of workers.

    Args:
        items: Items to chunk
        num_workers: Number of workers, uses config default if None

    Returns:
        List of chunks

    Example:
        ```python
        # Split work evenly across available CPUs
        chunks = chunk_for_workers(items)
        results = parallel_map(process_chunk, chunks, use_processes=True)
        ```
    """
    if num_workers is None:
        num_workers = get_parallel_config().max_workers_cpu

    if len(items) <= num_workers:
        return [[item] for item in items]

    chunk_size = (len(items) + num_workers - 1) // num_workers
    return batch_items(items, chunk_size)
