"""
Thread safety utilities for concurrent operations.

This module provides thread-safe data structures and synchronization primitives
for safe concurrent access to shared resources in the sleep analysis framework.
"""

import threading
from typing import Any, Dict, Optional, Callable, TypeVar
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RWLock:
    """
    Read-Write Lock implementation for optimized concurrent access.

    Multiple readers can hold the lock simultaneously, but writers have exclusive access.
    This is ideal for scenarios where reads are frequent and writes are infrequent.

    Example:
        ```python
        lock = RWLock()

        # Multiple readers can acquire simultaneously
        with lock.reader():
            data = shared_resource.read()

        # Writers get exclusive access
        with lock.writer():
            shared_resource.write(data)
        ```
    """

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(threading.Lock())
        self._write_ready = threading.Condition(threading.Lock())

    def acquire_read(self):
        """Acquire a read lock. Multiple readers can hold the lock simultaneously."""
        self._read_ready.acquire()
        try:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """Release a read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """Acquire a write lock. Writers have exclusive access."""
        self._write_ready.acquire()
        self._writers += 1
        self._write_ready.release()

        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """Release a write lock."""
        self._read_ready.release()
        self._write_ready.acquire()
        self._writers -= 1
        if self._writers == 0:
            self._write_ready.notify_all()
        self._write_ready.release()

    def reader(self):
        """Context manager for read operations."""
        return _ReadContext(self)

    def writer(self):
        """Context manager for write operations."""
        return _WriteContext(self)


class _ReadContext:
    """Context manager for read lock."""

    def __init__(self, lock: RWLock):
        self.lock = lock

    def __enter__(self):
        self.lock.acquire_read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release_read()
        return False


class _WriteContext:
    """Context manager for write lock."""

    def __init__(self, lock: RWLock):
        self.lock = lock

    def __enter__(self):
        self.lock.acquire_write()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release_write()
        return False


class ThreadSafeCache:
    """
    Thread-safe cache with read-write locking for concurrent access.

    This cache allows multiple concurrent readers but ensures exclusive access for writers.
    Ideal for caching computed results that are expensive to generate.

    Example:
        ```python
        cache = ThreadSafeCache()

        # Thread-safe read
        if value := cache.get(key):
            return value

        # Thread-safe write
        cache.set(key, expensive_computation())
        ```
    """

    def __init__(self):
        self._cache: Dict[Any, Any] = {}
        self._lock = RWLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'clears': 0
        }
        self._stats_lock = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Thread-safe cache retrieval.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self._lock.reader():
            value = self._cache.get(key, default)

        with self._stats_lock:
            if value is not None and key in self._cache:
                self._stats['hits'] += 1
            else:
                self._stats['misses'] += 1

        return value

    def set(self, key: Any, value: Any) -> None:
        """
        Thread-safe cache insertion.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock.writer():
            self._cache[key] = value

        with self._stats_lock:
            self._stats['sets'] += 1

    def get_or_compute(self, key: Any, compute_fn: Callable[[], T]) -> T:
        """
        Get cached value or compute and cache it if not present.

        Uses double-checked locking pattern to minimize write lock contention.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached

        Returns:
            Cached or computed value
        """
        # First check with read lock (fast path)
        value = self.get(key)
        if value is not None:
            return value

        # Not found, acquire write lock to compute
        with self._lock.writer():
            # Double-check in case another thread computed it
            if key in self._cache:
                with self._stats_lock:
                    self._stats['hits'] += 1
                return self._cache[key]

            # Compute and cache
            value = compute_fn()
            self._cache[key] = value

        with self._stats_lock:
            self._stats['sets'] += 1

        return value

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock.writer():
            self._cache.clear()

        with self._stats_lock:
            self._stats['clears'] += 1

    def contains(self, key: Any) -> bool:
        """Check if key exists in cache."""
        with self._lock.reader():
            return key in self._cache

    def size(self) -> int:
        """Get current cache size."""
        with self._lock.reader():
            return len(self._cache)

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, sets, and clears counts
        """
        with self._stats_lock:
            return self._stats.copy()

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as a percentage (0-100)
        """
        stats = self.get_stats()
        total = stats['hits'] + stats['misses']
        if total == 0:
            return 0.0
        return (stats['hits'] / total) * 100.0


def synchronized(lock_attr: str = '_lock'):
    """
    Decorator to synchronize method access using an instance lock.

    Args:
        lock_attr: Name of the lock attribute on the instance

    Example:
        ```python
        class MyClass:
            def __init__(self):
                self._lock = threading.Lock()

            @synchronized()
            def thread_safe_method(self):
                # This method is automatically synchronized
                pass
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr)
            with lock:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


class AtomicCounter:
    """
    Thread-safe atomic counter.

    Example:
        ```python
        counter = AtomicCounter()
        counter.increment()  # Thread-safe
        value = counter.value  # Thread-safe read
        ```
    """

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= delta
            return self._value

    @property
    def value(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value

    def reset(self, value: int = 0) -> None:
        """Reset counter to specified value."""
        with self._lock:
            self._value = value
