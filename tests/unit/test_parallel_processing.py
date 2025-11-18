"""
Unit tests for parallel processing utilities.
"""

import pytest
import time
import pandas as pd
import numpy as np
from sleep_analysis.utils.parallel import (
    ParallelConfig,
    ParallelExecutor,
    parallel_map,
    batch_items,
    chunk_for_workers,
    get_parallel_config,
    set_parallel_config,
    enable_parallel_processing,
    disable_parallel_processing
)
from sleep_analysis.utils.thread_safety import (
    RWLock,
    ThreadSafeCache,
    AtomicCounter
)


@pytest.fixture(autouse=True)
def enable_parallel_for_tests():
    """Enable parallel processing for these specific tests."""
    enable_parallel_processing()
    yield
    # The session-level fixture will handle final cleanup


class TestParallelConfig:
    """Test ParallelConfig functionality."""

    def test_default_config(self):
        """Test default parallel configuration."""
        config = ParallelConfig()
        assert config.enabled is True
        assert config.max_workers_cpu is not None
        assert config.max_workers_io is not None
        assert config.chunk_size > 0

    def test_custom_config(self):
        """Test custom parallel configuration."""
        config = ParallelConfig(
            enabled=False,
            max_workers_cpu=2,
            max_workers_io=4,
            chunk_size=5
        )
        assert config.enabled is False
        assert config.max_workers_cpu == 2
        assert config.max_workers_io == 4
        assert config.chunk_size == 5

    def test_global_config(self):
        """Test global configuration management."""
        original_config = get_parallel_config()

        # Set new config
        new_config = ParallelConfig(max_workers_cpu=2)
        set_parallel_config(new_config)
        assert get_parallel_config().max_workers_cpu == 2

        # Restore original
        set_parallel_config(original_config)


class TestParallelExecutor:
    """Test ParallelExecutor functionality."""

    def test_threaded_map(self):
        """Test parallel map with threads."""
        def square(x):
            return x ** 2

        with ParallelExecutor() as executor:
            items = list(range(10))
            results = executor.map_threaded(square, items, desc="Testing threads")
            assert results == [x ** 2 for x in items]

    def test_process_map(self):
        """Test parallel map with processes."""
        def cube(x):
            return x ** 3

        with ParallelExecutor() as executor:
            items = list(range(10))
            results = executor.map_processes(cube, items, desc="Testing processes")
            assert results == [x ** 3 for x in items]

    def test_submit_tasks(self):
        """Test submitting heterogeneous tasks."""
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b

        with ParallelExecutor() as executor:
            tasks = [
                (add, (2, 3), {}),
                (multiply, (4, 5), {}),
                (add, (10, 20), {})
            ]
            results = executor.submit_tasks(tasks, desc="Testing mixed tasks")
            assert results == [5, 20, 30]

    def test_disabled_parallel(self):
        """Test executor with parallel processing disabled."""
        config = ParallelConfig(enabled=False)
        executor = ParallelExecutor(config)

        def identity(x):
            return x

        items = list(range(5))
        results = executor.map_threaded(identity, items)
        assert results == items


class TestParallelMap:
    """Test parallel_map convenience function."""

    def test_parallel_map_threads(self):
        """Test parallel_map with threads."""
        def double(x):
            return x * 2

        items = list(range(20))
        results = parallel_map(double, items, use_processes=False)
        assert results == [x * 2 for x in items]

    def test_parallel_map_processes(self):
        """Test parallel_map with processes."""
        def triple(x):
            return x * 3

        items = list(range(20))
        results = parallel_map(triple, items, use_processes=True)
        assert results == [x * 3 for x in items]

    def test_custom_workers(self):
        """Test parallel_map with custom worker count."""
        def identity(x):
            return x

        items = list(range(10))
        results = parallel_map(identity, items, max_workers=2, use_processes=True)
        assert results == items


class TestBatchingUtilities:
    """Test batching and chunking utilities."""

    def test_batch_items(self):
        """Test batch_items function."""
        items = list(range(25))
        batches = batch_items(items, batch_size=10)
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

    def test_batch_items_exact(self):
        """Test batch_items with exact division."""
        items = list(range(20))
        batches = batch_items(items, batch_size=5)
        assert len(batches) == 4
        assert all(len(batch) == 5 for batch in batches)

    def test_chunk_for_workers(self):
        """Test chunk_for_workers function."""
        items = list(range(100))
        chunks = chunk_for_workers(items, num_workers=4)
        assert len(chunks) == 4
        # All chunks should be roughly equal
        assert all(20 <= len(chunk) <= 30 for chunk in chunks)


class TestRWLock:
    """Test Read-Write Lock implementation."""

    def test_multiple_readers(self):
        """Test that multiple readers can acquire lock simultaneously."""
        lock = RWLock()
        read_count = 0

        def reader():
            nonlocal read_count
            with lock.reader():
                current = read_count
                time.sleep(0.01)  # Simulate read operation
                assert read_count == current  # Should not change during read
                return True

        # Multiple readers should be able to run concurrently
        with ParallelExecutor() as executor:
            tasks = [(reader, (), {}) for _ in range(5)]
            results = executor.submit_tasks(tasks, use_processes=False)
            assert all(results)

    def test_writer_exclusivity(self):
        """Test that writers have exclusive access."""
        lock = RWLock()
        shared_value = [0]

        def writer(value):
            with lock.writer():
                current = shared_value[0]
                time.sleep(0.01)  # Simulate write operation
                shared_value[0] = current + value

        # Submit multiple writers
        with ParallelExecutor() as executor:
            tasks = [(writer, (1,), {}) for _ in range(10)]
            executor.submit_tasks(tasks, use_processes=False)

        # Final value should be sum of all increments
        assert shared_value[0] == 10


class TestThreadSafeCache:
    """Test ThreadSafeCache implementation."""

    def test_basic_get_set(self):
        """Test basic cache get/set operations."""
        cache = ThreadSafeCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None

    def test_get_with_default(self):
        """Test cache get with default value."""
        cache = ThreadSafeCache()
        assert cache.get("missing", "default") == "default"

    def test_get_or_compute(self):
        """Test get_or_compute functionality."""
        cache = ThreadSafeCache()
        compute_count = [0]

        def expensive_compute():
            compute_count[0] += 1
            return "computed_value"

        # First call should compute
        value1 = cache.get_or_compute("key1", expensive_compute)
        assert value1 == "computed_value"
        assert compute_count[0] == 1

        # Second call should use cache
        value2 = cache.get_or_compute("key1", expensive_compute)
        assert value2 == "computed_value"
        assert compute_count[0] == 1  # Should not recompute

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ThreadSafeCache()
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['sets'] == 1

    def test_hit_rate(self):
        """Test cache hit rate calculation."""
        cache = ThreadSafeCache()
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        hit_rate = cache.get_hit_rate()
        assert hit_rate == pytest.approx(66.67, rel=0.1)

    def test_clear(self):
        """Test cache clearing."""
        cache = ThreadSafeCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None

    def test_concurrent_access(self):
        """Test thread-safe concurrent cache access."""
        cache = ThreadSafeCache()

        def worker(i):
            key = f"key{i % 5}"  # Use 5 keys to create collisions
            value = cache.get_or_compute(key, lambda: f"value{i % 5}")
            return value

        with ParallelExecutor() as executor:
            tasks = [(worker, (i,), {}) for i in range(50)]
            results = executor.submit_tasks(tasks, use_processes=False)

        # Should have exactly 5 unique values
        assert len(set(results)) == 5


class TestAtomicCounter:
    """Test AtomicCounter implementation."""

    def test_increment(self):
        """Test counter increment."""
        counter = AtomicCounter()
        assert counter.value == 0
        counter.increment()
        assert counter.value == 1
        counter.increment(5)
        assert counter.value == 6

    def test_decrement(self):
        """Test counter decrement."""
        counter = AtomicCounter(10)
        counter.decrement()
        assert counter.value == 9
        counter.decrement(3)
        assert counter.value == 6

    def test_reset(self):
        """Test counter reset."""
        counter = AtomicCounter(5)
        counter.reset()
        assert counter.value == 0
        counter.reset(10)
        assert counter.value == 10

    def test_concurrent_increments(self):
        """Test thread-safe concurrent increments."""
        counter = AtomicCounter()

        def increment_worker():
            for _ in range(100):
                counter.increment()

        with ParallelExecutor() as executor:
            tasks = [(increment_worker, (), {}) for _ in range(10)]
            executor.submit_tasks(tasks, use_processes=False)

        # Should be exactly 1000 (10 workers × 100 increments each)
        assert counter.value == 1000


class TestEnableDisableParallel:
    """Test enable/disable parallel processing."""

    def test_disable_enable(self):
        """Test disabling and enabling parallel processing."""
        original_config = get_parallel_config()

        # Disable
        disable_parallel_processing()
        assert get_parallel_config().enabled is False

        # Enable
        enable_parallel_processing()
        assert get_parallel_config().enabled is True

        # Restore original
        set_parallel_config(original_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
