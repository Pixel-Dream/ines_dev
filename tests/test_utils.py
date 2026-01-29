"""Unit tests for INES utils module."""

import numpy as np
import pytest
from ines.utils import (
    parallelize,
    setup_hpc_environment,
    batch_process,
    timer,
    memory_usage,
    chunk_array,
)


class TestParallelize:
    """Tests for parallelize decorator."""
    
    def test_basic_parallelization(self):
        """Test basic parallel execution."""
        @parallelize(n_jobs=2)
        def simple_func(data):
            return [x * 2 for x in data]
        
        data = list(range(10))
        result = simple_func(data)
        
        # Result should be doubled
        assert len(result) > 0
    
    def test_single_job(self):
        """Test with single job (no parallelization)."""
        @parallelize(n_jobs=1)
        def simple_func(data):
            return [x * 2 for x in data]
        
        data = list(range(5))
        result = simple_func(data)
        
        assert len(result) > 0


class TestSetupHPCEnvironment:
    """Tests for HPC environment setup."""
    
    def test_basic_setup(self):
        """Test basic environment setup."""
        config = setup_hpc_environment(n_threads=4)
        
        assert 'n_threads' in config
        assert config['n_threads'] == 4
        assert 'cpu_count' in config
    
    def test_gpu_setup(self):
        """Test GPU configuration."""
        config = setup_hpc_environment(gpu_id=0)
        
        assert 'gpu_id' in config
        assert 'cuda_available' in config
    
    def test_memory_limit(self):
        """Test memory limit setting."""
        config = setup_hpc_environment(memory_limit="16GB")
        
        assert 'memory_limit' in config
        assert config['memory_limit'] == "16GB"


class TestBatchProcess:
    """Tests for batch processing."""
    
    def test_basic_batch_processing(self):
        """Test basic batch processing."""
        data = np.arange(100)
        
        def process_func(batch):
            return np.mean(batch)
        
        results = batch_process(data, process_func, batch_size=20, show_progress=False)
        
        assert len(results) == 5  # 100 / 20
        assert all(isinstance(r, (float, np.floating)) for r in results)
    
    def test_uneven_batches(self):
        """Test batch processing with uneven batch sizes."""
        data = np.arange(103)
        
        def process_func(batch):
            return len(batch)
        
        results = batch_process(data, process_func, batch_size=25, show_progress=False)
        
        # Should have 5 batches: 25, 25, 25, 25, 3
        assert len(results) == 5
        assert results[-1] == 3  # Last batch should have 3 elements


class TestTimer:
    """Tests for timer decorator."""
    
    def test_timer_decorator(self, capsys):
        """Test timer decorator."""
        @timer
        def slow_function():
            import time
            time.sleep(0.01)
            return 42
        
        result = slow_function()
        
        assert result == 42
        
        captured = capsys.readouterr()
        assert "slow_function executed" in captured.out
        assert "seconds" in captured.out


class TestMemoryUsage:
    """Tests for memory usage decorator."""
    
    def test_memory_decorator(self):
        """Test memory usage decorator."""
        @memory_usage
        def allocate_memory():
            # Allocate some memory
            _ = np.zeros((1000, 1000))
            return True
        
        # Should not raise error
        result = allocate_memory()
        assert result is True


class TestChunkArray:
    """Tests for array chunking."""
    
    def test_basic_chunking(self):
        """Test basic array chunking."""
        array = np.arange(100).reshape(10, 10)
        
        chunks = chunk_array(array, chunk_size=3, axis=0)
        
        # Should have 4 chunks: 3, 3, 3, 1
        assert len(chunks) == 4
        assert chunks[0].shape[0] == 3
        assert chunks[-1].shape[0] == 1
    
    def test_chunking_axis1(self):
        """Test chunking along axis 1."""
        array = np.arange(100).reshape(10, 10)
        
        chunks = chunk_array(array, chunk_size=4, axis=1)
        
        # Should have 3 chunks along axis 1
        assert len(chunks) == 3
        assert chunks[0].shape[1] == 4
        assert chunks[-1].shape[1] == 2
    
    def test_exact_chunks(self):
        """Test when array divides evenly into chunks."""
        array = np.arange(100).reshape(10, 10)
        
        chunks = chunk_array(array, chunk_size=5, axis=0)
        
        assert len(chunks) == 2
        assert all(c.shape[0] == 5 for c in chunks)
    
    def test_single_chunk(self):
        """Test when chunk size is larger than array."""
        array = np.arange(50).reshape(5, 10)
        
        chunks = chunk_array(array, chunk_size=10, axis=0)
        
        assert len(chunks) == 1
        assert chunks[0].shape == array.shape
    
    def test_invalid_axis(self):
        """Test invalid axis raises error."""
        array = np.arange(100).reshape(10, 10)
        
        with pytest.raises(ValueError, match="axis"):
            chunk_array(array, chunk_size=3, axis=2)


class TestEdgeCases:
    """Test edge cases for utils module."""
    
    def test_empty_batch_process(self):
        """Test batch processing with empty data."""
        data = np.array([])
        
        def process_func(batch):
            return np.mean(batch)
        
        results = batch_process(data, process_func, batch_size=10, show_progress=False)
        
        # Should handle empty data gracefully
        assert isinstance(results, list)
    
    def test_single_element_chunk(self):
        """Test chunking with single element."""
        array = np.array([[1]])
        
        chunks = chunk_array(array, chunk_size=1, axis=0)
        
        assert len(chunks) == 1
        assert chunks[0].shape == (1, 1)
