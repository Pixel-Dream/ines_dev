"""
Utilities module for INES package.

HPC helpers, multiprocessing decorators, and performance optimization utilities.
"""

import functools
import multiprocessing as mp
from typing import Callable, Any, Optional, List, Tuple
import numpy as np
import warnings
import time


def parallelize(
    func: Optional[Callable] = None,
    n_jobs: int = -1,
    backend: str = "multiprocessing",
) -> Callable:
    """
    Decorator for parallelizing function execution across multiple cores.
    
    Parameters
    ----------
    func : Callable, optional
        Function to parallelize (automatically set when used as decorator)
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all available cores)
    backend : str, default="multiprocessing"
        Parallelization backend: "multiprocessing" or "threading"
        
    Returns
    -------
    Callable
        Decorated function with parallel execution
        
    Examples
    --------
    >>> @parallelize(n_jobs=4)
    >>> def compute_scores(data):
    >>>     return expensive_computation(data)
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Determine number of jobs
            n_cores = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
            
            # Check if input is list-like for batching
            if len(args) > 0 and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
                data = args[0]
                
                # Split data into chunks
                chunk_size = max(1, len(data) // n_cores)
                chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
                
                # Parallel execution
                if backend == "multiprocessing":
                    with mp.Pool(n_cores) as pool:
                        results = pool.map(f, chunks)
                else:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=n_cores) as executor:
                        results = list(executor.map(f, chunks))
                
                # Flatten results if needed
                if isinstance(results[0], list):
                    results = [item for sublist in results for item in sublist]
                
                return results
            else:
                # Single execution if data not suitable for batching
                return f(*args, **kwargs)
        
        return wrapper
    
    # Handle both @parallelize and @parallelize()
    if func is None:
        return decorator
    else:
        return decorator(func)


def setup_hpc_environment(
    n_threads: Optional[int] = None,
    gpu_id: Optional[int] = None,
    memory_limit: Optional[str] = None,
) -> dict:
    """
    Configure HPC environment for optimal performance.
    
    Sets environment variables for NumPy, PyTorch, and other libraries
    to optimize performance on HPC clusters.
    
    Parameters
    ----------
    n_threads : int, optional
        Number of threads for BLAS/LAPACK operations
    gpu_id : int, optional
        GPU device ID to use
    memory_limit : str, optional
        Memory limit (e.g., "32GB")
        
    Returns
    -------
    dict
        Environment configuration
        
    Examples
    --------
    >>> config = setup_hpc_environment(n_threads=16, gpu_id=0)
    """
    import os
    
    config = {}
    
    # Set threading for NumPy/BLAS
    if n_threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
        config['n_threads'] = n_threads
    
    # Set GPU device
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        config['gpu_id'] = gpu_id
        
        # Check if PyTorch is available
        try:
            import torch
            if torch.cuda.is_available():
                config['cuda_available'] = True
                config['cuda_device'] = torch.cuda.get_device_name(gpu_id)
            else:
                warnings.warn("CUDA not available despite GPU ID being set")
                config['cuda_available'] = False
        except ImportError:
            config['cuda_available'] = False
    
    # Memory limit (informational)
    if memory_limit is not None:
        config['memory_limit'] = memory_limit
    
    # System info
    config['cpu_count'] = mp.cpu_count()
    
    return config


def batch_process(
    data: Any,
    func: Callable,
    batch_size: int = 1000,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> List[Any]:
    """
    Process data in batches for memory efficiency.
    
    Parameters
    ----------
    data : Any
        Input data (must be indexable)
    func : Callable
        Function to apply to each batch
    batch_size : int, default=1000
        Size of each batch
    show_progress : bool, default=True
        Show progress bar
    n_jobs : int, default=1
        Number of parallel jobs
        
    Returns
    -------
    List[Any]
        Processed results
    """
    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    results = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch = data[start_idx:end_idx]
        batch_result = func(batch)
        results.append(batch_result)
        
        if show_progress and i % 10 == 0:
            print(f"Processed {i+1}/{n_batches} batches ({100*(i+1)/n_batches:.1f}%)")
    
    return results


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Parameters
    ----------
    func : Callable
        Function to time
        
    Returns
    -------
    Callable
        Decorated function that prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
        
        return result
    
    return wrapper


def memory_usage(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of a function.
    
    Parameters
    ----------
    func : Callable
        Function to monitor
        
    Returns
    -------
    Callable
        Decorated function that reports memory usage
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            print(f"{func.__name__} memory usage: {mem_diff:.2f} MB (total: {mem_after:.2f} MB)")
            
            return result
        except ImportError:
            warnings.warn("psutil not installed. Cannot monitor memory usage.")
            return func(*args, **kwargs)
    
    return wrapper


def chunk_array(
    array: np.ndarray,
    chunk_size: int,
    axis: int = 0,
) -> List[np.ndarray]:
    """
    Split array into chunks along specified axis.
    
    Parameters
    ----------
    array : np.ndarray
        Array to split
    chunk_size : int
        Size of each chunk
    axis : int, default=0
        Axis along which to split
        
    Returns
    -------
    List[np.ndarray]
        List of array chunks
    """
    n_elements = array.shape[axis]
    n_chunks = (n_elements + chunk_size - 1) // chunk_size
    
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_elements)
        
        if axis == 0:
            chunk = array[start_idx:end_idx]
        elif axis == 1:
            chunk = array[:, start_idx:end_idx]
        else:
            raise ValueError("Only axis=0 or axis=1 supported")
        
        chunks.append(chunk)
    
    return chunks


def validate_adata(adata) -> Tuple[bool, List[str]]:
    """
    Validate AnnData object for INES analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object to validate
        
    Returns
    -------
    Tuple[bool, List[str]]
        Validation status and list of issues
    """
    issues = []
    
    try:
        # Check basic structure
        if adata.X is None:
            issues.append("Missing expression matrix (X)")
        
        if adata.n_obs == 0:
            issues.append("No cells in dataset")
        
        if adata.n_vars == 0:
            issues.append("No genes in dataset")
        
        # Check for sparse vs dense
        if hasattr(adata.X, 'toarray'):
            if adata.X.nnz == 0:
                issues.append("Expression matrix is all zeros")
        else:
            if np.all(adata.X == 0):
                issues.append("Expression matrix is all zeros")
        
        # Check for required metadata
        if len(adata.var_names) == 0:
            issues.append("Missing gene names")
        
        if len(adata.obs_names) == 0:
            issues.append("Missing cell names")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
        
    except Exception as e:
        issues.append(f"Validation error: {str(e)}")
        return False, issues
