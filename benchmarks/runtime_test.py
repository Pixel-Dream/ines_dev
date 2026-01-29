"""
Runtime Test for INES Scalability.

Tests EMD computation scalability on large datasets with various configurations.
Measures performance across different:
- Dataset sizes (cells x genes)
- Parallelization settings
- Memory usage patterns
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings


def generate_synthetic_data(
    n_cells: int,
    n_genes: int,
    sparse_rate: float = 0.7,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate synthetic scRNA-seq data for testing.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_genes : int
        Number of genes
    sparse_rate : float
        Fraction of zeros (sparsity)
    random_state : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Synthetic expression matrix
    """
    np.random.seed(random_state)
    
    # Generate from negative binomial (typical for scRNA-seq)
    data = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    
    # Add sparsity
    mask = np.random.rand(n_cells, n_genes) < sparse_rate
    data[mask] = 0
    
    return data.astype(np.float32)


def benchmark_emd_computation(
    n_cells: int,
    n_genes: int,
    n_repeats: int = 3
) -> Dict:
    """
    Benchmark EMD computation time.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_genes : int
        Number of genes
    n_repeats : int
        Number of repetitions
        
    Returns
    -------
    Dict
        Timing results
    """
    from ines.metrics import calculate_ines_score
    
    # Generate data
    original = generate_synthetic_data(n_cells, n_genes)
    imputed = original + np.random.randn(n_cells, n_genes) * 0.1 * np.std(original)
    imputed = np.maximum(imputed, 0)
    
    times = []
    memory_usage = []
    
    for i in range(n_repeats):
        # Measure time
        start_time = time.time()
        scores = calculate_ines_score(original, imputed)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Measure memory (if psutil available)
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            memory_usage.append(mem_mb)
        except ImportError:
            pass
    
    return {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_memory_mb': np.mean(memory_usage) if memory_usage else None,
        'genes_per_second': n_genes / np.mean(times)
    }


def benchmark_parallel_scaling(
    n_cells: int = 10000,
    n_genes: int = 2000,
    n_jobs_list: List[int] = [1, 2, 4, 8, 16]
) -> pd.DataFrame:
    """
    Benchmark parallel scaling performance.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_genes : int
        Number of genes
    n_jobs_list : List[int]
        List of parallel job counts to test
        
    Returns
    -------
    pd.DataFrame
        Scaling results
    """
    from ines.utils import setup_hpc_environment
    
    results = []
    
    # Generate data once
    original = generate_synthetic_data(n_cells, n_genes)
    imputed = original + np.random.randn(n_cells, n_genes) * 0.1 * np.std(original)
    imputed = np.maximum(imputed, 0)
    
    for n_jobs in n_jobs_list:
        print(f"Testing with {n_jobs} jobs...")
        
        # Configure environment
        setup_hpc_environment(n_threads=n_jobs)
        
        # Time computation
        start_time = time.time()
        
        from ines.metrics import calculate_ines_score
        scores = calculate_ines_score(original, imputed)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate speedup
        if n_jobs == 1:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        results.append({
            'n_jobs': n_jobs,
            'time_seconds': elapsed,
            'speedup': speedup,
            'efficiency': speedup / n_jobs
        })
    
    return pd.DataFrame(results)


def benchmark_dataset_sizes(
    cell_counts: List[int] = [1000, 5000, 10000, 20000, 50000],
    gene_counts: List[int] = [500, 1000, 2000, 5000],
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Benchmark across different dataset sizes.
    
    Parameters
    ----------
    cell_counts : List[int]
        List of cell counts to test
    gene_counts : List[int]
        List of gene counts to test
    output_dir : str
        Output directory
        
    Returns
    -------
    pd.DataFrame
        Scaling results
    """
    results = []
    
    total_tests = len(cell_counts) * len(gene_counts)
    test_num = 0
    
    for n_cells in cell_counts:
        for n_genes in gene_counts:
            test_num += 1
            print(f"\nTest {test_num}/{total_tests}: {n_cells} cells x {n_genes} genes")
            
            try:
                result = benchmark_emd_computation(n_cells, n_genes)
                results.append(result)
                
                print(f"  Time: {result['mean_time']:.2f}s "
                      f"({result['genes_per_second']:.1f} genes/s)")
                
            except Exception as e:
                print(f"  Error: {e}")
                warnings.warn(f"Failed test {n_cells}x{n_genes}: {e}")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / "runtime_scaling_results.csv", index=False)
    
    # Create visualizations
    plot_runtime_results(results_df, output_dir)
    
    return results_df


def plot_runtime_results(
    results_df: pd.DataFrame,
    output_dir: str
):
    """
    Visualize runtime scaling results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Runtime results
    output_dir : str
        Output directory
    """
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Time vs number of genes
    ax = axes[0, 0]
    for n_cells in results_df['n_cells'].unique():
        subset = results_df[results_df['n_cells'] == n_cells]
        ax.plot(subset['n_genes'], subset['mean_time'], marker='o', label=f'{n_cells} cells')
    ax.set_xlabel('Number of Genes')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Runtime vs Gene Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time vs number of cells
    ax = axes[0, 1]
    for n_genes in results_df['n_genes'].unique():
        subset = results_df[results_df['n_genes'] == n_genes]
        ax.plot(subset['n_cells'], subset['mean_time'], marker='o', label=f'{n_genes} genes')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Runtime vs Cell Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Throughput (genes/second)
    ax = axes[1, 0]
    scatter = ax.scatter(
        results_df['n_cells'],
        results_df['n_genes'],
        c=results_df['genes_per_second'],
        s=100,
        cmap='viridis',
        alpha=0.6
    )
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Number of Genes')
    ax.set_title('Throughput (Genes/Second)')
    plt.colorbar(scatter, ax=ax, label='Genes/Second')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Memory usage (if available)
    ax = axes[1, 1]
    if 'mean_memory_mb' in results_df.columns and results_df['mean_memory_mb'].notna().any():
        scatter = ax.scatter(
            results_df['n_cells'] * results_df['n_genes'],
            results_df['mean_memory_mb'],
            alpha=0.6,
            s=100
        )
        ax.set_xlabel('Total Elements (Cells × Genes)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Scaling')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Memory data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Memory Usage')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "runtime_scaling_plots.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def test_memory_efficiency(
    n_cells: int = 50000,
    n_genes: int = 5000,
    chunk_sizes: List[int] = [100, 500, 1000, 2000]
) -> pd.DataFrame:
    """
    Test memory efficiency with different chunking strategies.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_genes : int
        Number of genes
    chunk_sizes : List[int]
        List of chunk sizes to test
        
    Returns
    -------
    pd.DataFrame
        Memory efficiency results
    """
    from ines.utils import chunk_array, batch_process
    
    results = []
    
    # Generate large dataset
    print(f"Generating {n_cells} x {n_genes} dataset...")
    data = generate_synthetic_data(n_cells, n_genes)
    
    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk size: {chunk_size}")
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Chunk and process
            start_time = time.time()
            chunks = chunk_array(data, chunk_size, axis=1)
            
            # Simulate processing
            processed = [np.mean(chunk, axis=0) for chunk in chunks]
            
            end_time = time.time()
            mem_after = process.memory_info().rss / 1024 / 1024
            
            results.append({
                'chunk_size': chunk_size,
                'time_seconds': end_time - start_time,
                'memory_mb': mem_after - mem_before,
                'n_chunks': len(chunks)
            })
            
            print(f"  Time: {end_time - start_time:.2f}s, Memory: {mem_after - mem_before:.1f} MB")
            
        except ImportError:
            print("  psutil not available, skipping memory test")
            continue
        except Exception as e:
            print(f"  Error: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run INES runtime scalability tests")
    parser.add_argument("--test-type", type=str, default="scaling",
                       choices=["scaling", "parallel", "memory", "all"],
                       help="Type of test to run")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--max-cells", type=int, default=50000,
                       help="Maximum number of cells to test")
    parser.add_argument("--max-genes", type=int, default=5000,
                       help="Maximum number of genes to test")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.test_type in ["scaling", "all"]:
        print("=" * 50)
        print("Running dataset size scaling tests...")
        print("=" * 50)
        
        cell_counts = [1000, 5000, 10000, 20000, min(50000, args.max_cells)]
        gene_counts = [500, 1000, 2000, min(5000, args.max_genes)]
        
        results = benchmark_dataset_sizes(
            cell_counts=cell_counts,
            gene_counts=gene_counts,
            output_dir=args.output_dir
        )
        print("\nScaling test results:")
        print(results)
    
    if args.test_type in ["parallel", "all"]:
        print("\n" + "=" * 50)
        print("Running parallel scaling tests...")
        print("=" * 50)
        
        parallel_results = benchmark_parallel_scaling()
        parallel_results.to_csv(output_path / "parallel_scaling_results.csv", index=False)
        print("\nParallel scaling results:")
        print(parallel_results)
    
    if args.test_type in ["memory", "all"]:
        print("\n" + "=" * 50)
        print("Running memory efficiency tests...")
        print("=" * 50)
        
        memory_results = test_memory_efficiency()
        memory_results.to_csv(output_path / "memory_efficiency_results.csv", index=False)
        print("\nMemory efficiency results:")
        print(memory_results)
    
    print(f"\n✓ All tests complete! Results saved to {args.output_dir}")
