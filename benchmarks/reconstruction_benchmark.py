"""
Reconstruction Benchmark for INES Paper.

Compares INES-based probe selection against traditional methods:
- Co-expression networks
- Highly Variable Genes (HVG)
- Random selection

This script reproduces the key results for the paper demonstrating
INES superiority in identifying unimputable genes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings


def load_benchmark_data(
    data_path: str,
    dataset_name: str = "pbmc"
) -> 'AnnData':
    """
    Load benchmark dataset for reconstruction analysis.
    
    Parameters
    ----------
    data_path : str
        Path to data directory
    dataset_name : str
        Name of dataset to load
        
    Returns
    -------
    AnnData
        Loaded annotated data object
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy required for benchmark. Install with: pip install scanpy")
    
    data_file = Path(data_path) / f"{dataset_name}.h5ad"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")
    
    print(f"Loading {dataset_name} dataset from {data_file}")
    adata = sc.read_h5ad(data_file)
    
    return adata


def run_ines_selection(
    adata,
    n_probes: int,
    imputed_layer: str = "imputed"
) -> Dict:
    """
    Run INES-based probe selection.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_probes : int
        Number of probes to select
    imputed_layer : str
        Layer containing imputed data
        
    Returns
    -------
    Dict
        Selection results
    """
    from ines.design import optimize_gene_panel
    
    result = optimize_gene_panel(
        adata,
        n_probes=n_probes,
        imputed_layer=imputed_layer,
        strategy="ines"
    )
    
    return result


def run_hvg_selection(
    adata,
    n_probes: int
) -> Dict:
    """
    Run HVG-based probe selection (baseline).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_probes : int
        Number of probes to select
        
    Returns
    -------
    Dict
        Selection results
    """
    import scanpy as sc
    
    sc.pp.highly_variable_genes(adata, n_top_genes=n_probes)
    selected_genes = adata.var_names[adata.var['highly_variable']].tolist()
    
    return {
        'genes': selected_genes,
        'strategy': 'hvg',
        'n_probes': len(selected_genes)
    }


def run_coexpression_selection(
    adata,
    n_probes: int
) -> Dict:
    """
    Run co-expression network-based probe selection (baseline).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_probes : int
        Number of probes to select
        
    Returns
    -------
    Dict
        Selection results
    """
    # Compute gene-gene correlation matrix
    if hasattr(adata.X, 'toarray'):
        expr_data = adata.X.toarray()
    else:
        expr_data = adata.X
    
    # Calculate pairwise correlations
    gene_corr = np.corrcoef(expr_data.T)
    
    # Select genes with low average correlation (diverse set)
    avg_corr = np.mean(np.abs(gene_corr), axis=1)
    selected_indices = np.argsort(avg_corr)[:n_probes]
    selected_genes = adata.var_names[selected_indices].tolist()
    
    return {
        'genes': selected_genes,
        'strategy': 'coexpression',
        'n_probes': len(selected_genes)
    }


def run_random_selection(
    adata,
    n_probes: int,
    random_state: int = 42
) -> Dict:
    """
    Run random probe selection (negative control).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_probes : int
        Number of probes to select
    random_state : int
        Random seed
        
    Returns
    -------
    Dict
        Selection results
    """
    np.random.seed(random_state)
    selected_indices = np.random.choice(adata.n_vars, n_probes, replace=False)
    selected_genes = adata.var_names[selected_indices].tolist()
    
    return {
        'genes': selected_genes,
        'strategy': 'random',
        'n_probes': len(selected_genes)
    }


def evaluate_reconstruction(
    adata,
    selected_genes: List[str],
    n_neighbors: int = 15
) -> Dict:
    """
    Evaluate reconstruction quality using selected genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    selected_genes : List[str]
        Selected gene panel
    n_neighbors : int
        Number of neighbors for evaluation
        
    Returns
    -------
    Dict
        Evaluation metrics
    """
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    # Full dataset analysis
    adata_full = adata.copy()
    sc.pp.normalize_total(adata_full)
    sc.pp.log1p(adata_full)
    sc.pp.pca(adata_full)
    sc.pp.neighbors(adata_full, n_neighbors=n_neighbors)
    sc.tl.umap(adata_full)
    sc.tl.leiden(adata_full)
    
    # Subset dataset analysis
    adata_subset = adata[:, selected_genes].copy()
    sc.pp.normalize_total(adata_subset)
    sc.pp.log1p(adata_subset)
    sc.pp.pca(adata_subset)
    sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors)
    sc.tl.umap(adata_subset)
    sc.tl.leiden(adata_subset)
    
    # Calculate metrics
    ari = adjusted_rand_score(
        adata_full.obs['leiden'],
        adata_subset.obs['leiden']
    )
    
    # Silhouette score (if cell type labels available)
    if 'cell_type' in adata.obs.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs['cell_type'])
        
        silhouette_full = silhouette_score(adata_full.obsm['X_pca'], labels)
        silhouette_subset = silhouette_score(adata_subset.obsm['X_pca'], labels)
    else:
        silhouette_full = None
        silhouette_subset = None
    
    # Neighborhood preservation
    def neighborhood_preservation(adata1, adata2, k=n_neighbors):
        overlap_scores = []
        for i in range(min(1000, adata1.n_obs)):
            neighbors1 = set(adata1.obsp['distances'][i].indices)
            neighbors2 = set(adata2.obsp['distances'][i].indices)
            overlap = len(neighbors1 & neighbors2) / k
            overlap_scores.append(overlap)
        return np.mean(overlap_scores)
    
    neighborhood_score = neighborhood_preservation(adata_full, adata_subset)
    
    return {
        'ari': ari,
        'neighborhood_preservation': neighborhood_score,
        'silhouette_full': silhouette_full,
        'silhouette_subset': silhouette_subset,
        'n_genes': len(selected_genes)
    }


def run_full_benchmark(
    data_path: str,
    dataset_name: str = "pbmc",
    n_probes_list: List[int] = [100, 200, 300, 500],
    output_dir: str = "results",
    imputed_layer: str = "imputed"
) -> pd.DataFrame:
    """
    Run complete benchmark comparing all methods.
    
    Parameters
    ----------
    data_path : str
        Path to data directory
    dataset_name : str
        Name of dataset
    n_probes_list : List[int]
        List of probe panel sizes to test
    output_dir : str
        Output directory for results
    imputed_layer : str
        Layer containing imputed data
        
    Returns
    -------
    pd.DataFrame
        Benchmark results
    """
    # Load data
    adata = load_benchmark_data(data_path, dataset_name)
    
    results = []
    
    for n_probes in n_probes_list:
        print(f"\nBenchmarking with {n_probes} probes...")
        
        # Run all methods
        methods = {
            'INES': lambda: run_ines_selection(adata, n_probes, imputed_layer),
            'HVG': lambda: run_hvg_selection(adata, n_probes),
            'Co-expression': lambda: run_coexpression_selection(adata, n_probes),
            'Random': lambda: run_random_selection(adata, n_probes)
        }
        
        for method_name, method_func in methods.items():
            print(f"  Running {method_name}...")
            
            try:
                selection_result = method_func()
                eval_result = evaluate_reconstruction(
                    adata,
                    selection_result['genes']
                )
                
                results.append({
                    'method': method_name,
                    'n_probes': n_probes,
                    'dataset': dataset_name,
                    'ari': eval_result['ari'],
                    'neighborhood_preservation': eval_result['neighborhood_preservation'],
                    'silhouette_full': eval_result['silhouette_full'],
                    'silhouette_subset': eval_result['silhouette_subset']
                })
                
            except Exception as e:
                print(f"    Error with {method_name}: {e}")
                warnings.warn(f"Failed to run {method_name}: {e}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / f"{dataset_name}_benchmark_results.csv", index=False)
    
    # Create plots
    plot_benchmark_results(results_df, output_dir, dataset_name)
    
    return results_df


def plot_benchmark_results(
    results_df: pd.DataFrame,
    output_dir: str,
    dataset_name: str
):
    """
    Create visualization of benchmark results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    output_dir : str
        Output directory
    dataset_name : str
        Dataset name
    """
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot ARI
    sns.lineplot(
        data=results_df,
        x='n_probes',
        y='ari',
        hue='method',
        marker='o',
        ax=axes[0]
    )
    axes[0].set_title('Clustering Agreement (ARI)')
    axes[0].set_xlabel('Number of Probes')
    axes[0].set_ylabel('Adjusted Rand Index')
    
    # Plot neighborhood preservation
    sns.lineplot(
        data=results_df,
        x='n_probes',
        y='neighborhood_preservation',
        hue='method',
        marker='o',
        ax=axes[1]
    )
    axes[1].set_title('Neighborhood Preservation')
    axes[1].set_xlabel('Number of Probes')
    axes[1].set_ylabel('Preservation Score')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{dataset_name}_benchmark_plots.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run INES reconstruction benchmark")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--dataset", type=str, default="pbmc", help="Dataset name")
    parser.add_argument("--probes", type=int, nargs="+", default=[100, 200, 300, 500],
                       help="List of probe panel sizes to test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--imputed-layer", type=str, default="imputed",
                       help="Layer containing imputed data")
    
    args = parser.parse_args()
    
    results = run_full_benchmark(
        data_path=args.data_path,
        dataset_name=args.dataset,
        n_probes_list=args.probes,
        output_dir=args.output_dir,
        imputed_layer=args.imputed_layer
    )
    
    print("\nBenchmark complete!")
    print(results)
