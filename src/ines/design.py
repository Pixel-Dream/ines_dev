"""
Probe Selection module for INES package.

Implements greedy selection algorithms for designing optimized gene panels
for spatial transcriptomics experiments (MERFISH, Xenium).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import warnings


def greedy_probe_selection(
    ines_scores: np.ndarray,
    gene_names: np.ndarray,
    n_probes: int,
    expression_data: Optional[np.ndarray] = None,
    correlation_threshold: float = 0.3,
    max_iterations: int = 1000,
) -> Tuple[List[str], List[int], List[float]]:
    """
    Greedy selection algorithm for probe panel design.
    
    Selects genes with high INES scores (unimputable genes) that are poorly
    correlated with each other to maximize information content.
    
    Parameters
    ----------
    ines_scores : np.ndarray
        INES scores for each gene (higher = less imputable)
    gene_names : np.ndarray
        Names of genes corresponding to scores
    n_probes : int
        Number of probes to select
    expression_data : np.ndarray, optional
        Expression matrix for computing correlations (cells x genes)
    correlation_threshold : float, default=0.3
        Maximum allowed correlation between selected genes
    max_iterations : int, default=1000
        Maximum iterations for selection
        
    Returns
    -------
    Tuple[List[str], List[int], List[float]]
        Selected gene names, indices, and their INES scores
        
    Examples
    --------
    >>> scores = np.array([0.8, 0.6, 0.9, 0.3, 0.7])
    >>> genes = np.array(['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
    >>> selected, indices, scores = greedy_probe_selection(scores, genes, n_probes=3)
    """
    if len(ines_scores) != len(gene_names):
        raise ValueError("ines_scores and gene_names must have the same length")
    
    if n_probes > len(ines_scores):
        warnings.warn(f"n_probes ({n_probes}) exceeds number of genes ({len(ines_scores)}). "
                     f"Selecting all genes.")
        n_probes = len(ines_scores)
    
    # Sort genes by INES score (descending - higher score = less imputable)
    sorted_indices = np.argsort(ines_scores)[::-1]
    
    selected_indices = []
    selected_genes = []
    selected_scores = []
    
    # Compute correlation matrix if expression data provided
    correlation_matrix = None
    if expression_data is not None:
        correlation_matrix = np.corrcoef(expression_data.T)
    
    # Greedy selection
    for iteration in range(max_iterations):
        if len(selected_indices) >= n_probes:
            break
            
        for idx in sorted_indices:
            if idx in selected_indices:
                continue
            
            # Check correlation with already selected genes
            if correlation_matrix is not None and len(selected_indices) > 0:
                correlations = np.abs(correlation_matrix[idx, selected_indices])
                if np.any(correlations > correlation_threshold):
                    continue
            
            # Add gene to selection
            selected_indices.append(idx)
            selected_genes.append(gene_names[idx])
            selected_scores.append(ines_scores[idx])
            
            if len(selected_indices) >= n_probes:
                break
        
        # Break if no more genes can be added
        if len(selected_indices) == 0 or iteration > 0:
            break
    
    return selected_genes, selected_indices, selected_scores


def optimize_gene_panel(
    adata,
    n_probes: int,
    imputed_layer: str = "imputed",
    strategy: str = "ines",
    hvg_n_top: int = 2000,
    correlation_threshold: float = 0.3,
) -> Dict[str, any]:
    """
    Optimize gene panel selection for spatial transcriptomics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with expression data
    n_probes : int
        Number of probes to select
    imputed_layer : str, default="imputed"
        Layer containing imputed data
    strategy : str, default="ines"
        Selection strategy: "ines", "hvg", or "combined"
    hvg_n_top : int, default=2000
        Number of highly variable genes to consider
    correlation_threshold : float, default=0.3
        Correlation threshold for greedy selection
        
    Returns
    -------
    Dict[str, any]
        Dictionary containing selected genes and metadata
        
    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> result = optimize_gene_panel(adata, n_probes=300)
    >>> selected_genes = result['genes']
    """
    try:
        import scanpy as sc
        from .metrics import compute_gene_reliability_matrix
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}")
    
    # Compute INES scores
    ines_scores, gene_names = compute_gene_reliability_matrix(
        adata,
        imputed_layer=imputed_layer,
        n_top_genes=hvg_n_top
    )
    
    # Get expression data for correlation computation
    if hasattr(adata.X, 'toarray'):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X
    
    # Apply selection strategy
    if strategy == "ines":
        # Pure INES-based selection
        selected_genes, selected_indices, selected_scores = greedy_probe_selection(
            ines_scores,
            gene_names,
            n_probes,
            expression_data=expression_data,
            correlation_threshold=correlation_threshold
        )
    elif strategy == "hvg":
        # Fall back to highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_probes)
        selected_genes = adata.var_names[adata.var['highly_variable']].tolist()[:n_probes]
        selected_indices = np.where(adata.var['highly_variable'])[0][:n_probes]
        selected_scores = ines_scores[selected_indices].tolist()
    elif strategy == "combined":
        # Combine INES with HVG
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg_n_top)
        hvg_mask = adata.var['highly_variable'].values
        
        # Boost INES scores for HVGs
        combined_scores = ines_scores.copy()
        combined_scores[hvg_mask] *= 1.5
        
        selected_genes, selected_indices, selected_scores = greedy_probe_selection(
            combined_scores,
            gene_names,
            n_probes,
            expression_data=expression_data,
            correlation_threshold=correlation_threshold
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return {
        'genes': selected_genes,
        'indices': selected_indices,
        'ines_scores': selected_scores,
        'strategy': strategy,
        'n_probes': len(selected_genes),
        'correlation_threshold': correlation_threshold
    }


def evaluate_panel_coverage(
    adata,
    selected_genes: List[str],
    n_neighbors: int = 15,
) -> Dict[str, float]:
    """
    Evaluate how well a gene panel covers the transcriptomic space.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    selected_genes : List[str]
        List of selected gene names
    n_neighbors : int, default=15
        Number of neighbors for coverage analysis
        
    Returns
    -------
    Dict[str, float]
        Coverage metrics
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for this function")
    
    # Compute PCA on full dataset
    adata_full = adata.copy()
    sc.pp.pca(adata_full)
    
    # Compute PCA on selected genes
    adata_subset = adata[:, selected_genes].copy()
    sc.pp.pca(adata_subset)
    
    # Compare neighborhoods
    sc.pp.neighbors(adata_full, n_neighbors=n_neighbors)
    sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors)
    
    # Simple coverage metric (can be enhanced)
    coverage_score = np.mean([
        len(set(adata_full.obsp['distances'][i].indices) & 
            set(adata_subset.obsp['distances'][i].indices)) / n_neighbors
        for i in range(min(100, adata.n_obs))
    ])
    
    return {
        'coverage_score': coverage_score,
        'n_genes_selected': len(selected_genes),
        'total_genes': adata.n_vars
    }
