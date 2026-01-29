"""
Metrics module for INES package.

Contains vectorized Earth Mover's Distance (EMD) calculations and INES score computation
for evaluating imputation quality in scRNA-seq data.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Union, Tuple, Optional
import warnings


def earth_movers_distance(
    distribution1: np.ndarray,
    distribution2: np.ndarray,
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate Earth Mover's Distance (Wasserstein distance) between two distributions.
    
    Vectorized implementation for efficient computation across multiple gene pairs.
    
    Parameters
    ----------
    distribution1 : np.ndarray
        First distribution (can be 1D or 2D for vectorized computation)
    distribution2 : np.ndarray
        Second distribution (same shape as distribution1)
    weights1 : np.ndarray, optional
        Weights for first distribution
    weights2 : np.ndarray, optional
        Weights for second distribution
        
    Returns
    -------
    float or np.ndarray
        EMD value(s) between the distributions
        
    Examples
    --------
    >>> dist1 = np.array([0, 1, 2, 3, 4])
    >>> dist2 = np.array([1, 2, 3, 4, 5])
    >>> emd = earth_movers_distance(dist1, dist2)
    """
    # Handle edge cases
    if distribution1.shape != distribution2.shape:
        raise ValueError("Distributions must have the same shape")
    
    # Vectorized computation for 2D arrays (multiple genes)
    if distribution1.ndim == 2:
        results = np.zeros(distribution1.shape[0])
        for i in range(distribution1.shape[0]):
            results[i] = wasserstein_distance(
                distribution1[i], 
                distribution2[i],
                u_weights=weights1[i] if weights1 is not None else None,
                v_weights=weights2[i] if weights2 is not None else None
            )
        return results
    
    # Single distribution computation
    return wasserstein_distance(
        distribution1, 
        distribution2,
        u_weights=weights1,
        v_weights=weights2
    )


def calculate_ines_score(
    original_data: np.ndarray,
    imputed_data: np.ndarray,
    axis: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Calculate INES (Imputation Noise Evaluation Score) for genes.
    
    Lower INES scores indicate more reliable imputation.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original (non-imputed) expression matrix (cells x genes or genes x cells)
    imputed_data : np.ndarray
        Imputed expression matrix (same shape as original_data)
    axis : int, default=0
        Axis along which to compute EMD (0 for cell-wise, 1 for gene-wise)
    normalize : bool, default=True
        Whether to normalize scores to [0, 1] range
        
    Returns
    -------
    np.ndarray
        INES scores for each gene
        
    Examples
    --------
    >>> original = np.random.randn(1000, 100)  # 1000 cells, 100 genes
    >>> imputed = original + np.random.randn(1000, 100) * 0.1
    >>> scores = calculate_ines_score(original, imputed)
    """
    if original_data.shape != imputed_data.shape:
        raise ValueError("Original and imputed data must have the same shape")
    
    # Transpose if computing along genes
    if axis == 1:
        original_data = original_data.T
        imputed_data = imputed_data.T
    
    n_genes = original_data.shape[1]
    ines_scores = np.zeros(n_genes)
    
    # Calculate EMD for each gene across cells
    for gene_idx in range(n_genes):
        orig_dist = original_data[:, gene_idx]
        imp_dist = imputed_data[:, gene_idx]
        
        # Skip if all zeros
        if np.all(orig_dist == 0) and np.all(imp_dist == 0):
            ines_scores[gene_idx] = 0.0
            continue
            
        ines_scores[gene_idx] = wasserstein_distance(orig_dist, imp_dist)
    
    # Normalize to [0, 1] range
    if normalize and np.max(ines_scores) > 0:
        ines_scores = ines_scores / np.max(ines_scores)
    
    return ines_scores


def compute_gene_reliability_matrix(
    adata,
    imputed_layer: str = "imputed",
    raw_layer: str = "X",
    n_top_genes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gene reliability matrix from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with original and imputed counts
    imputed_layer : str, default="imputed"
        Layer name containing imputed data
    raw_layer : str, default="X"
        Layer name or 'X' for raw data
    n_top_genes : int, optional
        Number of top variable genes to analyze
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        INES scores and gene names
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for this function. Install with: pip install scanpy")
    
    # Extract data
    if raw_layer == "X":
        original = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    else:
        original = adata.layers[raw_layer]
        
    if imputed_layer in adata.layers:
        imputed = adata.layers[imputed_layer]
    else:
        raise ValueError(f"Layer '{imputed_layer}' not found in adata.layers")
    
    # Select top variable genes if specified
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        gene_mask = adata.var['highly_variable'].values
        original = original[:, gene_mask]
        imputed = imputed[:, gene_mask]
        gene_names = adata.var_names[gene_mask].values
    else:
        gene_names = adata.var_names.values
    
    # Calculate INES scores
    ines_scores = calculate_ines_score(original, imputed)
    
    return ines_scores, gene_names
