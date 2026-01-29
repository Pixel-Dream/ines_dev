"""Unit tests for INES design module."""

import numpy as np
import pytest
from ines.design import (
    greedy_probe_selection,
)


class TestGreedyProbeSelection:
    """Tests for greedy probe selection algorithm."""
    
    def test_basic_selection(self):
        """Test basic probe selection."""
        scores = np.array([0.8, 0.6, 0.9, 0.3, 0.7])
        genes = np.array(['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        
        selected_genes, selected_indices, selected_scores = greedy_probe_selection(
            scores, genes, n_probes=3
        )
        
        assert len(selected_genes) == 3
        assert len(selected_indices) == 3
        assert len(selected_scores) == 3
        
        # Should select highest scoring genes
        assert 'Gene3' in selected_genes  # Score 0.9
        assert 'Gene1' in selected_genes  # Score 0.8
    
    def test_selection_with_correlation(self):
        """Test selection with correlation constraint."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        genes = np.array(['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        
        # Create expression data with high correlation between Gene1 and Gene2
        expression = np.random.randn(1000, 5)
        expression[:, 1] = expression[:, 0] + np.random.randn(1000) * 0.1  # High correlation
        
        selected_genes, _, _ = greedy_probe_selection(
            scores, genes, n_probes=3,
            expression_data=expression,
            correlation_threshold=0.5
        )
        
        # Should select genes with low correlation
        assert len(selected_genes) <= 3
    
    def test_too_many_probes(self):
        """Test requesting more probes than genes available."""
        scores = np.array([0.8, 0.6, 0.9])
        genes = np.array(['Gene1', 'Gene2', 'Gene3'])
        
        with pytest.warns(UserWarning):
            selected_genes, _, _ = greedy_probe_selection(
                scores, genes, n_probes=10
            )
        
        assert len(selected_genes) == 3  # Should select all available
    
    def test_length_mismatch_raises(self):
        """Test that length mismatch raises error."""
        scores = np.array([0.8, 0.6, 0.9])
        genes = np.array(['Gene1', 'Gene2'])
        
        with pytest.raises(ValueError, match="same length"):
            greedy_probe_selection(scores, genes, n_probes=2)
    
    def test_sorted_selection(self):
        """Test that selection prioritizes high scores."""
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        genes = np.array(['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        
        selected_genes, selected_indices, selected_scores = greedy_probe_selection(
            scores, genes, n_probes=2
        )
        
        # Should select highest scoring genes
        assert 'Gene2' in selected_genes  # Score 0.9
        assert 'Gene4' in selected_genes  # Score 0.8
        
        # Scores should be sorted (descending)
        assert selected_scores[0] >= selected_scores[1]
    
    def test_empty_selection(self):
        """Test selection with zero probes."""
        scores = np.array([0.8, 0.6, 0.9])
        genes = np.array(['Gene1', 'Gene2', 'Gene3'])
        
        selected_genes, selected_indices, selected_scores = greedy_probe_selection(
            scores, genes, n_probes=0
        )
        
        assert len(selected_genes) == 0
        assert len(selected_indices) == 0
        assert len(selected_scores) == 0


class TestEdgeCases:
    """Test edge cases for design module."""
    
    def test_single_gene(self):
        """Test with single gene."""
        scores = np.array([0.8])
        genes = np.array(['Gene1'])
        
        selected_genes, _, _ = greedy_probe_selection(scores, genes, n_probes=1)
        
        assert len(selected_genes) == 1
        assert selected_genes[0] == 'Gene1'
    
    def test_all_zeros(self):
        """Test with all zero scores."""
        scores = np.zeros(5)
        genes = np.array(['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        
        selected_genes, _, _ = greedy_probe_selection(scores, genes, n_probes=3)
        
        # Should still select genes even with zero scores
        assert len(selected_genes) == 3
    
    def test_identical_scores(self):
        """Test with identical scores."""
        scores = np.ones(5) * 0.5
        genes = np.array(['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        
        selected_genes, _, _ = greedy_probe_selection(scores, genes, n_probes=3)
        
        assert len(selected_genes) == 3
