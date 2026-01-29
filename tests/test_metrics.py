"""Unit tests for INES metrics module."""

import numpy as np
import pytest
from ines.metrics import (
    earth_movers_distance,
    calculate_ines_score,
)


class TestEarthMoversDistance:
    """Tests for Earth Mover's Distance calculations."""
    
    def test_basic_emd(self):
        """Test basic EMD calculation."""
        dist1 = np.array([0, 1, 2, 3, 4])
        dist2 = np.array([1, 2, 3, 4, 5])
        
        emd = earth_movers_distance(dist1, dist2)
        
        assert isinstance(emd, (float, np.floating))
        assert emd >= 0
    
    def test_identical_distributions(self):
        """Test EMD of identical distributions."""
        dist = np.array([1, 2, 3, 4, 5])
        
        emd = earth_movers_distance(dist, dist)
        
        assert emd == pytest.approx(0.0, abs=1e-10)
    
    def test_vectorized_emd(self):
        """Test vectorized EMD computation."""
        dist1 = np.random.randn(10, 100)
        dist2 = np.random.randn(10, 100)
        
        emd_values = earth_movers_distance(dist1, dist2)
        
        assert isinstance(emd_values, np.ndarray)
        assert emd_values.shape == (10,)
        assert np.all(emd_values >= 0)
    
    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises error."""
        dist1 = np.array([1, 2, 3])
        dist2 = np.array([1, 2, 3, 4])
        
        with pytest.raises(ValueError, match="same shape"):
            earth_movers_distance(dist1, dist2)


class TestCalculateINESScore:
    """Tests for INES score calculation."""
    
    def test_basic_ines_score(self):
        """Test basic INES score calculation."""
        original = np.random.randn(1000, 100)
        imputed = original + np.random.randn(1000, 100) * 0.1
        
        scores = calculate_ines_score(original, imputed)
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (100,)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)  # Normalized
    
    def test_perfect_imputation(self):
        """Test INES score with perfect imputation."""
        original = np.random.randn(100, 10)
        imputed = original.copy()
        
        scores = calculate_ines_score(original, imputed)
        
        # Perfect imputation should give near-zero scores
        assert np.all(scores < 0.1)
    
    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises error."""
        original = np.random.randn(100, 50)
        imputed = np.random.randn(100, 60)
        
        with pytest.raises(ValueError, match="same shape"):
            calculate_ines_score(original, imputed)
    
    def test_all_zeros_handling(self):
        """Test handling of all-zero genes."""
        original = np.zeros((100, 10))
        imputed = np.zeros((100, 10))
        
        scores = calculate_ines_score(original, imputed)
        
        assert np.all(scores == 0.0)
    
    def test_normalization(self):
        """Test score normalization."""
        original = np.random.randn(100, 10)
        imputed = original + np.random.randn(100, 10) * 2.0
        
        # With normalization
        scores_norm = calculate_ines_score(original, imputed, normalize=True)
        assert np.max(scores_norm) == pytest.approx(1.0)
        
        # Without normalization
        scores_raw = calculate_ines_score(original, imputed, normalize=False)
        assert np.max(scores_raw) > 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_cell(self):
        """Test with single cell."""
        original = np.random.randn(1, 100)
        imputed = original + np.random.randn(1, 100) * 0.1
        
        scores = calculate_ines_score(original, imputed)
        
        assert scores.shape == (100,)
    
    def test_single_gene(self):
        """Test with single gene."""
        original = np.random.randn(1000, 1)
        imputed = original + np.random.randn(1000, 1) * 0.1
        
        scores = calculate_ines_score(original, imputed)
        
        assert scores.shape == (1,)
    
    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        original = np.array([]).reshape(0, 10)
        imputed = np.array([]).reshape(0, 10)
        
        # Empty arrays should return empty scores array
        scores = calculate_ines_score(original, imputed)
        assert len(scores) == 10
        assert np.all(scores == 0.0)
