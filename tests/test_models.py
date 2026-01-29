"""Unit tests for INES models module."""

import numpy as np
import pytest
from ines.models import ImputationSimulator, create_imputation_pipeline


class TestImputationSimulator:
    """Tests for ImputationSimulator class."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = ImputationSimulator(model_type="knn")
        
        assert simulator.model_type == "knn"
        assert simulator.device == "cpu"
    
    def test_knn_imputation(self):
        """Test k-NN imputation."""
        simulator = ImputationSimulator(model_type="knn", random_state=42)
        
        data = np.random.randn(1000, 100)
        imputed = simulator.simulate_imputation(data, dropout_rate=0.1)
        
        assert imputed.shape == data.shape
        assert np.all(imputed >= 0)  # Should be non-negative
    
    def test_magic_imputation(self):
        """Test MAGIC-style imputation."""
        simulator = ImputationSimulator(model_type="magic", random_state=42)
        
        data = np.abs(np.random.randn(500, 50))  # Positive values
        imputed = simulator.simulate_imputation(data, dropout_rate=0.1)
        
        assert imputed.shape == data.shape
    
    def test_dca_imputation(self):
        """Test DCA-style autoencoder imputation."""
        simulator = ImputationSimulator(model_type="dca", random_state=42)
        
        data = np.abs(np.random.randn(500, 50))
        imputed = simulator.simulate_imputation(data, dropout_rate=0.1)
        
        assert imputed.shape == data.shape
    
    def test_scvi_imputation(self):
        """Test scVI-style VAE imputation."""
        simulator = ImputationSimulator(model_type="scvi", random_state=42)
        
        data = np.abs(np.random.randn(500, 50))
        imputed = simulator.simulate_imputation(data, dropout_rate=0.1)
        
        assert imputed.shape == data.shape
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        simulator = ImputationSimulator(model_type="invalid")
        
        data = np.random.randn(100, 10)
        
        with pytest.raises(ValueError, match="Unknown model type"):
            simulator.simulate_imputation(data)
    
    def test_noise_addition(self):
        """Test noise addition during imputation."""
        simulator = ImputationSimulator(model_type="knn", random_state=42)
        
        data = np.random.randn(100, 10)
        
        # With noise
        imputed_noisy = simulator.simulate_imputation(data, noise_level=0.1)
        
        # Without noise
        imputed_clean = simulator.simulate_imputation(data, noise_level=0.0)
        
        # Should be different when noise is added
        assert not np.allclose(imputed_noisy, imputed_clean)
    
    def test_dropout_rate(self):
        """Test different dropout rates."""
        simulator = ImputationSimulator(model_type="knn", random_state=42)
        
        data = np.random.randn(100, 10)
        
        imputed_low = simulator.simulate_imputation(data, dropout_rate=0.05)
        imputed_high = simulator.simulate_imputation(data, dropout_rate=0.5)
        
        # Both should have correct shape
        assert imputed_low.shape == data.shape
        assert imputed_high.shape == data.shape
    
    def test_random_state_reproducibility(self):
        """Test random state for reproducibility."""
        data = np.random.randn(100, 10)
        
        simulator1 = ImputationSimulator(model_type="knn", random_state=42)
        imputed1 = simulator1.simulate_imputation(data, dropout_rate=0.1)
        
        simulator2 = ImputationSimulator(model_type="knn", random_state=42)
        imputed2 = simulator2.simulate_imputation(data, dropout_rate=0.1)
        
        # Should be identical with same random state
        assert np.allclose(imputed1, imputed2)


class TestBenchmarkImputation:
    """Tests for imputation benchmarking."""
    
    def test_basic_benchmark(self):
        """Test basic benchmarking."""
        simulator = ImputationSimulator(model_type="knn", random_state=42)
        
        data = np.abs(np.random.randn(100, 20))
        
        results = simulator.benchmark_imputation(
            data, n_trials=2, dropout_rates=[0.1, 0.2]
        )
        
        assert 'mse' in results
        assert 'mae' in results
        assert 'correlation' in results
        assert len(results['mse']) == 2
        assert len(results['dropout_rates']) == 2
    
    def test_benchmark_metrics_valid(self):
        """Test that benchmark metrics are valid."""
        simulator = ImputationSimulator(model_type="knn", random_state=42)
        
        data = np.abs(np.random.randn(100, 10))
        
        results = simulator.benchmark_imputation(data, n_trials=1, dropout_rates=[0.1])
        
        # All metrics should be non-negative
        assert all(mse >= 0 for mse in results['mse'])
        assert all(mae >= 0 for mae in results['mae'])
        
        # Correlation should be between -1 and 1
        assert all(-1 <= corr <= 1 for corr in results['correlation'])


class TestCreateImputationPipeline:
    """Tests for pipeline factory function."""
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline = create_imputation_pipeline(model_type="knn")
        
        assert isinstance(pipeline, ImputationSimulator)
        assert pipeline.model_type == "knn"
    
    def test_create_pipeline_with_kwargs(self):
        """Test pipeline creation with additional arguments."""
        pipeline = create_imputation_pipeline(
            model_type="magic",
            device="cpu",
            random_state=123
        )
        
        assert pipeline.model_type == "magic"
        assert pipeline.random_state == 123
