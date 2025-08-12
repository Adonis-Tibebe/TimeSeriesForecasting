import pytest
import numpy as np
from src.models.Models_utils import create_sequences, calculate_metrics


class TestCreateSequences:
    """Test cases for the create_sequences function."""
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation with default window size."""
        # Create simple sample data
        data = np.array([
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5]
        ])
        
        X, y = create_sequences(data, window=2)
        
        # Check shapes
        assert X.shape == (3, 2, 2)  # 3 sequences, 2 timesteps, 2 features
        assert y.shape == (3,)  # 3 target values
        
        # Check first sequence
        assert np.array_equal(X[0], data[:2])
        assert y[0] == data[2, 0]  # First target should be first feature of 3rd row
    
    def test_create_sequences_custom_window(self):
        """Test sequence creation with custom window size."""
        data = np.array([
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5]
        ])
        
        X, y = create_sequences(data, window=3)
        
        # Check shapes
        assert X.shape == (2, 3, 2)  # 2 sequences, 3 timesteps, 2 features
        assert y.shape == (2,)  # 2 target values
        
        # Check sequences
        assert np.array_equal(X[0], data[:3])
        assert y[0] == data[3, 0]


class TestCalculateMetrics:
    """Test cases for the calculate_metrics function."""
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        # Create simple test data with no zeros to avoid MAPE issues
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 1.9, 3.1])
        
        metrics = calculate_metrics(actual, predicted)
        
        # Check that all required metrics are present
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
        
        # Check metric types
        assert isinstance(metrics['MAE'], (float, np.floating))
        assert isinstance(metrics['RMSE'], (float, np.floating))
        assert isinstance(metrics['MAPE'], (float, np.floating))
    
    def test_calculate_metrics_identical_values(self):
        """Test metrics calculation with identical actual and predicted values."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        
        metrics = calculate_metrics(actual, predicted)
        
        # MAE and RMSE should be 0 for identical values
        assert metrics['MAE'] == 0.0
        assert metrics['RMSE'] == 0.0


class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from sequence creation to metrics calculation."""
        # Create sample time series data with no zeros
        data = np.random.randn(30, 2) + 1.0  # 30 timesteps, 2 features, shifted to avoid zeros
        
        # Create sequences
        X, y = create_sequences(data, window=5)
        
        # Simulate predictions (add some noise to actual values)
        predictions = y + np.random.normal(0, 0.1, y.shape)
        
        # Calculate metrics
        metrics = calculate_metrics(y, predictions)
        
        # Verify results
        assert X.shape[0] == y.shape[0]  # Same number of sequences
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
        
        # Metrics should be reasonable
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0
        assert metrics['MAPE'] >= 0


if __name__ == "__main__":
    pytest.main([__file__]) 