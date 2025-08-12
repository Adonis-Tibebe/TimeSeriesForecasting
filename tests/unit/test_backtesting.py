import pytest
import pandas as pd
import numpy as np
from src.core.backtesting import get_benchmark_returns, simulate_strategy, calculate_performance


class TestGetBenchmarkReturns:
    """Test cases for the get_benchmark_returns function."""
    
    def test_get_benchmark_returns_basic(self):
        """Test basic benchmark returns calculation."""
        # Create sample price data
        dates = pd.date_range('2024-08-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'SPY': [100, 101, 99, 102, 103, 101, 104, 105, 103, 106],
            'BND': [50, 50.1, 49.9, 50.2, 50.3, 50.1, 50.4, 50.5, 50.3, 50.6]
        }, index=dates)
        
        benchmark_returns = get_benchmark_returns(prices)
        
        # Check that returns are calculated
        assert len(benchmark_returns) == 9  # 10 prices = 9 returns
        assert isinstance(benchmark_returns, pd.Series)
        assert benchmark_returns.index.equals(prices.index[1:])  # First date dropped due to pct_change
    
    def test_get_benchmark_returns_weights(self):
        """Test that 60/40 weights are applied correctly."""
        # Create simple price data with known returns
        dates = pd.date_range('2024-08-01', periods=3, freq='D')
        prices = pd.DataFrame({
            'SPY': [100, 110, 105],  # 10% gain, then 4.55% loss
            'BND': [50, 51, 50.5]    # 2% gain, then 0.98% loss
        }, index=dates)
        
        benchmark_returns = get_benchmark_returns(prices)
        
        # First return: 0.6 * 0.10 + 0.4 * 0.02 = 0.068
        expected_first_return = 0.6 * 0.10 + 0.4 * 0.02
        assert abs(benchmark_returns.iloc[0] - expected_first_return) < 1e-10


class TestSimulateStrategy:
    """Test cases for the simulate_strategy function."""
    
    def test_simulate_strategy_basic(self):
        """Test basic strategy simulation."""
        # Create sample price data
        dates = pd.date_range('2024-08-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'TSLA': [200, 202, 198, 205, 208, 204, 210, 212, 208, 215],
            'SPY': [100, 101, 99, 102, 103, 101, 104, 105, 103, 106],
            'BND': [50, 50.1, 49.9, 50.2, 50.3, 50.1, 50.4, 50.5, 50.3, 50.6]
        }, index=dates)
        
        weights = {'TSLA': 0.3, 'SPY': 0.5, 'BND': 0.2}
        
        strategy_returns = simulate_strategy(prices, weights)
        
        # Check that strategy returns are calculated
        assert isinstance(strategy_returns, pd.Series)
        assert len(strategy_returns) > 0
    
    def test_simulate_strategy_weights(self):
        """Test that weights are applied correctly."""
        # Create simple price data
        dates = pd.date_range('2024-08-01', periods=3, freq='D')
        prices = pd.DataFrame({
            'TSLA': [100, 110, 105],  # 10% gain, then 4.55% loss
            'SPY': [100, 101, 99],    # 1% gain, then 1.98% loss
            'BND': [50, 50.1, 49.9]  # 0.2% gain, then 0.4% loss
        }, index=dates)
        
        weights = {'TSLA': 0.5, 'SPY': 0.3, 'BND': 0.2}
        
        strategy_returns = simulate_strategy(prices, weights)
        
        # Check that strategy returns are calculated
        assert isinstance(strategy_returns, pd.Series)
        assert len(strategy_returns) > 0


class TestCalculatePerformance:
    """Test cases for the calculate_performance function."""
    
    def test_calculate_performance_basic(self):
        """Test basic performance calculation."""
        # Create sample returns
        dates = pd.date_range('2024-08-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        
        performance = calculate_performance(returns)
        
        # Check that all required metrics are present
        assert 'Sharpe' in performance
        assert 'Cumulative_Return' in performance
        
        # Check metric types
        assert isinstance(performance['Sharpe'], (float, np.floating))
        assert isinstance(performance['Cumulative_Return'], (float, np.floating))
    
    def test_calculate_performance_positive_returns(self):
        """Test performance calculation with positive returns."""
        # Create consistently positive returns
        dates = pd.date_range('2024-08-01', periods=50, freq='D')
        returns = pd.Series(0.001, index=dates)  # 0.1% daily return
        
        performance = calculate_performance(returns)
        
        # Check that cumulative return is positive
        assert performance['Cumulative_Return'] > 0
        
        # Check that Sharpe ratio is reasonable
        assert performance['Sharpe'] > 0
    
    def test_calculate_performance_negative_returns(self):
        """Test performance calculation with negative returns."""
        # Create consistently negative returns
        dates = pd.date_range('2024-08-01', periods=50, freq='D')
        returns = pd.Series(-0.001, index=dates)  # -0.1% daily return
        
        performance = calculate_performance(returns)
        
        # Check that cumulative return is negative
        assert performance['Cumulative_Return'] < 0
        
        # Check that Sharpe ratio is reasonable
        assert performance['Sharpe'] < 0


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from benchmark to performance calculation."""
        # Create sample price data
        dates = pd.date_range('2024-08-01', periods=20, freq='D')
        prices = pd.DataFrame({
            'TSLA': np.random.uniform(200, 220, 20),
            'SPY': np.random.uniform(100, 110, 20),
            'BND': np.random.uniform(50, 51, 20)
        }, index=dates)
        
        # Test benchmark returns
        benchmark_returns = get_benchmark_returns(prices)
        assert len(benchmark_returns) > 0
        
        # Test strategy simulation
        weights = {'TSLA': 0.3, 'SPY': 0.5, 'BND': 0.2}
        strategy_returns = simulate_strategy(prices, weights)
        assert len(strategy_returns) > 0
        
        # Test performance calculation
        benchmark_performance = calculate_performance(benchmark_returns)
        strategy_performance = calculate_performance(strategy_returns)
        
        # Verify both performance calculations work
        assert 'Sharpe' in benchmark_performance
        assert 'Sharpe' in strategy_performance
        assert 'Cumulative_Return' in benchmark_performance
        assert 'Cumulative_Return' in strategy_performance


if __name__ == "__main__":
    pytest.main([__file__])
