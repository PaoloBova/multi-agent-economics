"""
Test suite for regime-switching functionality in market_for_finance.py

Following TDD principles: tests first, then minimal implementation.
"""

import pytest
import numpy as np
from typing import Dict, Any

# Import functions to test (will implement these next)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.models.market_for_finance import (
    transition_regimes,
    generate_regime_returns,
    build_regime_covariance,
    build_confusion_matrix,
    generate_forecast_signal,
    update_agent_beliefs,
    compute_portfolio_moments,
    optimize_portfolio
)


class TestRegimeTransition:
    """Test regime state evolution functions."""
    
    def test_transition_regimes_single_sector(self):
        """Test regime transition for a single sector."""
        current_regimes = {"tech": 0}
        transition_matrices = {
            "tech": np.array([[0.8, 0.2], [0.3, 0.7]])
        }
        
        # Set seed for reproducible testing
        np.random.seed(42)
        new_regimes = transition_regimes(current_regimes, transition_matrices)
        
        assert isinstance(new_regimes, dict)
        assert "tech" in new_regimes
        assert new_regimes["tech"] in [0, 1]
        
    def test_transition_regimes_multi_sector(self):
        """Test regime transition for multiple sectors."""
        current_regimes = {"tech": 0, "finance": 1, "healthcare": 0}
        transition_matrices = {
            "tech": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "finance": np.array([[0.6, 0.4], [0.5, 0.5]]),
            "healthcare": np.array([[0.9, 0.1], [0.2, 0.8]])
        }
        
        np.random.seed(42)
        new_regimes = transition_regimes(current_regimes, transition_matrices)
        
        assert len(new_regimes) == 3
        for sector in ["tech", "finance", "healthcare"]:
            assert sector in new_regimes
            assert new_regimes[sector] in [0, 1]
            
    def test_transition_matrices_probabilities_valid(self):
        """Test that transition matrices have valid probabilities."""
        current_regimes = {"tech": 0}
        
        # Invalid transition matrix (doesn't sum to 1)
        invalid_matrices = {
            "tech": np.array([[0.5, 0.3], [0.2, 0.6]])  # rows don't sum to 1
        }
        
        # Should handle this gracefully or raise appropriate error
        with pytest.raises((ValueError, AssertionError)):
            transition_regimes(current_regimes, invalid_matrices)


class TestRegimeReturns:
    """Test regime-dependent return generation."""
    
    def test_generate_regime_returns_single_regime(self):
        """Test return generation for single regime."""
        regimes = {"tech": 0}
        regime_params = {
            "tech": {
                0: {"mu": 0.08, "sigma": 0.15},
                1: {"mu": 0.03, "sigma": 0.25}
            }
        }
        
        np.random.seed(42)
        returns = generate_regime_returns(regimes, regime_params)
        
        assert isinstance(returns, dict)
        assert "tech" in returns
        assert isinstance(returns["tech"], float)
        
        # Returns should be reasonable (within ~3 sigma of mean)
        assert -0.5 < returns["tech"] < 0.5  
        
    def test_generate_regime_returns_multi_sector(self):
        """Test return generation for multiple sectors."""
        regimes = {"tech": 0, "finance": 1, "healthcare": 0}
        regime_params = {
            "tech": {0: {"mu": 0.08, "sigma": 0.15}, 1: {"mu": 0.03, "sigma": 0.25}},
            "finance": {0: {"mu": 0.06, "sigma": 0.12}, 1: {"mu": 0.02, "sigma": 0.20}},
            "healthcare": {0: {"mu": 0.07, "sigma": 0.10}, 1: {"mu": 0.04, "sigma": 0.18}}
        }
        
        np.random.seed(42)
        returns = generate_regime_returns(regimes, regime_params)
        
        assert len(returns) == 3
        for sector in ["tech", "finance", "healthcare"]:
            assert sector in returns
            assert isinstance(returns[sector], float)
            assert -0.5 < returns[sector] < 0.5


class TestStructuredCovariance:
    """Test structured covariance matrix construction."""
    
    def test_build_regime_covariance_two_sectors(self):
        """Test covariance matrix for two sectors."""
        regimes = {"tech": 0, "finance": 1}
        regime_volatilities = {
            "tech": {0: 0.15, 1: 0.25},
            "finance": {0: 0.12, 1: 0.20}
        }
        fixed_correlations = np.array([[1.0, 0.3], [0.3, 1.0]])
        sector_order = ["tech", "finance"]
        
        cov_matrix = build_regime_covariance(regimes, regime_volatilities, fixed_correlations, sector_order)
        
        assert cov_matrix.shape == (2, 2)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
        assert np.all(np.diag(cov_matrix) > 0)  # Positive diagonal
        
        # Check specific values
        expected_var_tech = 0.15**2  # regime 0 for tech
        expected_var_finance = 0.20**2  # regime 1 for finance
        expected_cov = 0.3 * 0.15 * 0.20  # correlation * vol1 * vol2
        
        assert np.isclose(cov_matrix[0, 0], expected_var_tech)
        assert np.isclose(cov_matrix[1, 1], expected_var_finance)
        assert np.isclose(cov_matrix[0, 1], expected_cov)
        assert np.isclose(cov_matrix[1, 0], expected_cov)
        
    def test_build_regime_covariance_positive_definite(self):
        """Test that covariance matrix is positive definite."""
        regimes = {"tech": 0, "finance": 1, "healthcare": 0}
        regime_volatilities = {
            "tech": {0: 0.15, 1: 0.25},
            "finance": {0: 0.12, 1: 0.20},
            "healthcare": {0: 0.10, 1: 0.18}
        }
        fixed_correlations = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0]
        ])
        sector_order = ["tech", "finance", "healthcare"]
        
        cov_matrix = build_regime_covariance(regimes, regime_volatilities, fixed_correlations, sector_order)
        
        # Check positive definite by computing eigenvalues
        eigenvals = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvals > 0), "Covariance matrix should be positive definite"


class TestConfusionMatrix:
    """Test confusion matrix-based forecasting."""
    
    def test_confusion_matrix_builder_quality_effect(self):
        """Test that higher quality leads to better confusion matrices."""
        low_quality_matrix = build_confusion_matrix(forecast_quality=0.1, K=2)
        high_quality_matrix = build_confusion_matrix(forecast_quality=0.9, K=2)
        
        # Higher quality should have higher diagonal (correct prediction) probability
        assert high_quality_matrix[0, 0] > low_quality_matrix[0, 0]
        assert high_quality_matrix[1, 1] > low_quality_matrix[1, 1]
        
        # Rows should sum to 1
        assert np.allclose(low_quality_matrix.sum(axis=1), 1.0)
        assert np.allclose(high_quality_matrix.sum(axis=1), 1.0)
        
    def test_generate_forecast_signal_distribution(self):
        """Test forecast signal generation follows confusion matrix distribution."""
        confusion_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        true_regime = 0
        
        # Generate many signals to test distribution
        np.random.seed(42)
        signals = [generate_forecast_signal(true_regime, confusion_matrix) for _ in range(1000)]
        
        # Should get roughly 80% correct (signal=0) and 20% incorrect (signal=1)
        signal_0_count = sum(1 for s in signals if s == 0)
        signal_1_count = sum(1 for s in signals if s == 1)
        
        assert 0.75 < signal_0_count / 1000 < 0.85  # ~80% correct
        assert 0.15 < signal_1_count / 1000 < 0.25  # ~20% incorrect


class TestBayesianBeliefs:
    """Test agent belief updating."""
    
    def test_update_agent_beliefs_single_sector(self):
        """Test belief update for single sector."""
        prior_beliefs = {"tech": np.array([0.6, 0.4])}
        forecast_signals = {"tech": 1}  # Signal for regime 1
        subjective_transitions = {"tech": np.array([[0.8, 0.2], [0.3, 0.7]])}
        confusion_matrices = {"tech": np.array([[0.7, 0.3], [0.4, 0.6]])}
        
        updated_beliefs = update_agent_beliefs(
            prior_beliefs, forecast_signals, subjective_transitions, confusion_matrices
        )
        
        assert "tech" in updated_beliefs
        assert len(updated_beliefs["tech"]) == 2
        assert np.isclose(updated_beliefs["tech"].sum(), 1.0)  # Probabilities sum to 1
        assert np.all(updated_beliefs["tech"] >= 0)  # Non-negative probabilities
        
    def test_portfolio_optimization_basic(self):
        """Test basic mean-variance portfolio optimization."""
        expected_returns = np.array([0.08, 0.06, 0.07])
        covariance_matrix = np.array([
            [0.0225, 0.0054, 0.003],
            [0.0054, 0.0144, 0.0048],
            [0.003, 0.0048, 0.01]
        ])
        risk_aversion = 2.0
        risk_free_rate = 0.03
        
        weights = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion, risk_free_rate)
        
        assert len(weights) == 3
        assert isinstance(weights, np.ndarray)
        # Weights should be finite
        assert np.all(np.isfinite(weights))


class TestMarketIntegration:
    """Test integration with existing market model."""
    
    def test_run_information_dynamics_initialization(self):
        """Test that run_information_dynamics initializes regime state correctly."""
        from multi_agent_economics.models.market_for_finance import run_information_dynamics, MarketState, MarketModel
        
        # Create mock model and config
        market_state = MarketState(
            prices={}, offers=[], trades=[], demand_profile={}, 
            supply_profile={}, index_values={}
        )
        
        class MockModel:
            def __init__(self):
                self.state = market_state
        
        model = MockModel()
        info_cfg = {"sectors": ["tech", "finance"]}
        
        # Run information dynamics
        returns = run_information_dynamics(model, info_cfg)
        
        # Check that regime state was initialized
        assert len(model.state.current_regimes) == 2
        assert "tech" in model.state.current_regimes
        assert "finance" in model.state.current_regimes
        assert all(regime in [0, 1] for regime in model.state.current_regimes.values())
        
        # Check that returns were generated
        assert isinstance(returns, dict)
        assert len(returns) == 2
        assert "tech" in returns
        assert "finance" in returns
        
        # Check that index values were updated
        assert len(model.state.index_values) == 2
        assert all(value > 0 for value in model.state.index_values.values())
        
    def test_enhanced_sector_forecast_integration(self):
        """Test that enhanced sector_forecast tool works with regime model."""
        from multi_agent_economics.core.tools import ToolRegistry
        
        # Create tool registry with enhanced sector_forecast
        registry = ToolRegistry()
        
        # Test the enhanced sector_forecast
        tool_func = registry.get_tool_function("sector_forecast")
        assert tool_func is not None
        
        # Test different quality tiers
        high_forecast = tool_func("tech", 5, "high")
        med_forecast = tool_func("tech", 5, "med") 
        low_forecast = tool_func("tech", 5, "low")
        
        # Check that higher tiers have better forecast quality
        assert high_forecast["forecast_quality"] > med_forecast["forecast_quality"]
        assert med_forecast["forecast_quality"] > low_forecast["forecast_quality"]
        
        # Check that all forecasts have regime predictions
        for forecast in [high_forecast, med_forecast, low_forecast]:
            assert "predicted_regime" in forecast
            assert "true_regime" in forecast
            assert "regime_accuracy" in forecast
            assert "forecast_attribute_value" in forecast
            assert forecast["predicted_regime"] in [0, 1]
            assert forecast["true_regime"] in [0, 1]
    
    def test_market_state_with_regime_fields(self):
        """Test that MarketState can be created with regime fields."""
        from multi_agent_economics.models.market_for_finance import MarketState
        
        # Test creating MarketState with regime fields
        market_state = MarketState(
            prices={}, offers=[], trades=[], demand_profile={}, 
            supply_profile={}, index_values={},
            current_regimes={"tech": 0, "finance": 1},
            regime_transition_matrices={"tech": np.array([[0.8, 0.2], [0.3, 0.7]])},
            regime_parameters={"tech": {0: {"mu": 0.08, "sigma": 0.15}}},
            agent_beliefs={"agent1": {"tech": np.array([0.6, 0.4])}},
            forecast_history=[{"sector": "tech", "predicted_regime": 0}]
        )
        
        assert market_state.current_regimes["tech"] == 0
        assert market_state.current_regimes["finance"] == 1
        assert len(market_state.forecast_history) == 1


# Helper function to run a quick smoke test
def test_imports():
    """Test that all required functions can be imported."""
    try:
        from multi_agent_economics.models.market_for_finance import (
            transition_regimes,
            generate_regime_returns,
            build_regime_covariance,
            run_information_dynamics
        )
        from multi_agent_economics.core.tools import ToolRegistry
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])