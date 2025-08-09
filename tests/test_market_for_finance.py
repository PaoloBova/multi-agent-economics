"""
Comprehensive test suite for market_for_finance.py

Tests all major components including:
- Regime history generation and integration
- Knowledge good impact calculation with explicit expected values
- Ex-post valuation resolution
- Sequential belief updating
- Surplus computation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.models.market_for_finance import (
    generate_regime_history_from_scenario,
    compute_knowledge_good_impact,
    resolve_ex_post_valuations,
    compute_surpluses,
    update_belief_with_forecast,
    compute_portfolio_moments,
    optimize_portfolio,
    Offer,
    MarketState,
    TradeData,
    ForecastData,
    RegimeParameters,
    PeriodData
)
from multi_agent_economics.models.scenario_templates import generate_boom_scenario


class TestMarketForFinance:
    """Test suite for market_for_finance.py functionality."""
    
    def test_regime_history_generation(self):
        """Test regime history generation from scenario templates."""
        # Create a deterministic scenario for testing
        scenario = generate_boom_scenario(sectors=["tech", "finance"], growth_factor=1.0)
        
        # Generate short history for testing
        history = generate_regime_history_from_scenario(scenario, num_periods=3)
        
        # Verify structure
        assert len(history) == 3
        for period_data in history:
            assert "period" in period_data
            assert "returns" in period_data
            assert "regimes" in period_data
            assert "index_values" in period_data
            
            # Verify sectors present
            assert "tech" in period_data["returns"]
            assert "finance" in period_data["returns"]
            assert "tech" in period_data["regimes"]
            assert "finance" in period_data["regimes"]
            assert "tech" in period_data["index_values"]
            assert "finance" in period_data["index_values"]
        
        # Verify index values change over time (boom scenario should generally increase)
        # Note: index values are updated with returns before storage, so first period != 100.0
        initial_tech = history[0]["index_values"]["tech"]
        initial_finance = history[0]["index_values"]["finance"]
        
        # Index values should be different from baseline 100.0 after first period
        assert initial_tech != 100.0
        assert initial_finance != 100.0
        
        # Later periods should have different index values
        assert history[1]["index_values"]["tech"] != initial_tech
        assert history[2]["index_values"]["tech"] != history[1]["index_values"]["tech"]
        
        # In boom scenario, should generally trend upward (but may have volatility)
        final_tech = history[2]["index_values"]["tech"]
        assert isinstance(final_tech, float) and final_tech > 0
        
        print("✓ Regime history generation test passed")
    
    def test_knowledge_good_impact_calculation_explicit_values(self):
        """Test knowledge good impact with explicit expected values."""
        
        # Create controlled scenario
        sectors = ["tech", "finance"]
        
        # Set up deterministic regime parameters
        regime_parameters = {
            "tech": {
                0: {"mu": 0.10, "sigma": 0.15},    # Bull market: 10% return, 15% vol
                1: {"mu": 0.02, "sigma": 0.25}     # Bear market: 2% return, 25% vol
            },
            "finance": {
                0: {"mu": 0.08, "sigma": 0.12},    # Bull market: 8% return, 12% vol  
                1: {"mu": 0.01, "sigma": 0.20}     # Bear market: 1% return, 20% vol
            }
        }
        
        # Fixed correlation matrix
        correlations = np.array([[1.0, 0.3], [0.3, 1.0]])  # 30% correlation
        
        # Set up market state with controlled values
        market_state = MarketState(
            prices={},
            offers=[],
            trades=[],
            demand_profile={},
            supply_profile={},
            index_values={"tech": 105.0, "finance": 103.0},  # 5% and 3% gains
            current_regimes={"tech": 0, "finance": 0},
            regime_transition_matrices={},
            regime_parameters=regime_parameters,
            regime_correlations=correlations,
            regime_history=[{
                "period": 0,
                "returns": {"tech": 0.05, "finance": 0.03},  # Actual returns this period
                "regimes": {"tech": 0, "finance": 0},
                "index_values": {"tech": 105.0, "finance": 103.0}
            }],
            current_period=0,
            risk_free_rate=0.02  # 2% risk-free rate
        )
        
        # Create knowledge good
        tech_forecast = Offer(
            good_id="tech_forecast_001",
            price=50.0,
            seller="analyst_001", 
            marketing_attributes={"innovation_level": "high", "data_source": "internal"}  # High quality, low cost methodology
        )
        
        # Add explicit forecast
        market_state.knowledge_good_forecasts["tech_forecast_001"] = ForecastData(
            sector="tech",
            predicted_regime=0,  # Predict bull market
            confidence_vector=[0.75, 0.25]  # 75% bull, 25% bear
        )
        
        # Create buyer with known beliefs and risk parameters
        class TestBuyerState:
            def __init__(self):
                self.buyer_id = "test_buyer"
                self.regime_beliefs = {
                    "tech": np.array([0.4, 0.6]),      # Initially bearish on tech (40% bull, 60% bear)
                    "finance": np.array([0.6, 0.4])    # Bullish on finance (60% bull, 40% bear)
                }
                self.risk_aversion = 2.0
            
            def dict(self):
                return {
                    "buyer_id": self.buyer_id,
                    "regime_beliefs": self.regime_beliefs.copy(),
                    "risk_aversion": self.risk_aversion
                }
            
            def __class__(self, **kwargs):
                instance = TestBuyerState()
                for key, value in kwargs.items():
                    setattr(instance, key, value)
                return instance
        
        buyer_state = TestBuyerState()
        
        # Calculate impact
        impact_result = compute_knowledge_good_impact(buyer_state, tech_forecast, market_state)
        
        # Verify structure
        assert isinstance(impact_result, dict)
        required_keys = ["economic_value", "beliefs_before", "beliefs_after", 
                        "expected_returns_before", "expected_returns_after",
                        "cov_matrix_before", "cov_matrix_after", 
                        "weights_before", "weights_after"]
        for key in required_keys:
            assert key in impact_result
        
        # Test explicit belief updating
        beliefs_before = impact_result["beliefs_before"]["tech"]
        beliefs_after = impact_result["beliefs_after"]["tech"]
        
        # Expected belief update using Bayesian formula:
        # posterior ∝ prior * likelihood
        # prior = [0.4, 0.6], likelihood = [0.75, 0.25] (confidence vector)
        expected_posterior_unnorm = np.array([0.4 * 0.75, 0.6 * 0.25])  # [0.3, 0.15]
        expected_posterior = expected_posterior_unnorm / np.sum(expected_posterior_unnorm)  # [0.667, 0.333]
        
        np.testing.assert_array_almost_equal(beliefs_after, expected_posterior, decimal=3)
        
        # Test expected returns calculation
        # Before: E[R_tech] = 0.4 * 0.10 + 0.6 * 0.02 = 0.052
        # After:  E[R_tech] = 0.667 * 0.10 + 0.333 * 0.02 = 0.0733
        expected_return_tech_before = 0.4 * 0.10 + 0.6 * 0.02
        expected_return_tech_after = expected_posterior[0] * 0.10 + expected_posterior[1] * 0.02
        
        assert abs(impact_result["expected_returns_before"][0] - expected_return_tech_before) < 1e-6
        assert abs(impact_result["expected_returns_after"][0] - expected_return_tech_after) < 1e-6
        
        # Finance beliefs shouldn't change (no forecast for finance)
        np.testing.assert_array_almost_equal(
            impact_result["beliefs_before"]["finance"], 
            impact_result["beliefs_after"]["finance"], 
            decimal=10
        )
        
        # Economic value should be positive (bullish forecast in bull market with positive returns)
        economic_value = impact_result["economic_value"]
        assert isinstance(economic_value, float)
        # Should be positive since forecast increases allocation to tech which had positive returns
        assert economic_value > 0, f"Expected positive economic value, got {economic_value}"
        
        print(f"✓ Knowledge good impact calculation test passed")
        print(f"  Beliefs before: tech={beliefs_before}, finance={impact_result['beliefs_before']['finance']}")
        print(f"  Beliefs after:  tech={beliefs_after}, finance={impact_result['beliefs_after']['finance']}")
        print(f"  Expected return before: tech={expected_return_tech_before:.6f}")
        print(f"  Expected return after:  tech={expected_return_tech_after:.6f}")
        print(f"  Economic value: {economic_value:.6f}")
    
    def test_sequential_knowledge_good_application(self):
        """Test sequential application of multiple knowledge goods with explicit calculations."""
        
        # Set up controlled scenario
        regime_parameters = {
            "tech": {
                0: {"mu": 0.08, "sigma": 0.15},    # Bull: 8% return, 15% vol
                1: {"mu": 0.01, "sigma": 0.25}     # Bear: 1% return, 25% vol
            }
        }
        
        correlations = np.array([[1.0]])  # Single sector
        
        market_state = MarketState(
            prices={},
            offers=[],
            trades=[],
            demand_profile={},
            supply_profile={},
            index_values={"tech": 104.0},  # 4% gain
            current_regimes={"tech": 0},
            regime_transition_matrices={},
            regime_parameters=regime_parameters,
            regime_correlations=correlations,
            regime_history=[PeriodData(
                period=0,
                returns={"tech": 0.04},
                regimes={"tech": 0},
                index_values={"tech": 104.0}
            )],
            current_period=0,
            risk_free_rate=0.02
        )
        
        # Create two knowledge goods
        forecast1 = Offer(good_id="forecast_1", price=30.0, seller="analyst_1", 
                         marketing_attributes={"innovation_level": "medium", "data_source": "internal"})
        forecast2 = Offer(good_id="forecast_2", price=40.0, seller="analyst_2", 
                         marketing_attributes={"innovation_level": "high", "data_source": "internal"})
        
        # Both predict bull market with different confidence
        market_state.knowledge_good_forecasts.update({
            "forecast_1": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.7, 0.3]),
            "forecast_2": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.8, 0.2])
        })
        
        # Create buyer starting with bearish beliefs
        class TestBuyerState:
            def __init__(self):
                self.buyer_id = "test_buyer"
                self.regime_beliefs = {"tech": np.array([0.3, 0.7])}  # 30% bull, 70% bear
                self.risk_aversion = 2.0
            
            def dict(self):
                return {"buyer_id": self.buyer_id, "regime_beliefs": self.regime_beliefs.copy(), "risk_aversion": self.risk_aversion}
            
            def __class__(self, **kwargs):
                instance = TestBuyerState()
                for key, value in kwargs.items():
                    setattr(instance, key, value)
                return instance
        
        buyer_state = TestBuyerState()
        market_state.offers = [forecast1, forecast2]
        market_state.buyers_state = [buyer_state]
        
        # Create trades for sequential application
        trades = [
            TradeData(buyer_id="test_buyer", seller_id="analyst_1", price=30.0, quantity=1, good_id="forecast_1"),
            TradeData(buyer_id="test_buyer", seller_id="analyst_2", price=40.0, quantity=1, good_id="forecast_2")
        ]
        
        # Calculate expected sequential belief updates manually
        
        # Initial beliefs: [0.3, 0.7]
        initial_beliefs = np.array([0.3, 0.7])
        
        # After forecast 1 (confidence [0.7, 0.3]):
        # posterior_1 ∝ [0.3 * 0.7, 0.7 * 0.3] = [0.21, 0.21]
        # normalized: [0.5, 0.5]
        posterior_1_unnorm = initial_beliefs * np.array([0.7, 0.3])
        posterior_1 = posterior_1_unnorm / np.sum(posterior_1_unnorm)
        expected_after_1 = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(posterior_1, expected_after_1, decimal=10)
        
        # After forecast 2 (confidence [0.8, 0.2]):
        # posterior_2 ∝ [0.5 * 0.8, 0.5 * 0.2] = [0.4, 0.1]
        # normalized: [0.8, 0.2]
        posterior_2_unnorm = posterior_1 * np.array([0.8, 0.2])
        posterior_2 = posterior_2_unnorm / np.sum(posterior_2_unnorm)
        expected_final = np.array([0.8, 0.2])
        np.testing.assert_array_almost_equal(posterior_2, expected_final, decimal=10)
        
        # Apply sequential resolution
        initial_beliefs_actual = buyer_state.regime_beliefs["tech"].copy()
        resolve_ex_post_valuations(trades, type('MockModel', (), {'state': market_state})(), None)
        final_beliefs_actual = buyer_state.regime_beliefs["tech"]
        
        # Verify sequential application worked correctly
        np.testing.assert_array_almost_equal(final_beliefs_actual, expected_final, decimal=6)
        
        # Verify both forecasts have individual economic values stored
        assert "forecast_1" in market_state.knowledge_good_impacts
        assert "forecast_2" in market_state.knowledge_good_impacts
        assert "test_buyer" in market_state.knowledge_good_impacts["forecast_1"]
        assert "test_buyer" in market_state.knowledge_good_impacts["forecast_2"]
        
        impact_1 = market_state.knowledge_good_impacts["forecast_1"]["test_buyer"]
        impact_2 = market_state.knowledge_good_impacts["forecast_2"]["test_buyer"]
        
        # Both should be positive (bullish forecasts in bull market)
        assert impact_1 > 0, f"First forecast should have positive impact, got {impact_1}"
        assert impact_2 > 0, f"Second forecast should have positive impact, got {impact_2}"
        
        print(f"✓ Sequential knowledge good application test passed")
        print(f"  Initial beliefs: {initial_beliefs_actual}")
        print(f"  Expected final:  {expected_final}")
        print(f"  Actual final:    {final_beliefs_actual}")
        print(f"  Impact 1: {impact_1:.6f}")
        print(f"  Impact 2: {impact_2:.6f}")
    
    def test_surplus_computation_explicit_values(self):
        """Test surplus computation with explicit expected values."""
        
        # Set up market state with known impacts
        market_state = MarketState(
            prices={},
            offers=[],
            trades=[],
            demand_profile={},
            supply_profile={},
            index_values={},
            knowledge_good_impacts={
                "forecast_A": {"buyer_1": 75.0, "buyer_2": 50.0},
                "forecast_B": {"buyer_1": 25.0}
            }
        )
        
        # Create buyer states
        class TestBuyerState:
            def __init__(self, buyer_id):
                self.buyer_id = buyer_id
                self.surplus = 0.0
        
        buyer_1 = TestBuyerState("buyer_1")
        buyer_2 = TestBuyerState("buyer_2")
        market_state.buyers_state = [buyer_1, buyer_2]
        
        # Create trades with known prices
        trades = [
            TradeData(buyer_id="buyer_1", seller_id="seller_A", price=60.0, quantity=1, good_id="forecast_A"),  # buyer_1 buys forecast_A for $60
            TradeData(buyer_id="buyer_1", seller_id="seller_B", price=30.0, quantity=1, good_id="forecast_B"),  # buyer_1 buys forecast_B for $30
            TradeData(buyer_id="buyer_2", seller_id="seller_A", price=60.0, quantity=1, good_id="forecast_A"),  # buyer_2 buys forecast_A for $60
        ]
        
        # Compute surpluses
        mock_model = type('MockModel', (), {'state': market_state})()
        compute_surpluses(trades, mock_model, None)
        
        # Calculate expected surpluses manually
        # buyer_1: impact_A + impact_B - price_A - price_B = 75.0 + 25.0 - 60.0 - 30.0 = 10.0
        # buyer_2: impact_A - price_A = 50.0 - 60.0 = -10.0
        
        expected_surplus_1 = 75.0 + 25.0 - 60.0 - 30.0  # 10.0
        expected_surplus_2 = 50.0 - 60.0                 # -10.0
        
        assert abs(buyer_1.surplus - expected_surplus_1) < 1e-10, f"Expected {expected_surplus_1}, got {buyer_1.surplus}"
        assert abs(buyer_2.surplus - expected_surplus_2) < 1e-10, f"Expected {expected_surplus_2}, got {buyer_2.surplus}"
        
        print(f"✓ Surplus computation test passed")
        print(f"  Buyer 1 surplus: {buyer_1.surplus} (expected: {expected_surplus_1})")
        print(f"  Buyer 2 surplus: {buyer_2.surplus} (expected: {expected_surplus_2})")
    
    def test_belief_updating_edge_cases(self):
        """Test belief updating with edge cases and explicit mathematical verification."""
        
        # Test 1: Perfect confidence forecast
        initial_belief = np.array([0.3, 0.7])
        perfect_forecast = {"confidence_vector": [1.0, 0.0]}  # 100% certain of regime 0
        
        updated = update_belief_with_forecast(initial_belief, perfect_forecast)
        expected = np.array([1.0, 0.0])  # Should become certain of regime 0
        np.testing.assert_array_almost_equal(updated, expected, decimal=10)
        
        # Test 2: No-information forecast (uniform confidence)
        uniform_forecast = {"confidence_vector": [0.5, 0.5]}  # No information
        updated_uniform = update_belief_with_forecast(initial_belief, uniform_forecast)
        # Should remain unchanged: [0.3, 0.7] * [0.5, 0.5] ∝ [0.15, 0.35] → [0.3, 0.7]
        np.testing.assert_array_almost_equal(updated_uniform, initial_belief, decimal=10)
        
        # Test 3: Contradictory forecast
        contradictory_forecast = {"confidence_vector": [0.2, 0.8]}  # Strongly favors regime 1
        updated_contra = update_belief_with_forecast(initial_belief, contradictory_forecast)
        # [0.3, 0.7] * [0.2, 0.8] ∝ [0.06, 0.56] → [0.097, 0.903]
        expected_contra = np.array([0.06, 0.56]) / (0.06 + 0.56)
        np.testing.assert_array_almost_equal(updated_contra, expected_contra, decimal=10)
        
        print(f"✓ Belief updating edge cases test passed")
        print(f"  Perfect confidence result: {updated}")
        print(f"  Uniform forecast result: {updated_uniform}")
        print(f"  Contradictory forecast result: {updated_contra}")
    
    def test_demand_transition_bayesian_updates_explicit_values(self):
        """Test demand transition with explicit Bayesian update calculations."""
        from multi_agent_economics.models.market_for_finance import update_buyer_preferences_from_knowledge_goods
        
        # Create a buyer with known initial attribute preferences
        class TestBuyerState:
            def __init__(self):
                self.buyer_id = "test_buyer"
                self.attr_mu = [0.5, 0.3]      # Prior means for 2 attributes
                self.attr_sigma2 = [0.1, 0.2]  # Prior variances for 2 attributes
                
        buyer_state = TestBuyerState()
        
        # Set up market state with one knowledge good with known impact
        market_state = MarketState(
            prices={},
            offers=[
                Offer(good_id="kg_test", price=50.0, seller="analyst", 
                      marketing_attributes={"innovation_level": "high", "data_source": "internal"})
            ],
            trades=[],
            demand_profile={},
            supply_profile={},
            index_values={}
        )
        
        # Add knowledge good impact
        market_state.knowledge_good_impacts = {
            "kg_test": {"test_buyer": 60.0}  # Economic impact = 60.0
        }
        
        # Update preferences
        update_buyer_preferences_from_knowledge_goods(buyer_state, market_state, obs_noise_var=1.0)
        
        # Calculate expected results manually using Bayesian update formula:
        # τ_post = τ0 + τL, μ_post = (τ0 * μ0 + x_j * impact / obs_noise_var) / τ_post
        
        # For attribute 0 (x_j = 0.8, impact = 60.0, obs_noise_var = 1.0):
        tau0_attr0 = 1 / 0.1  # = 10.0
        tauL_attr0 = (0.8 ** 2) / 1.0  # = 0.64
        tau_post_attr0 = tau0_attr0 + tauL_attr0  # = 10.64
        mu_post_attr0 = (tau0_attr0 * 0.5 + 0.8 * 60.0 / 1.0) / tau_post_attr0
        # = (10.0 * 0.5 + 48.0) / 10.64 = (5.0 + 48.0) / 10.64 = 53.0 / 10.64
        expected_mu_attr0 = 53.0 / 10.64
        expected_sigma2_attr0 = 1 / tau_post_attr0  # = 1 / 10.64
        
        # For attribute 1 (x_j = 0.4, impact = 60.0, obs_noise_var = 1.0):
        tau0_attr1 = 1 / 0.2  # = 5.0
        tauL_attr1 = (0.4 ** 2) / 1.0  # = 0.16
        tau_post_attr1 = tau0_attr1 + tauL_attr1  # = 5.16
        mu_post_attr1 = (tau0_attr1 * 0.3 + 0.4 * 60.0 / 1.0) / tau_post_attr1
        # = (5.0 * 0.3 + 24.0) / 5.16 = (1.5 + 24.0) / 5.16 = 25.5 / 5.16
        expected_mu_attr1 = 25.5 / 5.16
        expected_sigma2_attr1 = 1 / tau_post_attr1  # = 1 / 5.16
        
        # Verify results match expected calculations exactly
        assert abs(buyer_state.attr_mu[0] - expected_mu_attr0) < 1e-10, \
            f"Attr 0 mean: expected {expected_mu_attr0}, got {buyer_state.attr_mu[0]}"
        assert abs(buyer_state.attr_mu[1] - expected_mu_attr1) < 1e-10, \
            f"Attr 1 mean: expected {expected_mu_attr1}, got {buyer_state.attr_mu[1]}"
        assert abs(buyer_state.attr_sigma2[0] - expected_sigma2_attr0) < 1e-10, \
            f"Attr 0 variance: expected {expected_sigma2_attr0}, got {buyer_state.attr_sigma2[0]}"
        assert abs(buyer_state.attr_sigma2[1] - expected_sigma2_attr1) < 1e-10, \
            f"Attr 1 variance: expected {expected_sigma2_attr1}, got {buyer_state.attr_sigma2[1]}"
        
        print(f"✓ Demand transition Bayesian updates test passed")
        print(f"  Knowledge good: marketing_attributes={{'innovation_level': 'high', 'data_source': 'internal'}}, economic_impact=60.0")
        print(f"  Attr 0: μ {0.5:.6f} → {buyer_state.attr_mu[0]:.6f} (expected: {expected_mu_attr0:.6f})")
        print(f"  Attr 0: σ² {0.1:.6f} → {buyer_state.attr_sigma2[0]:.6f} (expected: {expected_sigma2_attr0:.6f})")
        print(f"  Attr 1: μ {0.3:.6f} → {buyer_state.attr_mu[1]:.6f} (expected: {expected_mu_attr1:.6f})")
        print(f"  Attr 1: σ² {0.2:.6f} → {buyer_state.attr_sigma2[1]:.6f} (expected: {expected_sigma2_attr1:.6f})")
    
    def test_demand_transition_poor_forecast_performance(self):
        """Test demand transition with poorly performing forecast leading to substantial parameter drops."""
        from multi_agent_economics.models.market_for_finance import update_buyer_preferences_from_knowledge_goods
        
        # Create a buyer with initially optimistic attribute beliefs
        class TestBuyerState:
            def __init__(self):
                self.buyer_id = "test_buyer"
                self.attr_mu = [2.5, 1.8]       # High prior means (optimistic about attributes)
                self.attr_sigma2 = [0.4, 0.3]   # Moderate prior variances
                
        buyer_state = TestBuyerState()
        
        # Set up market state with a knowledge good that has high attribute values but performs terribly
        market_state = MarketState(
            prices={},
            offers=[
                Offer(good_id="kg_terrible", price=100.0, seller="bad_analyst", 
                      marketing_attributes={"innovation_level": "high", "data_source": "proprietary"})  # High attributes
            ],
            trades=[],
            demand_profile={},
            supply_profile={},
            index_values={}
        )
        
        # Add knowledge good impact - terrible performance despite high attributes
        market_state.knowledge_good_impacts = {
            "kg_terrible": {"test_buyer": -120.0}  # Large negative economic impact
        }
        
        # Store initial values for comparison
        initial_mu_0 = buyer_state.attr_mu[0]
        initial_mu_1 = buyer_state.attr_mu[1]
        initial_sigma2_0 = buyer_state.attr_sigma2[0]
        initial_sigma2_1 = buyer_state.attr_sigma2[1]
        
        # Update preferences
        update_buyer_preferences_from_knowledge_goods(buyer_state, market_state, obs_noise_var=1.0)
        
        # Calculate expected results manually for verification
        # For attribute 0 (x_j = 0.9, impact = -120.0, obs_noise_var = 1.0):
        tau0_attr0 = 1 / 0.4  # = 2.5
        tauL_attr0 = (0.9 ** 2) / 1.0  # = 0.81
        tau_post_attr0 = tau0_attr0 + tauL_attr0  # = 3.31
        mu_post_attr0 = (tau0_attr0 * 2.5 + 0.9 * (-120.0) / 1.0) / tau_post_attr0
        # = (2.5 * 2.5 + (-108.0)) / 3.31 = (6.25 - 108.0) / 3.31 = -101.75 / 3.31
        expected_mu_attr0 = -101.75 / 3.31
        expected_sigma2_attr0 = 1 / tau_post_attr0  # = 1 / 3.31
        
        # For attribute 1 (x_j = 0.7, impact = -120.0, obs_noise_var = 1.0):
        tau0_attr1 = 1 / 0.3  # = 3.333...
        tauL_attr1 = (0.7 ** 2) / 1.0  # = 0.49
        tau_post_attr1 = tau0_attr1 + tauL_attr1  # = 3.823...
        mu_post_attr1 = (tau0_attr1 * 1.8 + 0.7 * (-120.0) / 1.0) / tau_post_attr1
        # = (3.333 * 1.8 + (-84.0)) / 3.823 = (6.0 - 84.0) / 3.823 = -78.0 / 3.823
        expected_mu_attr1 = -78.0 / (10/3 + 0.49)  # More precise calculation
        expected_mu_attr1 = -78.0 / (10/3 + 0.49)
        tau0_attr1_precise = 10/3
        tau_post_attr1_precise = tau0_attr1_precise + 0.49
        expected_mu_attr1 = (tau0_attr1_precise * 1.8 - 84.0) / tau_post_attr1_precise
        expected_sigma2_attr1 = 1 / tau_post_attr1_precise
        
        # Verify results match expected calculations exactly
        assert abs(buyer_state.attr_mu[0] - expected_mu_attr0) < 1e-10, \
            f"Attr 0 mean: expected {expected_mu_attr0}, got {buyer_state.attr_mu[0]}"
        assert abs(buyer_state.attr_mu[1] - expected_mu_attr1) < 1e-10, \
            f"Attr 1 mean: expected {expected_mu_attr1}, got {buyer_state.attr_mu[1]}"
        assert abs(buyer_state.attr_sigma2[0] - expected_sigma2_attr0) < 1e-10, \
            f"Attr 0 variance: expected {expected_sigma2_attr0}, got {buyer_state.attr_sigma2[0]}"
        assert abs(buyer_state.attr_sigma2[1] - expected_sigma2_attr1) < 1e-10, \
            f"Attr 1 variance: expected {expected_sigma2_attr1}, got {buyer_state.attr_sigma2[1]}"
        
        # Verify substantial drops in attribute means
        drop_attr0 = initial_mu_0 - buyer_state.attr_mu[0]
        drop_attr1 = initial_mu_1 - buyer_state.attr_mu[1]
        
        # Both drops should be substantial (much larger than initial values)
        assert drop_attr0 > 20.0, f"Attribute 0 mean should drop substantially, dropped by {drop_attr0}"
        assert drop_attr1 > 15.0, f"Attribute 1 mean should drop substantially, dropped by {drop_attr1}"
        
        # Final means should be negative (buyer now believes these attributes hurt performance)
        assert buyer_state.attr_mu[0] < 0, f"Attr 0 mean should be negative after poor performance, got {buyer_state.attr_mu[0]}"
        assert buyer_state.attr_mu[1] < 0, f"Attr 1 mean should be negative after poor performance, got {buyer_state.attr_mu[1]}"
        
        # Variances should decrease (buyer becomes more certain, but about negative values)
        assert buyer_state.attr_sigma2[0] < initial_sigma2_0, "Attr 0 variance should decrease"
        assert buyer_state.attr_sigma2[1] < initial_sigma2_1, "Attr 1 variance should decrease"
        
        print(f"✓ Demand transition poor forecast performance test passed")
        print(f"  Knowledge good: marketing_attributes={{'innovation_level': 'high', 'data_source': 'proprietary'}}, economic_impact=-120.0")
        print(f"  Attr 0: μ {initial_mu_0:.6f} → {buyer_state.attr_mu[0]:.6f} (drop: {drop_attr0:.6f})")
        print(f"  Attr 0: σ² {initial_sigma2_0:.6f} → {buyer_state.attr_sigma2[0]:.6f}")
        print(f"  Attr 1: μ {initial_mu_1:.6f} → {buyer_state.attr_mu[1]:.6f} (drop: {drop_attr1:.6f})")
        print(f"  Attr 1: σ² {initial_sigma2_1:.6f} → {buyer_state.attr_sigma2[1]:.6f}")
        print(f"  Buyer now believes high attribute values hurt performance!")


if __name__ == "__main__":
    test_suite = TestMarketForFinance()
    test_suite.test_regime_history_generation()
    test_suite.test_knowledge_good_impact_calculation_explicit_values()
    test_suite.test_sequential_knowledge_good_application()
    test_suite.test_surplus_computation_explicit_values()
    test_suite.test_belief_updating_edge_cases()
    test_suite.test_demand_transition_bayesian_updates_explicit_values()
    test_suite.test_demand_transition_poor_forecast_performance()
    print("✓ All market_for_finance tests passed")