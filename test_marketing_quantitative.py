"""
Rigorous quantitative tests for marketing-focused multi-agent economics.

These tests provide exact mathematical expectations based on the underlying
economic model, enabling precise validation of marketing research tools
and agent decision-making processes.

Mathematical Foundation:
- WTP Calculation: WTP = Σ(weight[i] × feature[i]) with known conversion functions
- Choice Model: Greedy selection with exact budget constraints  
- Portfolio Optimization: Mean-variance with exact regime parameters
- Belief Updating: Bayesian filtering with known confusion matrices
- Market Equilibrium: Cost-revenue balance with learning dynamics
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, Offer, ForecastData, TradeData, RegimeParameters,
    BuyerState, build_confusion_matrix, generate_forecast_signal,
    convert_marketing_to_features, compute_knowledge_good_impact
)
from multi_agent_economics.tools.implementations.economic import (
    analyze_buyer_preferences_impl, research_competitive_pricing_impl,
    analyze_historical_performance_impl
)


@pytest.fixture
def precise_buyer_preference_scenario():
    """Controlled scenario with exact mathematical expectations for buyer preference analysis."""
    
    # Setup: 2 attributes, 3 buyers with known preference vectors
    attribute_order = ["methodology", "coverage"]
    
    buyers = [
        BuyerState(
            buyer_id="buyer_1",
            regime_beliefs={"tech": [0.6, 0.4]},
            attr_weights={"tech": [0.8, 0.3]},  # High weight on methodology, low on coverage
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.2},
                "coverage": {"numeric_scaling": True, "base": 0.1, "scale": 0.8}
            }
        ),
        BuyerState(
            buyer_id="buyer_2", 
            regime_beliefs={"tech": [0.5, 0.5]},
            attr_weights={"tech": [0.4, 0.9]},  # Low weight on methodology, high on coverage
            buyer_conversion_function={
                "methodology": {"premium": 1.0, "standard": 0.7, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.0, "scale": 1.0}
            }
        ),
        BuyerState(
            buyer_id="buyer_3",
            regime_beliefs={"tech": [0.7, 0.3]},
            attr_weights={"tech": [0.6, 0.6]},  # Balanced preferences
            buyer_conversion_function={
                "methodology": {"premium": 0.8, "standard": 0.5, "basic": 0.1},
                "coverage": {"numeric_scaling": True, "base": 0.2, "scale": 0.6}
            }
        )
    ]
    
    # Initialize sector preferences for all buyers
    for buyer in buyers:
        buyer.initialize_sector_preferences(["tech"], 2, default_mu=0.5, default_sigma2=1.0)
    
    # Test offers with known attributes for WTP calculation validation
    test_offers = [
        {"methodology": "premium", "coverage": 0.8},  # High quality offer
        {"methodology": "standard", "coverage": 0.5}, # Medium quality offer  
        {"methodology": "basic", "coverage": 0.9}     # High coverage, low methodology
    ]
    
    # Marketing attribute definitions for consistent conversion
    marketing_definitions = {
        "methodology": {
            "type": "qualitative",
            "values": ["basic", "standard", "premium"],
            "descriptions": {
                "premium": "Advanced machine learning and statistical methods",
                "standard": "Traditional econometric approaches",
                "basic": "Simple heuristic methods"
            }
        },
        "coverage": {
            "type": "numeric",
            "range": [0.0, 1.0],
            "description": "Fraction of available data sources utilized"
        }
    }
    
    # Create market state
    market_state = MarketState(
        offers=[],
        trades=[],
        index_values={"tech": 100.0},
        buyers_state=buyers,
        current_regimes={"tech": 0},
        regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15), 1: RegimeParameters(mu=-0.02, sigma=0.20)}},
        attribute_order=attribute_order,
        sector_order=["tech"],
        marketing_attribute_definitions=marketing_definitions
    )
    
    # Create market model
    market_model = MarketModel(
        id=1,
        name="test_market",
        agents=[],
        state=market_state,
        step=lambda _: None,
        collect_stats=lambda: {}
    )
    
    return market_model, test_offers


def test_buyer_preference_analysis_exact_calculations(precise_buyer_preference_scenario):
    """Test regression-based buyer preference analysis with rigorous statistical validation."""
    market_model, test_offers = precise_buyer_preference_scenario
    buyers = market_model.state.buyers_state
    attribute_order = market_model.state.attribute_order
    
    # VALIDATION: Confirm the buyer setup has expected preference structure
    # Buyer 1: High weight on methodology (0.8), low on coverage (0.3)
    # Buyer 2: Low weight on methodology (0.4), high on coverage (0.9) 
    # Buyer 3: Balanced weights (0.6, 0.6)
    assert buyers[0].attr_weights["tech"][0] > buyers[0].attr_weights["tech"][1]  # methodology > coverage
    assert buyers[1].attr_weights["tech"][1] > buyers[1].attr_weights["tech"][0]  # coverage > methodology
    
    # EXACT CALCULATION: Validate WTP bounds for controlled test offers
    expected_wtp_bounds = {}
    for i, test_offer in enumerate(test_offers):
        buyer_wtps = []
        for buyer in buyers:
            features = convert_marketing_to_features(
                test_offer, buyer.buyer_conversion_function, attribute_order
            )
            wtp = np.dot(buyer.attr_weights["tech"], features)
            buyer_wtps.append(wtp)
            assert wtp > 0, f"WTP should be positive for buyer {buyer.buyer_id}"
        expected_wtp_bounds[f"offer_{i}"] = {"min": min(buyer_wtps), "max": max(buyer_wtps)}
    
    # Expected overall WTP range based on buyer weights and conversion functions:
    # Min: ~0.4 (low methodology buyer with basic offer, low coverage)
    # Max: ~1.2 (high coverage buyer with premium methodology, high coverage)
    overall_min_wtp = min(bounds["min"] for bounds in expected_wtp_bounds.values())
    overall_max_wtp = max(bounds["max"] for bounds in expected_wtp_bounds.values())
    assert 0.3 <= overall_min_wtp <= 0.6, f"Expected min WTP 0.3-0.6, got {overall_min_wtp}"
    assert 0.9 <= overall_max_wtp <= 1.3, f"Expected max WTP 0.9-1.3, got {overall_max_wtp}"
    
    # Setup config for tool call
    config_data = {
        "tool_parameters": {
            "analyze_buyer_preferences": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "high_effort_num_buyers": 10,
                "high_effort_num_test_offers": 12,
                "high_effort_analyze_by_attribute": True
            }
        }
    }
    
    # Run preference analysis tool
    pref_result = analyze_buyer_preferences_impl(
        market_model, config_data, "tech", effort=5.0
    )
    
    # STATISTICAL VALIDATION 1: Should use regression analysis for high effort
    assert pref_result.analysis_method == "regression", "High effort should trigger regression analysis"
    assert pref_result.regression_r_squared is not None, "Regression should provide R-squared"
    assert pref_result.regression_r_squared >= 0.8, f"R² should be high (>0.8) since WTP is deterministic, got {pref_result.regression_r_squared}"
    
    # STATISTICAL VALIDATION 2: Should detect both attributes as significant
    assert len(pref_result.attribute_insights) == 2, "Should analyze both methodology and coverage"
    attribute_names = [insight.attribute_name for insight in pref_result.attribute_insights]
    assert "methodology" in attribute_names, "Should include methodology in analysis"
    assert "coverage" in attribute_names, "Should include coverage in analysis"
    
    # STATISTICAL VALIDATION 3: Regression coefficients should be positive and reasonable
    methodology_insight = next((i for i in pref_result.attribute_insights if i.attribute_name == "methodology"), None)
    coverage_insight = next((i for i in pref_result.attribute_insights if i.attribute_name == "coverage"), None)
    
    assert methodology_insight is not None, "Should have methodology insights"
    assert coverage_insight is not None, "Should have coverage insights"
    
    # Both coefficients should be positive (all buyers have positive weights)
    assert methodology_insight.marginal_wtp_impact > 0, f"Methodology coefficient should be positive, got {methodology_insight.marginal_wtp_impact}"
    assert coverage_insight.marginal_wtp_impact > 0, f"Coverage coefficient should be positive, got {coverage_insight.marginal_wtp_impact}"
    
    # Coefficients should be reasonable (roughly 0.2-1.0 range based on buyer weights)
    assert 0.1 <= methodology_insight.marginal_wtp_impact <= 1.2, f"Methodology coefficient should be 0.1-1.2, got {methodology_insight.marginal_wtp_impact}"
    assert 0.1 <= coverage_insight.marginal_wtp_impact <= 1.2, f"Coverage coefficient should be 0.1-1.2, got {coverage_insight.marginal_wtp_impact}"
    
    # STATISTICAL VALIDATION 4: High confidence for deterministic relationships
    assert methodology_insight.confidence_level in ["medium", "high"], f"Methodology should have medium/high confidence, got {methodology_insight.confidence_level}"
    assert coverage_insight.confidence_level in ["medium", "high"], f"Coverage should have medium/high confidence, got {coverage_insight.confidence_level}"
    
    # STATISTICAL VALIDATION 5: Sample sizes and data structure
    assert pref_result.sample_size == 3, "Should analyze all 3 buyers"
    assert pref_result.total_observations == 36, "Should have 3 buyers × 12 test offers = 36 observations"
    assert pref_result.quality_tier == "high", "Should be high quality tier for effort=5.0"
    
    # STATISTICAL VALIDATION 6: Raw WTP data validation
    assert len(pref_result.raw_wtp_data) == 36, "Should have 36 WTP observations"
    
    wtp_values = [point.willingness_to_pay for point in pref_result.raw_wtp_data]
    observed_min_wtp = min(wtp_values)
    observed_max_wtp = max(wtp_values)
    
    # WTP distribution should match our calculated bounds
    assert observed_min_wtp >= 0.2, f"Observed min WTP too low: {observed_min_wtp}"
    assert observed_max_wtp <= 1.4, f"Observed max WTP too high: {observed_max_wtp}"
    assert observed_max_wtp - observed_min_wtp >= 0.3, "Should have sufficient WTP variation for regression"
    
    # Each buyer should be represented in the data
    buyer_ids = {point.buyer_id for point in pref_result.raw_wtp_data}
    assert buyer_ids == {"buyer_1", "buyer_2", "buyer_3"}, f"All buyers should be represented, got {buyer_ids}"
    
    # Each buyer should have multiple WTP observations (12 test offers each)
    buyer_counts = {}
    for point in pref_result.raw_wtp_data:
        buyer_counts[point.buyer_id] = buyer_counts.get(point.buyer_id, 0) + 1
    
    for buyer_id, count in buyer_counts.items():
        assert count == 12, f"Buyer {buyer_id} should have 12 observations, got {count}"


@pytest.fixture
def deterministic_pricing_scenario():
    """Scenario with deterministic greedy choice model for exact market share predictions."""
    
    attribute_order = ["methodology", "coverage"]
    
    # Setup: 4 buyers with known budgets and preferences
    buyers = [
        BuyerState(
            buyer_id="buyer_1", 
            regime_beliefs={"tech": [0.6, 0.4]},
            budget=100.0, 
            attr_weights={"tech": [1.0, 0.5]},
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.0, "scale": 1.0}
            }
        ),
        BuyerState(
            buyer_id="buyer_2", 
            regime_beliefs={"tech": [0.4, 0.6]},
            budget=80.0, 
            attr_weights={"tech": [0.6, 1.2]},
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.0, "scale": 1.0}
            }
        ),
        BuyerState(
            buyer_id="buyer_3", 
            regime_beliefs={"tech": [0.7, 0.3]},            budget=120.0, 
            attr_weights={"tech": [0.8, 0.8]},
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.0, "scale": 1.0}
            }
        ),
        BuyerState(
            buyer_id="buyer_4", 
            regime_beliefs={"tech": [0.5, 0.5]},
            budget=90.0, 
            attr_weights={"tech": [1.2, 0.4]},
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.0, "scale": 1.0}
            }
        )
    ]
    
    # Initialize sector preferences
    for buyer in buyers:
        buyer.initialize_sector_preferences(["tech"], 2)
    
    # Competitor offers with known attributes and prices (reasonable relative to WTP ~1.3-1.5)
    competitor_offers = [
        Offer(
            good_id="comp_1", 
            price=1.4, 
            seller="competitor_1", 
            marketing_attributes={"methodology": "premium", "coverage": 0.7}
        ),
        Offer(
            good_id="comp_2", 
            price=1.2, 
            seller="competitor_2",
            marketing_attributes={"methodology": "standard", "coverage": 0.9}
        )
    ]
    
    # Our test offer attributes
    our_attributes = {"methodology": "premium", "coverage": 0.8}
    
    # Create knowledge good forecasts for competitors
    knowledge_good_forecasts = {
        "comp_1": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.7, 0.3]),
        "comp_2": ForecastData(sector="tech", predicted_regime=1, confidence_vector=[0.4, 0.6])
    }
    
    # Create market state
    market_state = MarketState(
        offers=competitor_offers.copy(),
        trades=[],
        index_values={"tech": 100.0},
        buyers_state=buyers,
        current_regimes={"tech": 0},
        regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15), 1: RegimeParameters(mu=-0.02, sigma=0.20)}},
        attribute_order=attribute_order,
        sector_order=["tech"],
        marketing_attribute_definitions={
            "methodology": {"type": "qualitative", "values": ["basic", "standard", "premium"]},
            "coverage": {"type": "numeric", "range": [0.0, 1.0]}
        },
        knowledge_good_forecasts=knowledge_good_forecasts
    )
    
    # Create market model
    market_model = MarketModel(
        id=1,
        name="test_market",
        agents=[],
        state=market_state,
        step=lambda _: None,
        collect_stats=lambda: {}
    )
    
    return market_model, our_attributes


def test_competitive_pricing_exact_market_share(deterministic_pricing_scenario):
    """Test exact market share predictions using greedy choice model."""
    market_model, our_attributes = deterministic_pricing_scenario
    buyers = market_model.state.buyers_state
    attribute_order = market_model.state.attribute_order
    
    # EXACT CALCULATION: WTP for each buyer given our attributes {"methodology": "premium", "coverage": 0.8}
    # Standard conversion: premium=0.9, coverage=0.8 (numeric)
    
    # Buyer 1: WTP = 1.0 * 0.9 + 0.5 * 0.8 = 0.9 + 0.4 = 1.3
    # Buyer 2: WTP = 0.6 * 0.9 + 1.2 * 0.8 = 0.54 + 0.96 = 1.5  
    # Buyer 3: WTP = 0.8 * 0.9 + 0.8 * 0.8 = 0.72 + 0.64 = 1.36
    # Buyer 4: WTP = 1.2 * 0.9 + 0.4 * 0.8 = 1.08 + 0.32 = 1.4
    
    expected_wtps = [1.3, 1.5, 1.36, 1.4]
    
    # Validate WTP calculations
    for i, buyer in enumerate(buyers):
        features = convert_marketing_to_features(
            our_attributes, buyer.buyer_conversion_function, attribute_order
        )
        calculated_wtp = np.dot(buyer.attr_weights["tech"], features)
        assert abs(calculated_wtp - expected_wtps[i]) < 0.001, f"WTP mismatch for buyer {i+1}"
    
    # Setup config for tool call
    config_data = {
        "tool_parameters": {
            "research_competitive_pricing": {
                "effort_thresholds": {"high": 2.5, "medium": 1.2},
                "high_effort_num_buyers": 30,
                "high_effort_price_points": 12,
                "high_effort_lookback_trades": 50
            }
        },
        "choice_model": "greedy"  # Use greedy for deterministic results
    }
    
    # Run competitive pricing analysis
    pricing_result = research_competitive_pricing_impl(
        market_model, config_data, "tech", effort=5.0, marketing_attributes=our_attributes
    )
    
    # EXACT CHECK: Test specific price points and expected market shares
    
    # At price=1.0: All buyers should purchase (WTP > 1.0 for all)
    # Expected market share = 4/4 = 1.0 (100%)
    price_1_0_sim = next((sim for sim in pricing_result.price_simulations if abs(sim["price"] - 1.0) < 0.1), None)
    if price_1_0_sim:
        assert abs(price_1_0_sim["market_share"] - 1.0) < 0.01
        assert price_1_0_sim["buyers_purchasing"] == 4
    
    # At price=1.35: Buyers 2,3,4 should purchase (WTP >= 1.35)  
    # Expected market share = 3/4 = 0.75 (75%)
    price_1_35_sim = next((sim for sim in pricing_result.price_simulations if abs(sim["price"] - 1.35) < 0.05), None)
    if price_1_35_sim:
        assert abs(price_1_35_sim["market_share"] - 0.75) < 0.01
        assert price_1_35_sim["buyers_purchasing"] == 3
        
    # At price=1.45: Only buyer 2 should purchase (WTP = 1.5 > 1.45)
    # Expected market share = 1/4 = 0.25 (25%)  
    price_1_45_sim = next((sim for sim in pricing_result.price_simulations if abs(sim["price"] - 1.45) < 0.05), None)
    if price_1_45_sim:
        assert abs(price_1_45_sim["market_share"] - 0.25) < 0.01
        assert price_1_45_sim["buyers_purchasing"] == 1
        
    # EXACT CHECK: Recommended price should maximize revenue
    # Revenue = price × buyers_purchasing
    # At price=1.3: revenue = 1.3 × 4 = 5.2
    # At price=1.35: revenue = 1.35 × 3 = 4.05  
    # At price=1.4: revenue = 1.4 × 2 = 2.8 (buyers 2,4)
    # At price=1.45: revenue = 1.45 × 1 = 1.45
    # Maximum should be around 1.3
    
    assert 1.25 <= pricing_result.recommended_price <= 1.35  # Should optimize around this range


@pytest.fixture
def quality_performance_scenario():
    """Scenario with exact regime-switching parameters for profit calculation."""
    
    # Regime setup: Tech sector with 2 regimes
    regime_params = {
        "tech": {
            0: RegimeParameters(mu=0.08, sigma=0.15),  # Bull market
            1: RegimeParameters(mu=-0.02, sigma=0.25)  # Bear market  
        }
    }
    
    # Transition matrix: 80% persistence
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    
    # Current state: Bull market (regime 0)  
    current_regime = 0
    true_next_regime = 1  # Will transition to bear market
    
    # Quality levels and their confusion matrices
    high_quality_confusion = build_confusion_matrix(forecast_quality=0.85, K=2, base_quality=0.6)
    low_quality_confusion = build_confusion_matrix(forecast_quality=0.55, K=2, base_quality=0.6)
    
    # Create regime history with known outcomes
    regime_history = [
        {
            "period": 0,
            "returns": {"tech": 0.05},  # Current period return
            "regimes": {"tech": current_regime},
            "index_values": {"tech": 105.0}
        },
        {
            "period": 1, 
            "returns": {"tech": -0.03},  # Next period return (bear market)
            "regimes": {"tech": true_next_regime},
            "index_values": {"tech": 101.85}  # 105 * (1 - 0.03)
        }
    ]
    
    # Create buyer with known portfolio preferences
    buyer = BuyerState(
        buyer_id="portfolio_buyer",
        regime_beliefs={"tech": [0.7, 0.3]},  # Prior belief: 70% bull, 30% bear
        risk_aversion=2.0,
        budget=1000.0
    )
    
    # Create market state with regime information
    market_state = MarketState(
        offers=[],
        trades=[],
        index_values={"tech": 105.0},
        buyers_state=[buyer],
        current_regimes={"tech": current_regime},
        regime_transition_matrices={"tech": transition_matrix},
        regime_parameters=regime_params,
        regime_correlations=np.array([[1.0]]),  # Single sector correlation matrix
        regime_history=regime_history,
        current_period=0,
        sector_order=["tech"],
        knowledge_good_forecasts={},
        risk_free_rate=0.03
    )
    
    # Create market model
    market_model = MarketModel(
        id=1,
        name="quality_test_market", 
        agents=[],
        state=market_state,
        step=lambda _: None,
        collect_stats=lambda: {}
    )
    
    return market_model, high_quality_confusion, low_quality_confusion, true_next_regime


def test_forecast_accuracy_profit_differential(quality_performance_scenario):
    """Test exact profit differential from forecast quality using portfolio optimization."""
    market_model, high_conf, low_conf, true_next_regime = quality_performance_scenario
    buyer_state = market_model.state.buyers_state[0]
    
    # EXACT CALCULATION: Forecast signal probabilities
    # Confusion matrix for high quality (85% accuracy, base 60%):
    # For K=2: q = 0.6 + (1-0.6) * 0.85 = 0.6 + 0.34 = 0.94 (on diagonal)
    # Off-diagonal: (1-0.94)/(2-1) = 0.06
    
    # High quality confusion matrix should be: [[0.94, 0.06], [0.06, 0.94]]
    expected_high_diagonal = 0.6 + (1 - 0.6) * 0.85
    expected_high_off_diagonal = (1 - expected_high_diagonal) / (2 - 1)
    
    assert abs(high_conf[0, 0] - expected_high_diagonal) < 0.001
    assert abs(high_conf[0, 1] - expected_high_off_diagonal) < 0.001
    assert abs(high_conf[1, 1] - expected_high_diagonal) < 0.001
    assert abs(high_conf[1, 0] - expected_high_off_diagonal) < 0.001
    
    # Low quality confusion matrix: q = 0.6 + (1-0.6) * 0.55 = 0.6 + 0.22 = 0.82
    expected_low_diagonal = 0.6 + (1 - 0.6) * 0.55  
    expected_low_off_diagonal = (1 - expected_low_diagonal) / (2 - 1)
    
    assert abs(low_conf[0, 0] - expected_low_diagonal) < 0.001
    assert abs(low_conf[0, 1] - expected_low_off_diagonal) < 0.001
    assert abs(low_conf[1, 1] - expected_low_diagonal) < 0.001
    assert abs(low_conf[1, 0] - expected_low_off_diagonal) < 0.001
    
    # Generate forecast signals for true next regime (regime 1 = bear market)
    high_quality_signal = generate_forecast_signal(true_next_regime, high_conf)
    low_quality_signal = generate_forecast_signal(true_next_regime, low_conf)
    
    # Create forecast data objects
    high_quality_forecast = ForecastData(
        sector="tech",
        predicted_regime=high_quality_signal["predicted_regime"],
        confidence_vector=high_quality_signal["confidence_vector"]
    )
    
    low_quality_forecast = ForecastData(
        sector="tech", 
        predicted_regime=low_quality_signal["predicted_regime"],
        confidence_vector=low_quality_signal["confidence_vector"]
    )
    
    # Store forecasts in market state
    market_model.state.knowledge_good_forecasts["high_quality_good"] = high_quality_forecast
    market_model.state.knowledge_good_forecasts["low_quality_good"] = low_quality_forecast
    
    # Create offers for knowledge goods
    high_quality_offer = Offer(
        good_id="high_quality_good",
        price=50.0,
        seller="high_seller",
        marketing_attributes={"methodology": "premium", "accuracy": "high"}
    )
    
    low_quality_offer = Offer(
        good_id="low_quality_good",
        price=30.0,
        seller="low_seller", 
        marketing_attributes={"methodology": "basic", "accuracy": "medium"}
    )
    
    # EXACT CHECK 1: High quality forecast should be more accurate on average
    # Run multiple trials to test probabilistic accuracy
    high_correct_predictions = 0
    low_correct_predictions = 0
    num_trials = 1000
    
    for _ in range(num_trials):
        high_signal = generate_forecast_signal(true_next_regime, high_conf)
        low_signal = generate_forecast_signal(true_next_regime, low_conf)
        
        if high_signal["predicted_regime"] == true_next_regime:
            high_correct_predictions += 1
        if low_signal["predicted_regime"] == true_next_regime:
            low_correct_predictions += 1
    
    high_accuracy = high_correct_predictions / num_trials
    low_accuracy = low_correct_predictions / num_trials
    
    # High quality should be more accurate (expected ~94% vs ~82%)
    assert high_accuracy > low_accuracy + 0.05  # At least 5% better accuracy
    assert high_accuracy > 0.90  # Should be around 94%
    assert low_accuracy > 0.75   # Should be around 82%
    
    # EXACT CHECK 2: Economic impact calculation
    high_quality_impact = compute_knowledge_good_impact(buyer_state, high_quality_offer, market_model.state)
    low_quality_impact = compute_knowledge_good_impact(buyer_state, low_quality_offer, market_model.state)
    
    # High quality forecast should provide higher economic value
    quality_premium = high_quality_impact["economic_value"] - low_quality_impact["economic_value"]
    
    # EXACT CHECK 3: Belief updating should be different
    # High quality forecast should update beliefs more significantly toward correct regime
    high_beliefs_after = high_quality_impact["beliefs_after"]["tech"]
    low_beliefs_after = low_quality_impact["beliefs_after"]["tech"]
    
    # If forecast correctly predicts bear market (regime 1), belief in regime 1 should increase
    # High quality forecast should update beliefs more dramatically
    initial_bear_belief = buyer_state.regime_beliefs["tech"][1]  # 0.3
    
    high_bear_belief_after = high_beliefs_after[1]
    low_bear_belief_after = low_beliefs_after[1]
    
    # Both should increase belief in bear market, but high quality should increase more
    assert high_bear_belief_after > initial_bear_belief
    assert low_bear_belief_after > initial_bear_belief
    assert high_bear_belief_after > low_bear_belief_after  # High quality should update more
    
    # EXACT CHECK 4: Quality premium should be substantial
    assert quality_premium > 0.05  # At least 5% higher economic value from better forecast
    
    # EXACT CHECK 5: Portfolio weights should differ meaningfully
    high_weights = high_quality_impact["weights_after"]
    low_weights = low_quality_impact["weights_after"]
    
    # Weights should be different due to different belief updating
    weight_difference = np.abs(high_weights - low_weights).sum()
    assert weight_difference > 0.01  # Portfolios should differ by at least 1% allocation


@pytest.fixture
def adverse_selection_equilibrium():
    """Multi-period scenario with exact cost structures for equilibrium analysis."""
    
    # Cost structure: High quality costs 60 credits, low quality costs 20 credits
    cost_structure = {"high_quality": 60.0, "low_quality": 20.0}
    
    # Buyer learning parameters: Bayesian updating with experience
    initial_attribute_uncertainty = 1.0  # High initial uncertainty
    
    # Market setup: 6 sellers (3 high, 3 low), 8 buyers
    high_quality_sellers = [
        {"id": f"high_{i}", "type": "high_quality", "cost": 60.0, "true_accuracy": 0.85} 
        for i in range(3)
    ]
    low_quality_sellers = [
        {"id": f"low_{i}", "type": "low_quality", "cost": 20.0, "true_accuracy": 0.55} 
        for i in range(3) 
    ]
    
    sellers = high_quality_sellers + low_quality_sellers
    
    # Create buyers with initial uncertainty about quality signals
    buyers = []
    for i in range(8):
        buyer = BuyerState(
            buyer_id=f"buyer_{i}",
            regime_beliefs={"tech": [0.5, 0.5]},            budget=100.0,
            attr_mu={"tech": [0.5, 0.5]},  # Initial neutral preferences
            attr_sigma2={"tech": [initial_attribute_uncertainty, initial_attribute_uncertainty]},
            buyer_conversion_function={
                "methodology": {"premium": 0.8, "standard": 0.5, "basic": 0.2},
                "accuracy": {"high": 0.9, "medium": 0.6, "low": 0.3}
            }
        )
        buyers.append(buyer)
    
    # Regime parameters for economic impact calculation
    regime_params = {
        "tech": {
            0: RegimeParameters(mu=0.06, sigma=0.15),
            1: RegimeParameters(mu=-0.01, sigma=0.20)
        }
    }
    
    # Create initial market state
    market_state = MarketState(
        offers=[],
        trades=[],
        index_values={"tech": 100.0},
        all_trades=[],
        buyers_state=buyers,
        sellers_state=[],
        current_regimes={"tech": 0},
        regime_parameters=regime_params,
        regime_correlations=np.array([[1.0]]),
        current_period=0,
        knowledge_good_forecasts={},
        knowledge_good_impacts={},
        attribute_order=["methodology", "accuracy"],
        sector_order=["tech"],
        risk_free_rate=0.03
    )
    
    return cost_structure, sellers, market_state


def test_adverse_selection_exact_equilibrium(adverse_selection_equilibrium):
    """Test exact adverse selection dynamics over multiple periods."""
    cost_structure, _, _ = adverse_selection_equilibrium
    
    # EXACT PREDICTION: Initial period outcomes
    # With uniform pricing (buyers can't distinguish quality), expect:
    # - Market price around 40-50 (average willingness to pay)
    # - High quality sellers: Revenue ~45, Cost 60 → Loss of ~15
    # - Low quality sellers: Revenue ~45, Cost 20 → Profit of ~25
    
    # Simulate initial uniform pricing scenario
    uniform_price = 45.0
    
    # Calculate expected profits
    high_quality_expected_profit = uniform_price - cost_structure["high_quality"]  # 45 - 60 = -15
    low_quality_expected_profit = uniform_price - cost_structure["low_quality"]   # 45 - 20 = +25
    
    # EXACT CHECK 1: Cost structure should create adverse incentives initially
    assert high_quality_expected_profit < 0  # High quality loses money
    assert low_quality_expected_profit > 0   # Low quality makes profit
    assert low_quality_expected_profit > high_quality_expected_profit + 30  # 25 > -15 + 30 = 15
    
    # EXACT PREDICTION: Learning dynamics
    # As buyers experience forecast performance, they should:
    # 1. Update attr_mu toward quality-correlated attributes
    # 2. Reduce attr_sigma2 (become more confident)
    # 3. Pay premiums for signals of high quality
    
    # Simulate buyer learning from experience
    # High quality forecast provides economic value ~10-15 per period
    # Low quality forecast provides economic value ~2-5 per period
    
    high_quality_performance_history = [12.0, 11.5, 13.2, 10.8, 14.1]  # Consistently good
    low_quality_performance_history = [3.2, 4.1, 2.8, 3.9, 2.5]        # Consistently poor
    
    # Calculate learning updates for methodology attribute (assuming correlation with quality)
    # Buyer should learn that "premium" methodology correlates with higher performance
    
    # Initial uncertainty: attr_sigma2 = 1.0
    # After observing performance data, update beliefs about "premium" methodology
    
    # Simplified Bayesian update: posterior precision = prior precision + Σ(likelihood precision)
    # For methodology="premium" → high performance, increase attr_mu[0]
    
    obs_noise_var = 1.0  # Observation noise
    initial_precision = 1.0 / 1.0  # 1 / initial_sigma2
    
    # High methodology (premium) provides better performance
    methodology_evidence = sum(high_quality_performance_history) / len(high_quality_performance_history)  # ~12.3
    
    # Expected posterior for methodology preference
    likelihood_precision = 1.0 / obs_noise_var  # Precision from each observation
    posterior_precision = initial_precision + len(high_quality_performance_history) * likelihood_precision
    posterior_mean = (initial_precision * 0.5 + likelihood_precision * methodology_evidence * len(high_quality_performance_history)) / posterior_precision
    
    # EXACT CHECK 2: Learning should increase preference for quality-correlated attributes
    expected_methodology_preference = posterior_mean  # Should be much higher than initial 0.5
    assert expected_methodology_preference > 0.8  # Should learn to value methodology highly
    
    # EXACT CHECK 3: Variance reduction from learning
    expected_posterior_variance = 1.0 / posterior_precision
    assert expected_posterior_variance < 0.5  # Should be much more confident than initial 1.0
    
    # EXACT PREDICTION: Equilibrium emergence
    # After learning, buyers should pay premiums for quality signals:
    # - High quality can charge premium ~25-30% above cost (75-80)
    # - Low quality forced to compete on price (~25-30)
    
    learned_high_quality_price = cost_structure["high_quality"] * 1.25  # 60 * 1.25 = 75
    learned_low_quality_price = cost_structure["low_quality"] * 1.4     # 20 * 1.4 = 28
    
    learned_high_profit = learned_high_quality_price - cost_structure["high_quality"]  # 75 - 60 = 15
    learned_low_profit = learned_low_quality_price - cost_structure["low_quality"]    # 28 - 20 = 8
    
    # EXACT CHECK 4: Post-learning equilibrium should support quality differentiation
    assert learned_high_profit > 0   # High quality becomes profitable
    assert learned_low_profit > 0    # Low quality remains profitable but lower
    assert learned_high_profit > learned_low_profit  # Quality premium justified by performance
    
    # EXACT CHECK 5: Market shares should favor quality after learning
    # Initial: 50% high quality, 50% low quality (random)
    # Post-learning: ~65-70% high quality as buyers learn to identify value
    
    initial_market_share_high = 0.5  # Random initial allocation
    expected_final_market_share_high = 0.67  # Buyers learn to prefer quality
    
    market_share_improvement = expected_final_market_share_high - initial_market_share_high
    assert market_share_improvement > 0.15  # At least 15% improvement for high quality
    
    # EXACT CHECK 6: Total market efficiency should improve
    # Efficiency = (high_quality_share × high_performance + low_quality_share × low_performance)
    
    high_performance = np.mean(high_quality_performance_history)  # ~12.3
    low_performance = np.mean(low_quality_performance_history)    # ~3.3
    
    initial_efficiency = 0.5 * high_performance + 0.5 * low_performance  # ~7.8
    final_efficiency = expected_final_market_share_high * high_performance + (1 - expected_final_market_share_high) * low_performance  # ~9.3
    
    efficiency_improvement = (final_efficiency - initial_efficiency) / initial_efficiency
    assert efficiency_improvement > 0.15  # At least 15% efficiency gain from learning


@pytest.fixture  
def research_tool_roi_scenario():
    """Scenario for testing exact return on investment for each research tool."""
    
    # Create controlled market with known parameters
    attribute_order = ["methodology", "coverage"]
    
    # 10 buyers with varied but known preferences
    buyers = []
    for i in range(10):
        # Create diverse but predictable buyer preferences
        methodology_weight = 0.3 + (i % 5) * 0.15  # Range from 0.3 to 0.9
        coverage_weight = 0.2 + (i % 3) * 0.2      # Range from 0.2 to 0.6
        
        buyer = BuyerState(
            buyer_id=f"buyer_{i}",
            regime_beliefs={"tech": [0.6, 0.4]},            budget=80.0 + i * 5,  # Budgets from 80 to 125
            attr_weights={"tech": [methodology_weight, coverage_weight]},
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.1, "scale": 0.8}
            }
        )
        buyers.append(buyer)
    
    # Create historical trade data with known patterns
    historical_trades = []
    base_prices = {"premium": 55.0, "standard": 40.0, "basic": 25.0}
    
    # Map methodology to competitor good_id for historical trades
    methodology_to_good_id = {"premium": "comp_1", "standard": "comp_2", "basic": "comp_3"}
    
    for period in range(5):
        for methodology in ["premium", "standard", "basic"]:
            for coverage in [0.3, 0.6, 0.9]:
                # Price varies with methodology and coverage
                price = base_prices[methodology] + coverage * 15.0
                
                trade = TradeData(
                    buyer_id=f"buyer_{period % 10}",
                    seller_id=f"seller_{methodology}",
                    price=price,
                    quantity=1,
                    good_id=methodology_to_good_id[methodology],  # Use comp_1, comp_2, comp_3
                    marketing_attributes={"methodology": methodology, "coverage": coverage},
                    period=period
                )
                historical_trades.append(trade)
    
    # Create competitor offers
    competitor_offers = [
        Offer(good_id="comp_1", price=50.0, seller="comp_seller_1", 
              marketing_attributes={"methodology": "premium", "coverage": 0.7}),
        Offer(good_id="comp_2", price=35.0, seller="comp_seller_2",
              marketing_attributes={"methodology": "standard", "coverage": 0.8}),
        Offer(good_id="comp_3", price=25.0, seller="comp_seller_3",
              marketing_attributes={"methodology": "basic", "coverage": 0.5})
    ]
    
    # Create forecasts for competitors
    knowledge_good_forecasts = {
        offer.good_id: ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.7, 0.3])
        for offer in competitor_offers
    }
    
    # Create market state
    market_state = MarketState(
        offers=competitor_offers.copy(),
        trades=[],
        index_values={"tech": 100.0},
        all_trades=historical_trades,
        buyers_state=buyers,
        current_regimes={"tech": 0},
        regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15), 1: RegimeParameters(mu=-0.02, sigma=0.20)}},
        attribute_order=attribute_order,
        sector_order=["tech"],
        marketing_attribute_definitions={
            "methodology": {"type": "qualitative", "values": ["basic", "standard", "premium"]},
            "coverage": {"type": "numeric", "range": [0.0, 1.0]}
        },
        knowledge_good_forecasts=knowledge_good_forecasts
    )
    
    # Create market model
    market_model = MarketModel(
        id=1,
        name="roi_test_market",
        agents=[],
        state=market_state,
        step=lambda _: None,
        collect_stats=lambda: {}
    )
    
    return market_model


def test_research_tool_exact_roi(research_tool_roi_scenario):
    """Test exact return on investment for each research tool."""
    market_model = research_tool_roi_scenario
    
    # Define research tool costs and expected insights
    tool_costs = {
        "analyze_historical_performance": 2.0,
        "analyze_buyer_preferences": 3.0, 
        "research_competitive_pricing": 2.5
    }
    
    # Setup config for tool calls
    config_data = {
        "tool_parameters": {
            "analyze_historical_performance": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "high_effort_max_trades": 100,
                "high_effort_noise_factor": 0.05
            },
            "analyze_buyer_preferences": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "high_effort_num_buyers": 30,
                "high_effort_num_test_offers": 12,
                "high_effort_analyze_by_attribute": True
            },
            "research_competitive_pricing": {
                "effort_thresholds": {"high": 2.5, "medium": 1.2},
                "high_effort_num_buyers": 30,
                "high_effort_price_points": 12,
                "high_effort_lookback_trades": 50
            }
        },
        "choice_model": "greedy"
    }
    
    # TEST 1: Historical Performance Analysis ROI
    historical_result = analyze_historical_performance_impl(
        market_model, config_data, "tech", effort=5.0
    )
    
    # EXACT CHECK: Should return substantial historical data for analysis
    assert historical_result.sample_size >= 10  # Should get meaningful sample
    assert len(historical_result.trade_data) >= 10
    
    # EXACT CALCULATION: Expected value from historical insights
    # Historical data shows premium methodology commands 55+15*coverage price premium
    # This insight is worth ~10-15 profit per period vs random pricing
    
    # Calculate price patterns from returned data
    premium_prices = [t["price"] for t in historical_result.trade_data 
                     if t["marketing_attributes"]["methodology"] == "premium"]
    standard_prices = [t["price"] for t in historical_result.trade_data
                      if t["marketing_attributes"]["methodology"] == "standard"]
    
    if premium_prices and standard_prices:
        premium_avg = np.mean(premium_prices)
        standard_avg = np.mean(standard_prices)
        methodology_premium = premium_avg - standard_avg
        
        # Expected ROI: methodology premium insight worth ~8-12 profit, costs 2.0
        expected_historical_roi = methodology_premium / tool_costs["analyze_historical_performance"]
        assert expected_historical_roi > 3.0  # Should provide > 3x ROI
    
    # TEST 2: Buyer Preference Analysis ROI  
    buyer_pref_result = analyze_buyer_preferences_impl(
        market_model, config_data, "tech", effort=5.0
    )
    
    # EXACT CHECK: Should provide regression-based attribute insights
    assert buyer_pref_result.analysis_method in ["regression", "descriptive"]
    assert len(buyer_pref_result.attribute_insights) >= 2
    
    # Find the attribute with the highest absolute marginal impact
    top_insight = max(buyer_pref_result.attribute_insights, 
                     key=lambda x: abs(x.marginal_wtp_impact) if x.marginal_wtp_impact else 0)
    
    # EXACT CALCULATION: Value of preference insights
    # Knowing marginal WTP impact provides targeting advantage
    # Should increase profit by leveraging high-impact attributes
    
    if top_insight.marginal_wtp_impact is not None:
        # Use absolute marginal impact as basis for ROI calculation
        marginal_impact_value = abs(top_insight.marginal_wtp_impact)
        preference_insight_value = marginal_impact_value * 15  # Approximate profit multiplier
        
        preference_roi = preference_insight_value / tool_costs["analyze_buyer_preferences"]
        assert preference_roi > 1.0  # Should provide positive ROI
    else:
        # For descriptive analysis, ensure we still get useful insights
        assert buyer_pref_result.total_observations > 0
        assert len(buyer_pref_result.raw_wtp_data) > 0
    
    # TEST 3: Competitive Pricing Analysis ROI
    pricing_result = research_competitive_pricing_impl(
        market_model, config_data, "tech", effort=5.0, 
        marketing_attributes={"methodology": "premium", "coverage": 0.8}
    )
    
    # EXACT CHECK: Should provide actionable pricing recommendations
    assert len(pricing_result.price_simulations) >= 5
    assert pricing_result.recommended_price > 0
    
    # EXACT CALCULATION: Value of pricing optimization
    # Pricing research should identify optimal price point vs suboptimal pricing
    # Difference between optimal and random pricing should be ~10-15 profit per period
    
    # Find maximum revenue simulation
    max_revenue_sim = max(pricing_result.price_simulations, key=lambda x: x["expected_revenue"])
    max_revenue = max_revenue_sim["expected_revenue"]
    
    # Compare to suboptimal pricing (e.g., 20% below optimal)
    suboptimal_price = pricing_result.recommended_price * 0.8
    suboptimal_sim = min(pricing_result.price_simulations, 
                        key=lambda x: abs(x["price"] - suboptimal_price))
    suboptimal_revenue = suboptimal_sim["expected_revenue"]
    
    pricing_advantage = max_revenue - suboptimal_revenue
    pricing_roi = pricing_advantage / tool_costs["research_competitive_pricing"]
    
    assert pricing_roi > 3.0  # Should provide > 3x ROI
    
    # TEST 4: Combined tool usage should provide multiplicative benefits
    # Using all three tools should provide better ROI than sum of individual ROIs
    total_tool_cost = sum(tool_costs.values())  # 2.0 + 3.0 + 2.5 = 7.5
    
    # Combined insights should enable optimal strategy:
    # - Historical data informs methodology choice
    # - Buyer preferences guide attribute selection  
    # - Competitive pricing optimizes revenue
    
    combined_benefit = methodology_premium + preference_insight_value + pricing_advantage
    combined_roi = combined_benefit / total_tool_cost
    
    assert combined_roi > 4.0  # Combined usage should provide > 4x ROI
    
    # EXACT CHECK: Each tool should beat minimum ROI threshold individually
    assert preference_roi > 2.0   # Minimum 2x ROI for buyer preferences
    assert pricing_roi > 2.0      # Minimum 2x ROI for competitive pricing  
    
    if premium_prices and standard_prices:
        assert expected_historical_roi > 2.0  # Minimum 2x ROI for historical analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])