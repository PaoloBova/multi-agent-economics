"""
Comprehensive statistical property testing for buyer preference analysis tool.

These tests validate important statistical scenarios and properties of the buyer
preference analysis, including effort regime transitions, market heterogeneity,
attribute correlations, WTP distributions, and statistical power considerations.
"""

import numpy as np

from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, BuyerState, RegimeParameters,
    convert_marketing_to_features
)
from multi_agent_economics.tools.implementations.economic import (
    analyze_buyer_preferences_impl, generate_test_offers
)
from multi_agent_economics.tools.schemas import BuyerPreferenceResponse


class TestBuyerPreferenceStatisticalProperties:
    """Test statistical properties of buyer preference analysis across different scenarios."""

    def test_effort_regime_transitions(self):
        """Test statistical behavior transitions at effort thresholds."""
        # Setup: Create market with sufficient buyers for all effort levels
        buyers = []
        for i in range(40):  # Enough buyers for high effort testing
            # Create buyers with varied preferences to ensure WTP variation
            methodology_weight = 0.3 + 0.4 * (i / 39)  # Range from 0.3 to 0.7
            coverage_weight = 1.0 - methodology_weight
            
            buyer = BuyerState(
                buyer_id=f"buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "methodology": {"categorical": {"basic": 0.3, "advanced": 0.7, "cutting_edge": 1.0}},
                    "coverage": {"numeric": {"scale_factor": 1.0}}
                },
                attr_weights={"tech": [methodology_weight, coverage_weight]}  # Varied preferences
            )
            buyers.append(buyer)

        state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=["methodology", "coverage"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "methodology": {
                    "type": "qualitative", 
                    "values": ["basic", "advanced", "cutting_edge"]
                },
                "coverage": {
                    "type": "numeric", 
                    "range": [0.0, 1.0]
                }
            }
        )
        market_model = MarketModel(id=1, name="Test", agents=[], state=state, step=lambda: None, collect_stats=lambda: {})

        config_data = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 30,
                    "high_effort_num_test_offers": 12,
                    "high_effort_analyze_by_attribute": True,
                    "medium_effort_num_buyers": 15,
                    "medium_effort_num_test_offers": 6,
                    "medium_effort_analyze_by_attribute": False,
                    "low_effort_num_buyers": 5,
                    "low_effort_num_test_offers": 3,
                    "low_effort_analyze_by_attribute": False
                }
            }
        }

        # Test low effort regime (effort=1.0)
        low_result = analyze_buyer_preferences_impl(market_model, config_data, "tech", effort=1.0)
        
        # Statistical validation for low effort
        assert low_result.quality_tier == "low"
        assert low_result.analysis_method == "descriptive"
        assert low_result.sample_size == 5
        assert low_result.total_observations == 15  # 5 buyers × 3 test offers
        assert low_result.regression_r_squared is None  # No regression for low effort
        assert all(insight.marginal_wtp_impact is None for insight in low_result.attribute_insights)
        assert all(insight.confidence_level == "low" for insight in low_result.attribute_insights)

        # Test medium effort regime (effort=2.0)
        medium_result = analyze_buyer_preferences_impl(market_model, config_data, "tech", effort=2.0)
        
        # Statistical validation for medium effort
        assert medium_result.quality_tier == "medium"
        assert medium_result.analysis_method == "descriptive"  # Still descriptive, no regression
        assert medium_result.sample_size == 15
        assert medium_result.total_observations == 90  # 15 buyers × 6 test offers
        assert medium_result.regression_r_squared is None  # No regression for medium effort
        assert all(insight.marginal_wtp_impact is None for insight in medium_result.attribute_insights)
        # Medium effort with 90 observations should give medium confidence
        assert all(insight.confidence_level == "medium" for insight in medium_result.attribute_insights)

        # Test high effort regime (effort=4.0)
        np.random.seed(42)  # Reproducible test offers
        high_result = analyze_buyer_preferences_impl(market_model, config_data, "tech", effort=4.0)
        
        # Statistical validation for high effort
        assert high_result.quality_tier == "high"
        assert high_result.sample_size == 30
        assert high_result.total_observations == 360  # 30 buyers × 12 test offers
        
        # High effort should trigger regression analysis (≥30 observations + use_regression=True)
        assert high_result.analysis_method == "regression"
        assert high_result.regression_r_squared is not None
        
        # Regression should produce reasonable R² for moderate buyer consensus
        assert 0.4 <= high_result.regression_r_squared <= 0.95, f"Expected R² 0.4-0.95, got {high_result.regression_r_squared}"
        
        # All attributes should have regression coefficients
        assert all(insight.marginal_wtp_impact is not None for insight in high_result.attribute_insights)
        
        # With 360 observations, should have high statistical confidence
        high_confidence_count = sum(1 for insight in high_result.attribute_insights if insight.confidence_level == "high")
        assert high_confidence_count >= 1, "At least one attribute should have high confidence with 360 observations"

    def test_market_heterogeneity_scenarios(self):
        """Test statistical properties under different buyer preference patterns."""
        attribute_order = ["methodology", "coverage"]
        
        config_data = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 20,
                    "high_effort_num_test_offers": 8,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        # Scenario 1: HIGH CONSENSUS - All buyers have similar preferences
        consensus_buyers = []
        for i in range(25):
            buyer = BuyerState(
                buyer_id=f"consensus_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "methodology": {"categorical": {"basic": 0.2, "advanced": 0.6, "cutting_edge": 1.0}},
                    "coverage": {"numeric": {"scale_factor": 0.8}}
                },
                attr_weights={"tech": [0.7, 0.3]}  # Strong preference for methodology (consensus)
            )
            consensus_buyers.append(buyer)

        consensus_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=consensus_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=attribute_order,
            sector_order=["tech"],
            marketing_attribute_definitions={
                "methodology": {"type": "qualitative", "values": ["basic", "advanced", "cutting_edge"]},
                "coverage": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        consensus_market = MarketModel(id=1, name="Test", agents=[], state=consensus_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        consensus_result = analyze_buyer_preferences_impl(consensus_market, config_data, "tech", effort=4.0)

        # HIGH CONSENSUS: Should have high R² due to similar buyer preferences
        assert consensus_result.analysis_method == "regression"
        assert consensus_result.regression_r_squared >= 0.75, f"High consensus should have R² ≥ 0.75, got {consensus_result.regression_r_squared}"
        
        # Should have high confidence levels due to consistent pattern
        high_confidence_attrs = [insight for insight in consensus_result.attribute_insights if insight.confidence_level == "high"]
        assert len(high_confidence_attrs) >= 1, "High consensus should produce at least one high-confidence attribute"

        # Scenario 2: HIGH DISAGREEMENT - Buyers have very different preferences  
        disagreement_buyers = []
        preference_weights = [
            [0.9, 0.1],  # Strongly prefer methodology
            [0.1, 0.9],  # Strongly prefer coverage
            [0.5, 0.5],  # Neutral
            [0.8, 0.2],  # Moderately prefer methodology
            [0.2, 0.8]   # Moderately prefer coverage
        ]
        
        for i in range(25):
            weight_pattern = preference_weights[i % len(preference_weights)]
            buyer = BuyerState(
                buyer_id=f"disagreement_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "methodology": {"categorical": {"basic": 0.2, "advanced": 0.6, "cutting_edge": 1.0}},
                    "coverage": {"numeric": {"scale_factor": 0.8}}
                },
                attr_weights={"tech": weight_pattern}  # Diverse preferences
            )
            disagreement_buyers.append(buyer)

        disagreement_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=disagreement_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=attribute_order,
            sector_order=["tech"],
            marketing_attribute_definitions={
                "methodology": {"type": "qualitative", "values": ["basic", "advanced", "cutting_edge"]},
                "coverage": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        disagreement_market = MarketModel(id=1, name="Test", agents=[], state=disagreement_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        disagreement_result = analyze_buyer_preferences_impl(disagreement_market, config_data, "tech", effort=4.0)

        # HIGH DISAGREEMENT: Should have lower R² due to diverse buyer preferences
        assert disagreement_result.analysis_method == "regression"
        assert disagreement_result.regression_r_squared <= consensus_result.regression_r_squared, "Disagreement should have lower R² than consensus"
        assert 0.2 <= disagreement_result.regression_r_squared <= 0.7, f"High disagreement should have moderate R² 0.2-0.7, got {disagreement_result.regression_r_squared}"

        # Should still have some meaningful coefficients despite disagreement
        meaningful_coefficients = [insight for insight in disagreement_result.attribute_insights if insight.marginal_wtp_impact is not None and abs(insight.marginal_wtp_impact) > 0.05]
        assert len(meaningful_coefficients) >= 1, "Even with disagreement, should detect some attribute preferences"

        # Scenario 3: POLARIZED PREFERENCES - Clear segmentation
        polarized_buyers = []
        for i in range(25):
            if i < 12:
                # Segment 1: Methodology lovers
                buyer = BuyerState(
                    buyer_id=f"methodology_lover_{i+1}",
                    regime_beliefs={"tech": [0.6, 0.4]},
                    budget=100.0,
                    conversion_functions={
                        "methodology": {"categorical": {"basic": 0.2, "advanced": 0.6, "cutting_edge": 1.0}},
                        "coverage": {"numeric": {"scale_factor": 0.8}}
                    },
                    attr_weights={"tech": [0.95, 0.05]}  # Extreme preference for methodology
                )
            else:
                # Segment 2: Coverage lovers  
                buyer = BuyerState(
                    buyer_id=f"coverage_lover_{i+1}",
                    regime_beliefs={"tech": [0.6, 0.4]},
                    budget=100.0,
                    conversion_functions={
                        "methodology": {"categorical": {"basic": 0.2, "advanced": 0.6, "cutting_edge": 1.0}},
                        "coverage": {"numeric": {"scale_factor": 0.8}}
                    },
                    attr_weights={"tech": [0.05, 0.95]}  # Extreme preference for coverage
                )
            polarized_buyers.append(buyer)

        polarized_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=polarized_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=attribute_order,
            sector_order=["tech"],
            marketing_attribute_definitions={
                "methodology": {"type": "qualitative", "values": ["basic", "advanced", "cutting_edge"]},
                "coverage": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        polarized_market = MarketModel(id=1, name="Test", agents=[], state=polarized_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        polarized_result = analyze_buyer_preferences_impl(polarized_market, config_data, "tech", effort=4.0)

        # POLARIZED: Should have strong coefficients due to clear segmentation
        assert polarized_result.analysis_method == "regression"
        
        # Should detect strong preferences for both attributes
        strong_coefficients = [insight for insight in polarized_result.attribute_insights if insight.marginal_wtp_impact is not None and abs(insight.marginal_wtp_impact) > 0.3]
        assert len(strong_coefficients) >= 1, "Polarized preferences should produce strong coefficients"

        # R² could be moderate due to bimodal distribution but should still be meaningful
        assert 0.3 <= polarized_result.regression_r_squared <= 0.8, f"Polarized preferences should have meaningful R² 0.3-0.8, got {polarized_result.regression_r_squared}"

    def test_attribute_correlation_effects(self):
        """Test statistical properties under different marketing attribute correlation structures."""
        
        config_data = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 25,
                    "high_effort_num_test_offers": 10,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        # Create buyers with balanced preferences for all scenarios
        buyers = []
        for i in range(30):
            buyer = BuyerState(
                buyer_id=f"buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "attr_a": {"categorical": {"low": 0.2, "medium": 0.5, "high": 0.8}},
                    "attr_b": {"categorical": {"basic": 0.3, "standard": 0.6, "premium": 0.9}}, 
                    "attr_c": {"numeric": {"scale_factor": 1.0}}
                },
                attr_weights={"tech": [0.4, 0.4, 0.2]}  # Balanced weights
            )
            buyers.append(buyer)

        # Scenario 1: INDEPENDENT ATTRIBUTES - No correlation between marketing attributes
        independent_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=["attr_a", "attr_b", "attr_c"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "attr_a": {"type": "qualitative", "values": ["low", "medium", "high"]},
                "attr_b": {"type": "qualitative", "values": ["basic", "standard", "premium"]},
                "attr_c": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        independent_market = MarketModel(id=1, name="Test", agents=[], state=independent_state, step=lambda: None, collect_stats=lambda: {})

        # Use custom test offers to ensure independence
        np.random.seed(42)
        independent_offers = []
        for i in range(10):
            # Randomly assign attributes independently
            offer_attrs = {
                "attr_a": np.random.choice(["low", "medium", "high"]),
                "attr_b": np.random.choice(["basic", "standard", "premium"]),  
                "attr_c": np.random.uniform(0.0, 1.0)
            }
            independent_offers.append({"marketing_attributes": offer_attrs})

        # Mock generate_test_offers for independent scenario
        def mock_independent_offers(sector, num_offers, attribute_order, marketing_definitions):
            from multi_agent_economics.models.market_for_finance import Offer
            offers = []
            for i, offer_data in enumerate(independent_offers[:num_offers]):
                offer = Offer(
                    good_id=f"independent_test_{i}",
                    price=50.0,
                    seller="test_seller",
                    marketing_attributes=offer_data["marketing_attributes"]
                )
                offers.append(offer)
            return offers

        # Temporarily replace generate_test_offers
        import multi_agent_economics.tools.implementations.economic as econ_module
        original_generate = econ_module.generate_test_offers
        econ_module.generate_test_offers = mock_independent_offers

        try:
            independent_result = analyze_buyer_preferences_impl(independent_market, config_data, "tech", effort=4.0)
        finally:
            econ_module.generate_test_offers = original_generate

        # INDEPENDENT ATTRIBUTES: Should have stable regression with good R²
        assert independent_result.analysis_method == "regression"
        assert independent_result.regression_r_squared >= 0.4, f"Independent attributes should have decent R² ≥ 0.4, got {independent_result.regression_r_squared}"

        # All attributes should have meaningful coefficients
        independent_coeffs = [insight.marginal_wtp_impact for insight in independent_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(independent_coeffs) == 3, "Should have coefficients for all 3 independent attributes"
        assert all(abs(coeff) > 0.01 for coeff in independent_coeffs), "All independent attributes should have meaningful coefficients"

        # Scenario 2: CORRELATED ATTRIBUTES - Some attributes naturally correlate
        correlated_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=["attr_a", "attr_b", "attr_c"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "attr_a": {"type": "qualitative", "values": ["low", "medium", "high"]},
                "attr_b": {"type": "qualitative", "values": ["basic", "standard", "premium"]},
                "attr_c": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        correlated_market = MarketModel(id=1, name="Test", agents=[], state=correlated_state, step=lambda: None, collect_stats=lambda: {})

        # Create correlated test offers (attr_b correlates with attr_a)
        correlated_offers = []
        correlation_mapping = {"low": "basic", "medium": "standard", "high": "premium"}
        
        for i in range(10):
            attr_a_val = np.random.choice(["low", "medium", "high"])
            # attr_b correlated with attr_a (70% correlation, 30% random)
            if np.random.random() < 0.7:
                attr_b_val = correlation_mapping[attr_a_val]
            else:
                attr_b_val = np.random.choice(["basic", "standard", "premium"])
            
            offer_attrs = {
                "attr_a": attr_a_val,
                "attr_b": attr_b_val,
                "attr_c": np.random.uniform(0.0, 1.0)  # Independent
            }
            correlated_offers.append({"marketing_attributes": offer_attrs})

        def mock_correlated_offers(sector, num_offers, attribute_order, marketing_definitions):
            from multi_agent_economics.models.market_for_finance import Offer
            offers = []
            for i, offer_data in enumerate(correlated_offers[:num_offers]):
                offer = Offer(
                    good_id=f"correlated_test_{i}",
                    price=50.0,
                    seller="test_seller",
                    marketing_attributes=offer_data["marketing_attributes"]
                )
                offers.append(offer)
            return offers

        econ_module.generate_test_offers = mock_correlated_offers

        try:
            np.random.seed(42)
            correlated_result = analyze_buyer_preferences_impl(correlated_market, config_data, "tech", effort=4.0)
        finally:
            econ_module.generate_test_offers = original_generate

        # CORRELATED ATTRIBUTES: Should still work but potentially lower R² or coefficient instability
        assert correlated_result.analysis_method == "regression"
        
        # R² might be lower due to multicollinearity, but should still be meaningful
        assert correlated_result.regression_r_squared >= 0.25, f"Correlated attributes should still have meaningful R² ≥ 0.25, got {correlated_result.regression_r_squared}"
        
        # Should detect correlation effects
        correlated_coeffs = [insight.marginal_wtp_impact for insight in correlated_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(correlated_coeffs) >= 2, "Should have coefficients for at least 2 correlated attributes"

        # Scenario 3: REDUNDANT ATTRIBUTES - Multiple attributes measure same underlying factor
        redundant_buyers = []
        for i in range(30):
            buyer = BuyerState(
                buyer_id=f"redundant_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "quality_rating": {"categorical": {"poor": 0.1, "good": 0.5, "excellent": 0.9}},
                    "quality_score": {"categorical": {"low": 0.1, "medium": 0.5, "high": 0.9}},  # Redundant with quality_rating
                    "independent_attr": {"numeric": {"scale_factor": 0.8}}
                },
                attr_weights={"tech": [0.45, 0.45, 0.1]}  # Both quality measures get similar weight
            )
            redundant_buyers.append(buyer)

        redundant_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=redundant_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=["quality_rating", "quality_score", "independent_attr"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "quality_rating": {"type": "qualitative", "values": ["poor", "good", "excellent"]},
                "quality_score": {"type": "qualitative", "values": ["low", "medium", "high"]},
                "independent_attr": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        redundant_market = MarketModel(id=1, name="Test", agents=[], state=redundant_state, step=lambda: None, collect_stats=lambda: {})

        # Create perfectly redundant test offers (quality_rating and quality_score always match)
        redundant_offers = []
        redundancy_mapping = {"poor": "low", "good": "medium", "excellent": "high"}
        
        for i in range(10):
            quality_rating = np.random.choice(["poor", "good", "excellent"])
            quality_score = redundancy_mapping[quality_rating]  # Perfect redundancy
            
            offer_attrs = {
                "quality_rating": quality_rating,
                "quality_score": quality_score,
                "independent_attr": np.random.uniform(0.0, 1.0)
            }
            redundant_offers.append({"marketing_attributes": offer_attrs})

        def mock_redundant_offers(sector, num_offers, attribute_order, marketing_definitions):
            from multi_agent_economics.models.market_for_finance import Offer
            offers = []
            for i, offer_data in enumerate(redundant_offers[:num_offers]):
                offer = Offer(
                    good_id=f"redundant_test_{i}",
                    price=50.0,
                    seller="test_seller",
                    marketing_attributes=offer_data["marketing_attributes"]
                )
                offers.append(offer)
            return offers

        econ_module.generate_test_offers = mock_redundant_offers

        try:
            np.random.seed(42)
            redundant_result = analyze_buyer_preferences_impl(redundant_market, config_data, "tech", effort=4.0)
        finally:
            econ_module.generate_test_offers = original_generate

        # REDUNDANT ATTRIBUTES: Should handle multicollinearity gracefully  
        if redundant_result.analysis_method == "regression":
            # If regression succeeds, R² should still be meaningful
            assert redundant_result.regression_r_squared >= 0.2, f"Even with redundancy, should have some R² ≥ 0.2, got {redundant_result.regression_r_squared}"
            
            # May have coefficient instability - some might be near zero due to redundancy
            redundant_coeffs = [insight.marginal_wtp_impact for insight in redundant_result.attribute_insights if insight.marginal_wtp_impact is not None]
            
            # Should still detect the independent attribute clearly
            independent_attr_insight = next((insight for insight in redundant_result.attribute_insights if insight.attribute_name == "independent_attr"), None)
            assert independent_attr_insight is not None, "Should detect independent attribute even with redundant attributes"
            assert abs(independent_attr_insight.marginal_wtp_impact) > 0.01, "Independent attribute should have meaningful coefficient"
        else:
            # If regression fails due to perfect multicollinearity, should fallback gracefully
            assert redundant_result.analysis_method in ["descriptive", "insufficient_data"]
            assert "multicollinearity" in " ".join(redundant_result.warnings).lower() or "singular" in " ".join(redundant_result.warnings).lower(), "Should warn about multicollinearity issues"

    def test_wtp_distribution_properties(self):
        """Test statistical behavior under different WTP distribution characteristics."""
        
        config_data = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 25,
                    "high_effort_num_test_offers": 10,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        attribute_order = ["attr_a", "attr_b"]

        # Scenario 1: NORMAL WTP DISTRIBUTION - Balanced buyer preferences
        normal_buyers = []
        for i in range(30):
            # Create buyers with normally distributed preferences
            weight_a = max(0.1, min(0.9, np.random.normal(0.5, 0.15)))  # Mean=0.5, bounded
            weight_b = 1.0 - weight_a  # Ensures weights sum to 1
            
            buyer = BuyerState(
                buyer_id=f"normal_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "attr_a": {"categorical": {"low": 0.2, "medium": 0.5, "high": 0.8}},
                    "attr_b": {"numeric": {"scale_factor": 1.0}}
                },
                attr_weights={"tech": [weight_a, weight_b]}
            )
            normal_buyers.append(buyer)

        normal_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=normal_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=attribute_order,
            sector_order=["tech"],
            marketing_attribute_definitions={
                "attr_a": {"type": "qualitative", "values": ["low", "medium", "high"]},
                "attr_b": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        normal_market = MarketModel(id=1, name="Test", agents=[], state=normal_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        normal_result = analyze_buyer_preferences_impl(normal_market, config_data, "tech", effort=4.0)

        # NORMAL DISTRIBUTION: Should have good statistical properties
        assert normal_result.analysis_method == "regression"
        assert normal_result.regression_r_squared >= 0.3, f"Normal WTP distribution should have reasonable R² ≥ 0.3, got {normal_result.regression_r_squared}"
        
        # Should have good coefficient estimates  
        normal_coeffs = [insight.marginal_wtp_impact for insight in normal_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(normal_coeffs) == 2, "Should have coefficients for both attributes"
        assert all(abs(coeff) > 0.05 for coeff in normal_coeffs), "Normal distribution should produce meaningful coefficients"

        # Check WTP variance in raw data
        normal_wtp_values = [point.willingness_to_pay for point in normal_result.raw_wtp_data]
        normal_wtp_variance = np.var(normal_wtp_values)
        assert normal_wtp_variance > 0.01, f"Normal distribution should have decent WTP variance > 0.01, got {normal_wtp_variance}"

        # Scenario 2: SKEWED WTP DISTRIBUTION - Most buyers have low preferences, few have high
        skewed_buyers = []
        for i in range(30):
            # Create skewed distribution: 70% low preferences, 20% medium, 10% high
            if i < 21:  # 70% low preference buyers
                weight_a = np.random.uniform(0.1, 0.3)
            elif i < 27:  # 20% medium preference buyers  
                weight_a = np.random.uniform(0.4, 0.6)
            else:  # 10% high preference buyers
                weight_a = np.random.uniform(0.7, 0.9)
            
            weight_b = 1.0 - weight_a
            
            buyer = BuyerState(
                buyer_id=f"skewed_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "attr_a": {"categorical": {"low": 0.2, "medium": 0.5, "high": 0.8}},
                    "attr_b": {"numeric": {"scale_factor": 1.0}}
                },
                attr_weights={"tech": [weight_a, weight_b]}
            )
            skewed_buyers.append(buyer)

        skewed_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=skewed_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=attribute_order,
            sector_order=["tech"],
            marketing_attribute_definitions={
                "attr_a": {"type": "qualitative", "values": ["low", "medium", "high"]},
                "attr_b": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        skewed_market = MarketModel(id=1, name="Test", agents=[], state=skewed_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        skewed_result = analyze_buyer_preferences_impl(skewed_market, config_data, "tech", effort=4.0)

        # SKEWED DISTRIBUTION: Should still work but may have different statistical properties
        assert skewed_result.analysis_method == "regression"
        
        # R² might be lower due to skewness affecting linear regression assumptions
        assert skewed_result.regression_r_squared >= 0.15, f"Skewed WTP distribution should still have some explanatory power ≥ 0.15, got {skewed_result.regression_r_squared}"
        
        # Should still detect meaningful patterns
        skewed_coeffs = [insight.marginal_wtp_impact for insight in skewed_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(skewed_coeffs) >= 1, "Skewed distribution should still detect some attribute preferences"

        # Check that skewness is present in raw data
        skewed_wtp_values = [point.willingness_to_pay for point in skewed_result.raw_wtp_data]
        skewed_wtp_variance = np.var(skewed_wtp_values)
        
        # Skewed distribution should still have variation, but pattern may be different
        assert skewed_wtp_variance > 0.005, f"Skewed distribution should have WTP variance > 0.005, got {skewed_wtp_variance}"

        # Scenario 3: LOW-VARIANCE WTP DISTRIBUTION - All buyers have very similar preferences
        low_variance_buyers = []
        for i in range(30):
            # All buyers have very similar preferences (small variation around 0.5)
            weight_a = max(0.3, min(0.7, np.random.normal(0.5, 0.03)))  # Very low std dev
            weight_b = 1.0 - weight_a
            
            buyer = BuyerState(
                buyer_id=f"low_var_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "attr_a": {"categorical": {"low": 0.45, "medium": 0.50, "high": 0.55}},  # Small differences
                    "attr_b": {"numeric": {"scale_factor": 0.95}}  # Similar scaling
                },
                attr_weights={"tech": [weight_a, weight_b]}
            )
            low_variance_buyers.append(buyer)

        low_variance_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=low_variance_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=attribute_order,
            sector_order=["tech"],
            marketing_attribute_definitions={
                "attr_a": {"type": "qualitative", "values": ["low", "medium", "high"]},
                "attr_b": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        low_variance_market = MarketModel(id=1, name="Test", agents=[], state=low_variance_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        low_variance_result = analyze_buyer_preferences_impl(low_variance_market, config_data, "tech", effort=4.0)

        # LOW-VARIANCE DISTRIBUTION: Should trigger insufficient variation warning or have very low R²
        low_var_wtp_values = [point.willingness_to_pay for point in low_variance_result.raw_wtp_data]
        low_var_wtp_variance = np.var(low_var_wtp_values)
        
        # Check if variance is indeed low
        assert low_var_wtp_variance < normal_wtp_variance, "Low variance scenario should have lower WTP variance than normal scenario"
        
        if low_var_wtp_variance <= 1e-10:
            # If variance is extremely low, should trigger insufficient data analysis
            assert low_variance_result.analysis_method == "insufficient_data"
            assert "variation" in " ".join(low_variance_result.warnings).lower(), "Should warn about insufficient variation"
        else:
            # If some variance exists, regression may proceed but with very low R²
            if low_variance_result.analysis_method == "regression":
                # R² should be very low due to lack of variation
                assert low_variance_result.regression_r_squared <= 0.3, f"Low variance should produce low R² ≤ 0.3, got {low_variance_result.regression_r_squared}"
                
                # Coefficients may be unstable due to low signal-to-noise ratio
                low_var_coeffs = [insight.marginal_wtp_impact for insight in low_variance_result.attribute_insights if insight.marginal_wtp_impact is not None]
                
                # May have lower confidence levels due to low variation
                low_confidence_count = sum(1 for insight in low_variance_result.attribute_insights if insight.confidence_level == "low")
                assert low_confidence_count >= 1, "Low variance should produce at least some low-confidence attributes"

        # Comparative analysis: Normal distribution should outperform others
        assert normal_result.regression_r_squared >= skewed_result.regression_r_squared, "Normal distribution should have higher or equal R² than skewed"
        
        if low_variance_result.analysis_method == "regression":
            assert normal_result.regression_r_squared >= low_variance_result.regression_r_squared, "Normal distribution should have higher R² than low variance"

        # Check that different distributions produce different coefficient magnitudes
        normal_max_coeff = max(abs(coeff) for coeff in normal_coeffs)
        skewed_max_coeff = max(abs(coeff) for coeff in skewed_coeffs) if skewed_coeffs else 0
        
        # Normal distribution should produce more stable, meaningful coefficients
        assert normal_max_coeff > 0.1, "Normal distribution should produce substantial coefficients"

    def test_sample_size_statistical_power(self):
        """Test statistical power and confidence transitions at different sample sizes."""
        
        # Create large pool of buyers with consistent preferences for statistical power testing
        buyers = []
        for i in range(50):  # Large pool to sample from
            buyer = BuyerState(
                buyer_id=f"power_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "methodology": {"categorical": {"basic": 0.2, "advanced": 0.6, "cutting_edge": 1.0}},
                    "coverage": {"numeric": {"scale_factor": 0.8}}
                },
                attr_weights={"tech": [0.65, 0.35]}  # Clear preference for methodology
            )
            buyers.append(buyer)

        state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=["methodology", "coverage"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "methodology": {"type": "qualitative", "values": ["basic", "advanced", "cutting_edge"]},
                "coverage": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        market_model = MarketModel(id=1, name="Test", agents=[], state=state, step=lambda: None, collect_stats=lambda: {})

        # Scenario 1: VERY SMALL SAMPLE (10 observations) - Should use descriptive analysis
        small_config = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 2,  # 2 buyers × 5 offers = 10 observations
                    "high_effort_num_test_offers": 5,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        np.random.seed(42)
        small_result = analyze_buyer_preferences_impl(market_model, small_config, "tech", effort=4.0)
        
        # VERY SMALL SAMPLE: Should fallback to descriptive analysis
        assert small_result.total_observations == 10, f"Should have 10 observations (2×5), got {small_result.total_observations}"
        assert small_result.analysis_method == "descriptive", "10 observations should trigger descriptive analysis (< 30 threshold)"
        assert small_result.regression_r_squared is None, "Descriptive analysis should not have R²"
        
        # All attributes should have low confidence due to small sample
        confidence_levels = [insight.confidence_level for insight in small_result.attribute_insights]
        assert all(conf == "low" for conf in confidence_levels), "Small sample should produce low confidence for all attributes"

        # Scenario 2: REGRESSION THRESHOLD (30 observations) - Should just trigger regression
        threshold_config = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 5,  # 5 buyers × 6 offers = 30 observations (exactly at threshold)
                    "high_effort_num_test_offers": 6,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        np.random.seed(42)
        threshold_result = analyze_buyer_preferences_impl(market_model, threshold_config, "tech", effort=4.0)
        
        # REGRESSION THRESHOLD: Should trigger regression analysis
        assert threshold_result.total_observations == 30, f"Should have 30 observations (5×6), got {threshold_result.total_observations}"
        assert threshold_result.analysis_method == "regression", "30 observations should trigger regression analysis"
        assert threshold_result.regression_r_squared is not None, "Regression analysis should have R²"
        
        # With moderate sample size, should have medium confidence levels
        medium_or_high_count = sum(1 for insight in threshold_result.attribute_insights if insight.confidence_level in ["medium", "high"])
        assert medium_or_high_count >= 1, "30 observations should produce at least one medium/high confidence attribute"

        # Scenario 3: MODERATE SAMPLE (100 observations) - Should have good statistical power
        moderate_config = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 10,  # 10 buyers × 10 offers = 100 observations
                    "high_effort_num_test_offers": 10,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        np.random.seed(42)
        moderate_result = analyze_buyer_preferences_impl(market_model, moderate_config, "tech", effort=4.0)
        
        # MODERATE SAMPLE: Should have improved statistical properties
        assert moderate_result.total_observations == 100, f"Should have 100 observations (10×10), got {moderate_result.total_observations}"
        assert moderate_result.analysis_method == "regression", "100 observations should use regression"
        
        # Should have better R² than threshold case due to increased sample size
        assert moderate_result.regression_r_squared >= threshold_result.regression_r_squared, "100 observations should have R² ≥ 30 observations case"
        
        # Should have higher confidence levels
        high_confidence_count = sum(1 for insight in moderate_result.attribute_insights if insight.confidence_level == "high")
        threshold_high_count = sum(1 for insight in threshold_result.attribute_insights if insight.confidence_level == "high")
        assert high_confidence_count >= threshold_high_count, "100 observations should have ≥ high confidence attributes than 30 observations"

        # Scenario 4: LARGE SAMPLE (400 observations) - Should have high statistical power
        large_config = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 20,  # 20 buyers × 20 offers = 400 observations
                    "high_effort_num_test_offers": 20,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        np.random.seed(42)
        large_result = analyze_buyer_preferences_impl(market_model, large_config, "tech", effort=4.0)
        
        # LARGE SAMPLE: Should have excellent statistical properties
        assert large_result.total_observations == 400, f"Should have 400 observations (20×20), got {large_result.total_observations}"
        assert large_result.analysis_method == "regression", "400 observations should use regression"
        
        # Should have the best R² due to large sample size
        assert large_result.regression_r_squared >= moderate_result.regression_r_squared, "400 observations should have R² ≥ 100 observations case"
        
        # Should have mostly high confidence attributes
        large_high_confidence = sum(1 for insight in large_result.attribute_insights if insight.confidence_level == "high")
        assert large_high_confidence >= 1, "400 observations should produce at least one high confidence attribute"
        
        # All coefficients should be statistically meaningful with large sample
        large_coeffs = [insight.marginal_wtp_impact for insight in large_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(large_coeffs) == 2, "Should detect both attributes with 400 observations"
        assert all(abs(coeff) > 0.01 for coeff in large_coeffs), "Large sample should produce meaningful coefficients"

        # Power analysis: Check monotonic improvement with sample size
        r_squared_progression = [
            threshold_result.regression_r_squared,
            moderate_result.regression_r_squared, 
            large_result.regression_r_squared
        ]
        
        # R² should generally improve (or at least not decrease significantly) with larger samples
        assert r_squared_progression[1] >= r_squared_progression[0] - 0.05, "R² should not decrease significantly from 30 to 100 observations"
        assert r_squared_progression[2] >= r_squared_progression[1] - 0.05, "R² should not decrease significantly from 100 to 400 observations"

        # Confidence level progression: Larger samples should have more high-confidence results
        confidence_progression = [
            sum(1 for insight in threshold_result.attribute_insights if insight.confidence_level == "high"),
            sum(1 for insight in moderate_result.attribute_insights if insight.confidence_level == "high"),
            sum(1 for insight in large_result.attribute_insights if insight.confidence_level == "high")
        ]
        
        # Generally expect more high-confidence attributes with larger samples
        assert confidence_progression[2] >= confidence_progression[0], "400 observations should have ≥ high confidence attributes than 30 observations"

        # Test edge case: Just below regression threshold (29 observations)
        below_threshold_config = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 29,  # 29 buyers × 1 offer = 29 observations (just below 30)
                    "high_effort_num_test_offers": 1,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        np.random.seed(42)
        below_threshold_result = analyze_buyer_preferences_impl(market_model, below_threshold_config, "tech", effort=4.0)
        
        # JUST BELOW THRESHOLD: Should use descriptive analysis
        assert below_threshold_result.total_observations == 29, f"Should have 29 observations, got {below_threshold_result.total_observations}"
        assert below_threshold_result.analysis_method == "descriptive", "29 observations should trigger descriptive analysis (< 30 threshold)"
        assert below_threshold_result.regression_r_squared is None, "Descriptive analysis should not have R²"

        # Confidence transitions around sample size thresholds
        # Small sample (10): All low confidence
        # Threshold sample (30): Some medium/high confidence  
        # Large sample (400): Mostly high confidence
        
        small_low_confidence = sum(1 for insight in small_result.attribute_insights if insight.confidence_level == "low")
        large_low_confidence = sum(1 for insight in large_result.attribute_insights if insight.confidence_level == "low")
        
        assert small_low_confidence >= large_low_confidence, "Small sample should have more low-confidence attributes than large sample"

    def test_marketing_attribute_type_effects(self):
        """Test statistical properties with different marketing attribute type combinations."""
        
        config_data = {
            "tool_parameters": {
                "analyze_buyer_preferences": {
                    "effort_thresholds": {"high": 3.0, "medium": 1.5},
                    "high_effort_num_buyers": 20,
                    "high_effort_num_test_offers": 8,
                    "high_effort_analyze_by_attribute": True
                }
            }
        }

        # Scenario 1: ALL QUALITATIVE ATTRIBUTES - Only categorical variables
        qualitative_buyers = []
        for i in range(25):
            buyer = BuyerState(
                buyer_id=f"qual_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "methodology": {"categorical": {"basic": 0.2, "advanced": 0.6, "cutting_edge": 1.0}},
                    "support_level": {"categorical": {"minimal": 0.1, "standard": 0.5, "premium": 0.9}},
                    "brand_reputation": {"categorical": {"unknown": 0.2, "known": 0.6, "prestigious": 1.0}}
                },
                attr_weights={"tech": [0.5, 0.3, 0.2]}
            )
            qualitative_buyers.append(buyer)

        qualitative_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=qualitative_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=["methodology", "support_level", "brand_reputation"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "methodology": {"type": "qualitative", "values": ["basic", "advanced", "cutting_edge"]},
                "support_level": {"type": "qualitative", "values": ["minimal", "standard", "premium"]},
                "brand_reputation": {"type": "qualitative", "values": ["unknown", "known", "prestigious"]}
            }
        )
        qualitative_market = MarketModel(id=1, name="Test", agents=[], state=qualitative_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        qualitative_result = analyze_buyer_preferences_impl(qualitative_market, config_data, "tech", effort=4.0)

        # ALL QUALITATIVE: Should work well with categorical conversion
        assert qualitative_result.analysis_method == "regression"
        assert qualitative_result.total_observations == 160  # 20 buyers × 8 offers
        assert qualitative_result.regression_r_squared >= 0.25, f"Qualitative attributes should have meaningful R² ≥ 0.25, got {qualitative_result.regression_r_squared}"
        
        # All attributes should be detected
        qualitative_coeffs = [insight.marginal_wtp_impact for insight in qualitative_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(qualitative_coeffs) == 3, "Should have coefficients for all 3 qualitative attributes"
        
        # Coefficients should reflect categorical conversion patterns (discrete jumps)
        assert all(abs(coeff) > 0.01 for coeff in qualitative_coeffs), "Qualitative attributes should produce meaningful coefficients"

        # Scenario 2: ALL NUMERIC ATTRIBUTES - Only continuous variables  
        numeric_buyers = []
        for i in range(25):
            buyer = BuyerState(
                buyer_id=f"num_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "performance_score": {"numeric": {"scale_factor": 1.0}},
                    "efficiency_rating": {"numeric": {"scale_factor": 0.8}},
                    "innovation_index": {"numeric": {"scale_factor": 1.2}}
                },
                attr_weights={"tech": [0.4, 0.35, 0.25]}
            )
            numeric_buyers.append(buyer)

        numeric_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=numeric_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
            attribute_order=["performance_score", "efficiency_rating", "innovation_index"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "performance_score": {"type": "numeric", "range": [0.0, 1.0]},
                "efficiency_rating": {"type": "numeric", "range": [0.0, 1.0]},
                "innovation_index": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        numeric_market = MarketModel(id=1, name="Test", agents=[], state=numeric_state, step=lambda: None, collect_stats=lambda: {})

        np.random.seed(42)
        numeric_result = analyze_buyer_preferences_impl(numeric_market, config_data, "tech", effort=4.0)

        # ALL NUMERIC: Should work well with linear scaling
        assert numeric_result.analysis_method == "regression"
        assert numeric_result.total_observations == 160  # 20 buyers × 8 offers
        assert numeric_result.regression_r_squared >= 0.25, f"Numeric attributes should have meaningful R² ≥ 0.25, got {numeric_result.regression_r_squared}"
        
        # All numeric attributes should be detected
        numeric_coeffs = [insight.marginal_wtp_impact for insight in numeric_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(numeric_coeffs) == 3, "Should have coefficients for all 3 numeric attributes"
        
        # Numeric coefficients should reflect linear scaling (smoother relationship)
        assert all(abs(coeff) > 0.01 for coeff in numeric_coeffs), "Numeric attributes should produce meaningful coefficients"

        # Scenario 3: MIXED ATTRIBUTE TYPES - Combination of categorical and numeric
        mixed_buyers = []
        for i in range(25):
            buyer = BuyerState(
                buyer_id=f"mixed_buyer_{i+1}",
                regime_beliefs={"tech": [0.6, 0.4]},
                budget=100.0,
                conversion_functions={
                    "methodology": {"categorical": {"basic": 0.2, "advanced": 0.6, "cutting_edge": 1.0}},  # Qualitative
                    "performance_score": {"numeric": {"scale_factor": 1.0}},  # Numeric
                    "support_level": {"categorical": {"minimal": 0.1, "standard": 0.5, "premium": 0.9}},  # Qualitative
                    "efficiency_rating": {"numeric": {"scale_factor": 0.8}}  # Numeric
                },
                attr_weights={"tech": [0.3, 0.3, 0.2, 0.2]}
            )
            mixed_buyers.append(buyer)

        mixed_state = MarketState(
            offers=[],
            trades=[],
            index_values={"tech": 100.0},
            buyers_state=mixed_buyers,
            current_regimes={"tech": 0},
            regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sbigma=0.15)}},
            attribute_order=["methodology", "performance_score", "support_level", "efficiency_rating"],
            sector_order=["tech"],
            marketing_attribute_definitions={
                "methodology": {"type": "qualitative", "values": ["basic", "advanced", "cutting_edge"]},
                "performance_score": {"type": "numeric", "range": [0.0, 1.0]},
                "support_level": {"type": "qualitative", "values": ["minimal", "standard", "premium"]},
                "efficiency_rating": {"type": "numeric", "range": [0.0, 1.0]}
            }
        )
        mixed_market = MarketModel(id=1, name="Test", agents=[], state=mixed_state, step=lambda: None, collect_stats=lambda: {})
    
        np.random.seed(42)
        mixed_result = analyze_buyer_preferences_impl(mixed_market, config_data, "tech", effort=4.0)

        # MIXED TYPES: Should handle both categorical and numeric conversions
        assert mixed_result.analysis_method == "regression"
        assert mixed_result.total_observations == 160  # 20 buyers × 8 offers
        assert mixed_result.regression_r_squared >= 0.25, f"Mixed attributes should have meaningful R² ≥ 0.25, got {mixed_result.regression_r_squared}"
        
        # All mixed attributes should be detected
        mixed_coeffs = [insight.marginal_wtp_impact for insight in mixed_result.attribute_insights if insight.marginal_wtp_impact is not None]
        assert len(mixed_coeffs) == 4, "Should have coefficients for all 4 mixed attributes"
        
        # Should handle different conversion types appropriately
        assert all(abs(coeff) > 0.005 for coeff in mixed_coeffs), "Mixed attributes should produce meaningful coefficients"

        # Check individual attribute types within mixed scenario
        methodology_insight = next((insight for insight in mixed_result.attribute_insights if insight.attribute_name == "methodology"), None)
        performance_insight = next((insight for insight in mixed_result.attribute_insights if insight.attribute_name == "performance_score"), None)
        
        assert methodology_insight is not None, "Should detect qualitative methodology attribute"
        assert performance_insight is not None, "Should detect numeric performance_score attribute"
        assert abs(methodology_insight.marginal_wtp_impact) > 0.005, "Qualitative methodology should have meaningful coefficient"
        assert abs(performance_insight.marginal_wtp_impact) > 0.005, "Numeric performance should have meaningful coefficient"

        # Comparative analysis: Different attribute type combinations
        r_squared_comparison = [
            ("qualitative", qualitative_result.regression_r_squared),
            ("numeric", numeric_result.regression_r_squared), 
            ("mixed", mixed_result.regression_r_squared)
        ]
        
        # All approaches should produce meaningful R² values
        for attr_type, r_squared in r_squared_comparison:
            assert r_squared >= 0.20, f"{attr_type} attributes should have R² ≥ 0.20, got {r_squared}"

        # Check coefficient magnitudes across different attribute types
        all_qualitative_magnitudes = [abs(coeff) for coeff in qualitative_coeffs]
        all_numeric_magnitudes = [abs(coeff) for coeff in numeric_coeffs]
        all_mixed_magnitudes = [abs(coeff) for coeff in mixed_coeffs]
        
        # All attribute types should produce substantial coefficients
        assert max(all_qualitative_magnitudes) > 0.05, "Qualitative attributes should have at least one substantial coefficient"
        assert max(all_numeric_magnitudes) > 0.05, "Numeric attributes should have at least one substantial coefficient"
        assert max(all_mixed_magnitudes) > 0.05, "Mixed attributes should have at least one substantial coefficient"

        # Confidence levels should be reasonable across all attribute types
        for result, result_name in [(qualitative_result, "qualitative"), (numeric_result, "numeric"), (mixed_result, "mixed")]:
            medium_or_high_count = sum(1 for insight in result.attribute_insights if insight.confidence_level in ["medium", "high"])
            assert medium_or_high_count >= 1, f"{result_name} attributes should produce at least one medium/high confidence result"

        # Test edge case: Single attribute type scenarios with different conversion patterns
        # Should handle both lookup-based (categorical) and scaling-based (numeric) conversions properly
        qualitative_warnings = qualitative_result.warnings
        numeric_warnings = numeric_result.warnings
        mixed_warnings = mixed_result.warnings
        
        # None of the scenarios should produce serious warnings with adequate sample sizes
        serious_warnings = ["insufficient", "failed", "error", "singular"]
        for warnings, scenario in [(qualitative_warnings, "qualitative"), (numeric_warnings, "numeric"), (mixed_warnings, "mixed")]:
            warning_text = " ".join(warnings).lower()
            serious_warning_detected = any(serious in warning_text for serious in serious_warnings)
            assert not serious_warning_detected, f"{scenario} scenario should not produce serious warnings: {warnings}"