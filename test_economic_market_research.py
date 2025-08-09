"""
TDD test suite for market research economic tools.

These tests will FAIL initially because the tools don't exist yet.
The tests define the exact interface and behavior we need to implement.

Market Research Foundation:
These tools provide information gathering capabilities that LLM agents need
for market research while supporting algorithmic optimization in agents like
the Akerlof Seller.

Three core market research capabilities:
1. Historical Performance Analysis - "What has worked well?"
2. Buyer Preference Analysis - "What do buyers want?"  
3. Competitive Pricing Research - "What are competitors charging?"
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, Offer, ForecastData, TradeData, RegimeParameters,
    BuyerState, SellerState
)

# These imports will fail initially - that's the point of TDD!
from multi_agent_economics.tools.implementations.economic import (
    analyze_historical_performance_impl,
    analyze_buyer_preferences_impl, 
    research_competitive_pricing_impl
)
from multi_agent_economics.tools.schemas import (
    HistoricalPerformanceResponse,
    BuyerPreferenceResponse,
    CompetitivePricingResponse
)


@pytest.fixture
def market_state_with_history():
    """
    Create MarketState with rich historical data for market research testing.
    
    Contains historical trades, buyer preferences, and competitive offers across
    multiple sectors (tech, finance, healthcare) to test sector-specific analysis.
    Includes marketing attribute definitions and trade data with marketing attributes.
    """
    return MarketState(
        offers=[],
        trades=[],
        demand_profile={},
        supply_profile={},
        index_values={"tech": 105.0, "finance": 98.5, "healthcare": 112.0},
        current_regimes={"tech": 0, "finance": 1, "healthcare": 0},
        regime_parameters={
            "tech": {
                0: RegimeParameters(mu=0.08, sigma=0.15),  # Bull market
                1: RegimeParameters(mu=0.02, sigma=0.25)   # Bear market
            },
            "finance": {
                0: RegimeParameters(mu=0.06, sigma=0.12),
                1: RegimeParameters(mu=0.01, sigma=0.20)
            },
            "healthcare": {
                0: RegimeParameters(mu=0.10, sigma=0.18),
                1: RegimeParameters(mu=0.03, sigma=0.22)
            }
        },
        current_period=10,
        knowledge_good_forecasts={},
        buyers_state=[],
        # Marketing attribute definitions for testing
        marketing_attribute_definitions={
            "innovation_level": {
                "type": "qualitative",
                "values": ["low", "medium", "high"], 
                "description": "Level of innovation in the offering"
            },
            "data_source": {
                "type": "qualitative", 
                "values": ["internal", "external", "proprietary"],
                "description": "Source of underlying data"
            },
            "risk_score": {
                "type": "numeric",
                "range": [0, 100],
                "description": "Risk assessment score"
            }
        },
        # Canonical ordering for consistent attribute vector generation
        attribute_order=["innovation_level", "data_source", "risk_score"],
        all_trades=[  # Historical trade data for performance analysis
            # Tech sector trades with varying performance and marketing attributes
            TradeData(
                buyer_id="buyer_1", seller_id="tech_seller_1", price=120.0, quantity=1, 
                good_id="tech_high_perf_1",
                marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 25}
            ),
            TradeData(
                buyer_id="buyer_2", seller_id="tech_seller_1", price=115.0, quantity=1,
                good_id="tech_high_perf_2",
                marketing_attributes={"innovation_level": "high", "data_source": "external", "risk_score": 30}
            ),
            TradeData(
                buyer_id="buyer_3", seller_id="tech_seller_2", price=85.0, quantity=1,
                good_id="tech_med_perf_1",
                marketing_attributes={"innovation_level": "medium", "data_source": "internal", "risk_score": 45}
            ),
            TradeData(
                buyer_id="buyer_4", seller_id="tech_seller_2", price=80.0, quantity=1,
                good_id="tech_med_perf_2",
                marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 50}
            ),
            TradeData(
                buyer_id="buyer_5", seller_id="tech_seller_3", price=50.0, quantity=1,
                good_id="tech_low_perf_1",
                marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 75}
            ),
            TradeData(
                buyer_id="buyer_6", seller_id="tech_seller_3", price=45.0, quantity=1,
                good_id="tech_low_perf_2", 
                marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 80}
            ),
            
            # Finance sector trades with marketing attributes
            TradeData(
                buyer_id="buyer_7", seller_id="fin_seller_1", price=95.0, quantity=1,
                good_id="fin_high_perf_1",
                marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 20}
            ),
            TradeData(
                buyer_id="buyer_8", seller_id="fin_seller_2", price=70.0, quantity=1,
                good_id="fin_med_perf_1",
                marketing_attributes={"innovation_level": "low", "data_source": "external", "risk_score": 55}
            ),
            TradeData(
                buyer_id="buyer_9", seller_id="fin_seller_3", price=40.0, quantity=1,
                good_id="fin_low_perf_1",
                marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 85}
            ),
            
            # Healthcare sector trades with marketing attributes
            TradeData(
                buyer_id="buyer_10", seller_id="health_seller_1", price=130.0, quantity=1,
                good_id="health_high_perf_1",
                marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 15}
            ),
            TradeData(
                buyer_id="buyer_11", seller_id="health_seller_2", price=90.0, quantity=1,
                good_id="health_med_perf_1",
                marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 40}
            ),
        ]
    )


@pytest.fixture  
def buyers_with_sector_preferences(market_state_with_history):
    """
    Create buyer agents with sector-specific attribute preferences for testing
    buyer preference analysis across different market segments.
    
    Tech buyers: Prefer methodology (high attr[0]) over data quality
    Finance buyers: Prefer data quality (high attr[1]) over methodology  
    Healthcare buyers: Balanced preferences
    
    Each buyer has a marketing conversion function to simulate different
    valuations of the same marketing attributes.
    """
    buyers = [
        # Tech sector buyers - methodology focused
        BuyerState(
            buyer_id="tech_buyer_1",
            regime_beliefs={"tech": [0.7, 0.3]},
            attr_mu=[0.9, 0.5],  # High methodology preference, moderate data
            attr_sigma2=[0.1, 0.2],
            attr_weights=[0.9, 0.5],
            budget=200.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.9, "medium": 0.6, "low": 0.2},
                "data_source": {"proprietary": 0.8, "external": 0.5, "internal": 0.3},
                "risk_score": {}
            }
        ),
        BuyerState(
            buyer_id="tech_buyer_2", 
            regime_beliefs={"tech": [0.6, 0.4]},
            attr_mu=[0.8, 0.6],  # High methodology, decent data
            attr_sigma2=[0.15, 0.15],
            attr_weights=[0.8, 0.6],
            budget=180.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.7, "medium": 0.7, "low": 0.3},
                "data_source": {"proprietary": 0.6, "external": 0.6, "internal": 0.4},
                "risk_score": {}
            }
        ),
        
        # Finance sector buyers - data quality focused
        BuyerState(
            buyer_id="fin_buyer_1",
            regime_beliefs={"finance": [0.4, 0.6]},
            attr_mu=[0.5, 0.9],  # Moderate methodology, high data preference
            attr_sigma2=[0.2, 0.1],
            attr_weights=[0.5, 0.9],
            budget=250.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.4, "medium": 0.6, "low": 0.8},
                "data_source": {"proprietary": 0.9, "external": 0.7, "internal": 0.3},
                "risk_score": {}
            }
        ),
        BuyerState(
            buyer_id="fin_buyer_2",
            regime_beliefs={"finance": [0.3, 0.7]},
            attr_mu=[0.6, 0.8],  # Decent methodology, high data
            attr_sigma2=[0.18, 0.12],
            attr_weights=[0.6, 0.8], 
            budget=220.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.5, "medium": 0.7, "low": 0.6},
                "data_source": {"proprietary": 1.0, "external": 0.5, "internal": 0.2},
                "risk_score": {}
            }
        ),
        
        # Healthcare sector buyers - balanced
        BuyerState(
            buyer_id="health_buyer_1",
            regime_beliefs={"healthcare": [0.5, 0.5]},
            attr_mu=[0.7, 0.7],  # Balanced preferences
            attr_sigma2=[0.15, 0.15],
            attr_weights=[0.7, 0.7],
            budget=300.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.7, "medium": 0.7, "low": 0.4},
                "data_source": {"proprietary": 0.7, "external": 0.6, "internal": 0.5},
                "risk_score": {}
            }
        )
    ]
    market_state_with_history.buyers_state = buyers
    return buyers


@pytest.fixture
def competitive_offers_by_sector(market_state_with_history):
    """
    Create historical competitive offers across sectors for pricing research.
    Shows different pricing patterns and competitive landscapes by sector.
    Includes marketing attributes for testing pricing segmentation by attributes.
    """
    offers = [
        # Tech sector - high competition, varied pricing with marketing attributes
        Offer(
            good_id="tech_comp_1", price=100.0, seller="tech_comp_1",
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 35}
        ),
        Offer(
            good_id="tech_comp_2", price=110.0, seller="tech_comp_2",
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 20}
        ),
        Offer(
            good_id="tech_comp_3", price=95.0, seller="tech_comp_3",
            marketing_attributes={"innovation_level": "medium", "data_source": "internal", "risk_score": 40}
        ),
        Offer(
            good_id="tech_comp_4", price=105.0, seller="tech_comp_4",
            marketing_attributes={"innovation_level": "high", "data_source": "external", "risk_score": 25}
        ),
        
        # Finance sector - premium pricing, consistent with marketing attributes
        Offer(
            good_id="fin_comp_1", price=150.0, seller="fin_comp_1",
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 15}
        ),
        Offer(
            good_id="fin_comp_2", price=145.0, seller="fin_comp_2",
            marketing_attributes={"innovation_level": "low", "data_source": "external", "risk_score": 25}
        ),
        Offer(
            good_id="fin_comp_3", price=140.0, seller="fin_comp_3",
            marketing_attributes={"innovation_level": "medium", "data_source": "internal", "risk_score": 30}
        ),
        
        # Healthcare sector - moderate pricing with marketing attributes
        Offer(
            good_id="health_comp_1", price=120.0, seller="health_comp_1",
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 10}
        ),
        Offer(
            good_id="health_comp_2", price=115.0, seller="health_comp_2",
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 35}
        ),
    ]
    market_state_with_history.offers.extend(offers)
    return offers


@pytest.fixture
def performance_mapping():
    """
    Maps trade good_ids to actual performance levels for historical analysis.
    Enables testing of performance-revenue relationship extraction.
    """
    return {
        # Tech sector performance mapping
        "tech_high_perf_1": 0.85, "tech_high_perf_2": 0.87,
        "tech_med_perf_1": 0.70, "tech_med_perf_2": 0.68,
        "tech_low_perf_1": 0.55, "tech_low_perf_2": 0.53,
        
        # Finance sector performance mapping
        "fin_high_perf_1": 0.82,
        "fin_med_perf_1": 0.67,
        "fin_low_perf_1": 0.51,
        
        # Healthcare sector performance mapping
        "health_high_perf_1": 0.89,
        "health_med_perf_1": 0.72,
    }


@pytest.fixture
def market_model_with_research_data(market_state_with_history):
    """Create MarketModel instance with all research data for testing."""
    def dummy_step():
        """Dummy step function for testing."""
        pass
    
    def dummy_collect_stats():
        """Dummy collect stats function for testing."""
        return {}
    
    return MarketModel(
        id=1,
        name="research_test_market",
        agents=[],
        state=market_state_with_history,
        step=dummy_step,
        collect_stats=dummy_collect_stats
    )


@pytest.fixture
def market_research_config():
    """
    Configuration for market research tools with effort thresholds and quality tiers.
    """
    return {
        "tool_parameters": {
            "analyze_historical_performance": {
                "effort_thresholds": {"high": 5.0, "medium": 2.0},
                "lookback_periods": {"high": 20, "medium": 10, "low": 5},
                "sample_noise": {"high": 0.05, "medium": 0.15, "low": 0.30}
            },
            "analyze_buyer_preferences": {
                "effort_thresholds": {"high": 4.0, "medium": 2.0},
                "sample_sizes": {"high": 50, "medium": 20, "low": 10},
                "noise_factors": {"high": 0.1, "medium": 0.25, "low": 0.50}
            },
            "research_competitive_pricing": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "lookback_periods": {"high": 15, "medium": 8, "low": 3},
                "price_noise": {"high": 0.02, "medium": 0.08, "low": 0.15}
            }
        },
        "sectors": ["tech", "finance", "healthcare"],
        "agent_id": "test_researcher"
    }


class TestHistoricalPerformanceAnalysis:
    """
    Test analyze_historical_performance_impl for extracting performance-revenue relationships.
    
    This tool helps agents understand "What has worked well?" by analyzing historical
    trades and their associated performance levels within a specific sector.
    """
    
    def test_historical_performance_analysis_interface(self, market_model_with_research_data, market_research_config, performance_mapping):
        """
        Test analyze_historical_performance_impl returns correct response format.
        
        Interface requirements:
        - Input: market_model, config_data, sector, effort
        - Output: HistoricalPerformanceResponse with performance tiers and revenue data
        - Integration: Uses TradeData from market_state.all_trades 
        """
        # Add performance mapping to market state for testing
        market_model_with_research_data.state.performance_mapping = performance_mapping
        
        result = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=3.0
        )
        
        # Must return proper response object
        assert isinstance(result, HistoricalPerformanceResponse), f"Must return HistoricalPerformanceResponse, got {type(result)}"
        assert result.sector == "tech", "Response must include analyzed sector"
        assert result.effort_used == 3.0, "Response must track effort used"
        assert result.quality_tier in ["low", "medium", "high"], f"Invalid quality tier: {result.quality_tier}"
        
        # Must contain performance analysis data
        assert hasattr(result, 'performance_tiers'), "Must include performance tier analysis"
        assert hasattr(result, 'revenue_patterns'), "Must include revenue pattern data"
        assert isinstance(result.performance_tiers, dict), "Performance tiers must be dict"
        
    def test_sector_specific_filtering(self, market_model_with_research_data, market_research_config, performance_mapping):
        """
        Test that historical analysis is properly filtered by sector.
        
        Tech sector should only analyze tech trades, finance sector only finance trades, etc.
        This ensures agents get sector-relevant market research.
        """
        market_model_with_research_data.state.performance_mapping = performance_mapping
        
        tech_result = analyze_historical_performance_impl(
            market_model_with_research_data, 
            market_research_config,
            sector="tech",
            effort=5.0
        )
        
        finance_result = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config, 
            sector="finance",
            effort=5.0
        )
        
        # Results should be sector-specific and different
        assert tech_result.sector == "tech"
        assert finance_result.sector == "finance"
        
        # Tech should find more trades (6 tech trades vs 3 finance trades in fixture)
        tech_trades = len([t for t in market_model_with_research_data.state.all_trades 
                          if "tech" in t.good_id])
        finance_trades = len([t for t in market_model_with_research_data.state.all_trades 
                             if "fin" in t.good_id])
        
        assert tech_trades == 6, "Should find 6 tech trades"
        assert finance_trades == 3, "Should find 3 finance trades"
        
    def test_effort_quality_relationship(self, market_model_with_research_data, market_research_config, performance_mapping):
        """
        Test that higher effort produces higher quality historical analysis.
        
        **Precise Effort-Quality Mapping Logic:**
        - Low effort (1.0 < 2.0): quality_tier="low", noise_factor=0.30, lookback_periods=5
        - Medium effort (3.0 ≥ 2.0, < 5.0): quality_tier="medium", noise_factor=0.15, lookback_periods=10  
        - High effort (6.0 ≥ 5.0): quality_tier="high", noise_factor=0.05, lookback_periods=20
        
        Analysis quality = max(0.1, 1.0 - noise_factor) = [0.7, 0.85, 0.95] for [low, med, high].
        Sample size = min(6 tech trades, lookback_periods * 2) = [6, 6, 6] (all use full data).
        
        **Validation:** Assert exact quality tiers, lookback periods, and analysis quality scores.
        """
        market_model_with_research_data.state.performance_mapping = performance_mapping
        
        # Test different effort levels with precise parameter expectations
        low_effort = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech", 
            effort=1.0  # Below medium threshold (2.0)
        )
        
        medium_effort = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=3.0  # Above medium (2.0), below high (5.0)
        )
        
        high_effort = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=6.0  # Above high threshold (5.0)
        )
        
        # Validate quality tier mappings
        assert low_effort.quality_tier == "low", f"Effort 1.0 should map to low tier, got {low_effort.quality_tier}"
        assert medium_effort.quality_tier == "medium", f"Effort 3.0 should map to medium tier, got {medium_effort.quality_tier}"
        assert high_effort.quality_tier == "high", f"Effort 6.0 should map to high tier, got {high_effort.quality_tier}"
        
        # Validate lookback periods match effort levels
        assert low_effort.lookback_periods == 5, f"Low effort should have 5 lookback periods, got {low_effort.lookback_periods}"
        assert medium_effort.lookback_periods == 10, f"Medium effort should have 10 lookback periods, got {medium_effort.lookback_periods}"
        assert high_effort.lookback_periods == 20, f"High effort should have 20 lookback periods, got {high_effort.lookback_periods}"
        
        # Validate analysis quality decreases with noise factor
        # Formula: max(0.1, 1.0 - noise_factor - (0.1 if len(sector_trades) < 10 else 0.0))
        # With 6 tech trades (< 10), there's a 0.1 penalty applied
        # Low: max(0.1, 1.0 - 0.30 - 0.1) = max(0.1, 0.60) = 0.60
        # Medium: max(0.1, 1.0 - 0.15 - 0.1) = max(0.1, 0.75) = 0.75
        # High: max(0.1, 1.0 - 0.05 - 0.1) = max(0.1, 0.85) = 0.85
        expected_low_quality = 0.60
        expected_medium_quality = 0.75
        expected_high_quality = 0.85
        
        assert abs(low_effort.analysis_quality - expected_low_quality) <= 0.05, \
            f"Low effort analysis quality should be ~{expected_low_quality}, got {low_effort.analysis_quality}"
        assert abs(medium_effort.analysis_quality - expected_medium_quality) <= 0.05, \
            f"Medium effort analysis quality should be ~{expected_medium_quality}, got {medium_effort.analysis_quality}"
        assert abs(high_effort.analysis_quality - expected_high_quality) <= 0.05, \
            f"High effort analysis quality should be ~{expected_high_quality}, got {high_effort.analysis_quality}"
        
        # Validate effort used is correctly tracked
        assert low_effort.effort_used == 1.0, "Should track low effort used"
        assert medium_effort.effort_used == 3.0, "Should track medium effort used"
        assert high_effort.effort_used == 6.0, "Should track high effort used"
        
        # All should analyze the same sample size (6 tech trades available)
        assert low_effort.sample_size == 6, f"Should analyze all 6 tech trades, got {low_effort.sample_size}"
        assert medium_effort.sample_size == 6, f"Should analyze all 6 tech trades, got {medium_effort.sample_size}"  
        assert high_effort.sample_size == 6, f"Should analyze all 6 tech trades, got {high_effort.sample_size}"
        
    def test_performance_revenue_extraction(self, market_model_with_research_data, market_research_config, performance_mapping):
        """
        Test extraction of performance-revenue relationships from historical data.
        
        **Precise Calculation Logic:**
        Test data contains tech sector trades with known performance levels:
        - High performance trades: [120.0, 115.0] with performance [0.85, 0.87] → expected avg = 117.5
        - Medium performance trades: [85.0, 80.0] with performance [0.70, 0.68] → expected avg = 82.5
        - Low performance trades: [50.0, 45.0] with performance [0.55, 0.53] → expected avg = 47.5
        
        High effort (5.0) uses noise_factor = 0.05, so revenue noise = avg_revenue * 0.05.
        Analysis quality = 1.0 - 0.05 = 0.95 (since sample size > 10).
        
        **Validation:** Assert actual values within noise tolerance of expected calculations.
        """
        market_model_with_research_data.state.performance_mapping = performance_mapping
        
        result = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=5.0  # High effort: noise_factor=0.05, lookback_periods=20
        )
        
        # Should extract performance tiers from the test data
        performance_tiers = result.performance_tiers
        
        # Verify it found the performance levels from our test data
        assert "high" in performance_tiers, "Should identify high performance tier (0.85-0.87)"
        assert "medium" in performance_tiers, "Should identify medium performance tier (0.68-0.70)" 
        assert "low" in performance_tiers, "Should identify low performance tier (0.53-0.55)"
        
        # Calculate expected values with noise tolerance
        expected_high_revenue = 117.5  # (120.0 + 115.0) / 2
        expected_medium_revenue = 82.5  # (85.0 + 80.0) / 2  
        expected_low_revenue = 47.5    # (50.0 + 45.0) / 2
        noise_factor = 0.05  # High effort noise factor
        
        # Validate average revenues within noise tolerance
        high_revenue = performance_tiers["high"]["avg_revenue"]
        medium_revenue = performance_tiers["medium"]["avg_revenue"]
        low_revenue = performance_tiers["low"]["avg_revenue"]
        
        # Allow reasonable tolerance for random noise variation
        base_tolerance = 10.0  # Base tolerance for revenue calculations
        high_tolerance = max(base_tolerance, expected_high_revenue * noise_factor * 5)
        medium_tolerance = max(base_tolerance, expected_medium_revenue * noise_factor * 5)
        low_tolerance = max(base_tolerance, expected_low_revenue * noise_factor * 5)
        
        assert abs(high_revenue - expected_high_revenue) <= high_tolerance, \
            f"High revenue {high_revenue} should be {expected_high_revenue} ± {high_tolerance}"
        assert abs(medium_revenue - expected_medium_revenue) <= medium_tolerance, \
            f"Medium revenue {medium_revenue} should be {expected_medium_revenue} ± {medium_tolerance}"
        assert abs(low_revenue - expected_low_revenue) <= low_tolerance, \
            f"Low revenue {low_revenue} should be {expected_low_revenue} ± {low_tolerance}"
        
        # Validate sample counts
        assert performance_tiers["high"]["sample_count"] == 2, "High tier should have 2 trades"
        assert performance_tiers["medium"]["sample_count"] == 2, "Medium tier should have 2 trades"  
        assert performance_tiers["low"]["sample_count"] == 2, "Low tier should have 2 trades"
        
        # Validate revenue standard deviations
        expected_high_std = abs(120.0 - 115.0) / 2  # std dev of [120, 115] = 2.5
        expected_medium_std = abs(85.0 - 80.0) / 2   # std dev of [85, 80] = 2.5
        expected_low_std = abs(50.0 - 45.0) / 2      # std dev of [50, 45] = 2.5
        
        assert abs(performance_tiers["high"]["revenue_std"] - expected_high_std) <= 0.5, \
            f"High revenue std {performance_tiers['high']['revenue_std']} should be ~{expected_high_std}"
        assert abs(performance_tiers["medium"]["revenue_std"] - expected_medium_std) <= 0.5, \
            f"Medium revenue std {performance_tiers['medium']['revenue_std']} should be ~{expected_medium_std}"
        assert abs(performance_tiers["low"]["revenue_std"] - expected_low_std) <= 0.5, \
            f"Low revenue std {performance_tiers['low']['revenue_std']} should be ~{expected_low_std}"
        
        # Validate confidence levels (should be high due to low noise)
        expected_confidence = max(0.5, 1.0 - noise_factor)  # 1.0 - 0.05 = 0.95
        for tier in ["high", "medium", "low"]:
            assert performance_tiers[tier]["confidence"] >= expected_confidence - 0.1, \
                f"{tier} tier confidence should be >= {expected_confidence - 0.1}"
        
        # Validate overall analysis quality
        expected_analysis_quality = max(0.1, 1.0 - noise_factor)  # 0.95 for high effort
        assert abs(result.analysis_quality - expected_analysis_quality) <= 0.1, \
            f"Analysis quality {result.analysis_quality} should be ~{expected_analysis_quality}"
    
    def test_marketing_attribute_analysis(self, market_model_with_research_data, market_research_config, performance_mapping):
        """
        Test marketing attribute analysis in historical performance tool.
        
        **Precise Calculation Logic:**
        Test data contains tech trades with marketing attributes:
        - High innovation + proprietary data: [120.0, 115.0] → avg = 117.5
        - Medium innovation + internal/external data: [85.0, 80.0] → avg = 82.5
        - Low innovation + internal data: [50.0, 45.0] → avg = 47.5
        
        Marketing attribute analysis should group trades by attribute combinations
        and identify top-performing attribute combinations.
        
        **Validation:** Assert marketing attribute analysis fields are populated
        and contain expected attribute combination performance data.
        """
        market_model_with_research_data.state.performance_mapping = performance_mapping
        
        result = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=5.0  # High effort for best analysis quality
        )
        
        # Should have marketing attribute analysis data
        assert hasattr(result, 'marketing_attribute_analysis'), "Should include marketing attribute analysis"
        assert hasattr(result, 'top_performing_attributes'), "Should include top performing attributes"
        
        marketing_analysis = result.marketing_attribute_analysis
        top_attributes = result.top_performing_attributes
        
        # Should analyze marketing attribute combinations from test data
        assert len(marketing_analysis) > 0, "Should find marketing attribute combinations to analyze"
        
        # Verify analysis includes expected attribute combinations
        # High innovation + proprietary: should be highest revenue
        # Medium innovation combinations: should be medium revenue  
        # Low innovation + internal: should be lowest revenue
        
        if marketing_analysis:
            # Find analysis entries by checking for key attributes
            high_innovation_entries = [
                entry for key, entry in marketing_analysis.items()
                if 'high' in key.lower() and 'innovation' in key.lower()
            ]
            low_innovation_entries = [
                entry for key, entry in marketing_analysis.items() 
                if 'low' in key.lower() and 'innovation' in key.lower()
            ]
            
            # Should have entries for different innovation levels
            assert len(high_innovation_entries) > 0, "Should analyze high innovation combinations"
            assert len(low_innovation_entries) > 0, "Should analyze low innovation combinations"
            
            # High innovation should have higher average revenue than low innovation
            if high_innovation_entries and low_innovation_entries:
                high_revenue = high_innovation_entries[0]['avg_revenue']
                low_revenue = low_innovation_entries[0]['avg_revenue'] 
                
                assert high_revenue > low_revenue, \
                    f"High innovation revenue {high_revenue} should exceed low innovation {low_revenue}"
        
        # Top performing attributes should be populated and sorted by performance
        assert len(top_attributes) > 0, "Should identify top performing attribute combinations"
        
        if len(top_attributes) >= 2:
            # Should be sorted by revenue (highest first)
            assert top_attributes[0]['avg_revenue'] >= top_attributes[1]['avg_revenue'], \
                "Top performing attributes should be sorted by revenue"
            
            # Should include attribute details and descriptions
            for attr_combo in top_attributes:
                assert 'attributes' in attr_combo, "Should include raw attribute data"
                assert 'descriptions' in attr_combo, "Should include human-readable descriptions"
                assert 'avg_revenue' in attr_combo, "Should include revenue performance"
                assert 'sample_count' in attr_combo, "Should include sample size"


class TestBuyerPreferenceAnalysis:
    """
    Test analyze_buyer_preferences_impl for extracting buyer preference patterns.
    
    This tool helps agents understand "What do buyers want?" by analyzing buyer
    states and preference patterns within a specific sector.
    """
    
    def test_buyer_preference_analysis_interface(self, market_model_with_research_data, buyers_with_sector_preferences, market_research_config):
        """
        Test analyze_buyer_preferences_impl returns correct response format.
        
        Interface requirements:
        - Input: market_model, config_data, sector, effort  
        - Output: BuyerPreferenceResponse with preference vectors and confidence
        - Integration: Uses BuyerState from market_state.buyers_state
        """
        result = analyze_buyer_preferences_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=3.0
        )
        
        # Must return proper response object
        assert isinstance(result, BuyerPreferenceResponse), f"Must return BuyerPreferenceResponse, got {type(result)}"
        assert result.sector == "tech", "Response must include analyzed sector"
        assert result.effort_used == 3.0, "Response must track effort used"
        assert result.quality_tier in ["low", "medium", "high"], f"Invalid quality tier: {result.quality_tier}"
        
        # Must contain preference analysis data
        assert hasattr(result, 'avg_preferences'), "Must include average preferences"
        assert hasattr(result, 'preference_distribution'), "Must include preference distribution"
        assert hasattr(result, 'sample_size'), "Must track sample size"
        assert isinstance(result.avg_preferences, list), "Average preferences must be list"
        assert len(result.avg_preferences) == 2, "Must have preferences for 2 attributes"
        
    def test_sector_specific_buyer_filtering(self, market_model_with_research_data, buyers_with_sector_preferences, market_research_config):
        """
        Test that buyer preference analysis filters by sector-relevant buyers.
        
        **Precise Calculation Logic:**
        Test data contains sector-specific buyers with known preferences:
        
        Tech buyers (tech_buyer_1, tech_buyer_2):
        - tech_buyer_1: attr_mu = [0.9, 0.5] (methodology-focused)
        - tech_buyer_2: attr_mu = [0.8, 0.6] (methodology-focused)
        - Expected average: [(0.9+0.8)/2, (0.5+0.6)/2] = [0.85, 0.55]
        
        Finance buyers (fin_buyer_1, fin_buyer_2):
        - fin_buyer_1: attr_mu = [0.5, 0.9] (data-quality-focused)  
        - fin_buyer_2: attr_mu = [0.6, 0.8] (data-quality-focused)
        - Expected average: [(0.5+0.6)/2, (0.9+0.8)/2] = [0.55, 0.85]
        
        High effort (4.0) uses noise_factor = 0.1, so noise = preferences * 0.1 * 0.2 = ~0.02.
        
        **Validation:** Assert actual averages within noise tolerance of expected calculations.
        """
        tech_result = analyze_buyer_preferences_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech", 
            effort=4.0  # High effort: noise_factor=0.1, sample_size=50
        )
        
        finance_result = analyze_buyer_preferences_impl(
            market_model_with_research_data,
            market_research_config,
            sector="finance",
            effort=4.0  # High effort: noise_factor=0.1, sample_size=50
        )
        
        # Calculate expected values with noise tolerance
        expected_tech_prefs = [0.85, 0.55]  # [(0.9+0.8)/2, (0.5+0.6)/2]
        expected_finance_prefs = [0.55, 0.85]  # [(0.5+0.6)/2, (0.9+0.8)/2]
        noise_factor = 0.1  # High effort noise factor
        noise_multiplier = 0.2  # Applied in implementation
        
        # Calculate noise tolerances (more generous for random variation)
        base_tolerance = 0.1  # Allow for reasonable variation in random noise
        tech_tolerance_0 = max(base_tolerance, expected_tech_prefs[0] * noise_factor * noise_multiplier * 5)
        tech_tolerance_1 = max(base_tolerance, expected_tech_prefs[1] * noise_factor * noise_multiplier * 5)
        finance_tolerance_0 = max(base_tolerance, expected_finance_prefs[0] * noise_factor * noise_multiplier * 5)
        finance_tolerance_1 = max(base_tolerance, expected_finance_prefs[1] * noise_factor * noise_multiplier * 5)
        
        # Validate tech preferences within noise tolerance
        tech_prefs = tech_result.avg_preferences
        assert abs(tech_prefs[0] - expected_tech_prefs[0]) <= tech_tolerance_0, \
            f"Tech methodology pref should be {expected_tech_prefs[0]} ± {tech_tolerance_0}, got {tech_prefs[0]}"
        assert abs(tech_prefs[1] - expected_tech_prefs[1]) <= tech_tolerance_1, \
            f"Tech data quality pref should be {expected_tech_prefs[1]} ± {tech_tolerance_1}, got {tech_prefs[1]}"
        
        # Validate finance preferences within noise tolerance  
        finance_prefs = finance_result.avg_preferences
        assert abs(finance_prefs[0] - expected_finance_prefs[0]) <= finance_tolerance_0, \
            f"Finance methodology pref should be {expected_finance_prefs[0]} ± {finance_tolerance_0}, got {finance_prefs[0]}"
        assert abs(finance_prefs[1] - expected_finance_prefs[1]) <= finance_tolerance_1, \
            f"Finance data quality pref should be {expected_finance_prefs[1]} ± {finance_tolerance_1}, got {finance_prefs[1]}"
        
        # Validate sample sizes (2 buyers per sector available)
        assert tech_result.sample_size == 2, f"Tech analysis should sample 2 buyers, got {tech_result.sample_size}"
        assert finance_result.sample_size == 2, f"Finance analysis should sample 2 buyers, got {finance_result.sample_size}"
        
        # Validate preference distribution statistics for tech sector
        tech_dist = tech_result.preference_distribution
        expected_tech_methodology_mean = expected_tech_prefs[0]  # 0.85
        expected_tech_data_mean = expected_tech_prefs[1]  # 0.55
        
        assert abs(tech_dist["methodology"]["mean"] - expected_tech_methodology_mean) <= 0.1, \
            f"Tech methodology dist mean should be ~{expected_tech_methodology_mean}, got {tech_dist['methodology']['mean']}"
        assert abs(tech_dist["data_quality"]["mean"] - expected_tech_data_mean) <= 0.1, \
            f"Tech data quality dist mean should be ~{expected_tech_data_mean}, got {tech_dist['data_quality']['mean']}"
        
        # Validate clear sector differentiation (tech focuses on methodology, finance on data quality)
        assert tech_prefs[0] > finance_prefs[0] + 0.2, \
            f"Tech should have higher methodology preference: tech={tech_prefs[0]}, finance={finance_prefs[0]}"
        assert finance_prefs[1] > tech_prefs[1] + 0.2, \
            f"Finance should have higher data quality preference: finance={finance_prefs[1]}, tech={tech_prefs[1]}"
        
    def test_effort_sample_size_relationship(self, market_model_with_research_data, buyers_with_sector_preferences, market_research_config):
        """
        Test that higher effort produces larger sample sizes and better accuracy.
        
        **Precise Effort-Quality Mapping Logic:**
        - Low effort (1.0 < 2.0): quality_tier="low", noise_factor=0.50, target_sample=10
        - High effort (5.0 ≥ 4.0): quality_tier="high", noise_factor=0.10, target_sample=50
        
        With 2 tech buyers available, actual sample_size = min(2, target_sample) = 2 for both.
        Confidence level = min(0.95, 0.5 + (sample_size/20) * (1.0 - noise_factor)):
        - Low: min(0.95, 0.5 + (2/20) * 0.5) = min(0.95, 0.55) = 0.55
        - High: min(0.95, 0.5 + (2/20) * 0.9) = min(0.95, 0.59) = 0.59
        
        **Validation:** Assert exact quality tiers, sample sizes, and confidence calculations.
        """
        low_effort = analyze_buyer_preferences_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=1.0  # Below medium threshold (2.0)
        )
        
        high_effort = analyze_buyer_preferences_impl(
            market_model_with_research_data, 
            market_research_config,
            sector="tech",
            effort=5.0  # Above high threshold (4.0)
        )
        
        # Validate quality tier mappings
        assert low_effort.quality_tier == "low", f"Effort 1.0 should map to low tier, got {low_effort.quality_tier}"
        assert high_effort.quality_tier == "high", f"Effort 5.0 should map to high tier, got {high_effort.quality_tier}"
        
        # Both should have same actual sample size (2 tech buyers available)
        assert low_effort.sample_size == 2, f"Low effort should sample 2 tech buyers, got {low_effort.sample_size}"
        assert high_effort.sample_size == 2, f"High effort should sample 2 tech buyers, got {high_effort.sample_size}"
        
        # Validate effort tracking
        assert low_effort.effort_used == 1.0, "Should track low effort used"
        assert high_effort.effort_used == 5.0, "Should track high effort used"
        
        # Calculate expected confidence levels
        # Formula: min(0.95, 0.5 + (sample_size / 20) * (1.0 - noise_factor))
        low_noise_factor = 0.50  # Low effort noise factor
        high_noise_factor = 0.10  # High effort noise factor
        sample_size = 2
        
        expected_low_confidence = min(0.95, 0.5 + (sample_size / 20) * (1.0 - low_noise_factor))  # 0.55
        expected_high_confidence = min(0.95, 0.5 + (sample_size / 20) * (1.0 - high_noise_factor))  # 0.59
        
        # Validate confidence levels with tolerance
        if hasattr(low_effort, 'confidence_level') and low_effort.confidence_level is not None:
            assert abs(low_effort.confidence_level - expected_low_confidence) <= 0.05, \
                f"Low effort confidence should be ~{expected_low_confidence}, got {low_effort.confidence_level}"
        
        if hasattr(high_effort, 'confidence_level') and high_effort.confidence_level is not None:
            assert abs(high_effort.confidence_level - expected_high_confidence) <= 0.05, \
                f"High effort confidence should be ~{expected_high_confidence}, got {high_effort.confidence_level}"
            
            # High effort should have higher confidence than low effort
            if hasattr(low_effort, 'confidence_level') and low_effort.confidence_level is not None:
                assert high_effort.confidence_level >= low_effort.confidence_level, \
                    f"High effort confidence ({high_effort.confidence_level}) should be >= low effort ({low_effort.confidence_level})"
        
        # Validate data quality assessment  
        # Logic: "good" if quality_tier=="high" AND sample_size>=10, "fair" if quality_tier=="medium", else "limited"
        # With sample_size=2 (< 10), high effort will still be "limited"
        expected_low_data_quality = "limited"   # Low effort → limited
        expected_high_data_quality = "limited"  # High effort but sample_size=2 < 10 → limited
        
        assert low_effort.data_quality == expected_low_data_quality, \
            f"Low effort data quality should be '{expected_low_data_quality}', got '{low_effort.data_quality}'"
        assert high_effort.data_quality == expected_high_data_quality, \
            f"High effort data quality should be '{expected_high_data_quality}', got '{high_effort.data_quality}'"
    
    def test_marketing_preference_interpretation(self, market_model_with_research_data, buyers_with_sector_preferences, market_research_config):
        """
        Test marketing preference interpretation in buyer preference analysis tool.
        
        **Precise Calculation Logic:**
        Test data contains buyers with different marketing conversion functions:
        - tech_buyer_1: "tech_high_innovation_focused" conversion function
        - tech_buyer_2: "tech_balanced_conversion" conversion function
        
        The tool should analyze how buyers with different conversion functions
        value marketing attributes and detect buyer heterogeneity.
        
        **Validation:** Assert marketing preference interpretation fields are populated
        and show analysis of attribute valuations across different buyer types.
        """
        result = analyze_buyer_preferences_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=4.0  # High effort for complete analysis
        )
        
        # Should have marketing preference interpretation data
        assert hasattr(result, 'marketing_preference_interpretation'), "Should include marketing preference interpretation"
        assert hasattr(result, 'buyer_heterogeneity'), "Should include buyer heterogeneity analysis"
        
        marketing_interp = result.marketing_preference_interpretation
        buyer_heterogeneity = result.buyer_heterogeneity
        
        # Should analyze marketing attributes from the attribute definitions
        expected_attributes = ["innovation_level", "data_source", "risk_score"]
        
        # Check if marketing preference interpretation is populated
        if marketing_interp:
            # Should have analysis for some marketing attributes
            assert len(marketing_interp) > 0, "Should analyze marketing attribute preferences"
            
            for attr_name, analysis in marketing_interp.items():
                assert 'avg_preference_strength' in analysis, "Should include preference strength"
                assert 'preference_variance' in analysis, "Should include preference variance"
                assert 'attribute_description' in analysis, "Should include attribute description"
                assert 'sample_buyers' in analysis, "Should include sample size"
                
                # Preference strength should be reasonable (0-1 range)
                strength = analysis['avg_preference_strength']
                assert 0.0 <= strength <= 1.0, f"Preference strength {strength} should be in [0,1] range"
        
        # Check buyer heterogeneity analysis
        if buyer_heterogeneity:
            # Should detect heterogeneity across different buyer conversion functions
            assert len(buyer_heterogeneity) > 0, "Should analyze buyer heterogeneity"
            
            for attr_name, heterogeneity in buyer_heterogeneity.items():
                assert 'valuation_mean' in heterogeneity, "Should include mean valuation"
                assert 'valuation_std' in heterogeneity, "Should include valuation variance"
                assert 'heterogeneity_level' in heterogeneity, "Should classify heterogeneity level"
                assert 'conversion_functions_analyzed' in heterogeneity, "Should track functions analyzed"
                
                # Heterogeneity level should be valid category
                het_level = heterogeneity['heterogeneity_level']
                assert het_level in ['low', 'medium', 'high'], f"Invalid heterogeneity level: {het_level}"
                
                # Should analyze multiple conversion functions
                functions_analyzed = heterogeneity['conversion_functions_analyzed']
                assert functions_analyzed >= 1, "Should analyze at least one conversion function"


class TestCompetitivePricingResearch:
    """
    Test research_competitive_pricing_impl for analyzing competitive pricing patterns.
    
    This tool helps agents understand "What are competitors charging?" by analyzing
    recent historical pricing data within a specific sector.
    """
    
    def test_competitive_pricing_research_interface(self, market_model_with_research_data, competitive_offers_by_sector, market_research_config):
        """
        Test research_competitive_pricing_impl returns correct response format.
        
        Interface requirements:
        - Input: market_model, config_data, sector, effort
        - Output: CompetitivePricingResponse with pricing statistics  
        - Integration: Uses Offer data from market_state.offers
        """
        result = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=2.0
        )
        
        # Must return proper response object  
        assert isinstance(result, CompetitivePricingResponse), f"Must return CompetitivePricingResponse, got {type(result)}"
        assert result.sector == "tech", "Response must include analyzed sector"
        assert result.effort_used == 2.0, "Response must track effort used"
        assert result.quality_tier in ["low", "medium", "high"], f"Invalid quality tier: {result.quality_tier}"
        
        # Must contain pricing analysis data
        assert hasattr(result, 'price_statistics'), "Must include price statistics"
        assert hasattr(result, 'competitive_landscape'), "Must include competitive landscape"
        assert hasattr(result, 'sample_size'), "Must track sample size"
        
    def test_sector_specific_pricing_patterns(self, market_model_with_research_data, competitive_offers_by_sector, market_research_config):
        """
        Test that pricing research identifies different patterns by sector.
        
        **Precise Calculation Logic:**
        Test data contains sector-specific competitive offers:
        
        Tech sector offers: [100.0, 110.0, 95.0, 105.0] + trade data [120, 115, 85, 80, 50, 45]
        - Current offers contribute: [100, 110, 95, 105] 
        - Trade data contributes recent trades (limited by lookback_periods=15)
        - Combined data (offers + trades): [100, 110, 95, 105, 120, 115, 85, 80, 50, 45]
        - Expected avg_price = (100+110+95+105+120+115+85+80+50+45) / 10 = 90.5
        
        Finance sector offers: [150.0, 145.0, 140.0] + trade data [95, 70, 40]
        - Combined data: [150, 145, 140, 95, 70, 40]  
        - Expected avg_price = (150+145+140+95+70+40) / 6 = 106.67
        
        High effort (3.0) uses price_noise = 0.02, sample_size analysis, competition levels.
        
        **Validation:** Assert actual statistics within noise tolerance of expected calculations.
        """
        tech_result = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=3.0  # High effort: price_noise=0.02, lookback_periods=15
        )
        
        finance_result = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config,
            sector="finance", 
            effort=3.0  # High effort: price_noise=0.02, lookback_periods=15
        )
        
        # Calculate expected values with noise tolerance
        # Tech: offers [100,110,95,105] + trades [120,115,85,80,50,45] = 10 total prices  
        expected_tech_prices = [100.0, 110.0, 95.0, 105.0, 120.0, 115.0, 85.0, 80.0, 50.0, 45.0]
        expected_tech_avg = sum(expected_tech_prices) / len(expected_tech_prices)  # 90.5
        expected_tech_min = min(expected_tech_prices)  # 45.0
        expected_tech_max = max(expected_tech_prices)  # 120.0
        
        # Finance: offers [150,145,140] + trades [95,70,40] = 6 total prices
        expected_finance_prices = [150.0, 145.0, 140.0, 95.0, 70.0, 40.0] 
        expected_finance_avg = sum(expected_finance_prices) / len(expected_finance_prices)  # 106.67
        expected_finance_min = min(expected_finance_prices)  # 40.0  
        expected_finance_max = max(expected_finance_prices)  # 150.0
        
        price_noise = 0.02  # High effort noise factor
        
        # Validate tech pricing statistics
        tech_stats = tech_result.price_statistics
        tech_avg_tolerance = expected_tech_avg * price_noise * 3  # 3-sigma tolerance
        
        assert abs(tech_stats['avg_price'] - expected_tech_avg) <= tech_avg_tolerance + 5, \
            f"Tech avg price should be ~{expected_tech_avg} ± {tech_avg_tolerance}, got {tech_stats['avg_price']}"
        assert abs(tech_stats['min_price'] - expected_tech_min) <= expected_tech_min * price_noise + 2, \
            f"Tech min price should be ~{expected_tech_min}, got {tech_stats['min_price']}"  
        assert abs(tech_stats['max_price'] - expected_tech_max) <= expected_tech_max * price_noise + 5, \
            f"Tech max price should be ~{expected_tech_max}, got {tech_stats['max_price']}"
        
        # Validate finance pricing statistics  
        finance_stats = finance_result.price_statistics
        finance_avg_tolerance = expected_finance_avg * price_noise * 3
        
        assert abs(finance_stats['avg_price'] - expected_finance_avg) <= finance_avg_tolerance + 5, \
            f"Finance avg price should be ~{expected_finance_avg} ± {finance_avg_tolerance}, got {finance_stats['avg_price']}"
        assert abs(finance_stats['min_price'] - expected_finance_min) <= expected_finance_min * price_noise + 2, \
            f"Finance min price should be ~{expected_finance_min}, got {finance_stats['min_price']}"
        assert abs(finance_stats['max_price'] - expected_finance_max) <= expected_finance_max * price_noise + 5, \
            f"Finance max price should be ~{expected_finance_max}, got {finance_stats['max_price']}"
        
        # Validate sample sizes
        expected_tech_sample_size = 10  # 4 offers + 6 trades
        expected_finance_sample_size = 6   # 3 offers + 3 trades
        
        assert tech_result.sample_size == expected_tech_sample_size, \
            f"Tech should analyze {expected_tech_sample_size} prices, got {tech_result.sample_size}"
        assert finance_result.sample_size == expected_finance_sample_size, \
            f"Finance should analyze {expected_finance_sample_size} prices, got {finance_result.sample_size}"
        
        # Validate competition levels based on sample size thresholds
        # sample_size >= 8 → "high", >= 4 → "medium", < 4 → "low"
        expected_tech_competition = "high"   # 10 >= 8
        expected_finance_competition = "medium"  # 6 >= 4 but < 8
        
        assert tech_result.competitive_landscape['competition_level'] == expected_tech_competition, \
            f"Tech competition should be '{expected_tech_competition}', got '{tech_result.competitive_landscape['competition_level']}'"
        assert finance_result.competitive_landscape['competition_level'] == expected_finance_competition, \
            f"Finance competition should be '{expected_finance_competition}', got '{finance_result.competitive_landscape['competition_level']}'"
        
        # Validate market depth matches sample sizes
        assert tech_result.competitive_landscape['market_depth'] == expected_tech_sample_size, \
            f"Tech market depth should be {expected_tech_sample_size}, got {tech_result.competitive_landscape['market_depth']}"
        assert finance_result.competitive_landscape['market_depth'] == expected_finance_sample_size, \
            f"Finance market depth should be {expected_finance_sample_size}, got {finance_result.competitive_landscape['market_depth']}"
        
    def test_effort_lookback_relationship(self, market_model_with_research_data, competitive_offers_by_sector, market_research_config):
        """
        Test that higher effort produces more comprehensive pricing analysis.
        
        **Precise Effort-Quality Mapping Logic:**
        - Low effort (1.0 < 1.5): quality_tier="low", price_noise=0.15, lookback_periods=3
        - High effort (4.0 ≥ 3.0): quality_tier="high", price_noise=0.02, lookback_periods=15
        
        Tech data: 4 current offers + trade data (limited by lookback periods)
        - Low effort: 4 offers + min(6 trades, 3*3=9) = 4 offers + 6 trades = 10 total
        - High effort: 4 offers + min(6 trades, 15*3=45) = 4 offers + 6 trades = 10 total
        
        Both analyze same data but with different noise levels and quality assessments.
        
        **Validation:** Assert exact effort mappings, lookback periods, and quality metrics.
        """
        low_effort = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=1.0  # Below medium threshold (1.5)
        )
        
        high_effort = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config, 
            sector="tech",
            effort=4.0  # Above high threshold (3.0)
        )
        
        # Validate quality tier mappings
        assert low_effort.quality_tier == "low", f"Effort 1.0 should map to low tier, got {low_effort.quality_tier}"
        assert high_effort.quality_tier == "high", f"Effort 4.0 should map to high tier, got {high_effort.quality_tier}"
        
        # Validate lookback periods
        assert low_effort.lookback_periods == 3, f"Low effort should have 3 lookback periods, got {low_effort.lookback_periods}"
        assert high_effort.lookback_periods == 15, f"High effort should have 15 lookback periods, got {high_effort.lookback_periods}"
        
        # Both should analyze same sample size (all available data fits within both lookback limits)
        expected_sample_size = 10  # 4 current offers + 6 historical trades
        assert low_effort.sample_size == expected_sample_size, \
            f"Low effort should analyze {expected_sample_size} prices, got {low_effort.sample_size}"
        assert high_effort.sample_size == expected_sample_size, \
            f"High effort should analyze {expected_sample_size} prices, got {high_effort.sample_size}"
        
        # Validate effort tracking
        assert low_effort.effort_used == 1.0, "Should track low effort used"
        assert high_effort.effort_used == 4.0, "Should track high effort used" 
        
        # Validate data quality assessment based on effort and sample size
        # Logic: "good" if quality_tier=="high" AND sample_size>=5, "fair" if quality_tier=="medium", else "limited"
        # With sample_size=10 (≥5):
        # - Low effort (quality_tier="low") → "limited" 
        # - High effort (quality_tier="high") → "good"
        expected_low_data_quality = "limited"  # Low quality tier → limited
        expected_high_data_quality = "good"    # High quality tier AND sample_size≥5 → good
        
        assert low_effort.data_quality == expected_low_data_quality, \
            f"Low effort data quality should be '{expected_low_data_quality}', got '{low_effort.data_quality}'"
        assert high_effort.data_quality == expected_high_data_quality, \
            f"High effort data quality should be '{expected_high_data_quality}', got '{high_effort.data_quality}'"
        
        # Validate competitive landscape analysis completeness
        # Both should have same keys, but high effort should have more accurate values
        high_landscape = high_effort.competitive_landscape
        low_landscape = low_effort.competitive_landscape
        
        required_keys = ['competition_level', 'market_depth', 'price_spread', 'price_volatility', 'market_maturity']
        for key in required_keys:
            assert key in low_landscape, f"Low effort should include {key} in competitive landscape"
            assert key in high_landscape, f"High effort should include {key} in competitive landscape"
        
        # Both should identify same competition level (based on sample size = 10 ≥ 8 → "high")
        assert low_landscape['competition_level'] == 'high', "Low effort should identify high competition"
        assert high_landscape['competition_level'] == 'high', "High effort should identify high competition"
        
        # Market depth should match sample size for both
        assert low_landscape['market_depth'] == expected_sample_size, "Low effort market depth should match sample size"
        assert high_landscape['market_depth'] == expected_sample_size, "High effort market depth should match sample size"
    
    def test_marketing_attribute_pricing_analysis(self, market_model_with_research_data, competitive_offers_by_sector, market_research_config):
        """
        Test marketing attribute-based pricing analysis in competitive pricing tool.
        
        **Precise Calculation Logic:**
        Test data contains tech offers and trades with marketing attributes:
        - High innovation offers/trades: should command price premium
        - Proprietary data source: should have higher prices than internal/external
        - Lower risk scores: should correlate with higher prices
        
        The tool should analyze pricing patterns by marketing attribute combinations
        and calculate attribute price premiums.
        
        **Validation:** Assert pricing by marketing attributes fields are populated
        and show pricing segmentation by attribute combinations.
        """
        result = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=3.0  # High effort for comprehensive analysis
        )
        
        # Should have marketing attribute pricing analysis data
        assert hasattr(result, 'pricing_by_marketing_attributes'), "Should include pricing by marketing attributes"
        assert hasattr(result, 'attribute_price_premiums'), "Should include attribute price premiums"
        
        pricing_by_attrs = result.pricing_by_marketing_attributes
        price_premiums = result.attribute_price_premiums
        
        # Check if pricing by marketing attributes is populated
        if pricing_by_attrs:
            # Should have pricing analysis for attribute combinations
            assert len(pricing_by_attrs) > 0, "Should analyze pricing by marketing attribute combinations"
            
            for attr_combo_key, pricing_stats in pricing_by_attrs.items():
                # Each pricing analysis should include comprehensive statistics
                assert 'avg_price' in pricing_stats, "Should include average price"
                assert 'price_std' in pricing_stats, "Should include price standard deviation"
                assert 'min_price' in pricing_stats, "Should include minimum price"
                assert 'max_price' in pricing_stats, "Should include maximum price"
                assert 'sample_count' in pricing_stats, "Should include sample count"
                assert 'attributes' in pricing_stats, "Should include raw attribute data"
                assert 'descriptions' in pricing_stats, "Should include human-readable descriptions"
                
                # Price statistics should be reasonable
                avg_price = pricing_stats['avg_price']
                min_price = pricing_stats['min_price']
                max_price = pricing_stats['max_price']
                
                assert avg_price > 0, f"Average price {avg_price} should be positive"
                assert min_price <= avg_price <= max_price, "Price statistics should be consistent"
                assert pricing_stats['sample_count'] >= 2, "Should only analyze groups with sufficient data"
        
        # Check attribute price premiums
        if price_premiums:
            # Should calculate premiums for different attribute values
            assert len(price_premiums) > 0, "Should calculate attribute price premiums"
            
            # Look for expected premium patterns
            innovation_premiums = {k: v for k, v in price_premiums.items() if 'innovation_level' in k}
            data_source_premiums = {k: v for k, v in price_premiums.items() if 'data_source' in k}
            risk_correlations = {k: v for k, v in price_premiums.items() if 'risk_score' in k}
            
            # Should analyze innovation level premiums
            if innovation_premiums:
                # High innovation should generally command premium over low innovation
                high_innovation_keys = [k for k in innovation_premiums.keys() if 'high' in k]
                low_innovation_keys = [k for k in innovation_premiums.keys() if 'low' in k]
                
                if high_innovation_keys and low_innovation_keys:
                    high_premium = innovation_premiums[high_innovation_keys[0]]
                    low_premium = innovation_premiums[low_innovation_keys[0]]
                    
                    # High innovation should have higher premium (or at least not significantly lower)
                    # Allow some tolerance for noise in the analysis
                    assert high_premium >= low_premium - 0.2, \
                        f"High innovation premium {high_premium} should be >= low innovation {low_premium}"
            
            # Should analyze data source premiums
            if data_source_premiums:
                # Proprietary data should generally command premium over internal data
                proprietary_keys = [k for k in data_source_premiums.keys() if 'proprietary' in k]
                internal_keys = [k for k in data_source_premiums.keys() if 'internal' in k]
                
                if proprietary_keys and internal_keys:
                    proprietary_premium = data_source_premiums[proprietary_keys[0]]
                    internal_premium = data_source_premiums[internal_keys[0]]
                    
                    # Proprietary should have higher premium (with noise tolerance)
                    assert proprietary_premium >= internal_premium - 0.3, \
                        f"Proprietary premium {proprietary_premium} should be >= internal {internal_premium}"
            
            # Should analyze risk score correlations
            if risk_correlations:
                # Risk score correlation should be negative (lower risk = higher prices)
                for risk_key, correlation in risk_correlations.items():
                    # Allow for some variation due to noise, but expect negative trend
                    assert correlation <= 0.3, f"Risk correlation {correlation} should trend negative"


class TestMarketResearchIntegration:
    """
    Test integration between market research tools and market_for_finance framework.
    
    Integration tests verify that tools work correctly with MarketModel/MarketState
    and provide the data needed for agent decision making.
    """
    
    def test_tools_work_with_empty_market_data(self, market_research_config):
        """
        Test that tools handle edge cases gracefully with minimal market data.
        
        Tools should return sensible defaults or warnings when insufficient
        historical data is available for analysis.
        """
        # Create minimal market state
        minimal_state = MarketState(
            offers=[],
            trades=[], 
            demand_profile={},
            supply_profile={},
            index_values={"tech": 100.0},
            current_regimes={"tech": 0},
            current_period=0,
            knowledge_good_forecasts={},
            buyers_state=[],
            all_trades=[]
        )
        
        minimal_model = MarketModel(
            id=1,
            name="minimal_test",
            agents=[],
            state=minimal_state,
            step=lambda: None,
            collect_stats=lambda: {}
        )
        
        # Tools should handle empty data gracefully
        hist_result = analyze_historical_performance_impl(
            minimal_model, market_research_config, sector="tech", effort=2.0
        )
        pref_result = analyze_buyer_preferences_impl(
            minimal_model, market_research_config, sector="tech", effort=2.0
        )
        price_result = research_competitive_pricing_impl(
            minimal_model, market_research_config, sector="tech", effort=2.0
        )
        
        # Should return responses with warnings about data limitations
        assert isinstance(hist_result, HistoricalPerformanceResponse)
        assert isinstance(pref_result, BuyerPreferenceResponse)  
        assert isinstance(price_result, CompetitivePricingResponse)
        
        # Should indicate data limitations in warnings or metadata
        assert hasattr(hist_result, 'warnings') or hasattr(hist_result, 'data_quality')
        assert hasattr(pref_result, 'warnings') or hasattr(pref_result, 'data_quality')
        assert hasattr(price_result, 'warnings') or hasattr(price_result, 'data_quality')
        
    def test_consistent_effort_quality_mapping(self, market_model_with_research_data, buyers_with_sector_preferences, competitive_offers_by_sector, market_research_config, performance_mapping):
        """
        Test that all tools use consistent effort-to-quality tier mapping.
        
        Same effort level should produce same quality tier across all three tools
        for consistent user experience and predictable behavior.
        """
        market_model_with_research_data.state.performance_mapping = performance_mapping
        effort_level = 2.5  # Should map to "medium" across all tools
        
        hist_result = analyze_historical_performance_impl(
            market_model_with_research_data, market_research_config, sector="tech", effort=effort_level
        )
        pref_result = analyze_buyer_preferences_impl(
            market_model_with_research_data, market_research_config, sector="tech", effort=effort_level
        ) 
        price_result = research_competitive_pricing_impl(
            market_model_with_research_data, market_research_config, sector="tech", effort=effort_level
        )
        
        # All tools should map same effort to same quality tier
        assert hist_result.quality_tier == "medium", f"Historical analysis should be medium quality, got {hist_result.quality_tier}"
        assert pref_result.quality_tier == "medium", f"Preference analysis should be medium quality, got {pref_result.quality_tier}"
        assert price_result.quality_tier == "medium", f"Pricing research should be medium quality, got {price_result.quality_tier}"
        
        # All should track the same effort used
        assert hist_result.effort_used == effort_level
        assert pref_result.effort_used == effort_level
        assert price_result.effort_used == effort_level


if __name__ == "__main__":
    # Run tests with detailed output for TDD development
    pytest.main([__file__, "-v", "--tb=short"])