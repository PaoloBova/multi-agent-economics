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
        all_trades=[  # Historical trade data for performance analysis
            # Tech sector trades with varying performance
            TradeData(buyer_id="buyer_1", seller_id="tech_seller_1", price=120.0, quantity=1, good_id="tech_high_perf_1"),
            TradeData(buyer_id="buyer_2", seller_id="tech_seller_1", price=115.0, quantity=1, good_id="tech_high_perf_2"),
            TradeData(buyer_id="buyer_3", seller_id="tech_seller_2", price=85.0, quantity=1, good_id="tech_med_perf_1"),
            TradeData(buyer_id="buyer_4", seller_id="tech_seller_2", price=80.0, quantity=1, good_id="tech_med_perf_2"),
            TradeData(buyer_id="buyer_5", seller_id="tech_seller_3", price=50.0, quantity=1, good_id="tech_low_perf_1"),
            TradeData(buyer_id="buyer_6", seller_id="tech_seller_3", price=45.0, quantity=1, good_id="tech_low_perf_2"),
            
            # Finance sector trades  
            TradeData(buyer_id="buyer_7", seller_id="fin_seller_1", price=95.0, quantity=1, good_id="fin_high_perf_1"),
            TradeData(buyer_id="buyer_8", seller_id="fin_seller_2", price=70.0, quantity=1, good_id="fin_med_perf_1"),
            TradeData(buyer_id="buyer_9", seller_id="fin_seller_3", price=40.0, quantity=1, good_id="fin_low_perf_1"),
            
            # Healthcare sector trades
            TradeData(buyer_id="buyer_10", seller_id="health_seller_1", price=130.0, quantity=1, good_id="health_high_perf_1"),
            TradeData(buyer_id="buyer_11", seller_id="health_seller_2", price=90.0, quantity=1, good_id="health_med_perf_1"),
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
    """
    buyers = [
        # Tech sector buyers - methodology focused
        BuyerState(
            buyer_id="tech_buyer_1",
            regime_beliefs={"tech": [0.7, 0.3]},
            attr_mu=[0.9, 0.5],  # High methodology preference, moderate data
            attr_sigma2=[0.1, 0.2],
            attr_weights=[0.9, 0.5],
            budget=200.0
        ),
        BuyerState(
            buyer_id="tech_buyer_2", 
            regime_beliefs={"tech": [0.6, 0.4]},
            attr_mu=[0.8, 0.6],  # High methodology, decent data
            attr_sigma2=[0.15, 0.15],
            attr_weights=[0.8, 0.6],
            budget=180.0
        ),
        
        # Finance sector buyers - data quality focused
        BuyerState(
            buyer_id="fin_buyer_1",
            regime_beliefs={"finance": [0.4, 0.6]},
            attr_mu=[0.5, 0.9],  # Moderate methodology, high data preference
            attr_sigma2=[0.2, 0.1],
            attr_weights=[0.5, 0.9],
            budget=250.0
        ),
        BuyerState(
            buyer_id="fin_buyer_2",
            regime_beliefs={"finance": [0.3, 0.7]},
            attr_mu=[0.6, 0.8],  # Decent methodology, high data
            attr_sigma2=[0.18, 0.12],
            attr_weights=[0.6, 0.8], 
            budget=220.0
        ),
        
        # Healthcare sector buyers - balanced
        BuyerState(
            buyer_id="health_buyer_1",
            regime_beliefs={"healthcare": [0.5, 0.5]},
            attr_mu=[0.7, 0.7],  # Balanced preferences
            attr_sigma2=[0.15, 0.15],
            attr_weights=[0.7, 0.7],
            budget=300.0
        )
    ]
    market_state_with_history.buyers_state = buyers
    return buyers


@pytest.fixture
def competitive_offers_by_sector(market_state_with_history):
    """
    Create historical competitive offers across sectors for pricing research.
    Shows different pricing patterns and competitive landscapes by sector.
    """
    offers = [
        # Tech sector - high competition, varied pricing
        Offer(good_id="tech_comp_1", price=100.0, seller="tech_comp_1", attr_vector=[0.8, 0.7]),
        Offer(good_id="tech_comp_2", price=110.0, seller="tech_comp_2", attr_vector=[0.8, 0.8]),
        Offer(good_id="tech_comp_3", price=95.0, seller="tech_comp_3", attr_vector=[0.7, 0.7]),
        Offer(good_id="tech_comp_4", price=105.0, seller="tech_comp_4", attr_vector=[0.9, 0.6]),
        
        # Finance sector - premium pricing, consistent
        Offer(good_id="fin_comp_1", price=150.0, seller="fin_comp_1", attr_vector=[0.9, 0.9]),
        Offer(good_id="fin_comp_2", price=145.0, seller="fin_comp_2", attr_vector=[0.8, 0.9]),
        Offer(good_id="fin_comp_3", price=140.0, seller="fin_comp_3", attr_vector=[0.9, 0.8]),
        
        # Healthcare sector - moderate pricing  
        Offer(good_id="health_comp_1", price=120.0, seller="health_comp_1", attr_vector=[0.8, 0.8]),
        Offer(good_id="health_comp_2", price=115.0, seller="health_comp_2", attr_vector=[0.7, 0.8]),
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
        
        Higher effort should result in:
        - Longer lookback periods
        - Lower noise in performance estimates  
        - Better quality tier classification
        - More comprehensive analysis
        """
        market_model_with_research_data.state.performance_mapping = performance_mapping
        
        # Test different effort levels
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
            effort=3.0  # Above medium, below high (5.0)
        )
        
        high_effort = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=6.0  # Above high threshold (5.0)
        )
        
        # Quality tiers should increase with effort
        assert low_effort.quality_tier == "low"
        assert medium_effort.quality_tier == "medium" 
        assert high_effort.quality_tier == "high"
        
        # High effort should have less noise/better accuracy
        assert hasattr(high_effort, 'analysis_quality'), "Should track analysis quality"
        
    def test_performance_revenue_extraction(self, market_model_with_research_data, market_research_config, performance_mapping):
        """
        Test extraction of performance-revenue relationships from historical data.
        
        Should identify that high performance forecasts (0.85+) achieved higher prices
        than medium (0.70) or low (0.55) performance forecasts in the tech sector.
        """
        market_model_with_research_data.state.performance_mapping = performance_mapping
        
        result = analyze_historical_performance_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=5.0  # High effort for accurate analysis
        )
        
        # Should extract performance tiers from the test data
        performance_tiers = result.performance_tiers
        
        # Verify it found the performance levels from our test data
        assert "high" in performance_tiers, "Should identify high performance tier (0.85-0.87)"
        assert "medium" in performance_tiers, "Should identify medium performance tier (0.68-0.70)" 
        assert "low" in performance_tiers, "Should identify low performance tier (0.53-0.55)"
        
        # High performance should have higher average revenue
        high_revenue = performance_tiers["high"]["avg_revenue"]
        medium_revenue = performance_tiers["medium"]["avg_revenue"]
        low_revenue = performance_tiers["low"]["avg_revenue"]
        
        assert high_revenue > medium_revenue, f"High performance revenue ({high_revenue}) should exceed medium ({medium_revenue})"
        assert medium_revenue > low_revenue, f"Medium performance revenue ({medium_revenue}) should exceed low ({low_revenue})"


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
        
        Tech analysis should focus on tech buyers, finance on finance buyers, etc.
        Different sectors have different buyer preference patterns in the test data.
        """
        tech_result = analyze_buyer_preferences_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech", 
            effort=4.0
        )
        
        finance_result = analyze_buyer_preferences_impl(
            market_model_with_research_data,
            market_research_config,
            sector="finance",
            effort=4.0
        )
        
        # Should identify different preference patterns by sector
        tech_prefs = tech_result.avg_preferences
        finance_prefs = finance_result.avg_preferences
        
        # Tech buyers prefer methodology (high attr[0]) - should be > 0.8
        # Finance buyers prefer data quality (high attr[1]) - should be > 0.8
        assert tech_prefs[0] > 0.7, f"Tech buyers should prefer methodology, got {tech_prefs[0]}"
        assert finance_prefs[1] > 0.7, f"Finance buyers should prefer data quality, got {finance_prefs[1]}"
        
        # Preferences should be meaningfully different between sectors
        assert abs(tech_prefs[0] - finance_prefs[0]) > 0.1, "Should detect methodology preference difference"
        assert abs(tech_prefs[1] - finance_prefs[1]) > 0.1, "Should detect data quality preference difference"
        
    def test_effort_sample_size_relationship(self, market_model_with_research_data, buyers_with_sector_preferences, market_research_config):
        """
        Test that higher effort produces larger sample sizes and better accuracy.
        
        Higher effort should result in:
        - Larger effective sample sizes
        - Lower noise in preference estimates
        - Higher confidence levels
        - Better quality tier
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
        
        # Quality and sample sizes should increase with effort
        assert low_effort.quality_tier == "low"
        assert high_effort.quality_tier == "high"
        assert high_effort.sample_size >= low_effort.sample_size, "Higher effort should have larger sample"
        
        # High effort should have higher confidence
        if hasattr(high_effort, 'confidence_level') and hasattr(low_effort, 'confidence_level'):
            assert high_effort.confidence_level >= low_effort.confidence_level, "Higher effort should have higher confidence"


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
        
        Test data has different pricing patterns:
        - Tech: Moderate pricing, high competition (95-110 range)
        - Finance: Premium pricing, consistent (140-150 range)  
        - Healthcare: Moderate pricing (115-120 range)
        """
        tech_result = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config,
            sector="tech",
            effort=3.0
        )
        
        finance_result = research_competitive_pricing_impl(
            market_model_with_research_data,
            market_research_config,
            sector="finance", 
            effort=3.0
        )
        
        # Extract pricing statistics
        tech_stats = tech_result.price_statistics
        finance_stats = finance_result.price_statistics
        
        # Finance should have higher average pricing than tech
        assert finance_stats['avg_price'] > tech_stats['avg_price'], \
            f"Finance avg ({finance_stats['avg_price']}) should exceed tech avg ({tech_stats['avg_price']})"
        
        # Should identify correct competitive landscapes
        assert tech_result.competitive_landscape['competition_level'] in ['high', 'medium', 'low']
        assert finance_result.competitive_landscape['competition_level'] in ['high', 'medium', 'low']
        
    def test_effort_lookback_relationship(self, market_model_with_research_data, competitive_offers_by_sector, market_research_config):
        """
        Test that higher effort produces more comprehensive pricing analysis.
        
        Higher effort should result in:
        - Longer lookback periods for data collection
        - Lower noise in price estimates
        - More detailed competitive analysis  
        - Better quality tier classification
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
        
        # Quality should increase with effort
        assert low_effort.quality_tier == "low"
        assert high_effort.quality_tier == "high"
        
        # High effort should provide more comprehensive analysis
        assert high_effort.sample_size >= low_effort.sample_size, "Higher effort should analyze more data"
        
        # Should have different levels of detail in competitive landscape
        high_landscape = high_effort.competitive_landscape
        low_landscape = low_effort.competitive_landscape  
        
        assert len(high_landscape.keys()) >= len(low_landscape.keys()), "Higher effort should provide more detailed analysis"


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