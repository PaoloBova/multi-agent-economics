"""
Comprehensive test suite for economic research tools.

Tests the actual simplified implementations with realistic scenarios and quantitative expectations.
No mocks - uses real market data and validates actual tool behavior.
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
from multi_agent_economics.tools.implementations.economic import (
    analyze_historical_performance_impl,
    analyze_buyer_preferences_impl, 
    research_competitive_pricing_impl,
    sector_forecast_impl
)
from multi_agent_economics.tools.schemas import (
    HistoricalPerformanceResponse,
    BuyerPreferenceResponse,
    CompetitivePricingResponse,
    SectorForecastResponse
)


@pytest.fixture
def market_state_with_data():
    """
    Create MarketState with comprehensive historical data for testing.
    Contains sector-specific trades, buyers with preferences, and competitive offers.
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
                0: RegimeParameters(mu=0.08, sigma=0.15),
                1: RegimeParameters(mu=0.02, sigma=0.25)
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
        knowledge_good_forecasts={
            # Map trade good_ids to their sectors
            "tech_forecast_1": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.8, 0.2]),
            "tech_forecast_2": ForecastData(sector="tech", predicted_regime=1, confidence_vector=[0.3, 0.7]),
            "tech_forecast_3": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.9, 0.1]),
            "tech_forecast_4": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.7, 0.3]),
            "tech_forecast_5": ForecastData(sector="tech", predicted_regime=1, confidence_vector=[0.4, 0.6]),
            "tech_forecast_6": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.85, 0.15]),
            "finance_forecast_1": ForecastData(sector="finance", predicted_regime=1, confidence_vector=[0.2, 0.8]),
            "finance_forecast_2": ForecastData(sector="finance", predicted_regime=0, confidence_vector=[0.6, 0.4]),
            "finance_forecast_3": ForecastData(sector="finance", predicted_regime=1, confidence_vector=[0.1, 0.9]),
            "healthcare_forecast_1": ForecastData(sector="healthcare", predicted_regime=0, confidence_vector=[0.9, 0.1]),
            "healthcare_forecast_2": ForecastData(sector="healthcare", predicted_regime=1, confidence_vector=[0.3, 0.7])
        },
        buyers_state=[],  # Will be populated by fixture
        # Marketing attribute definitions
        marketing_attribute_definitions={
            "innovation_level": {
                "type": "qualitative",
                "values": ["low", "medium", "high"], 
                "description": "Level of innovation in methodology"
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
        # Canonical attribute ordering
        attribute_order=["innovation_level", "data_source", "risk_score"],
        # Historical trade data
        all_trades=[
            # Tech sector trades - high prices for good forecasts
            TradeData(
                buyer_id="buyer_1", seller_id="seller_1", price=120.0, quantity=1, 
                good_id="tech_forecast_1", period=8,
                marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 25}
            ),
            TradeData(
                buyer_id="buyer_2", seller_id="seller_2", price=115.0, quantity=1,
                good_id="tech_forecast_2", period=9,
                marketing_attributes={"innovation_level": "high", "data_source": "external", "risk_score": 30}
            ),
            TradeData(
                buyer_id="buyer_3", seller_id="seller_3", price=85.0, quantity=1,
                good_id="tech_forecast_3", period=7,
                marketing_attributes={"innovation_level": "medium", "data_source": "internal", "risk_score": 45}
            ),
            TradeData(
                buyer_id="buyer_4", seller_id="seller_4", price=80.0, quantity=1,
                good_id="tech_forecast_4", period=6,
                marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 50}
            ),
            TradeData(
                buyer_id="buyer_5", seller_id="seller_5", price=50.0, quantity=1,
                good_id="tech_forecast_5", period=9,
                marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 75}
            ),
            TradeData(
                buyer_id="buyer_6", seller_id="seller_6", price=45.0, quantity=1,
                good_id="tech_forecast_6", period=8,
                marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 80}
            ),
            
            # Finance sector trades
            TradeData(
                buyer_id="buyer_7", seller_id="seller_7", price=95.0, quantity=1,
                good_id="finance_forecast_1", period=7,
                marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 20}
            ),
            TradeData(
                buyer_id="buyer_8", seller_id="seller_8", price=70.0, quantity=1,
                good_id="finance_forecast_2", period=8,
                marketing_attributes={"innovation_level": "low", "data_source": "external", "risk_score": 55}
            ),
            TradeData(
                buyer_id="buyer_9", seller_id="seller_9", price=40.0, quantity=1,
                good_id="finance_forecast_3", period=9,
                marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 85}
            ),
            
            # Healthcare sector trades
            TradeData(
                buyer_id="buyer_10", seller_id="seller_10", price=130.0, quantity=1,
                good_id="healthcare_forecast_1", period=6,
                marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 15}
            ),
            TradeData(
                buyer_id="buyer_11", seller_id="seller_11", price=90.0, quantity=1,
                good_id="healthcare_forecast_2", period=7,
                marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 40}
            ),
        ]
    )


@pytest.fixture  
def buyers_with_preferences(market_state_with_data):
    """
    Create buyers with sector-specific preferences for testing preference analysis.
    Tech buyers prefer innovation, finance buyers prefer data quality.
    """
    buyers = [
        # Tech sector buyers - innovation focused
        BuyerState(
            buyer_id="tech_buyer_1",
            regime_beliefs={"tech": [0.7, 0.3]},
            attr_mu={"tech": [0.9, 0.5, 0.6]},     # High innovation preference
            attr_sigma2={"tech": [0.1, 0.2, 0.15]},
            attr_weights={"tech": [0.9, 0.5, 0.6]},
            budget=200.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.9, "medium": 0.6, "low": 0.2},
                "data_source": {"proprietary": 0.8, "external": 0.5, "internal": 0.3},
                "risk_score": {"numeric_scaling": True, "base": 0.0, "scale": 0.01}
            }
        ),
        BuyerState(
            buyer_id="tech_buyer_2", 
            regime_beliefs={"tech": [0.6, 0.4]},
            attr_mu={"tech": [0.8, 0.6, 0.5]},     # High innovation, decent data
            attr_sigma2={"tech": [0.15, 0.15, 0.2]},
            attr_weights={"tech": [0.8, 0.6, 0.5]},
            budget=180.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.7, "medium": 0.7, "low": 0.3},
                "data_source": {"proprietary": 0.6, "external": 0.6, "internal": 0.4},
                "risk_score": {"numeric_scaling": True, "base": 0.0, "scale": 0.008}
            }
        ),
        
        # Finance sector buyers - data quality focused  
        BuyerState(
            buyer_id="finance_buyer_1",
            regime_beliefs={"finance": [0.4, 0.6]},
            attr_mu={"finance": [0.5, 0.9, 0.4]},  # High data preference
            attr_sigma2={"finance": [0.2, 0.1, 0.25]},
            attr_weights={"finance": [0.5, 0.9, 0.4]},
            budget=250.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.4, "medium": 0.6, "low": 0.8},
                "data_source": {"proprietary": 0.9, "external": 0.7, "internal": 0.3},
                "risk_score": {"numeric_scaling": True, "base": 0.0, "scale": 0.005}
            }
        ),
        BuyerState(
            buyer_id="finance_buyer_2",
            regime_beliefs={"finance": [0.3, 0.7]},
            attr_mu={"finance": [0.6, 0.8, 0.3]},  # Decent innovation, high data
            attr_sigma2={"finance": [0.18, 0.12, 0.2]},
            attr_weights={"finance": [0.6, 0.8, 0.3]}, 
            budget=220.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.5, "medium": 0.7, "low": 0.6},
                "data_source": {"proprietary": 1.0, "external": 0.5, "internal": 0.2},
                "risk_score": {"numeric_scaling": True, "base": 0.0, "scale": 0.003}
            }
        ),
        
        # Healthcare buyer - balanced preferences
        BuyerState(
            buyer_id="healthcare_buyer_1",
            regime_beliefs={"healthcare": [0.5, 0.5]},
            attr_mu={"healthcare": [0.7, 0.7, 0.6]},  # Balanced
            attr_sigma2={"healthcare": [0.15, 0.15, 0.18]},
            attr_weights={"healthcare": [0.7, 0.7, 0.6]},
            budget=300.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.7, "medium": 0.7, "low": 0.4},
                "data_source": {"proprietary": 0.7, "external": 0.6, "internal": 0.5},
                "risk_score": {"numeric_scaling": True, "base": 0.0, "scale": 0.007}
            }
        )
    ]
    market_state_with_data.buyers_state = buyers
    return buyers


@pytest.fixture
def competitive_offers(market_state_with_data):
    """Create current competitive offers across sectors."""
    offers = [
        # Tech sector competitive offers
        Offer(
            good_id="tech_comp_1", price=100.0, seller="competitor_1",
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 35}
        ),
        Offer(
            good_id="tech_comp_2", price=110.0, seller="competitor_2",
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 20}
        ),
        Offer(
            good_id="tech_comp_3", price=95.0, seller="competitor_3",
            marketing_attributes={"innovation_level": "medium", "data_source": "internal", "risk_score": 40}
        ),
        
        # Finance sector competitive offers
        Offer(
            good_id="finance_comp_1", price=150.0, seller="fin_competitor_1",
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 15}
        ),
        Offer(
            good_id="finance_comp_2", price=145.0, seller="fin_competitor_2",
            marketing_attributes={"innovation_level": "low", "data_source": "external", "risk_score": 25}
        ),
    ]
    
    # Add competitive forecasts to knowledge base
    market_state_with_data.knowledge_good_forecasts.update({
        "tech_comp_1": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.6, 0.4]),
        "tech_comp_2": ForecastData(sector="tech", predicted_regime=0, confidence_vector=[0.8, 0.2]),
        "tech_comp_3": ForecastData(sector="tech", predicted_regime=1, confidence_vector=[0.3, 0.7]),
        "finance_comp_1": ForecastData(sector="finance", predicted_regime=1, confidence_vector=[0.2, 0.8]),
        "finance_comp_2": ForecastData(sector="finance", predicted_regime=0, confidence_vector=[0.7, 0.3]),
    })
    
    market_state_with_data.offers.extend(offers)
    return offers


@pytest.fixture
def market_model(market_state_with_data):
    """Create MarketModel instance with test data."""
    return MarketModel(
        id=1,
        name="test_market", 
        agents=[],
        state=market_state_with_data,
        step=lambda: None,
        collect_stats=lambda: {}
    )


@pytest.fixture
def tool_config():
    """Configuration for economic research tools."""
    return {
        "tool_parameters": {
            "analyze_historical_performance": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5}
            },
            "analyze_buyer_preferences": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5}
            },
            "research_competitive_pricing": {
                "effort_thresholds": {"high": 2.5, "medium": 1.2}
            },
            "sector_forecast": {
                "effort_thresholds": {"high": 5.0, "medium": 2.0}
            }
        },
        "seller_id": "test_seller",
    }


class TestSectorForecast:
    """Test sector_forecast_impl with different effort levels and regime scenarios."""
    
    def test_sector_forecast_interface(self, market_model, tool_config):
        """Test sector forecast returns correct response format."""
        result = sector_forecast_impl(
            market_model, 
            tool_config, 
            sector="tech", 
            horizon=1, 
            effort=3.0
        )
        
        assert isinstance(result, SectorForecastResponse)
        assert result.sector == "tech"
        assert result.effort_used == 3.0
        assert result.quality_tier in ["low", "medium", "high"]
        assert hasattr(result, 'forecast')
        assert result.forecast.sector == "tech"
        
    def test_effort_quality_mapping(self, market_model, tool_config):
        """Test effort maps to correct quality tiers."""
        low_result = sector_forecast_impl(market_model, tool_config, "tech", 1, 1.0)
        medium_result = sector_forecast_impl(market_model, tool_config, "tech", 1, 3.0) 
        high_result = sector_forecast_impl(market_model, tool_config, "tech", 1, 6.0)
        
        assert low_result.quality_tier == "low"
        assert medium_result.quality_tier == "medium"
        assert high_result.quality_tier == "high"
        
        # Higher effort should produce more accurate forecasts (higher confidence)
        assert np.max(high_result.forecast.confidence_vector) >= np.max(low_result.forecast.confidence_vector)
        
    def test_sector_forecasts_quantitative(self, market_model, tool_config):
        """Test forecasts produce expected quantitative results."""
        result = sector_forecast_impl(
            market_model, 
            tool_config, 
            sector="tech", 
            horizon=3, 
            effort=4.0
        )
        
        assert result.forecast.sector == "tech"
        assert len(result.forecast.confidence_vector) == 2
        # Confidence vector should have two values based on two regimes. W
        # can compute this based on the regime parameters defined in the market state.
        # The exact calculation will be as follows:
        # Given medium effort, the quality mapping and base forecast rate gives
        # us a confidence vector of [base * quality_mapping["medium"], (1 - base * quality_mapping["medium"])]
        tool_params = tool_config["tool_parameters"]["sector_forecast"]
        base_quality = tool_params.get("base_forecast_quality", 0.6)
        effort_level_quality_mapping = tool_params.get("effort_level_quality_mapping", {
            "high": 0.9, "medium": 0.7, "low": 0.5
        })
        quality_tier = "medium" # Based on effort 4.0, which is medium
        forecast_quality = effort_level_quality_mapping.get(quality_tier, 0.5)
        true_likelihood = base_quality + (1 - base_quality) * forecast_quality
        
        # Which row of our confidence matrix we get corresponds to the true next
        # regime, which is precomputed in the market state.
        true_next_regime = market_model.state.regime_history[market_model.state.current_period + 1].regimes["tech"]
        expected_confidence = [true_likelihood, (1 - true_likelihood)] if true_next_regime == 0 else [(1 - true_likelihood), true_likelihood]
        assert np.allclose(result.forecast.confidence_vector, expected_confidence)

class TestHistoricalPerformanceAnalysis:
    """Test analyze_historical_performance_impl with actual trade data filtering."""
    
    def test_historical_analysis_interface(self, market_model, tool_config):
        """Test historical performance analysis returns correct format."""
        
        result = analyze_historical_performance_impl(
            market_model, tool_config, sector="tech", effort=2.0
        )
        
        assert isinstance(result, HistoricalPerformanceResponse)
        assert result.sector == "tech"
        assert result.effort_used == 2.0
        assert result.quality_tier in ["low", "medium", "high"]
        assert isinstance(result.trade_data, list)
        assert result.sample_size >= 0
        assert isinstance(result.recommendation, str)
        
    def test_sector_filtering(self, market_model, tool_config):
        """Test that analysis filters trades by sector correctly."""
        
        tech_result = analyze_historical_performance_impl(
            market_model, tool_config, sector="tech", effort=3.0
        )
        finance_result = analyze_historical_performance_impl(
            market_model, tool_config, sector="finance", effort=3.0
        )
        
        # Tech should find 6 trades, finance should find 3 trades
        assert tech_result.sample_size == 6, f"Expected 6 tech trades, got {tech_result.sample_size}"
        assert finance_result.sample_size == 3, f"Expected 3 finance trades, got {finance_result.sample_size}"
        
        # Verify trades are sector-specific
        for trade in tech_result.trade_data:
            assert "tech" in trade["good_id"], f"Trade {trade['good_id']} should be from tech sector"
            
        for trade in finance_result.trade_data:
            assert "finance" in trade["good_id"], f"Trade {trade['good_id']} should be from finance sector"
            
    def test_effort_quality_relationship(self, market_model, tool_config):
        """Test higher effort produces better quality analysis."""
        
        low_effort = analyze_historical_performance_impl(
            market_model, tool_config, sector="tech", effort=1.0
        )
        high_effort = analyze_historical_performance_impl(
            market_model, tool_config, sector="tech", effort=4.0
        )
        
        assert low_effort.quality_tier == "low"
        assert high_effort.quality_tier == "high"
        
        # Same sample size (all available data) but different noise levels
        assert low_effort.sample_size == high_effort.sample_size == 6
        
        # High effort should have fewer warnings
        assert len(high_effort.warnings) <= len(low_effort.warnings)
        
    def test_price_data_with_noise(self, market_model, tool_config):
        """Test that price data includes appropriate noise based on effort."""
        
        # Get original prices from test data
        original_tech_prices = [120.0, 115.0, 85.0, 80.0, 50.0, 45.0]
        
        result = analyze_historical_performance_impl(
            market_model, tool_config, sector="tech", effort=3.0  # High effort, low noise
        )
        
        assert len(result.trade_data) == 6
        
        # Prices should be close to originals but with some noise
        returned_prices = [trade["price"] for trade in result.trade_data]
        
        for orig, noisy in zip(original_tech_prices, returned_prices):
            # Allow for 20% noise variation (generous tolerance)
            assert abs(noisy - orig) <= orig * 0.2, f"Noisy price {noisy} too far from original {orig}"


class TestBuyerPreferenceAnalysis:
    """Test analyze_buyer_preferences_impl with real buyer sampling and WTP calculations."""
    
    def test_buyer_preference_interface(self, market_model, buyers_with_preferences, tool_config):
        """Test buyer preference analysis returns correct format.""" 
        result = analyze_buyer_preferences_impl(
            market_model, tool_config, sector="tech", effort=2.0
        )
        
        assert isinstance(result, BuyerPreferenceResponse)
        assert result.sector == "tech"
        assert result.effort_used == 2.0
        assert result.quality_tier in ["low", "medium", "high"]
        assert isinstance(result.top_valued_attributes, list)
        assert result.sample_size >= 0
        assert isinstance(result.recommendation, str)
        
    def test_sector_specific_preferences(self, market_model, buyers_with_preferences, tool_config):
        """Test preferences analysis is sector-specific."""
        tech_result = analyze_buyer_preferences_impl(
            market_model, tool_config, sector="tech", effort=3.0
        )
        finance_result = analyze_buyer_preferences_impl(
            market_model, tool_config, sector="finance", effort=3.0
        )
        
        # Tech has 2 buyers, finance has 2 buyers
        assert tech_result.sample_size == 2
        assert finance_result.sample_size == 2
        
        # Should find different preference patterns
        assert tech_result.sector == "tech"
        assert finance_result.sector == "finance"
        
        # Should have analysis results
        if tech_result.top_valued_attributes and finance_result.top_valued_attributes:
            # Results should be different for different sectors
            tech_attrs = [attr["attribute"] for attr in tech_result.top_valued_attributes]
            finance_attrs = [attr["attribute"] for attr in finance_result.top_valued_attributes]
            
            # At least some differences expected due to different buyer preferences
            assert tech_attrs != finance_attrs or tech_result.top_valued_attributes != finance_result.top_valued_attributes
            
    def test_wtp_calculation_logic(self, market_model, buyers_with_preferences, tool_config):
        """Test that WTP calculations work correctly with buyer preferences."""
        result = analyze_buyer_preferences_impl(
            market_model, tool_config, sector="tech", effort=3.0
        )
        
        # Should generate test offers and calculate WTP
        if result.top_valued_attributes:
            assert len(result.top_valued_attributes) <= 3  # Top 3 attributes
            
            for attr_info in result.top_valued_attributes:
                assert "attribute" in attr_info
                assert "average_wtp" in attr_info
                assert "importance" in attr_info
                
                # WTP should be positive for valued attributes
                assert attr_info["average_wtp"] >= 0
                assert attr_info["importance"] >= 0
                
    def test_effort_affects_analysis_detail(self, market_model, buyers_with_preferences, tool_config):
        """Test effort level affects analysis quality and detail."""
        low_result = analyze_buyer_preferences_impl(
            market_model, tool_config, sector="tech", effort=1.0
        )
        high_result = analyze_buyer_preferences_impl(
            market_model, tool_config, sector="tech", effort=4.0
        )
        
        assert low_result.quality_tier == "low"
        assert high_result.quality_tier == "high"
        
        # Both sample same buyers but with different analysis depth
        assert low_result.sample_size == high_result.sample_size == 2
        
        # High effort should provide more detailed analysis
        if high_result.top_valued_attributes:
            # High effort analyzes by attribute, low effort might not
            assert len(high_result.top_valued_attributes) >= len(low_result.top_valued_attributes)


class TestCompetitivePricingResearch:
    """Test research_competitive_pricing_impl with market simulation scenarios."""
    
    def test_competitive_pricing_interface(self, market_model, competitive_offers, tool_config):
        """Test competitive pricing research returns correct format."""
        result = research_competitive_pricing_impl(
            market_model, tool_config, sector="tech", effort=2.0
        )
        
        assert isinstance(result, CompetitivePricingResponse)
        assert result.sector == "tech"
        assert result.effort_used == 2.0
        assert result.quality_tier in ["low", "medium", "high"]
        assert isinstance(result.price_simulations, list)
        assert result.recommended_price >= 0
        assert isinstance(result.recommendation, str)
        
    def test_competitive_landscape_analysis(self, market_model, competitive_offers, buyers_with_preferences, tool_config):
        """Test analysis of competitive landscape by sector."""
        tech_result = research_competitive_pricing_impl(
            market_model, tool_config, sector="tech", effort=3.0
        )
        finance_result = research_competitive_pricing_impl(
            market_model, tool_config, sector="finance", effort=3.0
        )
        
        # Tech has 3 current offers + 6 historical trades = 9 total
        # Finance has 2 current offers + 3 historical trades = 5 total  
        assert tech_result.sector == "tech"
        assert finance_result.sector == "finance"
        
        # Should have different pricing recommendations for different sectors
        assert tech_result.recommended_price != finance_result.recommended_price
        
        # Should have price simulation data
        assert len(tech_result.price_simulations) > 0
        assert len(finance_result.price_simulations) > 0
        
    def test_market_simulation_logic(self, market_model, buyers_with_preferences, competitive_offers, tool_config):
        """Test market share simulation with actual choice model."""
        result = research_competitive_pricing_impl(
            market_model, tool_config, sector="tech", effort=3.0
        )
        
        # Should have price simulations with market share data
        assert len(result.price_simulations) > 0
        
        for simulation in result.price_simulations:
            assert "price" in simulation
            assert "market_share" in simulation
            assert "capture_rate" in simulation
            assert "buyers_purchasing" in simulation
            assert "expected_revenue" in simulation
            assert "competitive_position" in simulation
            
            # Market shares should be reasonable
            assert 0 <= simulation["market_share"] <= 1
            assert 0 <= simulation["capture_rate"] <= 1
            assert simulation["buyers_purchasing"] >= 0
            assert simulation["expected_revenue"] >= 0
            
        # Should recommend price with highest expected revenue
        best_sim = max(result.price_simulations, key=lambda x: x["expected_revenue"])
        assert abs(result.recommended_price - best_sim["price"]) < 0.01
        
    def test_no_competition_scenario(self, market_model, tool_config):
        """Test behavior when no competitive data is available."""
        # Clear all offers and trades to test no competition case
        market_model.state.offers = []
        market_model.state.all_trades = []
        
        result = research_competitive_pricing_impl(
            market_model, tool_config, sector="tech", effort=2.0
        )
        
        # Should handle gracefully
        assert result.recommended_price == 0.0  # No info available
        assert "No competitive data available" in result.recommendation
        assert len(result.warnings) > 0
        assert "No competitive activity found" in result.warnings[0]
        
    def test_effort_affects_simulation_quality(self, market_model, competitive_offers, buyers_with_preferences, tool_config):
        """Test effort level affects simulation comprehensiveness."""
        low_result = research_competitive_pricing_impl(
            market_model, tool_config, sector="tech", effort=1.0
        )
        high_result = research_competitive_pricing_impl(
            market_model, tool_config, sector="tech", effort=3.0
        )
        
        assert low_result.quality_tier == "low"
        assert high_result.quality_tier == "high"
        
        # High effort should test more price points
        assert len(high_result.price_simulations) >= len(low_result.price_simulations)
        
        # High effort may have more warnings due to testing more price points, but should have better quality analysis
        # Focus on non-error warnings (data quality warnings)
        high_data_warnings = [w for w in high_result.warnings if "Choice model simulation error" not in w]
        low_data_warnings = [w for w in low_result.warnings if "Choice model simulation error" not in w]
        
        # Data quality warnings should be same or fewer for high effort
        assert len(high_data_warnings) <= len(low_data_warnings)


class TestMarketResearchIntegration:
    """Test integration between tools and market framework."""
    
    def test_empty_market_handling(self, tool_config):
        """Test tools handle empty market data gracefully."""
        empty_state = MarketState(
            offers=[], trades=[], demand_profile={}, supply_profile={},
            index_values={"tech": 100.0}, current_regimes={"tech": 0},
            current_period=0, knowledge_good_forecasts={}, buyers_state=[], all_trades=[]
        )
        empty_model = MarketModel(
            id=1, 
            name="empty", 
            agents=[], 
            state=empty_state, 
            step=lambda: None, 
            collect_stats=lambda: {}
        )
        
        hist_result = analyze_historical_performance_impl(empty_model, tool_config, "tech", 2.0)
        pref_result = analyze_buyer_preferences_impl(empty_model, tool_config, "tech", 2.0)
        price_result = research_competitive_pricing_impl(empty_model, tool_config, "tech", 2.0)
        
        # Should return valid responses with appropriate warnings
        assert isinstance(hist_result, HistoricalPerformanceResponse)
        assert isinstance(pref_result, BuyerPreferenceResponse)
        assert isinstance(price_result, CompetitivePricingResponse)
        
        assert hist_result.sample_size == 0
        assert pref_result.sample_size == 0
        assert price_result.recommended_price == 0.0
        
        assert "No historical trade data available" in hist_result.warnings
        assert "No buyers available for analysis" in pref_result.warnings
        assert any("No competitive activity found" in warning for warning in price_result.warnings)
        
    def test_consistent_effort_mapping(self, market_model, buyers_with_preferences, competitive_offers, tool_config):
        """Test all tools use consistent effort-to-quality mapping."""
        effort = 2.5  # Should be medium for all tools
        
        hist_result = analyze_historical_performance_impl(market_model, tool_config, "tech", effort)
        pref_result = analyze_buyer_preferences_impl(market_model, tool_config, "tech", effort)
        price_result = research_competitive_pricing_impl(market_model, tool_config, "tech", effort)
        
        # All should have same quality tier for same effort
        assert hist_result.quality_tier == "medium"
        assert pref_result.quality_tier == "medium" 
        assert price_result.quality_tier == "high"  # Different threshold for pricing tool
        
        # All should track same effort used
        assert hist_result.effort_used == effort
        assert pref_result.effort_used == effort
        assert price_result.effort_used == effort
        
    def test_sector_consistency_across_tools(self, market_model, buyers_with_preferences, competitive_offers, tool_config):
        """Test all tools respect sector boundaries consistently."""
        
        # Test tech sector across all tools
        hist_tech = analyze_historical_performance_impl(market_model, tool_config, "tech", 3.0)
        pref_tech = analyze_buyer_preferences_impl(market_model, tool_config, "tech", 3.0)
        price_tech = research_competitive_pricing_impl(market_model, tool_config, "tech", 3.0)
        
        # All should analyze tech sector consistently
        assert hist_tech.sector == "tech"
        assert pref_tech.sector == "tech"
        assert price_tech.sector == "tech"
        
        # Should find expected amounts of tech-specific data
        assert hist_tech.sample_size == 6  # 6 tech trades
        assert pref_tech.sample_size == 2   # 2 tech buyers
        # price_tech analyzes competitive data which varies
        
    def test_realistic_value_ranges(self, market_model, buyers_with_preferences, competitive_offers, tool_config):
        """Test all tools return realistic values for economic scenarios."""
        
        hist_result = analyze_historical_performance_impl(market_model, tool_config, "tech", 3.0)
        pref_result = analyze_buyer_preferences_impl(market_model, tool_config, "tech", 3.0)
        price_result = research_competitive_pricing_impl(market_model, tool_config, "tech", 3.0)
        
        # Historical analysis - prices should be in realistic ranges
        if hist_result.trade_data:
            prices = [trade["price"] for trade in hist_result.trade_data]
            assert all(10 <= price <= 200 for price in prices), f"Unrealistic prices: {prices}"
            
        # Buyer preferences - WTP should be reasonable
        if pref_result.top_valued_attributes:
            wtps = [attr["average_wtp"] for attr in pref_result.top_valued_attributes]
            assert all(wtp >= 0 for wtp in wtps), f"Negative WTP values: {wtps}"
            
        # Competitive pricing - recommended price should be reasonable
        assert 0 <= price_result.recommended_price <= 300, f"Unrealistic recommended price: {price_result.recommended_price}"
        
        # Market shares should sum to reasonable totals
        if price_result.price_simulations:
            market_shares = [sim["market_share"] for sim in price_result.price_simulations]
            assert all(0 <= share <= 1 for share in market_shares), f"Invalid market shares: {market_shares}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
