"""
TDD test suite for AkerlofSeller behavior demonstrating adverse selection dynamics.

These tests will FAIL initially because AkerlofSeller doesn't exist yet.
The tests define the exact interface and behavior we need to implement.

Economic Foundation:
The tests validate Akerlof's "Market for Lemons" mechanism where information
asymmetry between sellers (who know true quality) and buyers (who only observe
attributes) leads to adverse selection and market failure.

Two-stage seller decision process:
1. Stage 1: Effort allocation (production quality choice with sunk costs)
2. Stage 2: Marketing (attribute claims + pricing for already-produced goods)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, Offer, ForecastData, TradeData, RegimeParameters,
    BuyerState, SellerState
)

# This import will fail initially - that's the point of TDD!
from akerlof_seller import AkerlofSeller


@pytest.fixture
def market_state():
    """
    Create realistic MarketState instance for testing seller decisions.
    
    Market setup represents single tech sector with regime switching:
    - Bull market (regime 0): 8% expected return, 15% volatility  
    - Bear market (regime 1): 2% expected return, 25% volatility
    - Current state: Bull market, index at 105 (5% gain)
    - Next period: Bear market transition (for forecast testing)
    """
    return MarketState(
        offers=[],
        trades=[],
        demand_profile={},
        supply_profile={},
        index_values={"tech": 105.0},
        current_regimes={"tech": 0},  # Currently bull market
        regime_parameters={
            "tech": {
                0: RegimeParameters(mu=0.08, sigma=0.15),  # Bull market params
                1: RegimeParameters(mu=0.02, sigma=0.25)   # Bear market params
            }
        },
        regime_history=[
            {
                "period": 0,
                "returns": {"tech": 0.05},     # Bull market realization
                "regimes": {"tech": 0},
                "index_values": {"tech": 105.0}
            },
            {
                "period": 1,
                "returns": {"tech": -0.02},    # Bear market transition
                "regimes": {"tech": 1},
                "index_values": {"tech": 102.9}
            }
        ],
        current_period=0,
        knowledge_good_forecasts={},
        buyers_state=[],
        # Marketing attribute system for testing
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
            }
        },
        attribute_order=["innovation_level", "data_source"]
    )


@pytest.fixture
def buyers_with_preferences(market_state):
    """
    Create buyer agents with known attribute preferences for predictable testing.
    
    Buyer preference structure (attr_mu):
    - Buyer 1: [0.8, 0.6] = High methodology preference (0.8), moderate data quality (0.6)
    - Buyer 2: [0.4, 0.5] = Lower methodology preference (0.4), moderate data quality (0.5)
    
    This heterogeneity creates realistic market demand with different WTP levels:
    - Quality-sensitive buyer 1 will pay more for high attributes
    - Price-sensitive buyer 2 provides lower bound on pricing
    
    Expected willingness-to-pay calculations:
    For premium attributes [0.9, 0.8]:
    - Buyer 1: 0.8 × 0.9 + 0.6 × 0.8 = 0.72 + 0.48 = 1.20
    - Buyer 2: 0.4 × 0.9 + 0.5 × 0.8 = 0.36 + 0.40 = 0.76
    - Average market WTP: (1.20 + 0.76) / 2 = 0.98
    """
    buyers = [
        BuyerState(
            buyer_id="buyer_1",
            regime_beliefs={"tech": [0.6, 0.4]},  # Moderately bullish
            attr_mu={"tech": [0.8, 0.6]},                   # High quality preference
            attr_sigma2={"tech": [0.1, 0.1]},               # Low uncertainty
            attr_weights={"tech": [0.8, 0.6]},              # Current weights match means
            budget=200.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.9, "medium": 0.6, "low": 0.2},
                "data_source": {"proprietary": 0.8, "external": 0.5, "internal": 0.3}
            }
        ),
        BuyerState(
            buyer_id="buyer_2",
            regime_beliefs={"tech": [0.3, 0.7]},  # Bearish
            attr_mu={"tech": [0.4, 0.5]},                   # Lower quality preference  
            attr_sigma2={"tech": [0.2, 0.15]},              # Higher uncertainty
            attr_weights={"tech": [0.4, 0.5]},              # Current weights match means
            budget=150.0,
            buyer_conversion_function={
                "innovation_level": {"high": 0.4, "medium": 0.5, "low": 0.3},
                "data_source": {"proprietary": 0.5, "external": 0.6, "internal": 0.4}
            }
        )
    ]
    market_state.buyers_state = buyers
    return buyers


@pytest.fixture
def historical_trades():
    """
    Create historical trading data showing clear price-performance relationships.
    
    This data enables sellers to learn the relationship between true forecast
    quality and market prices achieved. The pattern shows:
    
    - High performance forecasts (0.85 accuracy): commanded prices [120, 115] → avg 117.5
    - Medium performance forecasts (0.70 accuracy): commanded prices [85, 80] → avg 82.5  
    - Low performance forecasts (0.55 accuracy): commanded prices [50, 45] → avg 47.5
    
    This creates clear profit incentives for different effort levels:
    - High effort (cost 100): 117.5 - 100 = 17.5 profit
    - Medium effort (cost 60): 82.5 - 60 = 22.5 profit ← maximum
    - Low effort (cost 30): 47.5 - 30 = 17.5 profit
    
    The historical data demonstrates that medium effort maximizes profit in this
    market environment, which sellers should learn through analysis.
    """
    return [
        # High performance trades (0.85 accuracy achieved high prices)
        TradeData(buyer_id="buyer_1", seller_id="seller_1", price=120, quantity=1, good_id="high_perf_1"),
        TradeData(buyer_id="buyer_2", seller_id="seller_1", price=115, quantity=1, good_id="high_perf_2"),
        
        # Medium performance trades (0.70 accuracy achieved medium prices)
        TradeData(buyer_id="buyer_3", seller_id="seller_2", price=85, quantity=1, good_id="med_perf_1"),
        TradeData(buyer_id="buyer_4", seller_id="seller_2", price=80, quantity=1, good_id="med_perf_2"),
        
        # Low performance trades (0.55 accuracy achieved low prices)
        TradeData(buyer_id="buyer_5", seller_id="seller_3", price=50, quantity=1, good_id="low_perf_1"),
        TradeData(buyer_id="buyer_6", seller_id="seller_3", price=45, quantity=1, good_id="low_perf_2")
    ]


@pytest.fixture
def market_model(market_state):
    """
    Create MarketModel instance wrapping the market state for AkerlofSeller.
    """
    def dummy_step():
        """Dummy step function for testing."""
        pass
    
    def dummy_collect_stats():
        """Dummy collect stats function for testing."""
        return {}
    
    return MarketModel(
        id=1,
        name="test_market",
        agents=[],
        state=market_state,
        step=dummy_step,
        collect_stats=dummy_collect_stats
    )


@pytest.fixture
def context_config():
    """
    Create context configuration data that sector_forecast_impl requires.
    This includes tool parameters and market data needed for economic tools.
    """
    return {
        "tool_parameters": {
            "sector_forecast": {
                "effort_thresholds": {"high": 5.0, "medium": 2.0},
                "quality_tiers": {
                    "high": {"noise_factor": 0.1, "confidence": 0.9},
                    "medium": {"noise_factor": 0.3, "confidence": 0.7},
                    "low": {"noise_factor": 0.6, "confidence": 0.5}
                }
            }
        },
        "market_data": {
            "sectors": {
                "tech": {"mean": 0.10, "std": 0.20},
                "finance": {"mean": 0.08, "std": 0.15},
                "healthcare": {"mean": 0.12, "std": 0.25}
            }
        },
        "sector": "tech",
        "horizon": 1,
        # AkerlofSeller specific configuration
        "effort_costs": {
            'high': 100.0,    # Extensive research, premium data, sophisticated models
            'medium': 60.0,   # Standard research, decent data, basic models  
            'low': 30.0       # Minimal research, public data, simple heuristics
        },
        "marketing_costs": {
            'premium_claims': 10.0,   # Professional presentation, detailed methodology
            'standard_claims': 5.0,   # Basic presentation, standard methodology
            'basic_claims': 2.0       # Simple presentation, minimal methodology
        },
        "effort_quality_mapping": {
            'high': 0.85,     # High effort produces high-accuracy forecasts
            'medium': 0.70,   # Medium effort produces medium-accuracy forecasts
            'low': 0.55       # Low effort produces low-accuracy forecasts
        },
        "marketing_strategies": {
            'premium_claims': [0.9, 0.8],    # High claims, high marketing cost (10)
            'standard_claims': [0.7, 0.6],   # Medium claims, medium marketing cost (5)
            'basic_claims': [0.5, 0.4]       # Low claims, low marketing cost (2)
        },
        "performance_tolerance": 0.05,
        "default_buyer_preferences": [0.5, 0.5],
        "default_wtp": 0.5,
        "wtp_pricing_factor": 0.9,
        "wtp_scaling_factor": 100.0,
        "similarity_threshold": 0.7,
        "competitor_discount": 0.95,
        "default_strategy": [0.7, 0.6],
        "effort_numeric_mapping": {
            'high': 6.0,    # Above high threshold (5.0)
            'medium': 3.0,  # Above medium threshold (2.0) but below high
            'low': 1.0      # Below medium threshold
        }
    }


class TestAkerlofSellerStage1:
    """
    Test Stage 1: Effort allocation (production decision with sunk costs).
    
    Stage 1 represents the core production decision where sellers choose how much
    effort/resources to invest in creating their knowledge good. This decision
    involves sunk costs that cannot be recovered regardless of marketing success.
    
    The economic logic:
    1. Sellers have different effort levels available (high/medium/low)  
    2. Higher effort produces higher quality forecasts but at higher cost
    3. Sellers must predict future marketing revenue to make optimal effort choice
    4. Profit maximization: choose effort with max(expected_revenue - effort_cost)
    """
    
    def test_seller_instantiation_with_cost_structure(self, market_model, context_config):
        """
        Test AkerlofSeller instantiation with required cost structure.
        
        The seller must have realistic cost structures that create economic
        trade-offs between quality levels. Cost structure represents:
        - High effort: Extensive research, premium data, sophisticated models (cost=100)
        - Medium effort: Standard research, decent data, basic models (cost=60)  
        - Low effort: Minimal research, public data, simple heuristics (cost=30)
        
        Marketing costs represent expense of claiming different attribute levels:
        - Premium claims: Professional presentation, detailed methodology (cost=10)
        - Standard claims: Basic presentation, standard methodology (cost=5)
        - Basic claims: Simple presentation, minimal methodology (cost=2)
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Basic instantiation
        assert seller.seller_id == "test_seller"
        assert seller.market_model == market_model
        
        # Required cost structures for economic realism
        assert hasattr(seller, 'effort_costs')
        assert hasattr(seller, 'marketing_costs')
        
        # Effort costs should create meaningful trade-offs
        assert seller.effort_costs['high'] == 100    # High quality, high cost
        assert seller.effort_costs['medium'] == 60   # Medium quality, medium cost
        assert seller.effort_costs['low'] == 30      # Low quality, low cost
        
        # Marketing costs should vary by claim level
        assert seller.marketing_costs['premium_claims'] == 10   # Expensive to claim premium
        assert seller.marketing_costs['standard_claims'] == 5   # Moderate to claim standard  
        assert seller.marketing_costs['basic_claims'] == 2      # Cheap to claim basic
    
    def test_stage1_effort_decision_interface(self, market_model, context_config, historical_trades):
        """
        Test stage1_effort_decision returns correct format for two-stage process.
        
        The effort decision must return both the chosen effort level and the
        generated forecast information. This supports the two-stage structure:
        
        Stage 1 output: (effort_level, forecast_info)
        - effort_level: string indicating chosen production approach
        - forecast_info: tuple of (forecast_id, forecast_data) for stage 2 marketing
        
        The forecast must be immediately generated and stored in market state
        so buyers can access it after marketing decisions are made.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        result = seller.stage1_effort_decision(historical_trades, round_num=1)
        
        # Must return tuple for two-stage structure
        assert isinstance(result, tuple), "Stage 1 must return (effort_level, forecast_info) tuple"
        assert len(result) == 2, "Must return exactly effort level and forecast info"
        
        effort_level, forecast_info = result
        
        # Effort level must be valid choice
        assert effort_level in ['high', 'medium', 'low'], f"Invalid effort level: {effort_level}"
        
        # Forecast info must be properly structured
        assert isinstance(forecast_info, tuple), "Forecast info must be (forecast_id, forecast_data) tuple"
        assert len(forecast_info) == 2, "Forecast info must have ID and data"
        
        forecast_id, forecast_data = forecast_info
        assert isinstance(forecast_id, str), "Forecast ID must be string"
        assert forecast_id.startswith("test_seller"), "Forecast ID must include seller identifier"
        assert isinstance(forecast_data, ForecastData), "Must return proper ForecastData object"
        assert forecast_data.sector == "tech", "Forecast must be for correct sector"
    
    def test_profit_maximizing_effort_choice_calculation(self, market_model, context_config, historical_trades):
        """
        Test seller chooses effort level maximizing expected profit through analysis.
        
        Economic calculation that seller should perform:
        
        1. Analyze historical data to estimate revenue by performance level:
           - High performance (0.85): prices [120, 115] → average revenue 117.5
           - Medium performance (0.70): prices [85, 80] → average revenue 82.5
           - Low performance (0.55): prices [50, 45] → average revenue 47.5
        
        2. Calculate expected profit for each effort level:
           - High effort: 117.5 - 100 = 17.5 profit
           - Medium effort: 82.5 - 60 = 22.5 profit ← maximum
           - Low effort: 47.5 - 30 = 17.5 profit
        
        3. Choose medium effort (highest expected profit)
        
        This tests the core economic logic of profit maximization under uncertainty.
        Seller must learn market patterns and respond rationally to incentives.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Provide performance data so seller can analyze price-performance relationships
        performance_mapping = {
            "high_perf_1": 0.85, "high_perf_2": 0.85,  # High performance achieved high prices
            "med_perf_1": 0.70, "med_perf_2": 0.70,    # Medium performance achieved medium prices
            "low_perf_1": 0.55, "low_perf_2": 0.55     # Low performance achieved low prices
        }
        seller.add_performance_data(historical_trades, performance_mapping)
        
        effort_level, _ = seller.stage1_effort_decision(historical_trades, round_num=1)
        
        # Should choose medium effort (profit = 22.5, highest of the three options)
        assert effort_level == 'medium', \
            f"Expected 'medium' effort (profit 22.5), got '{effort_level}'. " \
            f"Check profit calculations: high=17.5, medium=22.5, low=17.5"

    def test_effort_choice_minimizing_losses_under_market_crash(self, market_model, context_config):
        """
        Test seller minimizes losses when all effort levels are unprofitable.
        
        Market crash scenario represents adverse market conditions where even
        high-quality forecasts command low prices. Economic calculation:
        
        Crash scenario prices vs production costs:
        - High effort: revenue 70, cost 100 → loss -30
        - Medium effort: revenue 45, cost 60 → loss -15  
        - Low effort: revenue 25, cost 30 → loss -5 ← minimum loss
        
        Rational seller behavior: choose low effort to minimize losses when
        market cannot support quality premiums. This represents the economic
        pressure that drives quality degradation during market stress.
        
        This scenario tests seller response to unprofitable market conditions,
        which is crucial for modeling market downturns and adverse selection spirals.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Create market crash scenario with prices below all production costs
        crash_trades = [
            TradeData(buyer_id="buyer_1", seller_id="seller_1", price=70, quantity=1, good_id="crash_high"),
            TradeData(buyer_id="buyer_2", seller_id="seller_2", price=45, quantity=1, good_id="crash_med"),
            TradeData(buyer_id="buyer_3", seller_id="seller_3", price=25, quantity=1, good_id="crash_low")
        ]
        
        # Map trades to performance levels (showing even good forecasts get low prices)
        crash_performance = {
            "crash_high": 0.85,  # High performance but only got price 70 (vs cost 100)
            "crash_med": 0.70,   # Medium performance but only got price 45 (vs cost 60)
            "crash_low": 0.55    # Low performance but only got price 25 (vs cost 30)
        }
        seller.add_performance_data(crash_trades, crash_performance)
        
        effort_level, _ = seller.stage1_effort_decision(crash_trades, round_num=1)
        
        # Should choose low effort (loss = -5, smallest of -30, -15, -5)
        assert effort_level == 'low', \
            f"Expected 'low' effort (loss -5), got '{effort_level}'. " \
            f"Loss calculations: high=-30, medium=-15, low=-5"
    
    def test_forecast_generation_quality_mapping_integration(self, market_model, context_config):
        """
        Test forecast generation properly maps effort levels to forecast quality.
        
        Quality mapping represents the core production function:
        - High effort → 0.85 forecast accuracy (sophisticated models, premium data)
        - Medium effort → 0.70 forecast accuracy (standard models, decent data)
        - Low effort → 0.55 forecast accuracy (basic models, public data)
        
        This mapping must integrate with existing confusion matrix framework:
        For binary regimes with base_quality = 0.6:
        - Effective accuracy = 0.6 + (1 - 0.6) × forecast_quality
        - High effort: 0.6 + 0.4 × 0.85 = 0.94 diagonal probability
        - Medium effort: 0.6 + 0.4 × 0.70 = 0.88 diagonal probability
        - Low effort: 0.6 + 0.4 × 0.55 = 0.82 diagonal probability
        
        Integration requirements:
        1. Generated forecasts must be stored in market_state for buyer access
        2. ForecastData objects must be valid for existing market clearing
        3. Quality differences must be realistic but not deterministic
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Test forecast generation for all effort levels
        effort_to_expected_quality = {
            'high': 0.85,    # Premium effort should yield high accuracy
            'medium': 0.70,  # Standard effort should yield medium accuracy
            'low': 0.55      # Minimal effort should yield low accuracy
        }
        
        for effort, expected_quality in effort_to_expected_quality.items():
            forecast_id, forecast_data = seller.generate_forecast_with_effort(effort, round_num=1)
            
            # Forecast must be stored for buyer access
            assert forecast_id in market_model.state.knowledge_good_forecasts, \
                f"Forecast {forecast_id} not stored in market state"
            stored_forecast = market_model.state.knowledge_good_forecasts[forecast_id]
            assert stored_forecast == forecast_data, "Stored forecast must match returned forecast"
            
            # Forecast must have valid structure for existing framework
            assert forecast_data.sector == "tech", "Forecast must target correct sector"
            assert isinstance(forecast_data.predicted_regime, int), "Must predict specific regime"
            assert forecast_data.predicted_regime in [0, 1], "Must predict valid regime (0 or 1)"
            assert len(forecast_data.confidence_vector) == 2, "Must have confidence for both regimes"
            assert abs(sum(forecast_data.confidence_vector) - 1.0) < 1e-10, "Confidence must sum to 1.0"
            
            # Quality should influence confidence (higher effort → more confident predictions)
            max_confidence = max(forecast_data.confidence_vector)
            if effort == 'high':
                assert max_confidence >= 0.75, f"High effort should be confident, got {max_confidence}"
            elif effort == 'low':
                # Low effort should be less confident, but allow some randomness
                assert max_confidence <= 0.90, f"Low effort shouldn't be too confident, got {max_confidence}"


class TestAkerlofSellerStage2:
    """
    Test Stage 2: Marketing decision (attribute claims + competitive pricing).
    
    Stage 2 represents the marketing decision where sellers choose how to present
    their already-produced knowledge goods to maximize revenue. Key aspects:
    
    1. Attribute claims can be chosen independently of true quality (asymmetric info)
    2. Pricing must consider competitive landscape and buyer willingness-to-pay  
    3. Marketing costs vary by attribute claims made
    4. Revenue optimization: max(price - marketing_cost) subject to competition
    
    This stage creates the conditions for adverse selection: low-quality producers
    can mimic high-quality attributes if the profit incentives align.
    """
    
    def test_stage2_marketing_decision_interface(self, market_model, context_config, buyers_with_preferences):
        """
        Test stage2_marketing_decision creates and posts valid market offer.
        
        Marketing decision must transform the produced forecast into a market
        offering that buyers can evaluate and purchase. Interface requirements:
        
        Inputs: (effort_level, good_id, market_competition)
        - effort_level: production quality (for cost accounting)  
        - good_id: identifier of forecast to be marketed
        - market_competition: competing offers (for pricing decisions)
        
        Output: Offer object with chosen attributes and competitive price
        
        Side effects: Offer must be posted to market_state.offers for buyer access
        
        This tests the complete marketing pipeline from production to market availability.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Marketing inputs
        effort_level = 'medium'  # Moderate production cost
        good_id = 'test_forecast_123'  # Identifier of produced forecast
        market_competition = []  # No existing competition
        
        offer = seller.stage2_marketing_decision(effort_level, good_id, market_competition)
        
        # Must return valid Offer object
        assert isinstance(offer, Offer), f"Must return Offer object, got {type(offer)}"
        assert offer.good_id == good_id, "Offer must reference correct forecast"
        assert offer.seller == "test_seller", "Offer must identify correct seller"
        
        # Pricing must be realistic
        assert isinstance(offer.price, float), "Price must be numeric"
        assert offer.price > 0, f"Price must be positive, got {offer.price}"
        
        # Marketing attributes must be valid
        assert isinstance(offer.marketing_attributes, dict), "Marketing attributes must be dict"
        assert "innovation_level" in offer.marketing_attributes, "Must have innovation_level"
        assert "data_source" in offer.marketing_attributes, "Must have data_source"
        assert offer.marketing_attributes["innovation_level"] in ["low", "medium", "high"], "Invalid innovation level"
        assert offer.marketing_attributes["data_source"] in ["internal", "external", "proprietary"], "Invalid data source"
        
        # Must be posted to market for buyer access
        assert offer in market_model.state.offers, "Offer must be added to market state"
    
    def test_buyer_preference_extraction_calculation(self, market_model, context_config, buyers_with_preferences):
        """
        Test seller correctly extracts average buyer preferences for pricing.
        
        Buyer preference extraction enables sellers to estimate market willingness-
        to-pay for different attribute combinations. Calculation process:
        
        Given buyer preferences:
        - Buyer 1: attr_mu = [0.8, 0.6] (high methodology, moderate data preference)  
        - Buyer 2: attr_mu = [0.4, 0.5] (lower methodology, moderate data preference)
        
        Average market preferences: [(0.8 + 0.4)/2, (0.6 + 0.5)/2] = [0.6, 0.55]
        
        This average represents the expected willingness-to-pay per unit of each
        attribute across the buyer population. Sellers use this to optimize their
        attribute claims for maximum revenue.
        
        Economic significance: Preference extraction allows sellers to identify
        which attributes buyers value most, creating incentives for both honest
        quality improvement and deceptive attribute inflation.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        avg_prefs = seller.extract_buyer_preferences()
        
        # Must return average preferences across buyer population
        assert isinstance(avg_prefs, (list, np.ndarray)), "Must return preference vector"
        assert len(avg_prefs) == 2, f"Must return 2 average preferences, got {len(avg_prefs)}"
        
        # Calculation verification: [(0.8+0.4)/2, (0.6+0.5)/2] = [0.6, 0.55]
        expected_avg_pref_1 = (0.8 + 0.4) / 2  # = 0.6
        expected_avg_pref_2 = (0.6 + 0.5) / 2  # = 0.55
        
        assert avg_prefs[0] == pytest.approx(expected_avg_pref_1, abs=0.01), \
            f"First preference average: expected {expected_avg_pref_1}, got {avg_prefs[0]}"
        assert avg_prefs[1] == pytest.approx(expected_avg_pref_2, abs=0.01), \
            f"Second preference average: expected {expected_avg_pref_2}, got {avg_prefs[1]}"
    
    def test_competitive_pricing_no_competition_calculation(self, market_model, context_config, buyers_with_preferences):
        """
        Test pricing strategy when seller faces no direct competition.
        
        No-competition pricing represents market power scenario where seller
        can charge closer to buyer willingness-to-pay without competitive pressure.
        
        Pricing calculation for premium attributes [0.9, 0.8]:
        1. Buyer willingness-to-pay calculation:
           - Buyer 1: 0.8 × 0.9 + 0.6 × 0.8 = 0.72 + 0.48 = 1.20
           - Buyer 2: 0.4 × 0.9 + 0.5 × 0.8 = 0.36 + 0.40 = 0.76
           - Average WTP: (1.20 + 0.76) / 2 = 0.98
        
        2. Pricing strategy: charge 90% of WTP to ensure demand
           - Expected price: 0.98 × 0.9 = 0.882
        
        This scenario tests seller's ability to extract value when market
        power exists, representing early market conditions before competition
        intensifies and drives prices down.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Premium attribute claims
        proposed_attrs = [0.9, 0.8]  # High methodology, high data quality claims
        market_competition = []  # No competing offers
        
        price = seller.estimate_competitive_price(proposed_attrs, market_competition)
        
        # Calculate expected pricing
        # Average buyer WTP: 0.6 × 0.9 + 0.55 × 0.8 = 0.54 + 0.44 = 0.98
        expected_wtp = 0.6 * 0.9 + 0.55 * 0.8  # Using avg preferences [0.6, 0.55]
        expected_price = expected_wtp * 0.9  # 90% of WTP pricing strategy
        
        assert price == pytest.approx(expected_price, abs=0.01), \
            f"No-competition pricing: expected {expected_price:.3f}, got {price:.3f}. " \
            f"Calculation: WTP={expected_wtp:.3f} × 0.9 = {expected_price:.3f}"
    
    def test_competitive_pricing_with_similar_competition_calculation(self, market_model, context_config, buyers_with_preferences):
        """
        Test pricing adjustment under competitive pressure from similar offerings.
        
        Competitive pricing represents market conditions where multiple sellers
        offer similar attribute combinations, forcing price competition.
        
        Competition scenario:
        - Proposed attributes: [0.8, 0.7] (high-medium attribute claims)
        - Competing offers with similar attributes and prices: [100, 110, 95]
        - Similarity threshold: 0.7 (offerings within 70% similarity compete directly)
        
        Competitive pricing calculation:
        1. Identify similar competitors using attribute similarity
        2. Calculate average competitor price: (100 + 110 + 95) / 3 = 101.67
        3. Apply competitive discount: 101.67 × 0.95 = 96.58 (5% below average)
        
        Economic rationale: Sellers must price below competitors to win market
        share, creating downward pressure on prices that squeezes profit margins
        and can make high-cost quality production unprofitable.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        proposed_attrs = {"innovation_level": "high", "data_source": "external"}  # Equivalent to [0.8, 0.7]
        
        # Create competing offers with varying similarity levels
        competing_offers = [
            # Similar competitors (should influence pricing)
            Offer(good_id="comp1", price=100, seller="comp1", 
                  marketing_attributes={"innovation_level": "high", "data_source": "external"}),  # Similar
            Offer(good_id="comp2", price=110, seller="comp2", 
                  marketing_attributes={"innovation_level": "high", "data_source": "proprietary"}),  # Very similar
            Offer(good_id="comp3", price=95, seller="comp3", 
                  marketing_attributes={"innovation_level": "medium", "data_source": "external"}),   # Similar
            # Dissimilar competitor (should not influence pricing)
            Offer(good_id="comp4", price=150, seller="comp4", 
                  marketing_attributes={"innovation_level": "low", "data_source": "internal"})   # Very different
        ]
        
        price = seller.estimate_competitive_price(proposed_attrs, competing_offers)
        
        # Calculate expected competitive price
        # Only similar offers should affect pricing (first 3 offers)
        similar_prices = [100, 110, 95]  # Excludes dissimilar offer at 150
        avg_competitor_price = sum(similar_prices) / len(similar_prices)  # = 101.67
        competitive_discount = 0.05  # 5% below market average
        expected_competitive_price = avg_competitor_price * (1 - competitive_discount)  # = 96.58
        
        assert price == pytest.approx(expected_competitive_price, abs=1.0), \
            f"Competitive pricing: expected {expected_competitive_price:.2f}, got {price:.2f}. " \
            f"Calculation: avg_price={avg_competitor_price:.2f} × 0.95 = {expected_competitive_price:.2f}"
    
    def test_attribute_similarity_euclidean_calculation(self, market_model, context_config):
        """
        Test precise Euclidean distance similarity calculation for competition analysis.
        
        Attribute similarity determines which competitors directly threaten seller's
        market position. Formula: similarity = 1 - (euclidean_distance / max_distance)
        
        For 2-dimensional attribute vectors: max_distance = √2 ≈ 1.414
        
        Test cases with reference [0.8, 0.7]:
        1. [0.8, 0.7]: distance = 0 → similarity = 1 - 0/1.414 = 1.000 (identical)
        2. [0.8, 0.8]: distance = 0.1 → similarity = 1 - 0.1/1.414 = 0.929 (very similar)
        3. [0.7, 0.6]: distance = √(0.1² + 0.1²) = 0.141 → similarity = 0.900 (similar)
        4. [0.2, 0.3]: distance = √(0.6² + 0.4²) = 0.721 → similarity = 0.490 (different)
        
        Economic significance: Similarity threshold (e.g., 0.7) determines competitive
        scope. Higher thresholds mean more offers compete directly, increasing pricing pressure.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        reference_attrs = [0.8, 0.7]
        
        # Test cases with expected calculations
        test_cases = [
            ([0.8, 0.7], 0.0, 1.000),       # Identical attributes
            ([0.8, 0.8], 0.1, 0.929),       # Small difference (0.1)
            ([0.7, 0.6], 0.141, 0.900),     # Medium difference (√0.02 ≈ 0.141)
            ([0.2, 0.3], 0.721, 0.490)      # Large difference (√0.52 ≈ 0.721)
        ]
        
        max_distance = np.sqrt(2)  # Maximum possible distance for 2D vectors
        
        for test_attrs, expected_distance, expected_similarity in test_cases:
            similarity = seller.calculate_attribute_similarity(reference_attrs, test_attrs)
            
            # Verify similarity calculation
            actual_distance = np.sqrt(sum((r - t)**2 for r, t in zip(reference_attrs, test_attrs)))
            calculated_similarity = 1 - (actual_distance / max_distance)
            
            # Test distance calculation accuracy  
            assert actual_distance == pytest.approx(expected_distance, abs=0.01), \
                f"Distance for {test_attrs}: expected {expected_distance}, got {actual_distance}"
            
            # Test similarity calculation accuracy
            assert similarity == pytest.approx(calculated_similarity, abs=0.01), \
                f"Similarity calculation error for {test_attrs}"
            assert similarity == pytest.approx(expected_similarity, abs=0.05), \
                f"Similarity for {test_attrs}: expected {expected_similarity}, got {similarity}"
    
    def test_optimal_marketing_strategy_selection(self, market_model, context_config, buyers_with_preferences):
        """
        Test seller selects profit-maximizing marketing strategy given buyer preferences.
        
        Marketing strategy optimization involves choosing attribute claims and pricing
        to maximize net revenue after marketing costs. Decision process:
        
        Available strategies with buyer preferences [0.6, 0.55]:
        1. Premium claims [0.9, 0.8]: WTP = 0.6×0.9 + 0.55×0.8 = 0.98, cost = 10
        2. Standard claims [0.7, 0.6]: WTP = 0.6×0.7 + 0.55×0.6 = 0.75, cost = 5  
        3. Basic claims [0.5, 0.4]: WTP = 0.6×0.5 + 0.55×0.4 = 0.52, cost = 2
        
        Net revenues (assuming 90% of WTP pricing):
        1. Premium: 0.98 × 0.9 - 10 = 0.882 - 10 = -9.118 (loss)
        2. Standard: 0.75 × 0.9 - 5 = 0.675 - 5 = -4.325 (smaller loss)
        3. Basic: 0.52 × 0.9 - 2 = 0.468 - 2 = -1.532 (smallest loss)
        
        Should choose basic claims (minimizes losses in low-budget market).
        
        Economic insight: When buyer budgets are low relative to marketing costs,
        even optimal strategy may yield losses, but rational sellers minimize harm.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        effort_level = 'low'  # Low production cost (30) for clear profit calculation
        good_id = 'strategy_test'
        
        offer = seller.stage2_marketing_decision(effort_level, good_id, [])
        
        # Should select strategy that maximizes net revenue
        # Expected marketing attribute combinations for decision
        valid_attribute_options = [
            {"innovation_level": "high", "data_source": "proprietary"},  # Premium claims  
            {"innovation_level": "medium", "data_source": "external"},   # Standard claims
            {"innovation_level": "low", "data_source": "internal"}       # Basic claims
        ]
        
        assert offer.marketing_attributes in valid_attribute_options, \
            f"Must choose from valid attribute options, got {offer.marketing_attributes}"
        
        # Price should reflect buyer WTP for chosen attributes
        # Convert marketing attributes to average numeric attributes for WTP calculation
        from multi_agent_economics.models.market_for_finance import get_average_attribute_vector
        avg_attrs = get_average_attribute_vector(offer.marketing_attributes, market_model.state.buyers_state, market_model.state.attribute_order)
        avg_buyer_prefs = [0.65, 0.55]  # From buyers_with_preferences fixture (averaged)
        expected_wtp = sum(pref * attr for pref, attr in zip(avg_buyer_prefs, avg_attrs))
        expected_price = expected_wtp * 0.9  # 90% of WTP pricing
        
        assert offer.price == pytest.approx(expected_price, abs=0.01), \
            f"Price should match WTP calculation: expected {expected_price:.3f}, got {offer.price:.3f}"


class TestAkerlofSellerIntegration:
    """
    Test integration between AkerlofSeller and existing market_for_finance framework.
    
    Integration tests verify that AkerlofSeller works correctly with the existing
    market simulation infrastructure, including proper data flow, format compatibility,
    and economic consistency with buyer behavior and market clearing mechanisms.
    """
    
    def test_historical_revenue_analysis_by_performance(self, market_model, context_config, historical_trades):
        """
        Test seller analyzes historical trades to extract performance-revenue relationships.
        
        Historical analysis enables sellers to learn market patterns and make informed
        production decisions. The analysis must correctly group trades by performance
        level and calculate average revenues for profit estimation.
        
        Expected analysis from historical_trades fixture:
        - High performance (0.85): trades with prices [120, 115] → average 117.5
        - Medium performance (0.70): trades with prices [85, 80] → average 82.5
        - Low performance (0.55): trades with prices [50, 45] → average 47.5
        
        This analysis provides the foundation for profit-maximizing effort decisions
        in Stage 1, creating realistic adaptive behavior based on market feedback.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Map historical trades to performance levels
        performance_mapping = {
            "high_perf_1": 0.85, "high_perf_2": 0.85,  # High performance group
            "med_perf_1": 0.70, "med_perf_2": 0.70,    # Medium performance group  
            "low_perf_1": 0.55, "low_perf_2": 0.55     # Low performance group
        }
        seller.add_performance_data(historical_trades, performance_mapping)
        
        # Test revenue analysis for each performance level
        performance_levels = [0.85, 0.70, 0.55]
        expected_revenues = [
            [120, 115],  # High performance achieved these prices
            [85, 80],    # Medium performance achieved these prices
            [50, 45]     # Low performance achieved these prices
        ]
        expected_averages = [117.5, 82.5, 47.5]
        
        for performance, expected_prices, expected_avg in zip(performance_levels, expected_revenues, expected_averages):
            revenues = seller.analyze_historical_revenues(performance, historical_trades)
            
            # Should find correct prices for this performance level
            assert set(revenues) == set(expected_prices), \
                f"Performance {performance}: expected prices {expected_prices}, got {revenues}"
            
            # Average should match calculation
            actual_avg = sum(revenues) / len(revenues)
            assert actual_avg == pytest.approx(expected_avg, abs=0.1), \
                f"Performance {performance}: expected average {expected_avg}, got {actual_avg}"
    
    def test_market_posting_creates_accessible_offer(self, market_model, context_config):
        """
        Test create_and_post_offer properly integrates with market state for buyer access.
        
        Market posting must create properly formatted offers that buyers can evaluate
        and purchase through the existing market clearing mechanism. Requirements:
        
        1. Offer object must have all required fields for buyer choice models
        2. Offer must be added to market_state.offers for buyer visibility
        3. Pricing and attributes must be realistic for market clearing
        4. Integration must not disrupt existing buyer-seller interactions
        
        This tests the critical interface between seller decisions and buyer access,
        ensuring the two-stage seller process properly feeds into market dynamics.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Marketing decision inputs
        good_id = "integration_forecast_789"
        marketing_attrs = {"innovation_level": "high", "data_source": "external"}  # High-medium attribute claims
        price = 95.0  # Competitive price
        
        initial_offer_count = len(market_model.state.offers)
        
        offer = seller.create_and_post_offer(good_id, marketing_attrs, price)
        
        # Offer must be properly formatted  
        assert offer.good_id == good_id, "Offer must reference correct forecast"
        assert offer.marketing_attributes == marketing_attrs, "Offer must have chosen attributes"
        assert offer.price == price, "Offer must have chosen price"
        assert offer.seller == "test_seller", "Offer must identify seller"
        
        # Must be accessible through market state
        assert len(market_model.state.offers) == initial_offer_count + 1, "Must add one offer to market"
        assert market_model.state.offers[-1] == offer, "Most recent offer must be the created one"
        assert offer in market_model.state.offers, "Offer must be findable in market offers"
    
    def test_complete_two_stage_decision_process(self, market_model, context_config, buyers_with_preferences, historical_trades):
        """
        Test complete seller decision process from effort choice through market posting.
        
        End-to-end test validates that both stages work together correctly:
        
        Stage 1 flow:
        1. Analyze historical data to estimate profit by effort level
        2. Choose profit-maximizing effort level  
        3. Generate forecast with appropriate quality for chosen effort
        4. Store forecast in market state for Stage 2 access
        
        Stage 2 flow:
        1. Extract buyer preferences for WTP estimation
        2. Analyze competitive landscape for pricing strategy  
        3. Choose attribute claims maximizing net revenue
        4. Create and post offer to market for buyer access
        
        Integration requirements:
        - Stage 1 output must properly feed Stage 2 input
        - Forecast generated in Stage 1 must be accessible in Stage 2
        - Final offer must be buyable through existing market clearing
        - Economic consistency between effort costs and pricing decisions
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Prepare seller with performance data for Stage 1 analysis
        performance_mapping = {
            "high_perf_1": 0.85, "high_perf_2": 0.85,
            "med_perf_1": 0.70, "med_perf_2": 0.70,
            "low_perf_1": 0.55, "low_perf_2": 0.55
        }
        seller.add_performance_data(historical_trades, performance_mapping)
        
        # Stage 1: Effort decision and forecast generation
        effort_level, forecast_info = seller.stage1_effort_decision(historical_trades, round_num=1)
        
        # Verify Stage 1 outputs
        assert effort_level in ['high', 'medium', 'low'], f"Invalid effort level: {effort_level}"
        
        forecast_id, forecast_data = forecast_info
        assert isinstance(forecast_id, str), "Forecast ID must be string"
        assert isinstance(forecast_data, ForecastData), "Must generate valid forecast"
        assert forecast_id in market_model.state.knowledge_good_forecasts, "Forecast must be stored"
        
        # Stage 2: Marketing decision using Stage 1 output
        competitive_offers = []  # No competition for simplicity
        offer = seller.stage2_marketing_decision(effort_level, forecast_id, competitive_offers)
        
        # Verify Stage 2 outputs  
        assert isinstance(offer, Offer), "Must create valid offer object"
        assert offer.good_id == forecast_id, "Offer must reference Stage 1 forecast"
        assert offer.seller == "test_seller", "Offer must identify seller"
        assert offer in market_model.state.offers, "Offer must be posted to market"
        
        # Verify economic consistency
        assert offer.price > 0, "Price must be positive"
        assert isinstance(offer.marketing_attributes, dict), "Must have valid marketing attributes"
        assert "innovation_level" in offer.marketing_attributes, "Must have innovation level"
        assert "data_source" in offer.marketing_attributes, "Must have data source"
        
        # Integration success: Buyers can now evaluate and purchase this offer
        # through existing market clearing mechanisms


class TestAkerlofDeathSpiralScenarios:
    """
    Test scenarios demonstrating Akerlof's "Market for Lemons" death spiral dynamics.
    
    These scenarios test the core adverse selection mechanisms that lead to market
    failure when information asymmetries prevent quality differentiation.
    
    Death spiral progression:
    1. Mixed quality market with varied pricing
    2. Competitive pressure reduces profit margins  
    3. High-quality producers become unprofitable, exit or reduce quality
    4. Average market quality falls, buyer expectations decline
    5. Only low-quality producers remain profitable
    6. Market failure: "bad drives out good" (Akerlof's key insight)
    """
    
    def test_low_quality_mimicking_incentive_structure(self, market_model, context_config, buyers_with_preferences):
        """
        Test that low-quality producers can profitably mimic high-quality attributes.
        
        Mimicking scenario demonstrates the core information asymmetry problem:
        Low-quality producers can claim high attributes at marketing cost, potentially
        earning higher profits than honest attribute claims.
        
        Mimicking profit analysis (scaling up buyer budgets to show clear incentives):
        - Low-effort production cost: 30 (sunk cost)
        - Mimicking strategy: claim premium attributes [0.9, 0.8] for marketing cost 10
        - Total cost: 30 + 10 = 40
        
        With scaled buyer preferences (10x higher WTP):
        - Buyer WTP for premium: 6.0 × 0.9 + 5.5 × 0.8 = 9.8
        - Market price (90% WTP): 9.8 × 0.9 = 8.82  
        - Mimicking profit: 8.82 - 40 = -31.18 (still loss in this example)
        
        Vs honest low-quality strategy:
        - Basic attributes [0.5, 0.4]: WTP = 6.0 × 0.5 + 5.5 × 0.4 = 5.2
        - Price: 5.2 × 0.9 = 4.68
        - Honest cost: 30 + 2 = 32  
        - Honest profit: 4.68 - 32 = -27.32
        
        Even with losses, mimicking yields higher profit (-31.18 > -27.32), creating
        incentive for deceptive attribute claims.
        
        Economic significance: Information asymmetry allows low-quality producers
        to free-ride on reputation of high-quality attributes, undermining market
        efficiency and creating adverse selection pressure.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Scale up buyer preferences to create clear mimicking incentives
        for buyer in buyers_with_preferences:
            buyer.attr_mu = [pref * 10 for pref in buyer.attr_mu]  # 10x higher WTP
            buyer.attr_weights = [pref * 10 for pref in buyer.attr_weights]
        
        # Low-effort production (but seller can claim any attributes in marketing)
        effort_level = 'low'  # Production cost = 30
        good_id = 'mimicking_test'
        market_competition = []  # No competition to isolate mimicking incentive
        
        offer = seller.stage2_marketing_decision(effort_level, good_id, market_competition)
        
        # Should choose premium attributes despite low production effort
        # (because mimicking yields higher profit than honest claims)
        expected_premium_attrs = {"innovation_level": "high", "data_source": "proprietary"}  # Highest attribute claims
        assert offer.marketing_attributes == expected_premium_attrs, \
            f"Expected mimicking with {expected_premium_attrs}, got {offer.marketing_attributes}. " \
            f"Low-quality seller should claim premium attributes when profitable."
        
        # Verify pricing matches premium attribute claims
        from multi_agent_economics.models.market_for_finance import get_average_attribute_vector
        avg_attrs = get_average_attribute_vector(expected_premium_attrs, market_model.state.buyers_state, market_model.state.attribute_order)
        avg_buyer_prefs = [0.6, 0.55]  # From buyers_with_preferences fixture
        expected_premium_wtp = sum(pref * attr for pref, attr in zip(avg_buyer_prefs, avg_attrs))
        expected_price = expected_premium_wtp * 0.9
        
        assert offer.price == pytest.approx(expected_price, abs=1.0), \
            f"Mimicking price should match premium WTP: expected {expected_price:.2f}, got {offer.price:.2f}"
        
        # Economic verification: mimicking should be more profitable than honesty
        mimicking_profit = offer.price - (30 + 10)  # Low production + premium marketing
        
        # Calculate honest strategy profit for comparison
        honest_attrs = [0.5, 0.4]  # Basic attributes matching low effort
        honest_wtp = sum(pref * attr for pref, attr in zip(avg_buyer_prefs, honest_attrs))
        honest_price = honest_wtp * 0.9
        honest_profit = honest_price - (30 + 2)  # Low production + basic marketing
        
        assert mimicking_profit > honest_profit, \
            f"Mimicking should be more profitable: mimicking={mimicking_profit:.2f}, honest={honest_profit:.2f}"
    
    def test_high_quality_unprofitability_under_competitive_pressure(self, market_model, context_config, buyers_with_preferences):
        """
        Test high-quality producers face losses under competitive pricing pressure.
        
        Competitive pressure scenario demonstrates how price competition can make
        high-quality production unprofitable even when product quality is superior.
        
        High-quality cost structure:
        - High-effort production: cost = 100 (sophisticated models, premium data)
        - Premium marketing: cost = 10 (professional presentation)
        - Total cost: 110
        
        Competitive market scenario:
        - Multiple competitors offering similar attributes at low prices [80, 85, 75]
        - Average competitor price: 80
        - Competitive pricing: 80 × 0.95 = 76 (5% below average)
        
        High-quality profit: 76 - 110 = -34 (substantial loss)
        
        Meanwhile, low-quality producers with same competitive price:
        - Low production cost: 30, marketing cost: 10, total: 40  
        - Profit: 76 - 40 = 36 (healthy profit)
        
        Economic outcome: Competition drives prices below high-quality cost recovery,
        making only low-quality production viable. This creates adverse selection
        pressure that drives high-quality producers from the market.
        
        Market failure mechanism: Price competition + information asymmetry =
        systematic disadvantage for quality producers = market quality degradation.
        """
        seller = AkerlofSeller("test_seller", market_model, context_config)
        
        # Create competitive market with low prices
        competitive_offers = [
            Offer(good_id="comp1", price=80, seller="comp1", 
                  marketing_attributes={"innovation_level": "high", "data_source": "proprietary"}),
            Offer(good_id="comp2", price=85, seller="comp2", 
                  marketing_attributes={"innovation_level": "high", "data_source": "proprietary"}), 
            Offer(good_id="comp3", price=75, seller="comp3", 
                  marketing_attributes={"innovation_level": "high", "data_source": "external"})
        ]
        
        # High-effort production (expensive but high quality)
        effort_level = 'high'  # Production cost = 100
        good_id = 'high_quality_squeeze_test'
        
        offer = seller.stage2_marketing_decision(effort_level, good_id, competitive_offers)
        
        # Should be forced to price competitively despite high costs
        competitor_prices = [80, 85, 75]
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices)  # = 80
        expected_competitive_price = avg_competitor_price * 0.95  # = 76
        
        assert offer.price <= expected_competitive_price + 5, \
            f"Must price competitively: expected ~{expected_competitive_price}, got {offer.price}"
        
        # Calculate actual profitability
        high_production_cost = 100
        premium_marketing_cost = 10  
        total_cost = high_production_cost + premium_marketing_cost  # = 110
        actual_profit = offer.price - total_cost
        
        # Should result in substantial loss (price < cost recovery)
        assert actual_profit < -20, \
            f"High-quality should face major loss under competition: profit={actual_profit:.2f}, " \
            f"cost={total_cost}, price={offer.price:.2f}"
        
        # Verify death spiral condition: high-quality unprofitable, low-quality profitable
        # If low-quality seller faced same price
        low_cost_total = 30 + 10  # Low production + premium marketing
        low_quality_profit = offer.price - low_cost_total
        
        assert low_quality_profit > 0, \
            f"Low-quality should remain profitable at same price: profit={low_quality_profit:.2f}"
        assert actual_profit < low_quality_profit, \
            f"Profit gap creates adverse selection: high={actual_profit:.2f}, low={low_quality_profit:.2f}"


if __name__ == "__main__":
    # Run tests with detailed output for TDD development
    pytest.main([__file__, "-v", "--tb=short"])