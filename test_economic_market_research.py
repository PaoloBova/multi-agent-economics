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
    BuyerState, SellerState, PeriodData
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
def market_state_with_data(regime_parameters_setup, transition_matrices_setup, correlation_matrix_setup, 
                          regime_history_data, comprehensive_trade_history, agent_beliefs_setup):
    """
    Create comprehensive MarketState with all components for realistic simulation testing.
    Uses all the specialized fixtures to build a complete market environment.
    """
    
    # Create comprehensive knowledge_good_forecasts for all trade good_ids
    knowledge_good_forecasts = {}
    
    # Add forecasts for all trades in comprehensive_trade_history  
    for trade in comprehensive_trade_history:
        if trade.good_id not in knowledge_good_forecasts:
            # Determine sector from good_id
            if "tech_" in trade.good_id:
                sector = "tech"
            elif "finance_" in trade.good_id:
                sector = "finance"
            elif "healthcare_" in trade.good_id:
                sector = "healthcare"
            else:
                continue
                
            # Create realistic forecast data based on period regime
            period_data = regime_history_data[trade.period]
            regime = period_data.regimes[sector]
            
            # Generate realistic confidence vector based on regime
            if regime == 0:  # Good regime
                confidence = [0.75, 0.25]  # High confidence in good regime
            else:  # Bad regime  
                confidence = [0.25, 0.75]  # High confidence in bad regime
                
            knowledge_good_forecasts[trade.good_id] = ForecastData(
                sector=sector, 
                predicted_regime=regime, 
                confidence_vector=confidence
            )
    
    # Add some legacy forecasts for backwards compatibility
    knowledge_good_forecasts.update({
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
    })
    
    # Get current period index values from regime history
    current_period = 10
    current_index_values = regime_history_data[current_period].index_values
    current_regimes = regime_history_data[current_period].regimes
    
    return MarketState(
        offers=[],
        trades=[],
        index_values=current_index_values,
        
        # Regime-switching infrastructure
        current_regimes=current_regimes,
        regime_transition_matrices=transition_matrices_setup,
        regime_parameters=regime_parameters_setup,
        regime_correlations=correlation_matrix_setup,
        
        # Regime history and simulation state
        regime_history=regime_history_data,
        current_period=current_period,
        
        # Agent beliefs system
        agent_beliefs=agent_beliefs_setup["agent_beliefs"],
        agent_subjective_transitions=agent_beliefs_setup["agent_subjective_transitions"],
        
        # Knowledge goods and forecasting
        knowledge_good_forecasts=knowledge_good_forecasts,
        knowledge_good_impacts={},  # Will be populated during simulation
        
        # Trade history
        all_trades=comprehensive_trade_history,
        
        # Agent states (will be populated by other fixtures)
        buyers_state=[],
        sellers_state=[],
        
        # Marketing attribute system
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
        attribute_order=["innovation_level", "data_source", "risk_score"],
        
        # Risk parameters
        risk_free_rate=0.03
    )


@pytest.fixture  
def buyers_with_preferences(market_state_with_data, agent_beliefs_setup):
    """
    Create buyers with sector-specific preferences for testing preference analysis.
    Tech buyers prefer innovation, finance buyers prefer data quality.
    Initializes beliefs for all sectors using the agent beliefs setup.
    """
    # Base beliefs from agent setup, with some individual variation
    base_beliefs = agent_beliefs_setup["agent_beliefs"]
    
    buyers = [
        # Tech sector buyers - innovation focused
        BuyerState(
            buyer_id="tech_buyer_1",
            regime_beliefs={
                "tech": [0.75, 0.25],        # Slightly optimistic about tech
                "finance": base_beliefs["finance"],
                "healthcare": base_beliefs["healthcare"]
            },
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
            regime_beliefs={
                "tech": [0.65, 0.35],        # Moderately optimistic about tech
                "finance": base_beliefs["finance"],
                "healthcare": base_beliefs["healthcare"]
            },
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
            regime_beliefs={
                "tech": base_beliefs["tech"],
                "finance": [0.35, 0.65],     # Pessimistic about finance
                "healthcare": base_beliefs["healthcare"]
            },
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
            regime_beliefs={
                "tech": base_beliefs["tech"],
                "finance": [0.25, 0.75],     # More pessimistic about finance
                "healthcare": base_beliefs["healthcare"]
            },
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
            regime_beliefs={
                "tech": base_beliefs["tech"],
                "finance": base_beliefs["finance"],
                "healthcare": [0.6, 0.4]     # Slightly optimistic about healthcare
            },
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
def competitive_offers(market_state_with_data, regime_history_data):
    """Create current competitive offers across sectors with regime-appropriate pricing."""
    
    # Get current regimes from period 10 (tech=1 [bad], finance=0 [good], healthcare=1 [bad])
    current_period = 10
    current_regimes = regime_history_data[current_period].regimes
    
    offers = [
        # Tech sector competitive offers - lower prices due to bad regime (regime=1)
        Offer(
            good_id="tech_comp_1", price=75.0, seller="competitor_1",  # Lower price for bad regime
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 45}
        ),
        Offer(
            good_id="tech_comp_2", price=85.0, seller="competitor_2",  # Lower price for bad regime
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 30}
        ),
        Offer(
            good_id="tech_comp_3", price=70.0, seller="competitor_3",  # Lowest price for bad regime + poor attributes
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 55}
        ),
        
        # Finance sector competitive offers - higher prices due to good regime (regime=0)
        Offer(
            good_id="finance_comp_1", price=115.0, seller="fin_competitor_1",  # Good price for good regime
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 25}
        ),
        Offer(
            good_id="finance_comp_2", price=105.0, seller="fin_competitor_2",  # Decent price for good regime
            marketing_attributes={"innovation_level": "low", "data_source": "external", "risk_score": 35}
        ),
        
        # Healthcare sector competitive offers - lower prices due to bad regime (regime=1)
        Offer(
            good_id="healthcare_comp_1", price=80.0, seller="health_competitor_1",  # Lower price for bad regime
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 50}
        ),
    ]
    
    # Add competitive forecasts to knowledge base with realistic confidence based on regimes
    market_state_with_data.knowledge_good_forecasts.update({
        # Tech forecasts - regime 1 (bad)
        "tech_comp_1": ForecastData(sector="tech", predicted_regime=1, confidence_vector=[0.3, 0.7]),
        "tech_comp_2": ForecastData(sector="tech", predicted_regime=1, confidence_vector=[0.2, 0.8]),  
        "tech_comp_3": ForecastData(sector="tech", predicted_regime=1, confidence_vector=[0.4, 0.6]),
        
        # Finance forecasts - regime 0 (good)
        "finance_comp_1": ForecastData(sector="finance", predicted_regime=0, confidence_vector=[0.8, 0.2]),
        "finance_comp_2": ForecastData(sector="finance", predicted_regime=0, confidence_vector=[0.7, 0.3]),
        
        # Healthcare forecasts - regime 1 (bad)
        "healthcare_comp_1": ForecastData(sector="healthcare", predicted_regime=1, confidence_vector=[0.3, 0.7]),
    })
    
    market_state_with_data.offers.extend(offers)
    return offers


@pytest.fixture
def regime_parameters_setup():
    """Set up realistic regime parameters for each sector."""
    return {
        "tech": {
            0: RegimeParameters(mu=0.12, sigma=0.18),  # High growth, high volatility
            1: RegimeParameters(mu=0.04, sigma=0.25)   # Low growth, higher volatility
        },
        "finance": {
            0: RegimeParameters(mu=0.08, sigma=0.15),  # Moderate growth, moderate volatility
            1: RegimeParameters(mu=0.02, sigma=0.22)   # Low growth, higher volatility
        },
        "healthcare": {
            0: RegimeParameters(mu=0.10, sigma=0.20),  # Steady growth, moderate volatility
            1: RegimeParameters(mu=0.05, sigma=0.28)   # Low growth, high volatility
        }
    }


@pytest.fixture
def transition_matrices_setup():
    """Set up transition matrices for regime switching."""
    return {
        "tech": np.array([
            [0.75, 0.25],  # From regime 0: 75% stay in 0, 25% switch to 1
            [0.30, 0.70]   # From regime 1: 30% switch to 0, 70% stay in 1
        ]),
        "finance": np.array([
            [0.80, 0.20],  # More persistent than tech
            [0.25, 0.75]
        ]),
        "healthcare": np.array([
            [0.85, 0.15],  # Most persistent sector
            [0.20, 0.80]
        ])
    }


@pytest.fixture
def correlation_matrix_setup():
    """Set up cross-sector correlation matrix."""
    return np.array([
        [1.00, 0.60, 0.40],  # tech correlations
        [0.60, 1.00, 0.45],  # finance correlations  
        [0.40, 0.45, 1.00]   # healthcare correlations
    ])


@pytest.fixture
def regime_history_data():
    """Hardcoded regime history data for reproducible tests (generated with seed=42)."""
    return [
        # Period 0
        PeriodData(
            period=0,
            returns={'tech': 0.20940854754202187, 'finance': -0.010418146257660625, 'healthcare': 0.2295377076201385},
            regimes={'tech': 0, 'finance': 1, 'healthcare': 0},
            index_values={'tech': 120.94085475420218, 'finance': 98.95818537423395, 'healthcare': 122.95377076201386}
        ),
        # Period 1
        PeriodData(
            period=1,
            returns={'tech': 0.3941453741534445, 'finance': 0.12185619383002065, 'healthcare': 0.3021030569613053},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 0},
            index_values={'tech': 168.6091332017346, 'finance': 111.01685319226371, 'healthcare': 160.0984807741378}
        ),
        # Period 2
        PeriodData(
            period=2,
            returns={'tech': 0.03549461053170862, 'finance': 0.1613840065378947, 'healthcare': -0.07975695398748943},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 174.59384871681917, 'finance': 128.9331977536605, 'healthcare': 147.32951360956795}
        ),
        # Period 3
        PeriodData(
            period=3,
            returns={'tech': 0.03616864435735376, 'finance': 0.2024667621426991, 'healthcare': -0.3766852793324411},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 180.9086715380394, 'finance': 155.03788483554845, 'healthcare': 91.83265462163516}
        ),
        # Period 4
        PeriodData(
            period=4,
            returns={'tech': 0.3838167784058797, 'finance': 0.046133554927019656, 'healthcare': 0.06890789731261868},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 250.34445503345717, 'finance': 162.19033361137818, 'healthcare': 98.16064975624798}
        ),
        # Period 5
        PeriodData(
            period=5,
            returns={'tech': -0.13645467351842222, 'finance': -0.010038077515886756, 'healthcare': 0.3152831498930589},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 216.18378415471943, 'finance': 160.56225447025963, 'healthcare': 129.10904860694717}
        ),
        # Period 6
        PeriodData(
            period=6,
            returns={'tech': 0.11757049954717189, 'finance': -0.07865663934338506, 'healthcare': 0.2803125753888929},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 241.60061965178775, 'finance': 147.9329671282316, 'healthcare': 165.29993852797028}
        ),
        # Period 7
        PeriodData(
            period=7,
            returns={'tech': -0.099751856994784, 'finance': 0.11279574740807938, 'healthcare': 0.296893090905611},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 217.50050919043144, 'finance': 164.61917672175534, 'healthcare': 214.37634820404685}
        ),
        # Period 8
        PeriodData(
            period=8,
            returns={'tech': 0.011087929402939869, 'finance': 0.034834445661606685, 'healthcare': -0.19570439807348547},
            regimes={'tech': 1, 'finance': 0, 'healthcare': 0},
            index_values={'tech': 219.9121394814384, 'finance': 170.35359448812775, 'healthcare': 172.42195401758192}
        ),
        # Period 9
        PeriodData(
            period=9,
            returns={'tech': -0.13996105209867715, 'finance': -0.08958284570223503, 'healthcare': 0.1268857765256044},
            regimes={'tech': 1, 'finance': 0, 'healthcare': 0},
            index_values={'tech': 189.13300507034523, 'finance': 155.0928347182767, 'healthcare': 194.29984754316487}
        ),
        # Period 10
        PeriodData(
            period=10,
            returns={'tech': -0.12923050007648967, 'finance': 0.17175144332613018, 'healthcare': 0.33867986629886626},
            regimes={'tech': 1, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 164.69125224413526, 'finance': 181.73025293068167, 'healthcare': 260.1052939309741}
        ),
        # Period 11
        PeriodData(
            period=11,
            returns={'tech': 0.2728200297790496, 'finance': -0.14622299343212242, 'healthcare': 0.35790115321974975},
            regimes={'tech': 1, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 209.62232458572922, 'finance': 155.15711134998065, 'healthcare': 353.1972785874317}
        ),
        # Period 12
        PeriodData(
            period=12,
            returns={'tech': 0.26625464803095567, 'finance': 0.28343600428562343, 'healthcare': 0.029837165957506524},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 265.43524283773326, 'finance': 199.13422302751871, 'healthcare': 363.7356844043846}
        ),
        # Period 13
        PeriodData(
            period=13,
            returns={'tech': 0.30063592162056435, 'finance': 0.304315137932521, 'healthcare': 0.14469075055812952},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 345.2346116988335, 'finance': 259.7337815752235, 'healthcare': 416.36487358562994}
        ),
        # Period 14
        PeriodData(
            period=14,
            returns={'tech': 0.13566847228287082, 'finance': 0.03514889743011988, 'healthcare': 0.07569301742994064},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 392.07206404718437, 'finance': 268.86313762294816, 'healthcare': 447.880787219162}
        ),
        # Period 15
        PeriodData(
            period=15,
            returns={'tech': -0.4568922286502232, 'finance': -0.03838874930468801, 'healthcare': 0.2582391594559435},
            regimes={'tech': 1, 'finance': 0, 'healthcare': 1},
            index_values={'tech': 212.93738491317322, 'finance': 258.54181803546896, 'healthcare': 563.5411452471046}
        ),
        # Period 16
        PeriodData(
            period=16,
            returns={'tech': 0.2847723811863733, 'finance': 0.12931266644895267, 'healthcare': -0.0059520407534077585},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 0},
            index_values={'tech': 273.5760710584969, 'finance': 291.97454991419534, 'healthcare': 560.1869253843718}
        ),
        # Period 17
        PeriodData(
            period=17,
            returns={'tech': 0.16831685827833903, 'finance': 0.03360671803331335, 'healthcare': 0.23462510055479502},
            regimes={'tech': 1, 'finance': 0, 'healthcare': 0},
            index_values={'tech': 319.6235358391947, 'finance': 301.78685628606524, 'healthcare': 691.6208390821614}
        ),
        # Period 18
        PeriodData(
            period=18,
            returns={'tech': 0.1733016498716237, 'finance': 0.1191582908269834, 'healthcare': 0.10102269132849219},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 0},
            index_values={'tech': 375.0148219379292, 'finance': 337.7472622751613, 'healthcare': 761.4902376251114}
        ),
        # Period 19
        PeriodData(
            period=19,
            returns={'tech': 0.07777431599247356, 'finance': 0.14794044898002642, 'healthcare': 0.06678781832978693},
            regimes={'tech': 0, 'finance': 0, 'healthcare': 0},
            index_values={'tech': 404.1813432011909, 'finance': 387.7137438979234, 'healthcare': 812.3485092755236}
        ),
    ]


@pytest.fixture
def comprehensive_trade_history():
    """Comprehensive trade history distributed across periods 1-10, tied to regime states."""
    return [
        # Period 1 trades - all sectors in good regimes (tech=0, finance=0, healthcare=0)
        TradeData(
            buyer_id="buyer_1", seller_id="seller_1", price=135.0, quantity=1,
            good_id="tech_forecast_p1_1", period=1,
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 20}
        ),
        TradeData(
            buyer_id="buyer_2", seller_id="seller_2", price=125.0, quantity=1,
            good_id="tech_forecast_p1_2", period=1,
            marketing_attributes={"innovation_level": "high", "data_source": "external", "risk_score": 25}
        ),
        TradeData(
            buyer_id="buyer_7", seller_id="seller_7", price=110.0, quantity=1,
            good_id="finance_forecast_p1_1", period=1,
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 18}
        ),
        TradeData(
            buyer_id="buyer_10", seller_id="seller_10", price=140.0, quantity=1,
            good_id="healthcare_forecast_p1_1", period=1,
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 15}
        ),
        
        # Period 2 trades - tech/finance good, healthcare bad (tech=0, finance=0, healthcare=1)
        TradeData(
            buyer_id="buyer_3", seller_id="seller_3", price=130.0, quantity=1,
            good_id="tech_forecast_p2_1", period=2,
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 30}
        ),
        TradeData(
            buyer_id="buyer_8", seller_id="seller_8", price=105.0, quantity=1,
            good_id="finance_forecast_p2_1", period=2,
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 22}
        ),
        TradeData(
            buyer_id="buyer_11", seller_id="seller_11", price=75.0, quantity=1,
            good_id="healthcare_forecast_p2_1", period=2,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 65}
        ),
        
        # Period 3 trades - similar to period 2
        TradeData(
            buyer_id="buyer_4", seller_id="seller_4", price=120.0, quantity=1,
            good_id="tech_forecast_p3_1", period=3,
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 35}
        ),
        TradeData(
            buyer_id="buyer_9", seller_id="seller_9", price=100.0, quantity=1,
            good_id="finance_forecast_p3_1", period=3,
            marketing_attributes={"innovation_level": "low", "data_source": "external", "risk_score": 40}
        ),
        TradeData(
            buyer_id="buyer_11", seller_id="seller_11", price=70.0, quantity=1,
            good_id="healthcare_forecast_p3_1", period=3,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 70}
        ),
        
        # Period 4 trades - same regime pattern continues
        TradeData(
            buyer_id="buyer_1", seller_id="seller_1", price=145.0, quantity=1,
            good_id="tech_forecast_p4_1", period=4,
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 18}
        ),
        TradeData(
            buyer_id="buyer_2", seller_id="seller_3", price=115.0, quantity=1,
            good_id="tech_forecast_p4_2", period=4,
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 28}
        ),
        TradeData(
            buyer_id="buyer_7", seller_id="seller_7", price=108.0, quantity=1,
            good_id="finance_forecast_p4_1", period=4,
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 20}
        ),
        
        # Period 5 trades
        TradeData(
            buyer_id="buyer_5", seller_id="seller_5", price=90.0, quantity=1,
            good_id="tech_forecast_p5_1", period=5,
            marketing_attributes={"innovation_level": "medium", "data_source": "internal", "risk_score": 45}
        ),
        TradeData(
            buyer_id="buyer_8", seller_id="seller_8", price=95.0, quantity=1,
            good_id="finance_forecast_p5_1", period=5,
            marketing_attributes={"innovation_level": "low", "data_source": "external", "risk_score": 50}
        ),
        TradeData(
            buyer_id="buyer_10", seller_id="seller_10", price=65.0, quantity=1,
            good_id="healthcare_forecast_p5_1", period=5,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 75}
        ),
        
        # Period 6 trades
        TradeData(
            buyer_id="buyer_3", seller_id="seller_2", price=110.0, quantity=1,
            good_id="tech_forecast_p6_1", period=6,
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 40}
        ),
        TradeData(
            buyer_id="buyer_9", seller_id="seller_9", price=85.0, quantity=1,
            good_id="finance_forecast_p6_1", period=6,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 55}
        ),
        
        # Period 7 trades
        TradeData(
            buyer_id="buyer_6", seller_id="seller_6", price=85.0, quantity=1,
            good_id="tech_forecast_p7_1", period=7,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 60}
        ),
        TradeData(
            buyer_id="buyer_7", seller_id="seller_7", price=92.0, quantity=1,
            good_id="finance_forecast_p7_1", period=7,
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 45}
        ),
        TradeData(
            buyer_id="buyer_11", seller_id="seller_11", price=60.0, quantity=1,
            good_id="healthcare_forecast_p7_1", period=7,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 80}
        ),
        
        # Period 8 trades - tech regime switches to bad (tech=1, finance=0, healthcare=0)
        TradeData(
            buyer_id="buyer_1", seller_id="seller_1", price=75.0, quantity=1,  # Lower price for tech due to regime 1
            good_id="tech_forecast_p8_1", period=8,
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 50}
        ),
        TradeData(
            buyer_id="buyer_8", seller_id="seller_8", price=100.0, quantity=1,
            good_id="finance_forecast_p8_1", period=8,
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 25}
        ),
        TradeData(
            buyer_id="buyer_10", seller_id="seller_10", price=135.0, quantity=1,  # Healthcare back to good regime
            good_id="healthcare_forecast_p8_1", period=8,
            marketing_attributes={"innovation_level": "high", "data_source": "proprietary", "risk_score": 20}
        ),
        
        # Period 9 trades - tech still bad, others good
        TradeData(
            buyer_id="buyer_2", seller_id="seller_2", price=70.0, quantity=1,  # Low price for tech
            good_id="tech_forecast_p9_1", period=9,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 65}
        ),
        TradeData(
            buyer_id="buyer_4", seller_id="seller_4", price=65.0, quantity=1,  # Very low price for tech
            good_id="tech_forecast_p9_2", period=9,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 70}
        ),
        TradeData(
            buyer_id="buyer_9", seller_id="seller_9", price=98.0, quantity=1,
            good_id="finance_forecast_p9_1", period=9,
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 35}
        ),
        TradeData(
            buyer_id="buyer_11", seller_id="seller_11", price=130.0, quantity=1,
            good_id="healthcare_forecast_p9_1", period=9,
            marketing_attributes={"innovation_level": "high", "data_source": "external", "risk_score": 25}
        ),
        
        # Period 10 trades - tech bad, finance good, healthcare bad (tech=1, finance=0, healthcare=1)
        TradeData(
            buyer_id="buyer_5", seller_id="seller_5", price=55.0, quantity=1,  # Very low for tech
            good_id="tech_forecast_p10_1", period=10,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 85}
        ),
        TradeData(
            buyer_id="buyer_6", seller_id="seller_6", price=60.0, quantity=1,  # Low for tech
            good_id="tech_forecast_p10_2", period=10,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 80}
        ),
        TradeData(
            buyer_id="buyer_7", seller_id="seller_7", price=105.0, quantity=1,  # Good for finance
            good_id="finance_forecast_p10_1", period=10,
            marketing_attributes={"innovation_level": "medium", "data_source": "proprietary", "risk_score": 30}
        ),
        TradeData(
            buyer_id="buyer_8", seller_id="seller_8", price=102.0, quantity=1,  # Good for finance
            good_id="finance_forecast_p10_2", period=10,
            marketing_attributes={"innovation_level": "medium", "data_source": "external", "risk_score": 32}
        ),
        TradeData(
            buyer_id="buyer_10", seller_id="seller_10", price=68.0, quantity=1,  # Low for healthcare
            good_id="healthcare_forecast_p10_1", period=10,
            marketing_attributes={"innovation_level": "low", "data_source": "internal", "risk_score": 75}
        )
    ]


@pytest.fixture
def agent_beliefs_setup():
    """Set up agent beliefs and subjective transition matrices."""
    return {
        "agent_beliefs": {
            # Initial beliefs about regime probabilities (may differ from truth)
            "tech": [0.6, 0.4],      # Agent slightly underestimates good regime likelihood  
            "finance": [0.3, 0.7],   # Agent thinks bad regime more likely (truth is opposite)
            "healthcare": [0.7, 0.3] # Agent overestimates good regime
        },
        "agent_subjective_transitions": {
            # Agent's subjective transition matrices (may be biased)
            "tech": np.array([
                [0.70, 0.30],  # Agent thinks tech regimes less persistent than truth (0.75, 0.25)
                [0.35, 0.65]   # Agent underestimates bad regime persistence (truth: 0.30, 0.70)
            ]),
            "finance": np.array([
                [0.85, 0.15],  # Agent thinks finance more persistent than truth (0.80, 0.20)
                [0.20, 0.80]   # Agent thinks bad regime more persistent than truth (0.25, 0.75)
            ]),
            "healthcare": np.array([
                [0.90, 0.10],  # Agent thinks healthcare very persistent (truth: 0.85, 0.15)
                [0.15, 0.85]   # Agent underestimates regime switching (truth: 0.20, 0.80)
            ])
        }
    }


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
