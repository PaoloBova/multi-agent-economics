"""
Economic tool factory - simplified wrappers that only handle budget and logging.

Wrappers call implementations directly and return their results unchanged.
"""

from typing import Dict, List, Any
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool

from .implementations.economic import (
    sector_forecast_impl, post_to_market_impl,
    analyze_historical_performance_impl, analyze_buyer_preferences_impl, research_competitive_pricing_impl
)
from .schemas import (
    SectorForecastResponse, PostToMarketResponse,
    HistoricalPerformanceResponse, BuyerPreferenceResponse, CompetitivePricingResponse
)


def create_economic_tools(market_model, config_data: Dict[str, Any]) -> List[FunctionTool]:
    """Create economic tools with simple wrappers."""
    
    def handle_budget(market_model,
                      config_data: Dict[str, Any],
                      effort: float,
                      tool_name: str) -> float:
        """Handle budget management for economic tools."""
        
        # Budget management
        market_state = market_model.state
        org_id = config_data.get('org_id')
        if not org_id:
            raise ValueError("Agent ID not found in config_data")
        available_budget = market_state.budgets.get(org_id, 0.0)
        
        if effort > available_budget:
            effort = available_budget  # Limit effort to available budget
        
        # Deduct budget
        market_state.budgets[org_id] = available_budget - effort
        
        # Record action
        if hasattr(market_state, 'tool_usage'):
            if org_id not in market_state.tool_usage:
                market_state.tool_usage[org_id] = []
            market_state.tool_usage[org_id].append({
                "tool": tool_name, "effort": effort
            })
        
        return effort
    
    async def sector_forecast(
        sector: Annotated[str, "Sector to forecast (tech, finance, healthcare, energy)"],
        horizon: Annotated[int, "Number of periods to forecast (1-12)"], 
        effort: Annotated[float, "Credits to allocate (0.1-10.0)"]
    ) -> SectorForecastResponse:
        """Generate sector growth forecast using regime-switching analysis."""
        
        effort = handle_budget(market_model, config_data, effort, "sector_forecast")

        return sector_forecast_impl(market_model, config_data, sector, horizon, effort)
    
    
    async def post_to_market(
        notional: Annotated[float, "Notional amount of the structured note"],
        payoff_type: Annotated[str, "Type of payoff structure (linear, barrier, autocall, digital)"],
        underlying_forecast: Annotated[List[float], "Expected returns for underlying assets"],
        discount_rate: Annotated[float, "Risk-free discount rate"] = 0.03,
        effort: Annotated[float, "Credits to allocate"] = 1.0
    ) -> PostToMarketResponse:
        """Price structured financial instruments and post to market."""
        
        effort = handle_budget(market_model, config_data, effort, "post_to_market")
        return post_to_market_impl(market_model, config_data, notional, payoff_type, underlying_forecast, discount_rate, effort)
    
    
    async def analyze_historical_performance(
        sector: Annotated[str, "Sector to analyze (tech, finance, healthcare, energy)"],
        effort: Annotated[float, "Credits to allocate (0.1-10.0)"]
    ) -> HistoricalPerformanceResponse:
        """Analyze historical performance-revenue relationships within a sector for market research."""
        
        effort = handle_budget(market_model, config_data, effort, "analyze_historical_performance")
        return analyze_historical_performance_impl(market_model, config_data, sector, effort)
    
    
    async def analyze_buyer_preferences(
        sector: Annotated[str, "Sector to analyze (tech, finance, healthcare, energy)"],
        effort: Annotated[float, "Credits to allocate (0.1-10.0)"]
    ) -> BuyerPreferenceResponse:
        """Analyze buyer preference patterns within a sector for market research."""
        
        effort = handle_budget(market_model, config_data, effort, "analyze_buyer_preferences")
        return analyze_buyer_preferences_impl(market_model, config_data, sector, effort)
    
    
    async def research_competitive_pricing(
        sector: Annotated[str, "Sector to analyze (tech, finance, healthcare, energy)"],
        effort: Annotated[float, "Credits to allocate (0.1-10.0)"]
    ) -> CompetitivePricingResponse:
        """Research competitive pricing patterns within a sector for market research."""
        
        effort = handle_budget(market_model, config_data, effort, "research_competitive_pricing")
        return research_competitive_pricing_impl(market_model, config_data, sector, effort)
    
    
    # Return tools
    return [
        FunctionTool(sector_forecast, description="Generate sector growth forecast"),
        FunctionTool(post_to_market, description="Price structured financial instruments"),
        FunctionTool(analyze_historical_performance, description="Analyze historical performance-revenue relationships"),
        FunctionTool(analyze_buyer_preferences, description="Analyze buyer preference patterns"),
        FunctionTool(research_competitive_pricing, description="Research competitive pricing patterns")
    ]