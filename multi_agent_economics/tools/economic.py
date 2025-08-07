"""
Economic tool factory - simplified wrappers that only handle budget and logging.

Wrappers call implementations directly and return their results unchanged.
"""

from typing import Dict, List, Any
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool

from .implementations.economic import sector_forecast_impl, monte_carlo_var_impl, price_note_impl
from .schemas import SectorForecastResponse, MonteCarloVarResponse, PriceNoteResponse


def create_economic_tools(market_model, config_data: Dict[str, Any]) -> List[FunctionTool]:
    """Create economic tools with simple wrappers."""
    
    # Tool 1: Sector Forecast
    async def sector_forecast(
        sector: Annotated[str, "Sector to forecast (tech, finance, healthcare, energy)"],
        horizon: Annotated[int, "Number of periods to forecast (1-12)"], 
        effort: Annotated[float, "Credits to allocate (0.1-10.0)"]
    ) -> SectorForecastResponse:
        """Generate sector growth forecast using regime-switching analysis."""
        
        # Budget management
        market_state = market_model.state
        agent_id = config_data.get('agent_id')
        if not agent_id:
            raise ValueError("Agent ID not found in config_data")
        available_budget = market_state.budgets.get(agent_id, 0.0)
        
        if effort > available_budget:
            effort = available_budget  # Limit effort to available budget
        
        # Deduct budget
        market_state.budgets[agent_id] = available_budget - effort
        
        # Record action
        if hasattr(market_state, 'tool_usage'):
            if agent_id not in market_state.tool_usage:
                market_state.tool_usage[agent_id] = []
            market_state.tool_usage[agent_id].append({
                "tool": "sector_forecast", "effort": effort
            })
        
        # Call implementation - it handles ALL the work
        return sector_forecast_impl(market_model, config_data, sector, horizon, effort)
    
    
    # Tool 2: Monte Carlo VaR
    async def monte_carlo_var(
        portfolio_value: Annotated[float, "Portfolio value to analyze in USD"],
        volatility: Annotated[float, "Expected portfolio volatility (0.1 = 10%)"],
        confidence_level: Annotated[float, "Confidence level for VaR calculation (0.95 = 95%)"] = 0.95,
        effort: Annotated[float, "Credits to allocate"] = 1.0
    ) -> MonteCarloVarResponse:
        """Calculate portfolio Value at Risk using Monte Carlo simulation."""
        
        # Budget management  
        market_state = market_model.state
        agent_id = getattr(market_state, 'current_agent_id', 'unknown')
        available_budget = market_state.budgets.get(agent_id, 0.0)
        
        if effort > available_budget:
            effort = available_budget
        
        # Deduct budget
        market_state.budgets[agent_id] = market_state.budgets.get(agent_id, 0) - effort
        
        # Record action
        if hasattr(market_state, 'tool_usage'):
            if agent_id not in market_state.tool_usage:
                market_state.tool_usage[agent_id] = []
            market_state.tool_usage[agent_id].append({
                "tool": "monte_carlo_var", "effort": effort
            })
        
        # Call implementation
        return monte_carlo_var_impl(market_model, config_data, portfolio_value, volatility, confidence_level, effort)
    
    
    # Tool 3: Price Note
    async def price_note(
        notional: Annotated[float, "Notional amount of the structured note"],
        payoff_type: Annotated[str, "Type of payoff structure (linear, barrier, autocall, digital)"],
        underlying_forecast: Annotated[List[float], "Expected returns for underlying assets"],
        discount_rate: Annotated[float, "Risk-free discount rate"] = 0.03,
        effort: Annotated[float, "Credits to allocate"] = 1.0
    ) -> PriceNoteResponse:
        """Price structured financial instruments."""
        
        # Budget management
        market_state = market_model.state
        agent_id = getattr(market_state, 'current_agent_id', 'unknown')
        available_budget = market_state.budgets.get(agent_id, 0.0)
        
        if effort > available_budget:
            effort = available_budget
        
        # Deduct budget
        market_state.budgets[agent_id] = market_state.budgets.get(agent_id, 0) - effort
        
        # Record action
        if hasattr(market_state, 'tool_usage'):
            if agent_id not in market_state.tool_usage:
                market_state.tool_usage[agent_id] = []
            market_state.tool_usage[agent_id].append({
                "tool": "price_note", "effort": effort
            })
        
        # Call implementation
        return price_note_impl(market_model, config_data, notional, payoff_type, underlying_forecast, discount_rate, effort)
    
    
    # Return tools
    return [
        FunctionTool(sector_forecast, description="Generate sector growth forecast"),
        FunctionTool(monte_carlo_var, description="Calculate portfolio Value at Risk"), 
        FunctionTool(price_note, description="Price structured financial instruments")
    ]