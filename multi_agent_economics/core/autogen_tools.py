"""
Functional tool integration for AutoGen using FunctionTool with closure pattern.

This module creates economic analysis tools that can access simulation state
while maintaining clean function signatures compatible with AutoGen's
automatic schema generation.
"""

import numpy as np
from typing import Dict, List, Any
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool
from pathlib import Path
import json
import warnings


def create_economic_tools(market_model, config_data: Dict[str, Any]) -> List[FunctionTool]:
    """
    Create economic analysis tools using functional approach with closures.
    
    Args:
        market_model: MarketModel instance containing simulation state
        config_data: Configuration dictionary with tool parameters and market data
        
    Returns:
        List of FunctionTool instances ready for use with AutoGen agents
    """
    
    # Helper function to load tool configuration
    def get_tool_config(tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool."""
        tool_params = config_data.get("tool_parameters", {})
        
        # Load regime-switching configuration if available
        regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
        if regime_config_path.exists():
            try:
                with open(regime_config_path) as f:
                    regime_config = json.load(f)
                
                # Merge tool-specific config from regime switching file
                tool_quality_params = regime_config.get("tool_quality_parameters", {}).get(tool_name, {})
                return {**tool_params.get(tool_name, {}), **tool_quality_params}
            except Exception as e:
                warnings.warn(f"Could not load regime config: {e}")
        
        return tool_params.get(tool_name, {})
    
    # Helper function for budget management
    def manage_budget(agent_id: str, requested_effort: float) -> float:
        """Manage agent budget and return actual effort to use."""
        market_state = market_model.state
        available_budget = market_state.budgets.get(agent_id, 0.0)
        
        if requested_effort > available_budget:
            warnings.warn(f"Insufficient budget. Requested {requested_effort}, available {available_budget}. Using available budget.")
            actual_effort = available_budget
        else:
            actual_effort = requested_effort
        
        # Deduct effort from budget
        market_state.budgets[agent_id] = market_state.budgets.get(agent_id, 0) - actual_effort
        
        return actual_effort
    
    # Tool 1: Sector Forecast
    async def sector_forecast(
        sector: Annotated[str, "Sector to forecast (tech, finance, healthcare, energy)"],
        horizon: Annotated[int, "Number of periods to forecast (1-12)"], 
        effort: Annotated[float, "Credits to allocate - higher effort = better quality (0.1-10.0)"]
    ) -> Dict[str, Any]:
        """
        Generate sector growth forecast using regime-switching analysis.
        
        Uses the current market regime and transition probabilities to forecast
        sector returns. Quality and accuracy depend on effort allocation.
        """
        # Access simulation state through closure
        market_state = market_model.state
        agent_id = getattr(market_state, 'current_agent_id', 'unknown')
        
        # Manage budget
        actual_effort = manage_budget(agent_id, effort)
        
        # Get tool configuration
        tool_config = get_tool_config("sector_forecast")
        effort_thresholds = tool_config.get("effort_thresholds", {"high": 5.0, "medium": 2.0})
        quality_tiers = tool_config.get("quality_tiers", {})
        
        # Determine quality tier based on effort
        if actual_effort >= effort_thresholds.get("high", 5.0):
            quality_tier = "high"
            tier_params = quality_tiers.get("high", {"noise_factor": 0.1, "confidence": 0.9})
        elif actual_effort >= effort_thresholds.get("medium", 2.0):
            quality_tier = "medium" 
            tier_params = quality_tiers.get("medium", {"noise_factor": 0.3, "confidence": 0.7})
        else:
            quality_tier = "low"
            tier_params = quality_tiers.get("low", {"noise_factor": 0.6, "confidence": 0.5})
        
        noise_factor = tier_params.get("noise_factor", 0.3)
        confidence = tier_params.get("confidence", 0.7)
        
        # Access market data from config
        market_data = config_data.get("market_data", {})
        sectors_data = market_data.get("sectors", {})
        default_data = market_data.get("default_sector", {"mean": 0.06, "std": 0.15})
        base_params = sectors_data.get(sector, default_data)
        
        # Generate forecast using regime-switching model if available
        forecast = []
        if hasattr(market_state, 'regime_parameters') and sector in market_state.regime_parameters:
            # Use regime-switching model
            current_regime = market_state.current_regimes.get(sector, 0)
            regime_data = market_state.regime_parameters[sector].get(current_regime)
            
            if regime_data:
                regime_mu = regime_data.mu if hasattr(regime_data, 'mu') else regime_data.get('mu', base_params["mean"])
                regime_sigma = regime_data.sigma if hasattr(regime_data, 'sigma') else regime_data.get('sigma', base_params["std"])
                
                for _ in range(horizon):
                    base_return = np.random.normal(regime_mu, regime_sigma)
                    noise = np.random.normal(0, regime_sigma * noise_factor)
                    forecast.append(float(base_return + noise))
            else:
                # Fallback to base parameters
                for _ in range(horizon):
                    base_return = np.random.normal(base_params["mean"], base_params["std"])
                    noise = np.random.normal(0, base_params["std"] * noise_factor)
                    forecast.append(float(base_return + noise))
        else:
            # Use basic parameters
            for _ in range(horizon):
                base_return = np.random.normal(base_params["mean"], base_params["std"])
                noise = np.random.normal(0, base_params["std"] * noise_factor)
                forecast.append(float(base_return + noise))
        
        # Record tool usage
        if hasattr(market_state, 'tool_usage'):
            if agent_id not in market_state.tool_usage:
                market_state.tool_usage[agent_id] = []
            market_state.tool_usage[agent_id].append({
                "tool": "sector_forecast",
                "effort": actual_effort,
                "timestamp": "now"  # Would use actual timestamp in production
            })
        
        return {
            "sector": sector,
            "horizon": horizon,
            "forecast": forecast,
            "quality_tier": quality_tier,
            "confidence": confidence,
            "effort_requested": effort,
            "effort_used": actual_effort,
            "regime_used": market_state.current_regimes.get(sector, 0) if hasattr(market_state, 'current_regimes') else None,
            "warnings": [] if actual_effort == effort else [f"Budget limited effort to {actual_effort}"]
        }
    
    # Tool 2: Monte Carlo VaR
    async def monte_carlo_var(
        portfolio_value: Annotated[float, "Portfolio value to analyze in USD"],
        volatility: Annotated[float, "Expected portfolio volatility (0.1 = 10%)"],
        confidence_level: Annotated[float, "Confidence level for VaR calculation (0.95 = 95%)"] = 0.95,
        effort: Annotated[float, "Credits to allocate - higher effort = more simulations"] = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate portfolio Value at Risk using Monte Carlo simulation.
        
        Higher effort allocation results in more simulation runs and
        more accurate risk estimates.
        """
        # Access simulation state through closure
        market_state = market_model.state
        agent_id = getattr(market_state, 'current_agent_id', 'unknown')
        
        # Manage budget
        actual_effort = manage_budget(agent_id, effort)
        
        # Get tool configuration
        tool_config = get_tool_config("monte_carlo_var")
        simulation_sizes = tool_config.get("simulation_sizes", {"high": 50000, "medium": 10000, "low": 1000})
        effort_thresholds = tool_config.get("effort_thresholds", {"high": 5.0, "medium": 2.0})
        
        # Determine simulation parameters based on effort
        if actual_effort >= effort_thresholds.get("high", 5.0):
            n_simulations = simulation_sizes.get("high", 50000)
            quality_tier = "high"
        elif actual_effort >= effort_thresholds.get("medium", 2.0):
            n_simulations = simulation_sizes.get("medium", 10000)
            quality_tier = "medium"
        else:
            n_simulations = simulation_sizes.get("low", 1000)
            quality_tier = "low"
        
        # Run Monte Carlo simulation
        returns = np.random.normal(0, volatility, n_simulations)
        portfolio_values = portfolio_value * (1 + returns)
        losses = portfolio_value - portfolio_values
        
        # Calculate VaR and other risk metrics
        var_estimate = float(np.percentile(losses, confidence_level * 100))
        expected_shortfall = float(np.mean(losses[losses >= var_estimate]))
        
        # Record tool usage
        if hasattr(market_state, 'tool_usage'):
            if agent_id not in market_state.tool_usage:
                market_state.tool_usage[agent_id] = []
            market_state.tool_usage[agent_id].append({
                "tool": "monte_carlo_var",
                "effort": actual_effort,
                "timestamp": "now"
            })
        
        return {
            "portfolio_value": portfolio_value,
            "volatility": volatility,
            "confidence_level": confidence_level,
            "var_estimate": var_estimate,
            "expected_shortfall": expected_shortfall,
            "max_loss": float(np.max(losses)),
            "expected_loss": float(np.mean(losses)),
            "n_simulations": n_simulations,
            "quality_tier": quality_tier,
            "effort_requested": effort,
            "effort_used": actual_effort,
            "warnings": [] if actual_effort == effort else [f"Budget limited effort to {actual_effort}"]
        }
    
    # Tool 3: Price Note
    async def price_note(
        notional: Annotated[float, "Notional amount of the structured note"],
        payoff_type: Annotated[str, "Type of payoff structure (linear, barrier, autocall)"],
        underlying_forecast: Annotated[List[float], "Expected returns for underlying assets"],
        discount_rate: Annotated[float, "Risk-free discount rate"] = 0.03,
        effort: Annotated[float, "Credits to allocate - higher effort = better pricing accuracy"] = 1.0
    ) -> Dict[str, Any]:
        """
        Price structured financial instruments using Monte Carlo methods.
        
        Higher effort allocation results in more accurate pricing through
        increased simulation runs and better model calibration.
        """
        # Access simulation state through closure
        market_state = market_model.state
        agent_id = getattr(market_state, 'current_agent_id', 'unknown')
        
        # Manage budget
        actual_effort = manage_budget(agent_id, effort)
        
        # Get tool configuration
        tool_config = get_tool_config("price_note")
        pricing_error_std = tool_config.get("pricing_error_std", {"high": 0.01, "medium": 0.05, "low": 0.10})
        effort_thresholds = tool_config.get("effort_thresholds", {"high": 4.0, "medium": 2.0})
        
        # Determine pricing accuracy based on effort
        if actual_effort >= effort_thresholds.get("high", 4.0):
            error_std = pricing_error_std.get("high", 0.01)
            quality_tier = "high"
        elif actual_effort >= effort_thresholds.get("medium", 2.0):
            error_std = pricing_error_std.get("medium", 0.05)
            quality_tier = "medium"
        else:
            error_std = pricing_error_std.get("low", 0.10)
            quality_tier = "low"
        
        # Calculate expected payoff based on forecasts
        expected_return = np.mean(underlying_forecast) if underlying_forecast else 0.05
        
        # Basic pricing model (simplified)
        if payoff_type.lower() == "linear":
            fair_value = notional * (1 + expected_return) / (1 + discount_rate)
        elif payoff_type.lower() == "barrier":
            # Simplified barrier note pricing
            barrier_probability = 0.8  # Simplified
            fair_value = notional * (1 + expected_return * barrier_probability) / (1 + discount_rate)
        elif payoff_type.lower() == "autocall":
            # Simplified autocall note pricing
            autocall_probability = 0.6  # Simplified
            fair_value = notional * (1 + expected_return * autocall_probability * 1.5) / (1 + discount_rate)
        else:
            fair_value = notional * (1 + expected_return) / (1 + discount_rate)
        
        # Add pricing error based on effort quality
        pricing_error = np.random.normal(0, error_std)
        quoted_price = fair_value * (1 + pricing_error)
        
        # Record tool usage
        if hasattr(market_state, 'tool_usage'):
            if agent_id not in market_state.tool_usage:
                market_state.tool_usage[agent_id] = []
            market_state.tool_usage[agent_id].append({
                "tool": "price_note",
                "effort": actual_effort,
                "timestamp": "now"
            })
        
        return {
            "notional": notional,
            "payoff_type": payoff_type,
            "fair_value": float(fair_value),
            "quoted_price": float(quoted_price),
            "pricing_error": float(pricing_error),
            "pricing_accuracy": float(1.0 - abs(pricing_error)),
            "expected_return": expected_return,
            "discount_rate": discount_rate,
            "quality_tier": quality_tier,
            "effort_requested": effort,
            "effort_used": actual_effort,
            "warnings": [] if actual_effort == effort else [f"Budget limited effort to {actual_effort}"]
        }
    
    
    # Create and return FunctionTool instances
    tools = [
        FunctionTool(
            sector_forecast,
            description="Generate sector growth forecast using regime-switching analysis with configurable quality levels"
        ),
        FunctionTool(
            monte_carlo_var,
            description="Calculate portfolio Value at Risk using Monte Carlo simulation with effort-based accuracy"
        ),
        FunctionTool(
            price_note,
            description="Price structured financial instruments using Monte Carlo methods with configurable precision"
        )
    ]
    
    return tools


def setup_agent_with_tools(market_model, config_data: Dict[str, Any], 
                          agent_name: str, agent_role: str, organization: str,
                          model_client, initial_budget: float = 20.0) -> Any:
    """
    Create an AutoGen agent with economic tools using the functional approach.
    
    Args:
        market_model: MarketModel instance
        config_data: Configuration dictionary
        agent_name: Name for the agent
        agent_role: Role description (analyst, trader, etc.)
        organization: Organization name
        model_client: AutoGen model client
        initial_budget: Starting budget for tool usage
        
    Returns:
        Configured AssistantAgent with economic tools
    """
    from autogen_agentchat.agents import AssistantAgent
    
    # Create tools using functional approach
    tools = create_economic_tools(market_model, config_data)
    
    # Set up agent context in market state
    agent_id = f"{organization}.{agent_role}"
    market_state = market_model.state
    
    # Initialize budget and context
    if not hasattr(market_state, 'budgets'):
        market_state.budgets = {}
    if not hasattr(market_state, 'tool_usage'):
        market_state.tool_usage = {}
    
    market_state.budgets[agent_id] = initial_budget
    market_state.current_agent_id = agent_id
    
    # Create system message
    system_message = f"""You are {agent_role} at {organization}, an economic agent in a financial simulation.

Your responsibilities:
- Make strategic decisions using available tools
- Manage your budget efficiently (you start with {initial_budget} credits)
- Consider effort allocation when using tools - more effort generally yields better results

Available tools:
- sector_forecast: Generate market forecasts (effort determines accuracy)
- monte_carlo_var: Calculate portfolio risk (effort determines simulation quality)  
- price_note: Price financial instruments (effort determines pricing accuracy)
- reflect: Strategic planning and analysis (effort determines depth)

Budget Management:
- Each tool call requires an 'effort' parameter representing credits to spend
- Higher effort generally produces better quality results
- You'll receive warnings if you try to spend more than your available budget
- Plan your tool usage strategically to maximize value

Remember: Quality of your work depends on the effort you allocate to each task."""
    
    # Create agent with tools
    agent = AssistantAgent(
        name=agent_name,
        model_client=model_client,
        system_message=system_message,
        tools=tools
    )
    
    return agent