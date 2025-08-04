"""
Pure economic tools for agent use.

These tools take effort parameters and return results with warnings.
They access simulation state and configuration data through a wrapper.
"""

import json
import warnings
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def sector_forecast(sector: str, horizon: int, effort: float, 
                   _state=None, _config=None) -> Dict[str, Any]:
    """
    Generate sector growth forecast based on effort allocation.
    
    Args:
        sector: Sector to forecast (tech, finance, healthcare, energy)
        horizon: Number of periods to forecast
        effort: Credits allocated to this forecast (determines quality)
        _state: Simulation state (injected by wrapper)
        _config: Configuration data (injected by wrapper)
    
    Returns:
        Dictionary with forecast data, effort used, and any warnings
    """
    # Get agent ID from simulation state
    agent_id = _state.current_agent_id if _state else "unknown"
    
    # Check budget and adjust effort if necessary
    available_budget = _state.get_budget(agent_id) if _state else float('inf')
    original_effort = effort
    
    if effort > available_budget:
        warnings.warn(f"Insufficient budget. Requested {effort}, available {available_budget}. Using available budget.")
        effort = available_budget
    
    # Deduct effort from budget
    if _state:
        _state.deduct_budget(agent_id, effort)
        _state.record_tool_usage(agent_id, "sector_forecast", effort)
    
    # Load market data from config
    market_data = _config.get("market_data", {}) if _config else {}
    sectors_data = market_data.get("sectors", {})
    default_data = market_data.get("default_sector", {})
    base_params = sectors_data.get(sector, default_data)
    
    if not base_params:
        return {
            "error": "No market data configuration found for sector",
            "sector": sector,
            "effort_used": 0,
            "warnings": ["Configuration missing - unable to generate forecast"]
        }
    
    # Load regime-switching config for quality parameters
    from pathlib import Path
    import json
    
    regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
    if regime_config_path.exists():
        with open(regime_config_path) as f:
            regime_config = json.load(f)
        quality_params = regime_config.get("tool_quality_parameters", {}).get("sector_forecast", {}).get("quality_tiers", {})
    else:
        quality_params = {}
    
    # Determine quality tier based on effort (threshold-based)
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("sector_forecast_thresholds", {})
    
    if effort >= effort_thresholds.get("high", float('inf')):
        quality_tier = "high"
        tier_params = quality_params.get("high", {"noise_factor": 0.1, "confidence": 0.9})
    elif effort >= effort_thresholds.get("medium", float('inf')):
        quality_tier = "medium"
        tier_params = quality_params.get("medium", {"noise_factor": 0.3, "confidence": 0.7})
    else:
        quality_tier = "low"
        tier_params = quality_params.get("low", {"noise_factor": 0.6, "confidence": 0.5})
    
    noise_factor = tier_params.get("noise_factor", 0.3)
    confidence = tier_params.get("confidence", 0.7)
    
    # Generate forecast with quality-dependent noise
    forecast = []
    for _ in range(horizon):
        base_rate = np.random.normal(base_params["mean"], base_params["std"])
        noise = np.random.normal(0, base_params["std"] * noise_factor)
        forecast.append(float(base_rate + noise))
    
    result = {
        "sector": sector,
        "horizon": horizon,
        "forecast": forecast,
        "quality_tier": quality_tier,
        "confidence": confidence,
        "effort_requested": original_effort,
        "effort_used": effort,
        "warnings": [] if effort == original_effort else [f"Budget limited effort to {effort}"]
    }
    
    return result


def monte_carlo_var(portfolio: Dict[str, Any], effort: float,
                   _state=None, _config=None) -> Dict[str, Any]:
    """
    Calculate Value at Risk using Monte Carlo simulation.
    
    Args:
        portfolio: Portfolio data (value, volatility, etc.)
        effort: Credits allocated to this analysis
        _state: Simulation state (injected by wrapper)
        _config: Configuration data (injected by wrapper)
    
    Returns:
        Dictionary with VaR results, effort used, and any warnings
    """
    # Get agent ID and handle budget
    agent_id = _state.current_agent_id if _state else "unknown"
    available_budget = _state.get_budget(agent_id) if _state else float('inf')
    original_effort = effort
    
    if effort > available_budget:
        warnings.warn(f"Insufficient budget. Requested {effort}, available {available_budget}. Using available budget.")
        effort = available_budget
    
    # Deduct effort from budget
    if _state:
        _state.deduct_budget(agent_id, effort)
        _state.record_tool_usage(agent_id, "monte_carlo_var", effort)
    
    # Load regime-switching config for simulation parameters
    from pathlib import Path
    import json
    
    regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
    if regime_config_path.exists():
        with open(regime_config_path) as f:
            regime_config = json.load(f)
        sim_params = regime_config.get("tool_quality_parameters", {}).get("monte_carlo_var", {}).get("simulation_sizes", {})
    else:
        sim_params = {}
    
    # Load configuration
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("monte_carlo_thresholds", {})
    
    # Determine simulation parameters based on effort
    if effort >= effort_thresholds.get("high", float('inf')):
        n_simulations = sim_params.get("high", 50000)
        quality_tier = "high"
    elif effort >= effort_thresholds.get("medium", float('inf')):
        n_simulations = sim_params.get("medium", 10000)
        quality_tier = "medium"
    else:
        n_simulations = sim_params.get("low", 1000)
        quality_tier = "low"
    
    # Extract portfolio parameters
    portfolio_value = portfolio.get("value", 1000000)
    volatility = portfolio.get("volatility", 0.15)
    confidence = portfolio.get("confidence", 0.95)
    
    # Run Monte Carlo simulation
    returns = np.random.normal(0, volatility, n_simulations)
    portfolio_values = portfolio_value * (1 + returns)
    losses = portfolio_value - portfolio_values
    
    # Calculate VaR
    var_95 = float(np.percentile(losses, confidence * 100))
    
    result = {
        "var_95": var_95,
        "portfolio_value": portfolio_value,
        "confidence": confidence,
        "max_loss": float(np.max(losses)),
        "expected_loss": float(np.mean(losses)),
        "n_simulations": n_simulations,
        "quality_tier": quality_tier,
        "effort_requested": original_effort,
        "effort_used": effort,
        "warnings": [] if effort == original_effort else [f"Budget limited effort to {effort}"]
    }
    
    return result


def price_note(payoff_fn: Dict[str, Any], forecast: Dict[str, Any], 
               discount_curve: Dict[str, Any], effort: float,
               _state=None, _config=None) -> Dict[str, Any]:
    """
    Price a structured note given payoff function and market forecast.
    
    Args:
        payoff_fn: Payoff function parameters
        forecast: Market forecast data
        discount_curve: Discount rate information
        effort: Credits allocated to pricing analysis
        _state: Simulation state (injected by wrapper)
        _config: Configuration data (injected by wrapper)
    
    Returns:
        Dictionary with pricing results, effort used, and any warnings
    """
    # Get agent ID and handle budget
    agent_id = _state.current_agent_id if _state else "unknown"
    available_budget = _state.get_budget(agent_id) if _state else float('inf')
    original_effort = effort
    
    if effort > available_budget:
        warnings.warn(f"Insufficient budget. Requested {effort}, available {available_budget}. Using available budget.")
        effort = available_budget
    
    # Deduct effort from budget
    if _state:
        _state.deduct_budget(agent_id, effort)
        _state.record_tool_usage(agent_id, "price_note", effort)
    
    # Load regime-switching config for pricing parameters
    from pathlib import Path
    import json
    
    regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
    if regime_config_path.exists():
        with open(regime_config_path) as f:
            regime_config = json.load(f)
        pricing_params = regime_config.get("tool_quality_parameters", {}).get("price_note", {}).get("pricing_error_std", {})
    else:
        pricing_params = {}
    
    # Load configuration
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("price_note_thresholds", {})
    
    # Determine pricing accuracy based on effort
    if effort >= effort_thresholds.get("high", float('inf')):
        pricing_error_std = pricing_params.get("high", 0.01)
        quality_tier = "high"
    elif effort >= effort_thresholds.get("medium", float('inf')):
        pricing_error_std = pricing_params.get("medium", 0.05)
        quality_tier = "medium"
    else:
        pricing_error_std = pricing_params.get("low", 0.10)
        quality_tier = "low"
    
    # Extract parameters
    base_price = payoff_fn.get("notional", 100)
    growth_factor = np.mean(forecast.get("forecast", [0.05]))
    discount_rate = discount_curve.get("rate", 0.03)
    
    # Calculate fair price with effort-dependent error
    fair_price = base_price * (1 + growth_factor) / (1 + discount_rate)
    pricing_error = np.random.normal(0, pricing_error_std)
    final_price = fair_price * (1 + pricing_error)
    
    result = {
        "fair_price": float(fair_price),
        "quoted_price": float(final_price),
        "quality_tier": quality_tier,
        "growth_factor": float(growth_factor),
        "discount_rate": discount_rate,
        "pricing_error": float(pricing_error),
        "pricing_accuracy": 1.0 - abs(pricing_error),
        "effort_requested": original_effort,
        "effort_used": effort,
        "warnings": [] if effort == original_effort else [f"Budget limited effort to {effort}"]
    }
    
    return result


def reflect(topic: str, effort: float, _state=None, _config=None) -> Dict[str, Any]:
    """
    Generate strategic reflection and planning.
    
    Args:
        topic: Topic or situation to reflect on
        effort: Credits allocated to reflection
        _state: Simulation state (injected by wrapper)
        _config: Configuration data (injected by wrapper)
    
    Returns:
        Dictionary with reflection results, effort used, and any warnings
    """
    # Get agent ID and handle budget
    agent_id = _state.current_agent_id if _state else "unknown"
    available_budget = _state.get_budget(agent_id) if _state else float('inf')
    original_effort = effort
    
    if effort > available_budget:
        warnings.warn(f"Insufficient budget. Requested {effort}, available {available_budget}. Using available budget.")
        effort = available_budget
    
    # Deduct effort from budget
    if _state:
        _state.deduct_budget(agent_id, effort)
        _state.record_tool_usage(agent_id, "reflect", effort)
    
    # Load regime-switching config for reflection parameters
    from pathlib import Path
    import json
    
    regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
    if regime_config_path.exists():
        with open(regime_config_path) as f:
            regime_config = json.load(f)
        reflect_params = regime_config.get("tool_quality_parameters", {}).get("reflect", {}).get("reflection_parameters", {})
        confidence_formula = regime_config.get("tool_quality_parameters", {}).get("reflect", {}).get("confidence_formula", {})
    else:
        reflect_params = {}
        confidence_formula = {}
    
    # Load configuration
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("reflect_thresholds", {})
    
    # Determine reflection depth based on effort
    if effort >= effort_thresholds.get("high", float('inf')):
        quality_tier = "high"
        tier_params = reflect_params.get("high", {"depth": "deep", "num_insights": 5, "num_actions": 4})
    elif effort >= effort_thresholds.get("medium", float('inf')):
        quality_tier = "medium"
        tier_params = reflect_params.get("medium", {"depth": "moderate", "num_insights": 3, "num_actions": 2})
    else:
        quality_tier = "low"
        tier_params = reflect_params.get("low", {"depth": "shallow", "num_insights": 1, "num_actions": 1})
    
    reflection_depth = tier_params.get("depth", "moderate")
    num_insights = tier_params.get("num_insights", 3)
    num_actions = tier_params.get("num_actions", 2)
    
    # Generate reflection content (simplified - in practice would use LLM)
    insights = [f"Insight {i+1} about {topic}" for i in range(num_insights)]
    next_actions = [f"Action {i+1} based on reflection" for i in range(num_actions)]
    
    result = {
        "topic": topic,
        "reflection_depth": reflection_depth,
        "quality_tier": quality_tier,
        "insights": insights,
        "next_actions": next_actions,
        "confidence": min(
            confidence_formula.get("max", 0.9), 
            confidence_formula.get("base", 0.3) + effort * confidence_formula.get("multiplier", 0.2)
        ),  # Higher effort = higher confidence
        "effort_requested": original_effort,
        "effort_used": effort,
        "warnings": [] if effort == original_effort else [f"Budget limited effort to {effort}"]
    }
    
    return result
