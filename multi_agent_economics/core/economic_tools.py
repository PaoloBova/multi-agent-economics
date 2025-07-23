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
    sectors_data = market_data.get("sectors", {
        "tech": {"mean": 0.08, "std": 0.15},
        "finance": {"mean": 0.06, "std": 0.12},
        "healthcare": {"mean": 0.07, "std": 0.10},
        "energy": {"mean": 0.04, "std": 0.20}
    })
    default_data = market_data.get("default_sector", {"mean": 0.05, "std": 0.15})
    base_params = sectors_data.get(sector, default_data)
    
    # Determine quality tier based on effort (threshold-based)
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("sector_forecast_thresholds", {
        "high": 5.0,
        "medium": 2.0
    })
    
    if effort >= effort_thresholds.get("high", 5.0):
        quality_tier = "high"
        noise_factor = 0.1
        confidence = 0.9
    elif effort >= effort_thresholds.get("medium", 2.0):
        quality_tier = "medium"
        noise_factor = 0.3
        confidence = 0.7
    else:
        quality_tier = "low"
        noise_factor = 0.6
        confidence = 0.5
    
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
    
    # Load configuration
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("monte_carlo_thresholds", {
        "high": 4.0,
        "medium": 2.0
    })
    
    # Determine simulation parameters based on effort
    if effort >= effort_thresholds.get("high", 4.0):
        n_simulations = 50000
        quality_tier = "high"
    elif effort >= effort_thresholds.get("medium", 2.0):
        n_simulations = 10000
        quality_tier = "medium"
    else:
        n_simulations = 1000
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
    
    # Load configuration
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("price_note_thresholds", {
        "high": 6.0,
        "medium": 3.0
    })
    
    # Determine pricing accuracy based on effort
    if effort >= effort_thresholds.get("high", 6.0):
        pricing_error_std = 0.01
        quality_tier = "high"
    elif effort >= effort_thresholds.get("medium", 3.0):
        pricing_error_std = 0.05
        quality_tier = "medium"
    else:
        pricing_error_std = 0.10
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
    
    # Load configuration
    tool_params = _config.get("tool_parameters", {}) if _config else {}
    effort_thresholds = tool_params.get("reflect_thresholds", {
        "high": 2.0,
        "medium": 1.0
    })
    
    # Determine reflection depth based on effort
    if effort >= effort_thresholds.get("high", 2.0):
        reflection_depth = "deep"
        num_insights = 5
        num_actions = 4
    elif effort >= effort_thresholds.get("medium", 1.0):
        reflection_depth = "moderate"
        num_insights = 3
        num_actions = 2
    else:
        reflection_depth = "shallow"
        num_insights = 1
        num_actions = 1
    
    # Generate reflection content (simplified - in practice would use LLM)
    insights = [f"Insight {i+1} about {topic}" for i in range(num_insights)]
    next_actions = [f"Action {i+1} based on reflection" for i in range(num_actions)]
    
    result = {
        "topic": topic,
        "reflection_depth": reflection_depth,
        "insights": insights,
        "next_actions": next_actions,
        "confidence": min(0.9, 0.3 + effort * 0.2),  # Higher effort = higher confidence
        "effort_requested": original_effort,
        "effort_used": effort,
        "warnings": [] if effort == original_effort else [f"Budget limited effort to {effort}"]
    }
    
    return result
