"""
Core implementations for economic analysis tools.

These functions handle ALL parameter unpacking from market state and return
Pydantic response models directly. Wrappers only handle budget/logging.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ..schemas import SectorForecastResponse, MonteCarloVarResponse, PriceNoteResponse


def sector_forecast_impl(
    market_model, 
    config_data: Dict[str, Any], 
    sector: str, 
    horizon: int, 
    effort: float
) -> SectorForecastResponse:
    """
    Generate sector forecast with complete parameter unpacking.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to forecast
        horizon: Number of periods
        effort: Effort level allocated
    
    Returns:
        SectorForecastResponse: Complete Pydantic response
    """
    # Unpack parameters from market state and config
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("sector_forecast", {})
    market_data = config_data.get("market_data", {})
    
    # Determine quality tier from effort
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 5.0, "medium": 2.0})
    quality_tiers = tool_params.get("quality_tiers", {
        "high": {"noise_factor": 0.1, "confidence": 0.9},
        "medium": {"noise_factor": 0.3, "confidence": 0.7},
        "low": {"noise_factor": 0.6, "confidence": 0.5}
    })
    
    if effort >= effort_thresholds.get("high", 5.0):
        quality_tier = "high"
    elif effort >= effort_thresholds.get("medium", 2.0):
        quality_tier = "medium"
    else:
        quality_tier = "low"
    
    quality_params = quality_tiers.get(quality_tier, quality_tiers["low"])
    
    # Get sector parameters from market state
    regime_params = state.regime_parameters.get(sector, {})
    current_regime = state.current_regimes.get(sector)
    
    # Get base market data
    sectors_data = market_data.get("sectors", {})
    default_data = market_data.get("default_sector", {"mean": 0.06, "std": 0.15})
    base_params = sectors_data.get(sector, default_data)
    
    # Generate forecast
    forecast = []
    warnings = []
    
    # Use regime-switching model if available
    if regime_params and current_regime is not None and current_regime in regime_params:
        regime_data = regime_params[current_regime]
        
        # Extract regime parameters (handle both Pydantic objects and dicts)
        if hasattr(regime_data, 'mu'):
            regime_mu = regime_data.mu
            regime_sigma = regime_data.sigma
        else:
            regime_mu = regime_data.get('mu', base_params["mean"])
            regime_sigma = regime_data.get('sigma', base_params["std"])
        
        # Generate forecast with regime parameters
        for _ in range(horizon):
            base_return = np.random.normal(regime_mu, regime_sigma)
            noise = np.random.normal(0, regime_sigma * quality_params.get("noise_factor", 0.3))
            forecast.append(float(base_return + noise))
    else:
        # Use basic market data parameters
        warnings.append("No regime parameters available, using base market data")
        for _ in range(horizon):
            base_return = np.random.normal(base_params["mean"], base_params["std"])
            noise = np.random.normal(0, base_params["std"] * quality_params.get("noise_factor", 0.3))
            forecast.append(float(base_return + noise))
    
    # Check if sector exists in known sectors
    if sector not in sectors_data and sector not in state.regime_parameters:
        warnings.append(f"Sector '{sector}' not in known sectors, using default parameters")
    
    return SectorForecastResponse(
        sector=sector,
        horizon=horizon,
        forecast=forecast,
        quality_tier=quality_tier,
        confidence=quality_params.get("confidence", 0.5),
        effort_requested=effort,
        effort_used=effort,  # Could be adjusted based on budget constraints
        regime_used=current_regime,
        warnings=warnings
    )


def monte_carlo_var_impl(
    market_model,
    config_data: Dict[str, Any],
    portfolio_value: float,
    volatility: float,
    confidence_level: float,
    effort: float
) -> MonteCarloVarResponse:
    """
    Calculate Monte Carlo VaR with complete parameter unpacking.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        portfolio_value: Portfolio value to analyze
        volatility: Expected volatility
        confidence_level: VaR confidence level
        effort: Effort level allocated
    
    Returns:
        MonteCarloVarResponse: Complete Pydantic response
    """
    # Unpack parameters from config
    tool_params = config_data.get("tool_parameters", {}).get("monte_carlo_var", {})
    
    # Determine quality tier and simulation size from effort
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 5.0, "medium": 2.0})
    simulation_sizes = tool_params.get("simulation_sizes", {"high": 50000, "medium": 10000, "low": 1000})
    
    if effort >= effort_thresholds.get("high", 5.0):
        quality_tier = "high"
        n_simulations = simulation_sizes.get("high", 50000)
    elif effort >= effort_thresholds.get("medium", 2.0):
        quality_tier = "medium"
        n_simulations = simulation_sizes.get("medium", 10000)
    else:
        quality_tier = "low"
        n_simulations = simulation_sizes.get("low", 1000)
    
    # Run Monte Carlo simulation
    returns = np.random.normal(0, volatility, n_simulations)
    portfolio_values = portfolio_value * (1 + returns)
    losses = portfolio_value - portfolio_values
    
    # Calculate VaR at specified confidence level
    var_estimate = float(np.percentile(losses, confidence_level * 100))
    
    # Calculate Expected Shortfall (Conditional VaR)
    tail_losses = losses[losses >= var_estimate]
    expected_shortfall = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_estimate
    
    # Additional risk metrics
    max_loss = float(np.max(losses))
    expected_loss = float(np.mean(losses))
    
    warnings = []
    if n_simulations < 5000:
        warnings.append("Low simulation count may reduce accuracy")
    if confidence_level > 0.99:
        warnings.append("Very high confidence level may be unreliable")
    
    return MonteCarloVarResponse(
        portfolio_value=portfolio_value,
        volatility=volatility,
        confidence_level=confidence_level,
        var_estimate=var_estimate,
        expected_shortfall=expected_shortfall,
        max_loss=max_loss,
        expected_loss=expected_loss,
        n_simulations=n_simulations,
        quality_tier=quality_tier,
        effort_requested=effort,
        effort_used=effort,
        warnings=warnings
    )


def price_note_impl(
    market_model,
    config_data: Dict[str, Any],
    notional: float,
    payoff_type: str,
    underlying_forecast: List[float],
    discount_rate: float,
    effort: float
) -> PriceNoteResponse:
    """
    Price structured note with complete parameter unpacking.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        notional: Notional amount
        payoff_type: Type of payoff structure
        underlying_forecast: Forecast for underlying assets
        discount_rate: Risk-free discount rate
        effort: Effort level allocated
    
    Returns:
        PriceNoteResponse: Complete Pydantic response
    """
    # Unpack parameters from config
    tool_params = config_data.get("tool_parameters", {}).get("price_note", {})
    
    # Determine quality tier from effort
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 7.0, "medium": 3.5})
    pricing_error_std = tool_params.get("pricing_error_std", {"high": 0.005, "medium": 0.02, "low": 0.08})
    
    if effort >= effort_thresholds.get("high", 7.0):
        quality_tier = "high"
        error_std = pricing_error_std.get("high", 0.005)
    elif effort >= effort_thresholds.get("medium", 3.5):
        quality_tier = "medium" 
        error_std = pricing_error_std.get("medium", 0.02)
    else:
        quality_tier = "low"
        error_std = pricing_error_std.get("low", 0.08)
    
    # Calculate expected return from forecast
    expected_return = np.mean(underlying_forecast) if underlying_forecast else 0.05
    
    # Use risk-free rate from market state if not provided
    if discount_rate is None:
        discount_rate = market_model.state.risk_free_rate
    
    # Basic pricing models for different payoff types
    if payoff_type.lower() == "linear":
        fair_value = notional * (1 + expected_return) / (1 + discount_rate)
        
    elif payoff_type.lower() == "barrier":
        # Simplified barrier note pricing
        barrier_probability = 0.8
        barrier_coupon = expected_return * 1.2
        fair_value = notional * (1 + barrier_coupon * barrier_probability) / (1 + discount_rate)
        
    elif payoff_type.lower() == "autocall":
        # Simplified autocall note pricing
        autocall_probability = 0.6
        autocall_coupon = expected_return * 1.5
        fair_value = notional * (1 + autocall_coupon * autocall_probability) / (1 + discount_rate)
        
    elif payoff_type.lower() == "digital":
        # Digital/binary payoff
        strike_probability = 0.5
        digital_payout = expected_return * 2.0
        fair_value = notional * (1 + digital_payout * strike_probability) / (1 + discount_rate)
        
    else:
        # Default to linear pricing
        fair_value = notional * (1 + expected_return) / (1 + discount_rate)
    
    # Apply pricing error based on quality
    pricing_error = np.random.normal(0, error_std)
    quoted_price = fair_value * (1 + pricing_error)
    
    warnings = []
    if not underlying_forecast:
        warnings.append("No underlying forecast provided, using default expected return")
    if payoff_type.lower() not in ["linear", "barrier", "autocall", "digital"]:
        warnings.append(f"Unknown payoff type '{payoff_type}', using linear pricing")
    
    return PriceNoteResponse(
        notional=notional,
        payoff_type=payoff_type,
        fair_value=float(fair_value),
        quoted_price=float(quoted_price),
        pricing_error=float(pricing_error),
        pricing_accuracy=float(1.0 - abs(pricing_error)),
        expected_return=expected_return,
        discount_rate=discount_rate,
        quality_tier=quality_tier,
        effort_requested=effort,
        effort_used=effort,
        warnings=warnings
    )