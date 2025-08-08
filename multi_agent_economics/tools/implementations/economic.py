"""
Core implementations for economic analysis tools.

These functions handle ALL parameter unpacking from market state and return
Pydantic response models directly. Wrappers only handle budget/logging.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ..schemas import SectorForecastResponse, MonteCarloVarResponse, PostToMarketResponse
from ...models.market_for_finance import (
    ForecastData, build_confusion_matrix, generate_forecast_signal, 
    transition_regimes, categorical_draw
)


def sector_forecast_impl(
    market_model, 
    config_data: Dict[str, Any], 
    sector: str, 
    horizon: int, 
    effort: float
) -> SectorForecastResponse:
    """
    Generate sector forecast using market framework methods.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to forecast
        horizon: Number of periods (ignored for now)
        effort: Effort level allocated
    
    Returns:
        SectorForecastResponse: Complete response with embedded ForecastData
    """
    # Extract configuration parameters
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("sector_forecast", {})
    
    # Map effort to quality tier
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 5.0, "medium": 2.0})
    if effort >= effort_thresholds.get("high", 5.0):
        quality_tier = "high"
    elif effort >= effort_thresholds.get("medium", 2.0):
        quality_tier = "medium"
    else:
        quality_tier = "low"
    
    # Map quality tier to accuracy parameter
    effort_level_quality_mapping = tool_params.get("effort_level_quality_mapping", {
        "high": 0.9, "medium": 0.7, "low": 0.5
    })
    forecast_quality = effort_level_quality_mapping.get(quality_tier, 0.5)
    
    # Get current regime for the sector
    current_regime = state.current_regimes.get(sector, 0)
    
    # Determine number of regimes (K) from state
    if hasattr(state, 'regime_parameters') and sector in state.regime_parameters:
        K = len(state.regime_parameters[sector])
    elif hasattr(state, 'transition_matrices') and sector in state.transition_matrices:
        K = state.transition_matrices[sector].shape[0]
    else:
        # Default from config
        K = tool_params.get("default_num_regimes", 2)
    
    # Look up true next regime from pre-generated regime history
    next_period = state.current_period + 1
    if hasattr(state, 'regime_history') and len(state.regime_history) > next_period:
        # Use pre-generated regime from history
        true_next_regime = state.regime_history[next_period]["regimes"][sector]
    elif hasattr(state, 'transition_matrices') and sector in state.transition_matrices:
        # Fallback: simulate using transition matrix if no history available
        transition_matrix = state.transition_matrices[sector]
        transition_probs = transition_matrix[current_regime]
        true_next_regime = categorical_draw(transition_probs)
    else:
        # Last resort: use config parameters for fallback behavior
        regime_persistence = tool_params.get("default_regime_persistence", 0.8)
        if np.random.random() < regime_persistence:
            true_next_regime = current_regime
        else:
            # Uniform probability over other regimes
            other_regimes = [i for i in range(K) if i != current_regime]
            true_next_regime = np.random.choice(other_regimes) if other_regimes else current_regime
    
    # Use existing market_for_finance methods with config parameters
    base_quality = tool_params.get("base_forecast_quality", 0.6)
    confusion_matrix = build_confusion_matrix(forecast_quality, K, base_quality)
    forecast_result = generate_forecast_signal(true_next_regime, confusion_matrix)
    
    # Create ForecastData object
    forecast_data = ForecastData(
        sector=sector,
        predicted_regime=forecast_result["predicted_regime"],
        confidence_vector=forecast_result["confidence_vector"]
    )
    
    # Return SectorForecastResponse with embedded ForecastData
    return SectorForecastResponse(
        sector=sector,
        forecast=forecast_data,
        quality_tier=quality_tier,
        effort_used=effort
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


def post_to_market_impl(
    market_model,
    config_data: Dict[str, Any],
    notional: float,
    payoff_type: str,
    underlying_forecast: List[float],
    discount_rate: float,
    effort: float
) -> PostToMarketResponse:
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