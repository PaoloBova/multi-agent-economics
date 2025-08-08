"""
Core implementations for economic analysis tools.

These functions handle ALL parameter unpacking from market state and return
Pydantic response models directly. Wrappers only handle budget/logging.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ..schemas import (
    SectorForecastResponse, MonteCarloVarResponse, PostToMarketResponse,
    HistoricalPerformanceResponse, BuyerPreferenceResponse, CompetitivePricingResponse
)
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
        PostToMarketResponse: Complete Pydantic response
    """
    # Unpack parameters from config
    tool_params = config_data.get("tool_parameters", {}).get("post_to_market", {})
    
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
    
    return PostToMarketResponse(
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


# Market Research Tool Implementations

def analyze_historical_performance_impl(
    market_model,
    config_data: Dict[str, Any], 
    sector: str,
    effort: float
) -> HistoricalPerformanceResponse:
    """
    Analyze historical performance-revenue relationships within a sector.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to analyze
        effort: Effort level allocated
    
    Returns:
        HistoricalPerformanceResponse: Analysis of performance tiers and revenue patterns
    """
    # Extract configuration parameters
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("analyze_historical_performance", {})
    
    # Map effort to quality tier and analysis parameters
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 5.0, "medium": 2.0})
    lookback_periods_config = tool_params.get("lookback_periods", {"high": 20, "medium": 10, "low": 5})
    sample_noise_config = tool_params.get("sample_noise", {"high": 0.05, "medium": 0.15, "low": 0.30})
    
    if effort >= effort_thresholds.get("high", 5.0):
        quality_tier = "high"
        lookback_periods = lookback_periods_config.get("high", 20)
        noise_factor = sample_noise_config.get("high", 0.05)
    elif effort >= effort_thresholds.get("medium", 2.0):
        quality_tier = "medium"
        lookback_periods = lookback_periods_config.get("medium", 10)
        noise_factor = sample_noise_config.get("medium", 0.15)
    else:
        quality_tier = "low"
        lookback_periods = lookback_periods_config.get("low", 5)
        noise_factor = sample_noise_config.get("low", 0.30)
    
    # Filter trades by sector from historical data
    all_trades = getattr(state, 'all_trades', [])
    sector_trades = [trade for trade in all_trades if sector.lower() in trade.good_id.lower()]
    
    # Apply lookback period limitation (simulate data availability based on effort)
    current_period = getattr(state, 'current_period', 0)
    min_period = max(0, current_period - lookback_periods)
    
    # For testing, we'll use all sector trades but limit sample size based on effort
    sample_size = min(len(sector_trades), lookback_periods * 2) if sector_trades else 0
    
    warnings = []
    performance_mapping = getattr(state, 'performance_mapping', {})
    
    if not sector_trades:
        warnings.append(f"No historical trades found for sector '{sector}'")
        return HistoricalPerformanceResponse(
            sector=sector,
            performance_tiers={},
            revenue_patterns={},
            analysis_quality=0.0,
            sample_size=0,
            lookback_periods=lookback_periods,
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=warnings,
            data_quality="insufficient"
        )
    
    if not performance_mapping:
        warnings.append("No performance mapping available - using price-based heuristics")
    
    # Group trades by performance level
    performance_groups = {"high": [], "medium": [], "low": []}
    
    for trade in sector_trades[:sample_size]:
        if trade.good_id in performance_mapping:
            performance = performance_mapping[trade.good_id]
            # Add noise based on effort quality
            performance += np.random.normal(0, noise_factor * 0.1)
            
            if performance >= 0.8:
                performance_groups["high"].append(trade)
            elif performance >= 0.65:
                performance_groups["medium"].append(trade)
            else:
                performance_groups["low"].append(trade)
        else:
            # Fallback: use price as performance proxy (higher price = better performance)
            if trade.price >= 100:
                performance_groups["high"].append(trade)
            elif trade.price >= 70:
                performance_groups["medium"].append(trade)
            else:
                performance_groups["low"].append(trade)
    
    # Analyze performance tiers
    performance_tiers = {}
    revenue_patterns = {}
    
    for tier, trades in performance_groups.items():
        if trades:
            prices = [trade.price for trade in trades]
            avg_revenue = np.mean(prices)
            
            # Add analysis noise based on effort
            avg_revenue += np.random.normal(0, avg_revenue * noise_factor)
            
            performance_tiers[tier] = {
                "avg_revenue": float(avg_revenue),
                "revenue_std": float(np.std(prices)) if len(prices) > 1 else 0.0,
                "sample_count": len(trades),
                "confidence": max(0.5, 1.0 - noise_factor)
            }
            revenue_patterns[tier] = prices
    
    # Calculate analysis quality
    analysis_quality = max(0.1, 1.0 - noise_factor - (0.1 if len(sector_trades) < 10 else 0.0))
    
    if sample_size < 5:
        warnings.append("Small sample size may reduce analysis reliability")
    
    data_quality = "good" if quality_tier == "high" else "fair" if quality_tier == "medium" else "limited"
    
    return HistoricalPerformanceResponse(
        sector=sector,
        performance_tiers=performance_tiers,
        revenue_patterns=revenue_patterns,
        analysis_quality=analysis_quality,
        sample_size=sample_size,
        lookback_periods=lookback_periods,
        quality_tier=quality_tier,
        effort_used=effort,
        warnings=warnings,
        data_quality=data_quality
    )


def analyze_buyer_preferences_impl(
    market_model,
    config_data: Dict[str, Any],
    sector: str,
    effort: float
) -> BuyerPreferenceResponse:
    """
    Analyze buyer preference patterns within a sector.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to analyze
        effort: Effort level allocated
    
    Returns:
        BuyerPreferenceResponse: Analysis of buyer preferences and patterns
    """
    # Extract configuration parameters
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("analyze_buyer_preferences", {})
    
    # Map effort to quality tier and analysis parameters
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 4.0, "medium": 2.0})
    sample_sizes_config = tool_params.get("sample_sizes", {"high": 50, "medium": 20, "low": 10})
    noise_factors_config = tool_params.get("noise_factors", {"high": 0.1, "medium": 0.25, "low": 0.50})
    
    if effort >= effort_thresholds.get("high", 4.0):
        quality_tier = "high"
        target_sample_size = sample_sizes_config.get("high", 50)
        noise_factor = noise_factors_config.get("high", 0.1)
    elif effort >= effort_thresholds.get("medium", 2.0):
        quality_tier = "medium"
        target_sample_size = sample_sizes_config.get("medium", 20)
        noise_factor = noise_factors_config.get("medium", 0.25)
    else:
        quality_tier = "low"
        target_sample_size = sample_sizes_config.get("low", 10)
        noise_factor = noise_factors_config.get("low", 0.50)
    
    # Filter buyers by sector interest (buyers with regime beliefs or trading history in sector)
    all_buyers = getattr(state, 'buyers_state', [])
    sector_buyers = []
    
    for buyer in all_buyers:
        # Include buyer if they have regime beliefs for this sector or sector-specific ID
        if (hasattr(buyer, 'regime_beliefs') and sector in buyer.regime_beliefs) or \
           (hasattr(buyer, 'buyer_id') and sector.lower() in buyer.buyer_id.lower()):
            sector_buyers.append(buyer)
    
    warnings = []
    
    if not sector_buyers:
        warnings.append(f"No buyers found with interest in sector '{sector}'")
        return BuyerPreferenceResponse(
            sector=sector,
            avg_preferences=[0.5, 0.5],  # Default neutral preferences
            preference_distribution={},
            sample_size=0,
            confidence_level=0.0,
            preference_variance=[],
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=warnings,
            data_quality="insufficient"
        )
    
    # Sample buyers based on effort level
    sample_size = min(len(sector_buyers), target_sample_size)
    sampled_buyers = np.random.choice(sector_buyers, size=sample_size, replace=False) if sample_size > 0 else []
    
    if sample_size < 3:
        warnings.append("Small buyer sample may reduce preference accuracy")
    
    # Extract preference data
    all_preferences = []
    for buyer in sampled_buyers:
        if hasattr(buyer, 'attr_mu') and buyer.attr_mu:
            prefs = buyer.attr_mu[:2]  # Take first 2 attributes
            
            # Add noise based on effort quality
            noisy_prefs = [
                max(0.0, min(1.0, pref + np.random.normal(0, noise_factor * 0.2)))
                for pref in prefs
            ]
            all_preferences.append(noisy_prefs)
    
    if not all_preferences:
        warnings.append("No preference data available from sampled buyers")
        return BuyerPreferenceResponse(
            sector=sector,
            avg_preferences=[0.5, 0.5],
            preference_distribution={},
            sample_size=sample_size,
            confidence_level=0.0,
            preference_variance=[],
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=warnings,
            data_quality="limited"
        )
    
    # Calculate average preferences
    preferences_array = np.array(all_preferences)
    avg_preferences = np.mean(preferences_array, axis=0).tolist()
    preference_variance = np.var(preferences_array, axis=0).tolist()
    
    # Calculate preference distribution statistics
    preference_distribution = {}
    for i, attr_name in enumerate(["methodology", "data_quality"]):
        attr_preferences = preferences_array[:, i]
        preference_distribution[attr_name] = {
            "mean": float(np.mean(attr_preferences)),
            "std": float(np.std(attr_preferences)),
            "min": float(np.min(attr_preferences)),
            "max": float(np.max(attr_preferences))
        }
    
    # Calculate confidence level based on sample size and quality
    confidence_level = min(0.95, 0.5 + (sample_size / 20) * (1.0 - noise_factor))
    
    data_quality = "good" if quality_tier == "high" and sample_size >= 10 else \
                  "fair" if quality_tier == "medium" else "limited"
    
    return BuyerPreferenceResponse(
        sector=sector,
        avg_preferences=avg_preferences,
        preference_distribution=preference_distribution,
        sample_size=sample_size,
        confidence_level=confidence_level,
        preference_variance=preference_variance,
        quality_tier=quality_tier,
        effort_used=effort,
        warnings=warnings,
        data_quality=data_quality
    )


def research_competitive_pricing_impl(
    market_model,
    config_data: Dict[str, Any],
    sector: str,
    effort: float
) -> CompetitivePricingResponse:
    """
    Research competitive pricing patterns within a sector.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to analyze
        effort: Effort level allocated
    
    Returns:
        CompetitivePricingResponse: Analysis of competitive pricing patterns
    """
    # Extract configuration parameters
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("research_competitive_pricing", {})
    
    # Map effort to quality tier and analysis parameters
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 3.0, "medium": 1.5})
    lookback_periods_config = tool_params.get("lookback_periods", {"high": 15, "medium": 8, "low": 3})
    price_noise_config = tool_params.get("price_noise", {"high": 0.02, "medium": 0.08, "low": 0.15})
    
    if effort >= effort_thresholds.get("high", 3.0):
        quality_tier = "high"
        lookback_periods = lookback_periods_config.get("high", 15)
        price_noise = price_noise_config.get("high", 0.02)
    elif effort >= effort_thresholds.get("medium", 1.5):
        quality_tier = "medium"
        lookback_periods = lookback_periods_config.get("medium", 8)
        price_noise = price_noise_config.get("medium", 0.08)
    else:
        quality_tier = "low"
        lookback_periods = lookback_periods_config.get("low", 3)
        price_noise = price_noise_config.get("low", 0.15)
    
    # Filter offers by sector
    all_offers = getattr(state, 'offers', [])
    # Include historical trades as well for more comprehensive pricing data
    all_trades = getattr(state, 'all_trades', [])
    
    sector_offers = [offer for offer in all_offers if sector.lower() in offer.good_id.lower()]
    sector_trades = [trade for trade in all_trades if sector.lower() in trade.good_id.lower()]
    
    warnings = []
    
    # Combine offers and trades for comprehensive pricing analysis
    all_prices = []
    
    # Add current offers
    for offer in sector_offers:
        # Add noise based on effort quality
        noisy_price = offer.price * (1 + np.random.normal(0, price_noise))
        all_prices.append(max(0, noisy_price))
    
    # Add recent trade prices (limit by lookback_periods)
    recent_trades = sector_trades[-lookback_periods*3:] if sector_trades else []
    for trade in recent_trades:
        noisy_price = trade.price * (1 + np.random.normal(0, price_noise))
        all_prices.append(max(0, noisy_price))
    
    sample_size = len(all_prices)
    
    if sample_size == 0:
        warnings.append(f"No competitive pricing data found for sector '{sector}'")
        return CompetitivePricingResponse(
            sector=sector,
            price_statistics={"avg_price": 0.0, "price_std": 0.0, "min_price": 0.0, "max_price": 0.0},
            competitive_landscape={"competition_level": "unknown", "market_depth": 0},
            sample_size=0,
            lookback_periods=lookback_periods,
            price_ranges={},
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=warnings,
            data_quality="insufficient"
        )
    
    if sample_size < 3:
        warnings.append("Limited pricing data may reduce analysis accuracy")
    
    # Calculate price statistics
    price_statistics = {
        "avg_price": float(np.mean(all_prices)),
        "price_std": float(np.std(all_prices)) if len(all_prices) > 1 else 0.0,
        "min_price": float(np.min(all_prices)),
        "max_price": float(np.max(all_prices)),
        "median_price": float(np.median(all_prices))
    }
    
    # Analyze competitive landscape
    competition_level = "high" if sample_size >= 8 else "medium" if sample_size >= 4 else "low"
    price_spread = price_statistics["max_price"] - price_statistics["min_price"]
    price_cv = price_statistics["price_std"] / price_statistics["avg_price"] if price_statistics["avg_price"] > 0 else 0
    
    competitive_landscape = {
        "competition_level": competition_level,
        "market_depth": sample_size,
        "price_spread": float(price_spread),
        "price_volatility": price_cv,
        "market_maturity": "mature" if price_cv < 0.2 else "developing"
    }
    
    # Calculate price ranges by segment
    if len(all_prices) >= 3:
        price_ranges = {
            "premium": float(np.percentile(all_prices, 75)),
            "standard": float(np.percentile(all_prices, 50)),
            "budget": float(np.percentile(all_prices, 25))
        }
    else:
        price_ranges = {"standard": price_statistics["avg_price"]}
    
    data_quality = "good" if quality_tier == "high" and sample_size >= 5 else \
                  "fair" if quality_tier == "medium" else "limited"
    
    return CompetitivePricingResponse(
        sector=sector,
        price_statistics=price_statistics,
        competitive_landscape=competitive_landscape,
        sample_size=sample_size,
        lookback_periods=lookback_periods,
        price_ranges=price_ranges,
        quality_tier=quality_tier,
        effort_used=effort,
        warnings=warnings,
        data_quality=data_quality
    )