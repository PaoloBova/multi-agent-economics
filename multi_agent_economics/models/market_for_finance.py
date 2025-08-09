"""
Market for Finance Model

This module implements a market for finance model, simulating the dynamics of
supply and demand, information flow, and budget allocation among agents.
"""

import numpy
from pydantic import Field, BaseModel
from typing import Any, Callable, Union

import numpy as np


# ============================================================================
# Financial market dynamics
# ============================================================================

def transition_regimes(current_regimes: dict, transition_matrices: dict) -> dict:
    """
    Transition regimes for all sectors based on their transition matrices.
    
    Args:
        current_regimes: Dict mapping sector -> current regime index
        transition_matrices: Dict mapping sector -> transition matrix (K x K)
        
    Returns:
        Dict mapping sector -> new regime index
    """
    new_regimes = {}
    
    for sector, current_regime in current_regimes.items():
        if sector not in transition_matrices:
            raise ValueError(f"No transition matrix for sector {sector}")
            
        transition_matrix = transition_matrices[sector]
        
        # Validate transition matrix
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError(f"Transition matrix for {sector} has invalid probabilities")
            
        # Sample next regime based on current regime
        transition_probs = transition_matrix[current_regime]
        new_regime = categorical_draw(transition_probs)
        new_regimes[sector] = new_regime
        
    return new_regimes

def generate_regime_returns(regimes: dict, regime_params: dict) -> dict:
    """
    Generate returns for each sector based on their current regimes.
    
    Args:
        regimes: Dict mapping sector -> current regime index
        regime_params: Dict mapping sector -> {regime_idx -> {mu, sigma}}
        
    Returns:
        Dict mapping sector -> realized return
    """
    returns = {}
    
    for sector, regime in regimes.items():
        if sector not in regime_params:
            raise ValueError(f"No regime parameters for sector {sector}")
            
        if regime not in regime_params[sector]:
            raise ValueError(f"No parameters for regime {regime} in sector {sector}")
            
        params = regime_params[sector][regime]
        mu = params["mu"]
        sigma = params["sigma"]
        
        # Generate return from normal distribution
        return_value = np.random.normal(mu, sigma)
        returns[sector] = return_value
        
    return returns

def generate_regime_history(regimes: dict, regime_params: dict, transition_matrices: dict, history_length: int) -> list[dict]:
    """
    Generate a history of regime returns for each sector.
    
    Args:
        regimes: Dict mapping sector -> current regime index
        regime_params: Dict mapping sector -> {regime_idx -> {mu, sigma}}
        transition_matrices: Dict mapping sector -> transition matrix (K x K)
        history_length: Number of periods to generate returns for
        
    Returns:
        list of dicts containing returns, regimes, and index values by sector over the history
    """
    history = []
    # Initialize index values
    index_values = {sector: 100.0 for sector in regimes.keys()}
    
    for period in range(history_length):
        # Generate returns for current regimes
        returns = generate_regime_returns(regimes, regime_params)
        
        # Update index values with returns
        for sector, return_val in returns.items():
            index_values[sector] *= (1 + return_val)
        
        # Store period data
        data = {
            "period": period,
            "returns": returns.copy(),
            "regimes": regimes.copy(),
            "index_values": index_values.copy()
        }
        history.append(data)
        
        # Transition to next period's regimes
        regimes = transition_regimes(regimes, transition_matrices)
    
    return history

def generate_regime_history_from_scenario(scenario_config, num_periods: int) -> list[dict]:
    """
    Generate complete regime history from scenario template.
    
    Args:
        scenario_config: ScenarioConfig object containing regime setup
        num_periods: Number of periods to simulate
        
    Returns:
        List of dicts with period data: [{'period': 0, 'regimes': {...}, 'returns': {...}, 'index_values': {...}}, ...]
    """
    return generate_regime_history(
        regimes=scenario_config.initial_regimes,
        regime_params=scenario_config.regime_parameters,
        transition_matrices=scenario_config.transition_matrices,
        history_length=num_periods
    )


def build_regime_covariance(regimes: dict, regime_volatilities: dict, 
                           fixed_correlations: np.ndarray) -> np.ndarray:
    """
    Build covariance matrix using regime-dependent volatilities and fixed correlations.
    
    Implementation of Strategy 1 from simulation_backend_plan.md:
    Σ_ij = ρ_ij * σ_i^{s^(i)} * σ_j^{s^(j)}
    
    Args:
        regimes: Dict mapping sector -> current regime index
        regime_volatilities: Dict mapping sector -> {regime_idx -> volatility}
        fixed_correlations: Correlation matrix (n_sectors x n_sectors)
        
    Returns:
        Covariance matrix (n_sectors x n_sectors)
    """
    sectors = list(regimes.keys())
    n_sectors = len(sectors)
    
    if fixed_correlations.shape != (n_sectors, n_sectors):
        raise ValueError(f"Correlation matrix shape {fixed_correlations.shape} doesn't match {n_sectors} sectors")
        
    # Extract volatilities for current regimes
    volatilities = np.zeros(n_sectors)
    for i, sector in enumerate(sectors):
        regime = regimes[sector]
        if sector not in regime_volatilities or regime not in regime_volatilities[sector]:
            raise ValueError(f"Missing volatility for sector {sector}, regime {regime}")
        volatilities[i] = regime_volatilities[sector][regime]
    
    # Build covariance matrix: Σ_ij = ρ_ij * σ_i * σ_j
    covariance_matrix = np.outer(volatilities, volatilities) * fixed_correlations
    
    return covariance_matrix

def build_confusion_matrix(forecast_quality: float,
                           K: int,
                           base_quality: float = 0.6) -> np.ndarray:
    """
    Build confusion matrix for forecasting based on quality and number of regimes.
    Args:
        forecast_quality: Quality parameter [0, 1] (higher = more accurate)
        K: Number of regimes
        base_quality: Base forecasting accuracy (0.5 = random, 1.0 = perfect)
    Returns:
        Confusion matrix (K x K) where entry [i,j] = P(forecast=j | true=i)
    """
    # Calculate effective accuracy
    q = base_quality + (1 - base_quality) * forecast_quality
    q = np.clip(q, 1/K, 1.0)  # Ensure valid probability
    
    # Build confusion matrix: high probability on diagonal, low off-diagonal
    off_diagonal_prob = (1 - q) / (K - 1) if K > 1 else 0
    confusion_matrix = np.full((K, K), off_diagonal_prob)
    np.fill_diagonal(confusion_matrix, q)
    
    return confusion_matrix

def generate_forecast_signal(true_next_regime: int, confusion_matrix: np.ndarray) -> int:
    """
    Generate forecast signal based on true regime and confusion matrix.
    
    Args:
        true_next_regime: True regime that will occur next
        confusion_matrix: Confusion matrix P(forecast | true_regime)
        
    Returns:
        Forecast data (regime prediction and confidence vector)
    """
    forecast_probs = confusion_matrix[true_next_regime]
    return {"predicted_regime": categorical_draw(forecast_probs),
            "confidence_vector": forecast_probs.tolist()}

def update_belief_with_forecast(current_belief, forecast):
    """
    Update beliefs for a specific sector based on the forecast signal.
    """
    if isinstance(forecast, ForecastData):
        likelihood = forecast.confidence_vector
    else:
        # Legacy dictionary format
        likelihood = forecast["confidence_vector"]
    posterior = current_belief * likelihood
    posterior = posterior / np.sum(posterior) if np.sum(posterior) > 0 else current_belief
    return posterior

def update_agent_beliefs(prior_beliefs: dict,
                         forecasts: dict,
                         subjective_transitions: dict) -> dict:
    """
    Update agent beliefs using Bayesian filtering (HMM filter step).
    
    Two steps:
    1. Prediction: p̂_{t+1} = p_t * Π (transition)
    2. Update: p_{t+1} ∝ p̂_{t+1} * likelihood (Bayes rule)
    
    Args:
        prior_beliefs: Dict mapping sector -> belief vector
        forecasts: Dict mapping sector -> forecast data
        subjective_transitions: Dict mapping sector -> subjective transition matrix
        
    Returns:
        Dict mapping sector -> updated belief vector
    """
    updated_beliefs = {}
    
    for sector in prior_beliefs.keys():
        
        # Step 1: Prediction (prior transition)
        predicted_belief = prior_beliefs[sector] @ subjective_transitions[sector]
        
        if sector not in forecasts:
            # No forecast for this sector, just apply transition
            updated_beliefs[sector] = predicted_belief
            continue
        
        # Step 2: Update with forecast signal (Bayes rule)
        forecast = forecasts[sector]
        updated_beliefs[sector] = update_belief_with_forecast(predicted_belief, forecast)

    return updated_beliefs


def compute_portfolio_moments(agent_beliefs: dict, regime_returns: dict, 
                             regime_volatilities: dict, correlations: np.ndarray) -> tuple:
    """
    Compute expected returns and covariance matrix from agent beliefs.
    
    Implementation of mixture distribution moments from simulation_backend_plan.md:
    E[R] = Σ_s P(s) * μ^s
    Var[R] = Σ_s P(s) * [Σ^s + (μ^s - E[R])(μ^s - E[R])']
    
    Args:
        agent_beliefs: Dict mapping sector -> belief vector over regimes
        regime_returns: Dict mapping sector -> {regime_idx -> expected_return}
        regime_volatilities: Dict mapping sector -> {regime_idx -> volatility}
        correlations: Fixed correlation matrix between sectors
        
    Returns:
        Tuple of (expected_returns_vector, covariance_matrix)
    """
    sectors = list(agent_beliefs.keys())
    n_sectors = len(sectors)
    
    # Compute expected returns for each sector
    expected_returns = np.zeros(n_sectors)
    for i, sector in enumerate(sectors):
        beliefs = agent_beliefs[sector]
        sector_returns = regime_returns[sector]
        
        # E[R_i] = Σ_k P(k) * μ_i^k
        expected_return = sum(beliefs[k] * sector_returns[k] for k in range(len(beliefs)))
        expected_returns[i] = expected_return
    
    # Compute proper mixture covariance matrix
    import itertools
    import logging
    
    # Get number of regimes per sector
    n_regimes = [len(agent_beliefs[sector]) for sector in sectors]
    total_combinations = np.prod(n_regimes)
    
    if total_combinations > 1000:
        logging.warning(f"Computing covariance for {total_combinations} regime combinations")
    
    # Initialize covariance matrix
    covariance_matrix = np.zeros((n_sectors, n_sectors))
    
    # Enumerate all regime combinations
    regime_ranges = [range(n_regimes[i]) for i in range(n_sectors)]
    
    for regime_combo in itertools.product(*regime_ranges):
        # Compute joint probability of this regime combination
        joint_prob = 1.0
        for i, regime in enumerate(regime_combo):
            joint_prob *= agent_beliefs[sectors[i]][regime]
        
        # Compute regime-specific mean vector
        regime_means = np.zeros(n_sectors)
        regime_vols = np.zeros(n_sectors)
        
        for i, sector in enumerate(sectors):
            regime = regime_combo[i]
            regime_means[i] = regime_returns[sector][regime]
            regime_vols[i] = regime_volatilities[sector][regime]
        
        # Regime-specific covariance matrix
        regime_cov = np.outer(regime_vols, regime_vols) * correlations
        
        # Add to mixture: P(s) * [Σ^s + (μ^s - E[R])(μ^s - E[R])^T]
        mean_diff = regime_means - expected_returns
        covariance_matrix += joint_prob * (regime_cov + np.outer(mean_diff, mean_diff))
    
    return expected_returns, covariance_matrix


def optimize_portfolio(expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                      risk_aversion: float, risk_free_rate: float) -> np.ndarray:
    """
    Compute optimal portfolio weights using mean-variance optimization.
    
    Implementation of the portfolio optimization from simulation_backend_plan.md:
    w* = (1/γ) * Σ^{-1} * (μ - R_f * 1)
    
    Args:
        expected_returns: Expected return vector for assets
        covariance_matrix: Covariance matrix of returns
        risk_aversion: Risk aversion parameter γ
        risk_free_rate: Risk-free rate R_f
        
    Returns:
        Optimal portfolio weights
    """
    n_assets = len(expected_returns)
    excess_returns = expected_returns - risk_free_rate
    
    try:
        # Solve: w = (1/γ) * Σ^{-1} * (μ - R_f)
        weights = (1 / risk_aversion) * np.linalg.solve(covariance_matrix, excess_returns)
    except np.linalg.LinAlgError:
        # Fallback to equal weights if covariance matrix is singular
        weights = np.ones(n_assets) / n_assets
        
    return weights


def compute_knowledge_good_impact(buyer_state, knowledge_good, model_state):
    """
    Compute economic value of a knowledge good for a buyer.
    
    Knowledge goods provide sector-specific regime predictions with confidence levels.
    This function calculates the profit impact by comparing portfolio allocations
    with and without the knowledge good information.
    
    Args:
        buyer_state: State of the buyer agent (should contain regime beliefs and risk preferences)
        knowledge_good: Offer object representing the knowledge good
        model_state: MarketState containing current regime info, parameters, forecasts, etc.
        
    Returns:
        Dictionary with economic value, updated beliefs, expected returns, covariance matrices, and weights
    """
    # Look up the actual forecast for this knowledge good
    good_id = knowledge_good.good_id
    if good_id not in model_state.knowledge_good_forecasts:
        return 0.0  # No forecast available
    
    forecast = model_state.knowledge_good_forecasts[good_id]
    forecast_sector = forecast.sector
    
    # Get all sectors for portfolio construction
    sectors = list(model_state.current_regimes.keys())
    if not sectors or forecast_sector not in sectors:
        return 0.0

    current_beliefs = buyer_state.regime_beliefs

    # Get buyer's risk parameters (with defaults)
    risk_aversion = getattr(buyer_state, 'risk_aversion', 2.0)
    risk_free_rate = getattr(model_state, 'risk_free_rate', 0.03)
    
    # Create regime returns and volatilities dicts from regime parameters
    regime_returns = {}
    regime_volatilities = {}
    for sector, sector_params in model_state.regime_parameters.items():
        regime_returns[sector] = {regime: params.mu for regime, params in sector_params.items()}
        regime_volatilities[sector] = {regime: params.sigma for regime, params in sector_params.items()}
    
    # 1. Compute portfolio allocation with current beliefs
    expected_returns_before, cov_matrix_before = compute_portfolio_moments(
        current_beliefs, 
        regime_returns, 
        regime_volatilities, 
        model_state.regime_correlations
    )
    weights_before = optimize_portfolio(
        expected_returns_before, cov_matrix_before, risk_aversion, risk_free_rate
    )
    
    # 2. Create updated beliefs incorporating knowledge good forecast
    current_belief = buyer_state.regime_beliefs[forecast_sector]
    posterior = update_belief_with_forecast(current_belief, forecast)
    updated_beliefs = current_beliefs.copy()
    updated_beliefs[forecast_sector] = posterior

    # 3. Compute portfolio allocation with updated beliefs
    expected_returns_after, cov_matrix_after = compute_portfolio_moments(
        updated_beliefs, 
        regime_returns, 
        regime_volatilities, 
        model_state.regime_correlations
    )
    weights_after = optimize_portfolio(
        expected_returns_after, cov_matrix_after, risk_aversion, risk_free_rate
    )
    
    # 4. Calculate actual returns based on current period data from regime history
    current_period_data = model_state.regime_history[model_state.current_period]
    sector_returns = np.array([current_period_data.returns[sector]
                               for sector in current_period_data.returns])

    # 5. Calculate profits for both allocations
    profit_before = np.dot(weights_before, sector_returns)
    profit_after = np.dot(weights_after, sector_returns)
    
    # 6. Return the economic value (profit difference)
    economic_value = profit_after - profit_before

    return {"economic_value": economic_value,
            "beliefs_before": current_beliefs,
            "beliefs_after": updated_beliefs,
            "expected_returns_before": expected_returns_before,
            "expected_returns_after": expected_returns_after,
            "cov_matrix_before": cov_matrix_before,
            "cov_matrix_after": cov_matrix_after,
            "weights_before": weights_before,
            "weights_after": weights_after}

# ============================================================================
# Choice models for knowledge goods
# ============================================================================

def renormalize(probs):
    """
    Renormalize a 1D array of non-negative probabilities so they sum to 1.
    Zeros remain zeros.
    """
    probs = numpy.array(probs, dtype=float)
    total = probs.sum()
    if total > 0:
        return probs / total
    else:
        # If all zero, just return unchanged (or raise an error)
        return probs

def categorical_draw(probs):
    """
    Draw a single index j ∼ Categorical(probs).
    `probs` must sum to 1 (up to floating-point).
    """
    probs = numpy.array(probs, dtype=float)
    # Cumulative sums
    cum = numpy.cumsum(probs)
    u = numpy.random.rand()
    # find first index where cum ≥ u
    return int(numpy.searchsorted(cum, u, side='right'))


def convert_marketing_to_features(marketing_attributes: dict[str, Any], 
                                buyer_conversion_function: dict[str, dict[str, float]],
                                attribute_order: list[str]) -> list[float]:
    """
    Convert seller's marketing attributes to buyer's numeric features using canonical ordering.
    
    Args:
        marketing_attributes: Seller's original marketing attributes (e.g., {"methodology": "premium", "data_coverage": 0.85})
        buyer_conversion_function: Buyer's conversion mapping (e.g., {"methodology": {"premium": 0.9, "standard": 0.7}})
        attribute_order: Canonical order of attributes for consistent attribute vector generation
        
    Returns:
        List of numeric features for buyer's choice model, ordered according to attribute_order
    """
    features = []
    
    for attr_name in attribute_order:
        if attr_name in marketing_attributes and attr_name in buyer_conversion_function:
            attr_value = marketing_attributes[attr_name]
            attr_mapping = buyer_conversion_function[attr_name]
            
            # Handle both categorical and numeric attributes
            if isinstance(attr_value, (int, float)):
                # Numeric attribute - use as-is or apply scaling if defined
                if "numeric_scaling" in attr_mapping:
                    # Apply buyer-specific scaling: feature = base + scale * value
                    base = attr_mapping.get("base", 0.0)
                    scale = attr_mapping.get("scale", 1.0) 
                    feature = base + scale * attr_value
                else:
                    # Use numeric value directly
                    feature = float(attr_value)
            else:
                # Categorical attribute - look up in conversion mapping
                attr_value_str = str(attr_value)
                if attr_value_str in attr_mapping:
                    feature = attr_mapping[attr_value_str]
                else:
                    # Unknown attribute value - use default of 0.0
                    feature = 0.0
        else:
            # Missing attribute or no buyer conversion - use default
            feature = 0.0
        
        features.append(feature)
    
    return features


def get_average_attribute_vector(marketing_attributes: dict[str, Any], 
                               buyers: list, 
                               attribute_order: list[str]) -> list[float]:
    """
    Convert marketing attributes to average attribute vector across all buyers for competitive pricing.
    
    This function is used by the Akerlof Seller to estimate market perception of marketing attributes
    by averaging how all buyers would convert the attributes to numeric features.
    
    Args:
        marketing_attributes: Seller's marketing attributes (e.g., {"innovation": "high", "risk_score": 25})
        buyers: List of BuyerState objects with their conversion functions
        attribute_order: Canonical order of attributes for consistent vector generation
        
    Returns:
        List of averaged numeric features representing market perception of the attributes
    """
    if not buyers or not marketing_attributes or not attribute_order:
        return []
    
    # Convert using each buyer's conversion function
    all_conversions = []
    for buyer in buyers:
        try:
            features = convert_marketing_to_features(
                marketing_attributes, 
                buyer.buyer_conversion_function, 
                attribute_order
            )
            all_conversions.append(features)
        except (ValueError, KeyError, TypeError):
            continue  # Skip buyers without proper conversion functions
    
    if not all_conversions:
        return []
    
    # Average across all buyers
    num_features = len(all_conversions[0])
    return [
        sum(conv[i] for conv in all_conversions) / len(all_conversions) 
        for i in range(num_features)
    ]


def get_marketing_attribute_description(marketing_attributes: dict[str, Any], 
                                      marketing_definitions: dict[str, dict[str, Any]]) -> dict[str, str]:
    """
    Convert marketing attributes to human-readable descriptions using global definitions.
    
    Args:
        marketing_attributes: Marketing attributes to describe
        marketing_definitions: Global definitions of attributes and their meanings
        
    Returns:
        Dictionary mapping attribute names to descriptive strings
    """
    descriptions = {}
    
    for attr_name, attr_value in marketing_attributes.items():
        if attr_name in marketing_definitions:
            definition = marketing_definitions[attr_name]
            attr_type = definition.get("type", "categorical")
            
            if attr_type == "numeric":
                # For numeric attributes, provide range context
                range_info = definition.get("range", [0, 1])
                descriptions[attr_name] = f"{attr_value} (range: {range_info[0]}-{range_info[1]})"
            else:
                # For categorical attributes, use the value directly or map if available
                value_descriptions = definition.get("descriptions", {})
                descriptions[attr_name] = value_descriptions.get(str(attr_value), str(attr_value))
        else:
            descriptions[attr_name] = str(attr_value)
    
    return descriptions

def sample_cart_multidraw(offers, probs, budget, T=10000):
    """
    Build a “cart” by sampling offers without replacement until budget runs out
    or no probability mass remains.
    
    offers: list of objects with a .price attribute
    probs:  list of choice probabilities p_j  (will be renormalized internally)
    budget: total spend limit
    T:      maximum number of draws (optional; if None, infinite)
    """
    probs = renormalize(probs)
    cart = []
    remaining = budget
    draws = 0

    while remaining > 0 and probs.sum() > 0 and (T is None or draws < T):
        j = categorical_draw(probs)
        price = offers[j].price
        if price <= remaining:
            cart.append(offers[j])
            remaining -= price
            # zero‐out that offer to avoid repeats
            probs[j] = 0
            probs = renormalize(probs)
        # even if the item didn't fit, we count the draw
        draws += 1

    return cart

def solve_continuous_knapsack(U, p, B):
    # U, p are arrays of same length; B is budget
    # find λ>0 s.t. sum_j min(1, max(0, U[j]/λ)) * p[j] == B
    λ_low, λ_high = 1e-9, max(U)/1e-9
    for _ in range(50):
        λ = 0.5*(λ_low + λ_high)
        spend = sum(min(1, max(0, U[j]/λ)) * p[j] for j in range(len(U)))
        if spend > B:
            λ_low = λ
        else:
            λ_high = λ
    x = [min(1, max(0, U[j]/λ)) for j in range(len(U))]
    return x

def greedy_budget_choice(offers, V, B):
    """
    offers: list of Offer
    V[j]: buyer’s gross valuation for offer j
    B: initial budget
    """
    scores = [ ( (V[j]-offers[j].price)/offers[j].price, j )
               for j in range(len(offers)) ]
    # stable‐sort descending, but break exact ties by a coin flip
    scores.sort(key=lambda x: (x[0], numpy.random.random()), reverse=True)

    cart = []
    remaining = B
    for _, j in scores:
        price = offers[j].price
        net   = V[j] - price
        if net > 0 and price <= remaining:
            cart.append(offers[j])
            remaining -= price
    return cart

def choice_model(buyer_state, offers, config, model_state):
    """Dispatches to the appropriate choice model based on configuration."""
    # Calculate value of each offer to the buyer using canonical attribute ordering
    V = lambda offer: numpy.dot(buyer_state.attr_weights[model_state.knowledge_good_forecasts[offer.good_id].sector], 
                                convert_marketing_to_features(offer.marketing_attributes, 
                                                              buyer_state.buyer_conversion_function,
                                                              model_state.attribute_order))
    value_of = [V(offer) for offer in offers]
    buyer_state.value_of = value_of  # Store for later use
    if "cart_draws" not in config:
        config.cart_draws = None  # Default to no limit

    if config.choice_model == "greedy":
        choices = greedy_budget_choice(offers, value_of, buyer_state.budget)
    elif config.choice_model == "logit_cart":
        choice_probs = numpy.exp(value_of) / numpy.sum(numpy.exp(value_of))
        choices = sample_cart_multidraw(offers, choice_probs, buyer_state.budget, T=config.cart_draws)
    elif config.choice_model == "knapsack":
        p = [o.price for o in offers]
        fractional_choices = solve_continuous_knapsack(value_of, p, buyer_state.budget)
        # Use sample_cart_multidraw to convert factional choices to discrete choices
        choices = sample_cart_multidraw(offers, fractional_choices, buyer_state.budget, T=config.cart_draws)
    else:
        raise ValueError(f"Unknown choice model: {config.choice_model}")
    return choices

def clear_market(choices, _model, config):
    """
    choices: list of (buyer_id, Offer) pairs
    Since goods are non-rival, every choice is filled.
    Transform into TradeData objects.
    """
    trades = []
    if config.matching_rule == "non-rival":
        for buyer_id, offer in choices:
            trade = TradeData(
                buyer_id=buyer_id,
                seller_id=offer.seller,  # Use seller field from Offer
                price=offer.price,
                quantity=1,  # Default quantity
                good_id=offer.good_id
                period=_model.state.current_period
            )
            trades.append(trade)
    else:
        raise ValueError(f"Unknown matching rule: {config.matching_rule}")
    return trades

def resolve_ex_post_valuations(trades, model, market_cfg):
    """
    Resolve ex-post valuations for knowledge goods purchased by buyers.
    
    For each buyer, applies multiple knowledge goods sequentially to compute
    the individual economic impact of each knowledge good. Sequential application
    ensures proper belief updating when multiple goods affect the same sector.
    """
    # Create lookup structures
    buyer_lookup = {buyer_state.buyer_id: buyer_state for buyer_state in model.state.buyers_state}
    offer_lookup = {offer.good_id: offer for offer in model.state.offers}
    
    # Group knowledge good trades by buyer and sector
    buyer_sector_trades = {}
    for trade in trades:
        # Skip if not a knowledge good
        if trade.good_id not in model.state.knowledge_good_forecasts:
            continue
            
        # Get sector for this knowledge good
        forecast = model.state.knowledge_good_forecasts[trade.good_id]
        sector = forecast.sector
        
        # Group by (buyer_id, sector)
        key = (trade.buyer_id, sector)
        if key not in buyer_sector_trades:
            buyer_sector_trades[key] = []
        buyer_sector_trades[key].append(trade)
    
    # Process each buyer-sector group sequentially
    for (buyer_id, sector), sector_trades in buyer_sector_trades.items():
        
        # Get buyer state
        if buyer_id not in buyer_lookup:
            continue
        buyer_state = buyer_lookup[buyer_id]
        
        # Track evolving beliefs for this buyer
        current_beliefs = buyer_state.regime_beliefs.copy()
        
        # Apply each knowledge good sequentially
        for trade in sector_trades:
            # Get knowledge good offer
            if trade.good_id not in offer_lookup:
                continue
            knowledge_good = offer_lookup[trade.good_id]
            
            # Create temporary buyer state with current beliefs
            temp_buyer_state = buyer_state.__class__(**buyer_state.dict())
            temp_buyer_state.regime_beliefs = current_beliefs
            
            # Compute individual impact of this knowledge good
            impact_result = compute_knowledge_good_impact(temp_buyer_state, knowledge_good, model.state)
            
            # Store individual knowledge good impact
            if trade.good_id not in model.state.knowledge_good_impacts:
                model.state.knowledge_good_impacts[trade.good_id] = {}
            model.state.knowledge_good_impacts[trade.good_id][buyer_id] = impact_result["economic_value"]
            
            # Update beliefs for next knowledge good application
            current_beliefs = impact_result["beliefs_after"].copy()
        
        # Update the buyer's actual beliefs after processing all knowledge goods for this sector
        buyer_state.regime_beliefs[sector] = current_beliefs[sector]

def compute_surpluses(trades, model, market_cfg):
    """Apply transfers and record surplus."""
    for buyer_state in model.state.buyers_state:
        buyer_surplus = 0.0
        for trade in trades:
            if trade.buyer_id == buyer_state.buyer_id:
                # Add the economic value of knowledge goods
                if (trade.good_id in model.state.knowledge_good_impacts and 
                    trade.buyer_id in model.state.knowledge_good_impacts[trade.good_id]):
                    buyer_surplus += model.state.knowledge_good_impacts[trade.good_id][trade.buyer_id]
                
                # Subtract cost of purchase
                buyer_surplus -= trade.price * trade.quantity
        buyer_state.surplus = buyer_surplus
        
    for seller_state in model.state.sellers_state:
        seller_surplus = 0.0
        for trade in trades:
            if trade.seller_id == seller_state.org_id:
                seller_surplus += trade.price * trade.quantity
        seller_state.surplus = seller_surplus
        # Take care of any production costs from the seller
        # Assuming production_cost is defined in seller state
        seller_state.surplus -= seller_state.production_cost
        seller_state.total_profits += seller_state.surplus

def update_buyer_preferences_from_knowledge_goods(buyer_state, model_state, obs_noise_var=1.0):
    """
    Update buyer attribute preferences based on individual knowledge good economic impacts.
    
    This approach is more precise than using total surplus because we can directly
    attribute the economic value to specific attribute combinations.
    
    Args:
        buyer_state: Buyer with attr_mu[j] and attr_sigma2[j] preferences
        model_state: MarketState containing knowledge_good_impacts and offers
        obs_noise_var: Observation noise variance for Bayesian updates
    """
    buyer_id = buyer_state.buyer_id
    
    # Create lookup for offers by good_id
    offer_lookup = {offer.good_id: offer for offer in model_state.offers}
    
    # Process each knowledge good this buyer purchased
    for good_id, buyer_impacts in model_state.knowledge_good_impacts.items():
        if buyer_id not in buyer_impacts:
            continue
            
        # Get the economic impact and attribute vector for this knowledge good
        economic_impact = buyer_impacts[buyer_id]
        
        if good_id not in offer_lookup:
            continue
        
        # Retrieve sector
        sector = model_state.knowledge_good_forecasts[good_id].sector

        offer = offer_lookup[good_id]
        x = convert_marketing_to_features(offer.marketing_attributes, 
                                          buyer_state.buyer_conversion_function,
                                          model_state.attribute_order)
        # Bayesian update: economic_impact ≈ x·β + noise, noise~N(0,obs_noise_var)
        # This assumes the economic impact can be linearly attributed to attributes
        attr_mu = buyer_state.attr_mu.get(sector, [0.0] * len(x))
        attr_sigma2 = buyer_state.attr_sigma2.get(sector, [1.0] * len(x))
        for j, x_j in enumerate(x):
            if j >= len(attr_mu) or j >= len(attr_sigma2):
                continue  # Skip if buyer doesn't have preferences for this attribute
                
            # Prior precision τ0 and likelihood precision τL
            τ0 = 1 / attr_sigma2[j]
            τL = x_j**2 / obs_noise_var if x_j != 0 else 0
            
            if τL == 0:  # Skip zero attributes
                continue
                
            # Posterior precision & mean
            τ_post = τ0 + τL
            μ_post = (τ0 * attr_mu[j] + x_j * economic_impact / obs_noise_var) / τ_post

            buyer_state.attr_sigma2[sector][j] = 1 / τ_post
            buyer_state.attr_mu[sector][j] = μ_post

def transition_demand(model, market_cfg):
    """Update buyer preferences based on knowledge good performance."""
    for buyer_state in model.state.buyers_state:
        # Use the new knowledge good impact-based preference updating
        update_buyer_preferences_from_knowledge_goods(buyer_state, model.state)
        # Use new attr_mu and attr_sigma2 to update buyer preferences per sector
        # This will affect the next round's choice model inputs. Sampled from
        # normal distribution.
        for sector in model.state.current_regimes.keys():
            buyer_state.attr_weights[sector] = [0.0] * len(model.state.attribute_order)
            attr_mu = buyer_state.attr_mu.get(sector, [0.0] * len(model.state.attribute_order))
            attr_sigma2 = buyer_state.attr_sigma2.get(sector, [1.0] * len(model.state.attribute_order))
            for j in range(len(model.state.attribute_order)):
                buyer_state.attr_weights[sector][j] = numpy.random.normal(
                    loc=attr_mu[j],
                    scale=numpy.sqrt(attr_sigma2[j])
                )

        # TODO: Add novel demand shocks from market_cfg if needed in the future
        # For now, focusing on preference learning from knowledge good impacts

def run_market_dynamics(model, market_cfg):
    """Market dynamics: match supply and demand, clear trades"""
    offers = model.state.offers
    
    # 1. Each buyer runs choice model to select offers:

    buyers_state = model.state.buyers_state
    choices = []
    for buyer_state in buyers_state:
        offers_chosen = choice_model(buyer_state, offers, market_cfg, model.state)
        for offer in offers_chosen:
            choices.append((buyer_state.buyer_id, offer))
    
    # 2. Clearing mechanism: e.g., discrete double auction or posted‑price matching
    trades = clear_market(choices, model, market_cfg)
    model.state.all_trades.extend(trades)
    model.state.trades = trades  # Current round trades
    
    # 3. Resolve ex-post valuations and surpluses
    resolve_ex_post_valuations(trades, model, market_cfg)
    for trade in trades:
        compute_surpluses(trade, model, market_cfg)

    # 4. Demand-side shock for next round - update buyer preferences based on knowledge good performance
    transition_demand(model, market_cfg)

    return trades

def run_information_dynamics(model, info_cfg):
    """
    Complete information dynamics using regime-switching model.
    
    This function:
    1. Updates regime states for all sectors
    2. Generates regime-dependent returns
    3. Updates agent beliefs based on forecasts
    4. Records forecast accuracy for economic value measurement
    """
    
    # a) Evolve underlying regime states
    model.state.current_regimes = transition_regimes(
        model.state.current_regimes, 
        model.state.regime_transition_matrices
    )
    
    # b) Generate regime-dependent returns
    regime_returns = generate_regime_returns(
        model.state.current_regimes,
        model.state.regime_parameters
    )
    
    # Update index values with regime-dependent returns
    for sector, return_val in regime_returns.items():
        if sector not in model.state.index_values:
            model.state.index_values[sector] = 100.0  # Initial value
        model.state.index_values[sector] *= (1 + return_val)

    return regime_returns

# def allocate_budgets(model, alloc_cfg):
#     "Organizational rebalancing: principals allocate budgets across teams"
#     scores = aggregate_scores(model.state, alloc_cfg)
#     weights = alloc_cfg.pivot_function(scores)
#     redistribute_budget(model.state, weights)

def model_step(model, config):
    "Top‑level model step, invoked once per tick (round)"
    run_market_dynamics(model, config.market_params)
    run_information_dynamics(model, config.info_params)
    # TODO: Handle stochastic arrival of news or shocks
    # allocate_budgets(model, config.budget_params)
    model.state.current_period += 1

def collect_stats(model):
    "Collect statistics from the model state"
    stats = {
        "total_trades": len(model.state.trades),
        "average_price": sum(trade.price for trade in model.state.trades) / len(model.state.trades) if model.state.trades else 0,
    }
    return stats

class RegimeParameters(BaseModel):
    """Parameters for a single regime."""
    mu: float = Field(..., description="Expected return for this regime")
    sigma: float = Field(..., description="Volatility for this regime", gt=0)

class ForecastData(BaseModel):
    """Structure for knowledge good forecast data."""
    sector: str = Field(..., description="Sector being forecasted")
    predicted_regime: int = Field(..., description="Predicted regime index", ge=0)
    confidence_vector: list[float] = Field(..., description="Confidence probabilities for each regime")

class PeriodData(BaseModel):
    """Data for a single period in regime history."""
    period: int = Field(..., description="Period number", ge=0)
    returns: dict[str, float] = Field(..., description="Realized returns by sector")
    regimes: dict[str, int] = Field(..., description="Active regimes by sector")
    index_values: dict[str, float] = Field(..., description="Index values by sector")

class TradeData(BaseModel):
    """Structure for trade information."""
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller")
    price: float = Field(..., description="Trade price", gt=0)
    quantity: int = Field(..., description="Trade quantity", gt=0)
    good_id: str = Field(..., description="ID of the traded good")
    marketing_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Original marketing attributes from the winning offer"
    )
    buyer_conversion_used: dict[str, float] = Field(
        default_factory=dict,
        description="Buyer's converted numeric features used in choice model"
    )
    period: int = Field(..., description="Period in which the trade occurred", ge=0)
    
class Offer(BaseModel):
    """ Represents an offer in the market."""
    good_id: str = Field(..., description="Unique identifier for the good")
    price: float = Field(..., description="Price of the offer", gt=0)
    seller: str = Field(..., description="ID of the seller agent")
    marketing_attributes: dict[str, Any] = Field(
        default_factory=dict, 
        description="Original marketing attributes posted by seller (qualitative/numeric)"
    )

class BuyerState(BaseModel):
    """State of a buyer agent in the market."""
    buyer_id: str = Field(..., description="Unique identifier for the buyer")
    regime_beliefs: dict[str, list[float]] = Field(..., description="Beliefs about regime probabilities by sector")
    risk_aversion: float = Field(default=2.0, description="Risk aversion parameter", gt=0)
    attr_mu: dict[str, list[float]] = Field(default_factory=dict, description="Mean preferences for each attribute by sector")
    attr_sigma2: dict[str, list[float]] = Field(default_factory=dict, description="Variance of preferences for each attribute by sector")
    attr_weights: dict[str, list[float]] = Field(default_factory=dict, description="Current attribute weights for utility calculation by sector")
    budget: float = Field(default=100.0, description="Available budget for purchases", ge=0)
    # TODO: Should budget update dynamically based on trades and expectations?
    surplus: float = Field(default=0.0, description="Current period surplus")
    value_of: list[float] = Field(default_factory=list, description="Calculated values of current offers")
    
    # Marketing attribute system - buyer-specific conversion
    buyer_conversion_function: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Buyer's personal conversion from marketing attributes to numeric features"
    )
    
    def initialize_sector_preferences(self, sectors: list[str], num_attributes: int, 
                                    default_mu: float = 0.5, default_sigma2: float = 1.0):
        """Initialize sector-specific attribute preferences for this buyer."""
        for sector in sectors:
            if sector not in self.attr_mu:
                self.attr_mu[sector] = [default_mu] * num_attributes
            if sector not in self.attr_sigma2:
                self.attr_sigma2[sector] = [default_sigma2] * num_attributes
            if sector not in self.attr_weights:
                # Sample initial weights from prior
                self.attr_weights[sector] = [
                    np.random.normal(default_mu, np.sqrt(default_sigma2)) 
                    for _ in range(num_attributes)
                ]
    
    def ensure_sector_exists(self, sector: str, num_attributes: int):
        """Ensure a sector exists in buyer preferences, adding with defaults if not."""
        if sector not in self.attr_mu:
            self.attr_mu[sector] = [0.5] * num_attributes
        if sector not in self.attr_sigma2:
            self.attr_sigma2[sector] = [1.0] * num_attributes  
        if sector not in self.attr_weights:
            self.attr_weights[sector] = [
                np.random.normal(self.attr_mu[sector][i], np.sqrt(self.attr_sigma2[sector][i]))
                for i in range(num_attributes)
            ]

class SellerState(BaseModel):
    """State of a seller agent in the market."""
    org_id: str = Field(..., description="Organization/seller identifier")
    production_cost: float = Field(default=0.0, description="Cost of producing current offerings", ge=0)
    surplus: float = Field(default=0.0, description="Current period surplus")
    total_profits: float = Field(default=0.0, description="Cumulative profits over all periods")

class MarketState(BaseModel):
    """ Represents the state of a market in the simulation framework."""
    offers: list[Offer] = Field(..., description="List of offers available in the market")
    trades: list[TradeData] = Field(..., description="List of trades executed in the market")
    demand_profile: dict = Field(..., description="Demand profile for the market")
    supply_profile: dict = Field(..., description="Supply profile for the market")
    index_values: dict[str, float] = Field(..., description="Values of market indices")
    
    # Regime-switching state
    current_regimes: dict[str, int] = Field(default_factory=dict, description="Current regime for each sector")
    regime_transition_matrices: dict = Field(default_factory=dict, description="Transition matrices for regime switching")
    regime_parameters: dict[str, dict[int, RegimeParameters]] = Field(default_factory=dict, description="Parameters (mu, sigma) for each regime")
    regime_correlations: Any = Field(default=None, description="Fixed correlation matrix between sectors")
    
    # Agent beliefs and forecasting
    agent_beliefs: dict[str, list[float]] = Field(default_factory=dict, description="Agent beliefs about regimes")
    agent_subjective_transitions: dict = Field(default_factory=dict, description="Agent subjective transition matrices")
    forecast_history: list = Field(default_factory=list, description="History of forecasts and their accuracy")
    
    # Additional tracking
    all_trades: list = Field(default_factory=list, description="Complete history of all trades")
    buyers_state: list = Field(default_factory=list, description="State of buyer agents")
    sellers_state: list = Field(default_factory=list, description="State of seller agents")
    
    # Regime history and simulation state
    regime_history: list[PeriodData] = Field(default_factory=list, description="Pre-generated regime history")
    current_period: int = Field(default=0, description="Current simulation time step", ge=0)
    knowledge_good_impacts: dict[str, dict[str, float]] = Field(default_factory=dict, description="Impact of knowledge goods on buyer profits")
    knowledge_good_forecasts: dict[str, ForecastData] = Field(default_factory=dict, description="Mapping good_id -> forecast data")
    
    # Market research data
    performance_mapping: dict[str, float] = Field(default_factory=dict, description="Mapping good_id -> actual performance level for market research")
    
    # Marketing attribute system - global definitions
    marketing_attribute_definitions: dict[str, dict[str, Any]] = Field(
        default_factory=dict, 
        description="Global definitions of available marketing attributes and their valid values"
    )
    
    # Canonical ordering for consistent attribute vector generation
    attribute_order: list[str] = Field(
        default_factory=list,
        description="Canonical order of attributes for consistent attribute vector generation across all buyers"
    )
    
    # Risk parameters
    risk_free_rate: float = Field(default=0.03, description="Risk-free rate for portfolio optimization", ge=0)
    
    # Agent context and tool management
    current_agent_id: str = Field(default="unknown", description="Current agent context for tool calls")
    budgets: dict[str, float] = Field(default_factory=dict, description="Agent budgets for tool usage")
    tool_usage: dict[str, list[dict]] = Field(default_factory=dict, description="History of tool usage by agent")

class MarketModel(BaseModel):
    """ Represents a market model in the simulation framework."""
    model_config = {"arbitrary_types_allowed": True}
    
    id: int = Field(..., description="Unique identifier for the market")
    name: str = Field(..., description="Name of the market")
    agents: list = Field(..., description="List of agents participating in the market")
    state: MarketState = Field(..., description="Current state of the market, including prices, trades, and other relevant data")
    step: Callable = Field(..., description="Function to execute a step in the market model")
    collect_stats: Callable = Field(..., description="Function to collect statistics from the market model")