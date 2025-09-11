"""
Core implementations for economic analysis tools.

These functions handle ALL parameter unpacking from market state and return
Pydantic response models directly. Wrappers only handle budget/logging.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import uuid
from ..schemas import (
    SectorForecastResponse, PostToMarketResponse,
    HistoricalPerformanceResponse, BuyerPreferenceResponse, CompetitivePricingResponse,
    AttributeAnalysis, WTPDataPoint
)
from ...models.market_for_finance import (
    ForecastData, Offer, build_confusion_matrix, generate_forecast_signal, 
    transition_regimes, categorical_draw, get_marketing_attribute_description,
    convert_marketing_to_features, choice_model
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
        horizon: Number of periods
        effort: Effort level allocated
    
    Returns:
        SectorForecastResponse: Complete response with embedded ForecastData
    """
    # Extract configuration parameters
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("sector_forecast", {})
    
    # Map effort to quality attributes and forecast accuracy
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 5.0, "medium": 2.0})
    if effort >= effort_thresholds.get("high", 5.0):
        methodology = "premium"
        coverage = np.random.uniform(0.8, 1.0)  # High-quality coverage
        forecast_quality = 0.9
    elif effort >= effort_thresholds.get("medium", 2.0):
        methodology = "standard" 
        coverage = np.random.uniform(0.5, 0.7)  # Medium-quality coverage
        forecast_quality = 0.7
    else:
        methodology = "basic"
        coverage = np.random.uniform(0.1, 0.4)  # Low-quality coverage
        forecast_quality = 0.5
    
    # Create quality attributes dict
    quality_attributes = {
        "methodology": methodology,
        "coverage": coverage
    }
    
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
        true_next_regime = state.regime_history[next_period].regimes[sector]
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
    forecast_id = f"forecast_{uuid.uuid4().hex[:8]}"
    forecast_data = ForecastData(
        forecast_id=forecast_id,
        sector=sector,
        predicted_regime=forecast_result["predicted_regime"],
        confidence_vector=forecast_result["confidence_vector"]
    )
    
    # Add forecast to market state knowledge goods
    forecast_id = forecast_data.forecast_id
    state.knowledge_good_forecasts[forecast_id] = forecast_data
    
    # Return SectorForecastResponse with embedded ForecastData
    return SectorForecastResponse(
        sector=sector,
        forecast=forecast_data,
        quality_attributes=quality_attributes,
        effort_used=effort
    )


def post_to_market_impl(
    market_model,
    config_data: Dict[str, Any],
    forecast_id: str,
    price: float,
    marketing_attributes: Dict[str, Any]
) -> PostToMarketResponse:
    """
    Post an offer to the market.
    """
    if forecast_id not in market_model.state.knowledge_good_forecasts:
        return PostToMarketResponse(
            offer=None,
            status="error",
            message=f"Forecast ID {forecast_id} not found in market model state. No such knowledge good exists."
        )
    
    # Prepare the offer data
    offer_data = {
        "good_id": forecast_id,
        "seller_id": config_data.get("org_id"),
        "price": price,
        "marketing_attributes": marketing_attributes
    }
    market_model.state.offers.append(Offer(**offer_data))
    
    # TODO: I suspect that the correct approach might be to have agents convert
    # their forecast data into marketing evidence, that they can use to justify
    # their offers in terms of marketing attributes. This would look like:
    # evidence = [{
    #     "forecast_id": forecast_id,
    #     "attribute": "quality",
    #     "evidence": "This offer has high quality based on forecast analysis. We used the following methods to generate this forecast: "
    # }]
    # Or maybe the type of evidence should be more specific, like: if they know
    # a high quality forecast requires looking at a set of leading indicators and
    # running statistical analysis, then they can mention what indicators they used
    # and what statistical methods they applied to generate the forecast. This
    # information is an assertion of quality
    # Eventually, we could return warnings if attributes do not have sufficient evidence
    # and either edit the offer or return an error response and ask to re-submit.
    # However, the usefulness of this approach depends on how convincing it is to
    # see the agents deceptively generating fake evidence that passes the checks.
    # As this is hard to achieve, we will not implement this for now. Instead,
    # agents will have to work out what attributes they want to use in their
    # marketing, when there is an obvious mismatch between the forecast and the
    # marketing attributes they might want to use to maximise profits.
    
    return PostToMarketResponse(
        offer=Offer(**offer_data),
        status="success",
        message="Offer posted successfully"
    )


# Market Research Tool Implementations

def analyze_historical_performance_impl(
    market_model,
    config_data: Dict[str, Any], 
    sector: str,
    effort: float
) -> HistoricalPerformanceResponse:
    """
    Return historical trade data subset for agent analysis. Effort affects data quantity, 
    quality, and sector coverage.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to analyze
        effort: Effort level allocated
    
    Returns:
        HistoricalPerformanceResponse: Raw trade data for agent analysis
    """
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("analyze_historical_performance", {})
    
    # Map effort to data parameters
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 3.0, "medium": 1.5})
    if effort >= effort_thresholds.get("high", 3.0):
        quality_tier = "high"
        max_trades = tool_params.get("high_effort_max_trades", 100)
        noise_factor = tool_params.get("high_effort_noise_factor", 0.05)
    elif effort >= effort_thresholds.get("medium", 1.5):
        quality_tier = "medium"
        max_trades = tool_params.get("medium_effort_max_trades", 50)
        noise_factor = tool_params.get("medium_effort_noise_factor", 0.1)
    else:
        quality_tier = "low"
        max_trades = tool_params.get("low_effort_max_trades", 20)
        noise_factor = tool_params.get("low_effort_noise_factor", 0.2)

    # Get all historical trades
    all_trades = getattr(state, 'all_trades', [])
    warnings = []
    
    if not all_trades:
        warnings.append("No historical trade data available")
        return HistoricalPerformanceResponse(
            sector=sector,
            trade_data=[],
            sample_size=0,
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=warnings,
            recommendation="No historical data available for analysis"
        )
    
    # Primary filtering by sector using knowledge_good_forecasts
    sector_trades = [t for t in all_trades 
                    if hasattr(t, 'good_id') and t.good_id in state.knowledge_good_forecasts
                    and state.knowledge_good_forecasts[t.good_id].sector == sector]
    sampled_trades = np.random.choice(sector_trades, size=min(max_trades, len(sector_trades)), replace=False).tolist()
    # Apply noise based on effort quality (simulate data imperfections)
    trade_data = []
    for trade in sampled_trades:
        trade_record = {
            "good_id": trade.good_id,
            "price": float(trade.price * (1 + np.random.normal(0, noise_factor))),  # Add price noise
            "buyer_id": trade.buyer_id,
            "seller_id": trade.seller_id,
            "marketing_attributes": trade.marketing_attributes,
            "period": trade.period
        }
        trade_data.append(trade_record)
    
    # Generate recommendation based on data availability
    if trade_data:
        avg_price = np.mean([t["price"] for t in trade_data])
        recommendation = f"Found {len(trade_data)} historical trades. Average price: ${avg_price:.2f}. Analyze patterns as needed."
    else:
        recommendation = "No relevant historical data found for analysis"
    
    if len(sampled_trades) < 10:
        warnings.append("Limited sample size may reduce analysis reliability")
    if noise_factor > 0.15:
        warnings.append("Data quality limited by effort level - prices may have significant noise")
    
    return HistoricalPerformanceResponse(
        sector=sector,
        trade_data=trade_data,
        sample_size=len(trade_data),
        quality_tier=quality_tier,
        effort_used=effort,
        warnings=warnings,
        recommendation=recommendation
    )


def run_ols_regression(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Run OLS regression using numpy linear algebra.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        
    Returns:
        Dict with coefficients, r_squared, and statistics
    """
    n, p = X.shape
    X_with_intercept = np.column_stack([np.ones(n), X])
    
    try:
        # Solve normal equations: (X'X)β = X'y
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y
        beta = np.linalg.solve(XTX, XTy)
        
        # Calculate predictions and residuals
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred
        
        # Calculate R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum(residuals ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate standard errors
        mse = ss_res / (n - p - 1) if n > p + 1 else np.inf
        var_covar_matrix = mse * np.linalg.inv(XTX)
        std_errors = np.sqrt(np.diag(var_covar_matrix))
        
        return {
            'intercept': beta[0],
            'coefficients': beta[1:],  # Exclude intercept
            'std_errors': std_errors[1:],  # Exclude intercept SE
            'r_squared': r_squared,
            'n_obs': n,
            'residuals': residuals
        }
    except np.linalg.LinAlgError:
        # Handle singular matrix (perfect multicollinearity)
        return {
            'intercept': 0,
            'coefficients': np.zeros(p),
            'std_errors': np.full(p, np.inf),
            'r_squared': 0,
            'n_obs': n,
            'residuals': np.zeros(n)
        }


def generate_test_offers(sector: str, num_offers: int, attribute_order: List[str], 
                        marketing_definitions: Dict[str, Any]) -> List:
    """Generate representative test offers for preference analysis."""
    from ...models.market_for_finance import Offer
    
    test_offers = []
    for i in range(num_offers):
        # Create varied marketing attributes
        marketing_attributes = {}
        
        for attr_name in attribute_order:
            if attr_name in marketing_definitions:
                attr_def = marketing_definitions[attr_name]
                attr_type = attr_def.get('type', 'categorical')
                
                if attr_type == 'qualitative':
                    # Pick random value from available options
                    values = attr_def.get('values', ['low', 'medium', 'high'])
                    marketing_attributes[attr_name] = np.random.choice(values)
                elif attr_type == 'numeric':
                    # Sample from range
                    attr_range = attr_def.get('range', [0.0, 1.0])
                    marketing_attributes[attr_name] = np.random.uniform(attr_range[0], attr_range[1])
                else:
                    marketing_attributes[attr_name] = 'standard'
            else:
                # Default for unknown attributes
                marketing_attributes[attr_name] = np.random.choice(['low', 'medium', 'high'])
        
        offer = Offer(
            good_id=f"test_{sector}_{i}",
            price=50.0 + np.random.uniform(-20, 20),  # Vary prices
            seller_id="test_seller",
            marketing_attributes=marketing_attributes
        )
        test_offers.append(offer)
    
    return test_offers


def analyze_buyer_preferences_impl(
    market_model,
    config_data: Dict[str, Any],
    sector: str,
    effort: float
) -> BuyerPreferenceResponse:
    """
    Analyze buyer preference patterns by sampling buyers and testing their WTP for various offers.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to analyze
        effort: Effort level allocated
    
    Returns:
        BuyerPreferenceResponse: Top attributes buyers value most in this sector
    """
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("analyze_buyer_preferences", {})
    
    # Map effort to sample sizes
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 3.0, "medium": 1.5})
    if effort >= effort_thresholds.get("high", 3.0):
        quality_tier = "high"
        num_buyers = tool_params.get("high_effort_num_buyers", 30)
        num_test_offers = tool_params.get("high_effort_num_test_offers", 12)
        use_regression = tool_params.get("high_effort_analyze_by_attribute", True)
    elif effort >= effort_thresholds.get("medium", 1.5):
        quality_tier = "medium"
        num_buyers = tool_params.get("medium_effort_num_buyers", 15)
        num_test_offers = tool_params.get("medium_effort_num_test_offers", 6)
        use_regression = tool_params.get("medium_effort_analyze_by_attribute", False)
    else:
        quality_tier = "low"
        num_buyers = tool_params.get("low_effort_num_buyers", 5)
        num_test_offers = tool_params.get("low_effort_num_test_offers", 3)
        use_regression = tool_params.get("low_effort_analyze_by_attribute", False)
    
    all_buyers = state.buyers_state
    attribute_order = state.attribute_order

    if len(all_buyers) == 0:
        return BuyerPreferenceResponse(
            sector=sector,
            analysis_method="insufficient_data",
            attribute_insights=[],
            raw_wtp_data=[],
            sample_size=0,
            total_observations=0,
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=["No buyers available for analysis"],
            recommendation="Unable to analyze preferences - no buyers found"
        )

    # Sample from all buyers
    sampled_buyers = np.random.choice(all_buyers, size=min(num_buyers, len(all_buyers)), replace=False)
    for buyer in sampled_buyers:
        buyer.ensure_sector_exists(sector, len(attribute_order))

    # Generate representative test offers for this sector
    marketing_definitions = state.marketing_attribute_definitions
    test_offers = generate_test_offers(sector, num_test_offers, attribute_order, marketing_definitions)
    
    # Calculate WTP for each buyer-offer combination and build feature matrix
    raw_wtp_data = []
    feature_matrix = []
    wtp_vector = []
    
    for buyer in sampled_buyers:
        for offer in test_offers:
            features = convert_marketing_to_features(
                offer.marketing_attributes, 
                buyer.buyer_conversion_function,
                attribute_order
            )
            
            # WTP = sum of (attribute_weight * feature_value)
            wtp = np.dot(buyer.attr_weights.get(sector, [0.5] * len(attribute_order)), features)
            
            # Store in new schema format
            wtp_data_point = WTPDataPoint(
                buyer_id=buyer.buyer_id,
                offer_attributes=offer.marketing_attributes,
                willingness_to_pay=wtp
            )
            raw_wtp_data.append(wtp_data_point)
            
            # Build matrices for regression
            feature_matrix.append(features)
            wtp_vector.append(wtp)

    # Add warnings
    warnings = []
    if len(sampled_buyers) < 5:
        warnings.append("Limited buyer sample size may reduce analysis reliability")
    if len(test_offers) < 3:
        warnings.append("Limited offer variety may not capture full preference spectrum")
    
    total_observations = len(raw_wtp_data)
    
    # Determine analysis method and perform analysis
    attribute_insights = []
    analysis_method = "insufficient_data"
    regression_r_squared = None
    
    if total_observations >= 30 and use_regression and len(attribute_order) > 0:
        # High effort: Use OLS regression
        try:
            X = np.array(feature_matrix)
            y = np.array(wtp_vector)
            
            # Check for sufficient variation
            if np.var(y) > 1e-10 and X.shape[1] > 0:
                regression_results = run_ols_regression(X, y)
                analysis_method = "regression"
                regression_r_squared = regression_results["r_squared"]
                
                # Create AttributeAnalysis for each attribute
                for i, attr_name in enumerate(attribute_order):
                    # Extract coefficient for this attribute (intercept already excluded)
                    if i < len(regression_results["coefficients"]):
                        marginal_impact = regression_results["coefficients"][i]
                        
                        # Determine confidence based on standard error
                        std_error = regression_results["std_errors"][i] if i < len(regression_results["std_errors"]) else float('inf')
                        if abs(marginal_impact) > 2 * std_error:  # Roughly 95% confidence
                            confidence = "high"
                        elif abs(marginal_impact) > std_error:
                            confidence = "medium"
                        else:
                            confidence = "low"
                        
                        attribute_insights.append(AttributeAnalysis(
                            attribute_name=attr_name,
                            marginal_wtp_impact=marginal_impact,
                            average_feature_value=None,  # Not meaningful for business interpretation
                            sample_size=total_observations,
                            confidence_level=confidence
                        ))
                
                recommendation = f"Regression analysis with R² = {regression_r_squared:.3f}. Focus on attributes with highest marginal WTP impact and high confidence."
            else:
                warnings.append("Insufficient variation in WTP data for regression analysis")
                analysis_method = "insufficient_data"
                recommendation = "Insufficient variation in data - consider more diverse test offers or buyers"
        except Exception as e:
            warnings.append(f"Regression analysis failed: {str(e)}")
            analysis_method = "insufficient_data"
            recommendation = "Statistical analysis failed - falling back to descriptive approach"
    
    if analysis_method == "insufficient_data" and total_observations > 0:
        # Fall back to descriptive analysis
        analysis_method = "descriptive"
        
        # Calculate descriptive statistics per attribute
        for attr_name in attribute_order:
            # Simple confidence based on sample size
            if total_observations >= 20:
                confidence = "medium"
            elif total_observations >= 10:
                confidence = "low"
            else:
                confidence = "low"
            
            attribute_insights.append(AttributeAnalysis(
                attribute_name=attr_name,
                marginal_wtp_impact=None,  # No regression available
                average_feature_value=None,  # Not meaningful for descriptive analysis
                sample_size=total_observations,
                confidence_level=confidence
            ))
        
        recommendation = "Descriptive analysis only - increase effort and sample size for regression-based insights"

    return BuyerPreferenceResponse(
        sector=sector,
        analysis_method=analysis_method,
        attribute_insights=attribute_insights,
        regression_r_squared=regression_r_squared,
        raw_wtp_data=raw_wtp_data,
        sample_size=len(sampled_buyers),
        total_observations=total_observations,
        quality_tier=quality_tier,
        effort_used=effort,
        warnings=warnings,
        recommendation=recommendation
    )


def generate_typical_attributes(attribute_order: List[str], marketing_definitions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate typical/average marketing attributes for pricing simulation."""
    attributes = {}
    
    for attr_name in attribute_order:
        if attr_name in marketing_definitions:
            attr_def = marketing_definitions[attr_name]
            attr_type = attr_def.get('type', 'categorical')
            
            if attr_type == 'qualitative':
                # Pick middle-tier value
                values = attr_def.get('values', ['low', 'medium', 'high'])
                attributes[attr_name] = values[len(values) // 2] if values else 'medium'
            elif attr_type == 'numeric':
                # Pick middle of range  
                attr_range = attr_def.get('range', [0.0, 1.0])
                attributes[attr_name] = (attr_range[0] + attr_range[1]) / 2
            else:
                attributes[attr_name] = 'standard'
        else:
            attributes[attr_name] = 'medium'  # Default
    
    return attributes

def research_competitive_pricing_impl(
    market_model,
    config_data: Dict[str, Any],
    sector: str,
    effort: float,
    marketing_attributes: Dict[str, Any]
) -> CompetitivePricingResponse:
    """
    Research competitive pricing by simulating market share against real competitors.
    
    Args:
        market_model: Complete market model with state
        config_data: Full configuration data
        sector: Sector to analyze
        effort: Effort level allocated
        marketing_attributes: Attributes to test pricing for (if None, uses typical attributes)
    
    Returns:
        CompetitivePricingResponse: Competitive analysis with market share projections
    """
    state = market_model.state
    tool_params = config_data.get("tool_parameters", {}).get("research_competitive_pricing", {})
    
    # Map effort to simulation parameters
    effort_thresholds = tool_params.get("effort_thresholds", {"high": 2.5, "medium": 1.2})
    if effort >= effort_thresholds.get("high", 2.5):
        quality_tier = "high"
        num_buyers = tool_params.get("high_effort_num_buyers", 30)
        price_points = tool_params.get("high_effort_price_points", 12)
        lookback_trades = tool_params.get("high_effort_lookback_trades", 50)
    elif effort >= effort_thresholds.get("medium", 1.2):
        quality_tier = "medium"
        num_buyers = tool_params.get("medium_effort_num_buyers", 15)
        price_points = tool_params.get("medium_effort_price_points", 6)
        lookback_trades = tool_params.get("medium_effort_lookback_trades", 20)
    else:
        quality_tier = "low"
        num_buyers = tool_params.get("low_effort_num_buyers", 8)
        price_points = tool_params.get("low_effort_price_points", 4)
        lookback_trades = tool_params.get("low_effort_lookback_trades", 10) 
    
    # Get current competitive landscape from recent trades/offers
    all_trades = getattr(state, 'all_trades', [])
    current_offers = getattr(state, 'offers', [])
    
    # Filter to sector and recent activity
    recent_trades = [t for t in all_trades[-lookback_trades:] 
                    if hasattr(t, 'good_id') and t.good_id in state.knowledge_good_forecasts
                    and state.knowledge_good_forecasts[t.good_id].sector == sector]
    
    sector_offers = [o for o in current_offers 
                    if hasattr(o, 'good_id') and o.good_id in state.knowledge_good_forecasts
                    and state.knowledge_good_forecasts[o.good_id].sector == sector]
    
    warnings = []
    if not recent_trades and not sector_offers:
        warnings.append(f"No competitive activity found in {sector} sector")
        return CompetitivePricingResponse(
            sector=sector,
            price_simulations=[],
            recommended_price=0.0,
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=warnings,
            recommendation="No competitive data available - unable to provide pricing guidance"
        )
    
    # Convert recent trades to competitor offers for simulation
    competitor_offers = []
    for trade in recent_trades:
        competitor_offer = Offer(
            good_id=trade.good_id,
            price=trade.price,
            seller_id=trade.seller_id,
            marketing_attributes=trade.marketing_attributes
        )
        competitor_offers.append(competitor_offer)
    
    # Add current offers from other sellers
    competitor_offers.extend(sector_offers)
    
    if len(competitor_offers) == 0:
        warnings.append("No competitor offers available for analysis")
    
    # Sample buyers for simulation
    all_buyers = getattr(state, 'buyers_state', [])
    if not all_buyers:
        warnings.append("No buyers available for competitive simulation")
        return CompetitivePricingResponse(
            sector=sector,
            price_simulations=[],
            recommended_price=0.0,
            quality_tier=quality_tier,
            effort_used=effort,
            warnings=warnings,
            recommendation="Cannot simulate without buyers"
        )
    
    attribute_order = state.attribute_order
    
    sampled_buyers = np.random.choice(all_buyers, size=min(num_buyers, len(all_buyers)), replace=False)

    # Determine price range from competitive landscape
    competitor_prices = [o.price for o in competitor_offers]
    if competitor_prices:
        min_comp_price = min(competitor_prices)
        max_comp_price = max(competitor_prices)
        avg_comp_price = np.mean(competitor_prices)
        # Test prices from 20% below minimum to 20% above maximum competitor
        price_range = (min_comp_price * 0.8, max_comp_price * 1.2)
    else:
        avg_comp_price = 0.0
        price_range = (0.0, 0.0)

    # Test different price points
    test_prices = np.linspace(price_range[0], price_range[1], price_points)
    price_simulations = []

    # Create mock config for choice model
    choice_config = type('Config', (), {
        'choice_model': config_data.get('choice_model', 'greedy'),
        'cart_draws': config_data.get('cart_draws', None)
    })()
    
    for price in test_prices:
        # Create our candidate offer
        candidate_offer = Offer(
            good_id=f"candidate_offer_{sector}",
            price=price,
            seller_id="our_seller",
            marketing_attributes=marketing_attributes
        )
        
        # Create full offer set (competitors + our candidate)
        all_offers = competitor_offers + [candidate_offer]
        
        # Simulate buyer choices using actual choice model
        our_purchases = 0
        total_purchases = 0
        total_buyer_value = 0.0
        
        for buyer in sampled_buyers:
            # Run choice model to see what buyer would purchase
            try:
                buyer_choices = choice_model(buyer, all_offers, choice_config, state)
                
                # Count purchases
                for chosen_offer in buyer_choices:
                    total_purchases += 1
                    if chosen_offer.good_id == candidate_offer.good_id:
                        our_purchases += 1
                        # Calculate buyer's value for our offer
                        features = convert_marketing_to_features(
                            marketing_attributes,
                            buyer.buyer_conversion_function,
                            attribute_order
                        )
                        buyer_value = np.dot(buyer.attr_weights.get(sector, [0.5] * len(attribute_order)), features)
                        total_buyer_value += buyer_value
                        
            except Exception as e:
                # Handle any choice model errors gracefully
                warnings.append(f"Choice model simulation error at price {price:.2f}")
                continue
        
        # Calculate competitive metrics
        market_share = our_purchases / len(sampled_buyers) if len(sampled_buyers) > 0 else 0
        capture_rate = our_purchases / total_purchases if total_purchases > 0 else 0
        expected_revenue = our_purchases * price
        
        price_simulations.append({
            "price": float(price),
            "market_share": float(market_share),
            "capture_rate": float(capture_rate),
            "buyers_purchasing": int(our_purchases),
            "total_market_purchases": int(total_purchases),
            "expected_revenue": float(expected_revenue),
            "competitive_position": "premium" if price > avg_comp_price * 1.1 else 
                                   "discount" if price < avg_comp_price * 0.9 else "competitive"
        })
    
    # Find optimal price (highest expected revenue)
    if price_simulations:
        best_simulation = max(price_simulations, key=lambda x: x["expected_revenue"])
        recommended_price = best_simulation["price"]
        
        # Generate recommendation
        market_share = best_simulation["market_share"] * 100
        competitive_pos = best_simulation["competitive_position"]
        recommendation = f"Recommended price: ${recommended_price:.2f} ({competitive_pos} positioning, {market_share:.1f}% market share)"
    else:
        recommended_price = avg_comp_price
        recommendation = "Simulation failed - consider competitive pricing around average"
    
    if len(sampled_buyers) < 10:
        warnings.append("Small buyer sample may reduce accuracy")
    if len(competitor_offers) < 3:
        warnings.append("Limited competitive landscape may skew results")
    
    return CompetitivePricingResponse(
        sector=sector,
        price_simulations=price_simulations,
        recommended_price=recommended_price,
        quality_tier=quality_tier,
        effort_used=effort,
        warnings=warnings,
        recommendation=recommendation
    )
