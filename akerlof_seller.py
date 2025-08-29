"""
AkerlofSeller implementation for demonstrating Akerlof's "Market for Lemons" dynamics.

MARKETING-FOCUSED VERSION: This implementation focuses on marketing decisions only.
Sellers receive pre-assigned forecasts based on their quality type and must:
1. Assess the quality of assigned forecasts
2. Research market conditions using available tools
3. Choose optimal marketing strategies (attribute claims + competitive pricing)
4. Post offers to market

The asymmetric information between true quality and observable attributes creates
conditions for adverse selection and potential market failure.
"""

import itertools
import numpy as np
from typing import List, Tuple, Dict, Optional
from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, Offer, ForecastData, TradeData, RegimeParameters,
    BuyerState, SellerState, build_confusion_matrix, generate_forecast_signal,
    categorical_draw, get_average_attribute_vector
)
from multi_agent_economics.tools.implementations.economic import (
    analyze_historical_performance_impl, sector_forecast_impl,
    analyze_buyer_preferences_impl, research_competitive_pricing_impl
)


class AkerlofSeller:
    """
    
    AkerlofSeller adapted for marketing-focused decision making.
    
    This creates the conditions for Akerlof's "Market for Lemons" adverse selection:
    - Sellers know true quality but buyers only observe attributes
    - Low-quality producers can mimic high-quality attributes
    - Competition can make quality production unprofitable
    """

    def __init__(self, seller_id: str, market_model: MarketModel, context: Dict = {}):
        """
        Initialize AkerlofSeller with required cost structures.
        
        Args:
            seller_id: Unique identifier for this seller
            market_state: Current market state for decision making
        """
        self.seller_id = seller_id
        self.market_model = market_model
        self.market_state: MarketState = market_model.state
        self.context = context
        
        # Marketing cost structure - represents expense of different attribute claims
        # Get from context with sensible defaults
        self.marketing_costs = self.context.get('marketing_costs', {
            'premium_claims': 10.0,   # Professional presentation, detailed methodology
            'standard_claims': 5.0,   # Basic presentation, standard methodology
            'basic_claims': 2.0       # Simple presentation, minimal methodology
        })
        
        # Quality mapping for forecast generation
        # Get from context with sensible defaults
        self.effort_quality_mapping = self.context.get('effort_quality_mapping', {
            'high': 0.85,     # High effort produces high-accuracy forecasts
            'medium': 0.70,   # Medium effort produces medium-accuracy forecasts
            'low': 0.55       # Low effort produces low-accuracy forecasts
        })
        
        # Storage for historical performance analysis
        self.performance_data = {}  # Maps good_id -> actual performance
        self.historical_trades = []
        
    def add_performance_data(self, trades: List[TradeData], performance_mapping: Dict[str, float]):
        """
        Add historical performance data for learning price-performance relationships.
        
        Args:
            trades: List of historical trade data
            performance_mapping: Maps good_id -> actual forecast performance (0-1)
        """
        self.historical_trades.extend(trades)
        self.performance_data.update(performance_mapping)
        
    def analyze_historical_revenues(self, performance_level: float, trades: List[TradeData]) -> List[float]:
        """
        Analyze historical revenues achieved by forecasts with given performance level.
        
        Args:
            performance_level: Target performance level (e.g., 0.85 for high quality)
            trades: Historical trades to analyze
            
        Returns:
            List of prices achieved by forecasts with similar performance
        """
        matching_revenues = []
        
        for trade in trades:
            # 
            group_by_effort_level = 
            matching_revenues.append(trade.price)
        
        return matching_revenues
        
    def _convert_effort_to_numeric(self, effort_level: str) -> float:
        """
        Convert effort level string to numeric value based on context tool parameters.
        
        Args:
            effort_level: String effort level ('high', 'medium', 'low')
            
        Returns:
            Numeric effort value matching tool parameter thresholds
        """
        # Get effort thresholds from tool parameters
        tool_params = self.context.get('tool_parameters', {})
        sector_forecast_params = tool_params.get('sector_forecast', {})
        effort_thresholds = sector_forecast_params.get('effort_thresholds', {
            'high': 5.0, 
            'medium': 2.0
        })
        
        # Map effort level to numeric value
        if effort_level == 'high':
            return effort_thresholds.get('high', 5.0)
        elif effort_level == 'medium':
            return effort_thresholds.get('medium', 2.0)
        else:  # 'low'
            # Low effort should be below medium threshold
            medium_threshold = effort_thresholds.get('medium', 2.0)
            return medium_threshold * 0.5  # Half of medium threshold
    
    def get_assigned_forecasts(self) -> List[Tuple[str, ForecastData]]:
        """
        Get forecasts assigned to this seller for the current period.
        
        In the marketing-focused system, sellers receive pre-assigned forecasts
        based on their quality type rather than making production decisions.
        
        Returns:
            List of (forecast_id, forecast_data) tuples assigned to this seller
        """
        assigned_forecasts = []
        
        # Look for forecasts assigned to this seller in current period
        for forecast_id, forecast_data in self.market_state.knowledge_good_forecasts.items():
            if forecast_id.startswith(self.seller_id):
                assigned_forecasts.append((forecast_id, forecast_data))
        
        return assigned_forecasts
    
    
    def marketing_strategy_decision(self, market_competition: List[Offer] = None) -> List[Offer]:
        """
        Marketing-focused decision making: research market and create optimal marketing strategy.
        
        This replaces the two-stage decision process with pure marketing strategy:
        1. Get assigned forecasts for this period
        2. Research market conditions using available tools  
        3. Create optimal marketing strategies for each forecast
        4. Post offers to market
        
        Args:
            market_competition: List of competing offers (optional, will query market if not provided)
            
        Returns:
            List of Offer objects posted to market
        """
        # Get current market competition if not provided
        if market_competition is None:
            market_competition = self.market_state.offers.copy()
        
        # Get assigned forecasts for this period
        assigned_forecasts = self.get_assigned_forecasts()
        
        if not assigned_forecasts:
            # No forecasts assigned - return empty list
            return []
        
        # Research market conditions using available tools
        market_research = self.conduct_market_research()
        
        # Create marketing strategies for each assigned forecast
        offers = []
        for forecast_id, forecast_data in assigned_forecasts:
            offer = self.create_marketing_strategy(
                forecast_id, forecast_data, market_research, market_competition
            )
            offers.append(offer)
        
        return offers
    
    
    def conduct_market_research(self) -> Dict[str, Any]:
        """
        Use available research tools to gather market intelligence.
        
        Returns:
            Dictionary containing market research insights
        """
        research_budget = self.context.get('research_budget', 5.0)
        sector = self.context.get('sector', 'tech')
        
        research_results = {}
        
        # Historical performance analysis (cost: 2.0)
        if research_budget >= 2.0:
            try:
                historical_result = analyze_historical_performance_impl(
                    self.market_model, self.context, sector, effort=2.0
                )
                research_results['historical_performance'] = historical_result
                research_budget -= 2.0
            except Exception as e:
                research_results['historical_performance'] = None
        
        # Buyer preference analysis (cost: 3.0)  
        if research_budget >= 3.0:
            try:
                buyer_pref_result = analyze_buyer_preferences_impl(
                    self.market_model, self.context, sector, effort=3.0
                )
                research_results['buyer_preferences'] = buyer_pref_result
                research_budget -= 3.0
            except Exception as e:
                research_results['buyer_preferences'] = None
        
        # Competitive pricing analysis (cost: 2.5)
        if research_budget >= 2.5:
            try:
                # Use basic attributes for competitive analysis
                test_attributes = self.context.get('default_strategy', {"methodology": "standard", "coverage": 0.5})
                pricing_result = research_competitive_pricing_impl(
                    self.market_model, self.context, sector, effort=2.5,
                    marketing_attributes=test_attributes
                )
                research_results['competitive_pricing'] = pricing_result
                research_budget -= 2.5
            except Exception as e:
                research_results['competitive_pricing'] = None
        
        return research_results
    
    
    def create_marketing_strategy(self, forecast_id: str, forecast_data: ForecastData, 
                                 market_research: Dict[str, Any], market_competition: List[Offer]) -> Offer:
        """
        Create optimal marketing strategy for a specific forecast.
        
        Args:
            forecast_id: ID of the forecast to market
            forecast_data: The forecast data object
            market_research: Results from market research tools
            market_competition: Current market competition
            
        Returns:
            Offer object with optimal marketing strategy
        """
        # Analyze forecast quality to inform marketing strategy
        forecast_quality = self.assess_forecast_quality(forecast_data)
        
        # Choose marketing attributes based on research and forecast quality
        marketing_attributes = self.choose_marketing_attributes(
            forecast_quality, market_research, market_competition
        )
        
        # Set competitive price based on research
        price = self.set_optimal_price(
            marketing_attributes, market_research, market_competition
        )
        
        # Create and post offer
        offer = self.create_and_post_offer(forecast_id, marketing_attributes, price)
        
        return offer
    
    
    def assess_forecast_quality(self, forecast_data: ForecastData) -> str:
        """
        Assess the quality of an assigned forecast.
        
        Since sellers receive forecasts rather than create them, they need to
        assess quality from observable characteristics of the forecast.
        
        Args:
            forecast_data: The forecast to assess
            
        Returns:
            Quality assessment: "high", "medium", or "low"
        """
        # Assess quality based on confidence vector characteristics
        confidence_vector = forecast_data.confidence_vector
        
        # High quality: Strong confidence in prediction
        max_confidence = max(confidence_vector)
        confidence_spread = max_confidence - min(confidence_vector)
        
        if max_confidence >= 0.8 and confidence_spread >= 0.4:
            return "high"
        elif max_confidence >= 0.7 and confidence_spread >= 0.2:
            return "medium"
        else:
            return "low"
    
    
    def choose_marketing_attributes(self, forecast_quality: str, market_research: Dict[str, Any], 
                                   market_competition: List[Offer]) -> Dict[str, Any]:
        """
        Choose optimal marketing attributes based on forecast quality and market research.
        
        Args:
            forecast_quality: Assessed quality of the forecast ("high", "medium", "low")
            market_research: Results from market research tools
            market_competition: Current competitive offers
            
        Returns:
            Dictionary of marketing attributes
        """
        # Start with baseline attributes
        attributes = {"methodology": "standard", "coverage": 0.5}
        
        # Adjust based on forecast quality
        if forecast_quality == "high":
            attributes["methodology"] = "premium"
            attributes["coverage"] = 0.8
        elif forecast_quality == "medium":
            attributes["methodology"] = "standard" 
            attributes["coverage"] = 0.6
        else:  # low quality
            attributes["methodology"] = "basic"
            attributes["coverage"] = 0.4
        
        # Adjust based on buyer preference research
        if 'buyer_preferences' in market_research and market_research['buyer_preferences']:
            buyer_prefs = market_research['buyer_preferences']
            if buyer_prefs.top_valued_attributes:
                top_attribute = buyer_prefs.top_valued_attributes[0]
                attribute_name = top_attribute["attribute"]
                
                # Enhance the most valued attribute if we have high quality forecast
                if forecast_quality == "high" and attribute_name in attributes:
                    if attribute_name == "methodology":
                        attributes["methodology"] = "premium"
                    elif attribute_name == "coverage" and isinstance(attributes["coverage"], (int, float)):
                        attributes["coverage"] = min(0.9, attributes["coverage"] + 0.2)
        
        # Consider competitive positioning
        if market_competition:
            # Simple strategy: differentiate if we have high quality, follow if low quality
            if forecast_quality == "high":
                # Try to stand out with premium positioning
                attributes["methodology"] = "premium"
            elif forecast_quality == "low":
                # Try to match competition to avoid standing out negatively
                competitor_methodologies = [offer.marketing_attributes.get("methodology", "standard") 
                                          for offer in market_competition]
                if competitor_methodologies:
                    most_common = max(set(competitor_methodologies), key=competitor_methodologies.count)
                    attributes["methodology"] = most_common
        
        return attributes
    
    
    def set_optimal_price(self, marketing_attributes: Dict[str, Any], market_research: Dict[str, Any], 
                         market_competition: List[Offer]) -> float:
        """
        Set optimal price based on marketing attributes and market research.
        
        Args:
            marketing_attributes: Chosen marketing attributes
            market_research: Results from market research tools  
            market_competition: Current competitive offers
            
        Returns:
            Optimal price
        """
        # Use competitive pricing research if available
        if 'competitive_pricing' in market_research and market_research['competitive_pricing']:
            pricing_research = market_research['competitive_pricing']
            if pricing_research.recommended_price > 0:
                return pricing_research.recommended_price
        
        # Fallback to competitive pricing logic
        return self.estimate_competitive_price(marketing_attributes, market_competition)


    def stage2_marketing_decision(self, effort_level: str, good_id: str, market_competition: List[Offer]) -> Offer:
        """
        Stage 2: Create marketing strategy (attribute claims + competitive pricing).
        
        Economic logic:
        1. Choose attribute claims to maximize willingness-to-pay
        2. Set competitive price considering market competition
        3. Account for marketing costs in profit calculation
        
        Args:
            effort_level: Production effort level (for cost accounting)
            good_id: Identifier of forecast to be marketed
            market_competition: List of competing offers
            
        Returns:
            Offer object posted to market
        
        Note:
            This method assumes the seller has already produced the good
            and is now deciding how to market it.
        """
        # Extract buyer preferences for optimal attribute selection (sector-specific)
        sector = self.context.get('sector', 'tech')  # Get seller's sector
        avg_buyer_prefs = self.extract_buyer_preferences(sector)
        
        # Choose attribute claims (can be independent of true quality - asymmetric info!)
        # For now, use a simple strategy: claim high attributes if profitable
        proposed_attrs = self.choose_optimal_attributes(avg_buyer_prefs, market_competition)
        
        # Set competitive price
        price = self.estimate_competitive_price(proposed_attrs, market_competition)
        
        # Create and post offer
        offer = self.create_and_post_offer(good_id, proposed_attrs, price)
        
        return offer
        
    def extract_buyer_preferences(self, sector: Optional[str] = None) -> List[float]:
        """
        Extract average buyer preferences from market state for a specific sector.
        
        Args:
            sector: Sector to extract preferences for (defaults to seller's sector)
        
        Returns:
            List of average attribute preferences across all buyers for the sector
        """
        if sector is None:
            sector = self.context.get('sector', 'tech')  # Use seller's sector
            
        if not self.market_state.buyers_state:
            # Get default preferences from context
            default_prefs = self.context.get('default_buyer_preferences', [0.5, 0.5])
            return default_prefs
        
        # Get attribute order for consistent vector length
        attribute_order = getattr(self.market_state, 'attribute_order', [])
        if not attribute_order:
            default_prefs = self.context.get('default_buyer_preferences', [0.5, 0.5])
            return default_prefs
        
        # Extract sector-specific preferences from buyers
        all_prefs = []
        for buyer in self.market_state.buyers_state:
            # Ensure buyer has preferences for this sector
            if hasattr(buyer, 'ensure_sector_exists'):
                buyer.ensure_sector_exists(sector, len(attribute_order))
            
            if hasattr(buyer, 'attr_mu') and sector in buyer.attr_mu:
                all_prefs.append(buyer.attr_mu[sector])
        
        if not all_prefs:
            # Get default preferences from context
            default_prefs = self.context.get('default_buyer_preferences', [0.5] * len(attribute_order))
            return default_prefs
            
        # Average across buyers and attributes
        avg_prefs = np.mean(all_prefs, axis=0).tolist()
        
        return avg_prefs
        
    def choose_optimal_attributes(self, avg_buyer_prefs: List[float], competition: List[Offer]) -> List[float]:
        """
        Choose attribute claims to maximize expected revenue.
        
        Uses configurable strategies from context that match marketing cost structure.
        
        Args:
            avg_buyer_prefs: Average buyer preferences
            competition: Competing offers
            
        Returns:
            Chosen marketing attributes as a dictionary
        """
        # Get marketing strategies from context
        strategie_choices = self.context.get('marketing_strategies')
        # Find all combinations of strategies that combine the choices in strategies
        strategy_set = [dict(zip(strategie_choices.keys(), v))
                        for v in itertools.product(*strategie_choices.values())]

        # Get pricing parameters from context
        wtp_pricing_factor = self.context.get('wtp_pricing_factor', 0.9)  # Default 90% of WTP
        
        best_strategy = self.context.get('default_strategy')
        best_profit = -float('inf')

        for strategy in strategy_set:
            # Estimate revenue from this strategy
            avg_attrs = get_average_attribute_vector(strategy, self.market_state.buyers_state, self.market_state.attribute_order)

            wtp = self._estimate_willingness_to_pay_for_attrs(avg_attrs, avg_buyer_prefs)
            # Add up marketing costs for this strategy
            marketing_cost = sum(self.marketing_costs.get(k, {}).get(v, 0) for k, v in strategy.items())

            # Calculate expected profit (revenue - marketing cost)
            expected_profit = wtp * wtp_pricing_factor - marketing_cost
            
            # TODO: Research whether there is a way for Akerlof Sellers to estimate the expected demand for their offers under competition
            # This will enhance realism but is potentially too complex for now
            
            if expected_profit > best_profit:
                best_profit = expected_profit
                best_strategy = strategy

        return best_strategy
        
    def estimate_competitive_price(self, marketing_attrs: Dict, competition: List[Offer]) -> float:
        """
        Estimate competitive price for given marketing attributes.
        
        Args:
            marketing_attrs: Proposed marketing attributes dict
            competition: List of competing offers
            
        Returns:
            Competitive price estimate
        """
        if not competition:
            # No competition - price based on buyer willingness-to-pay
            wtp_pricing_factor = self.context.get('wtp_pricing_factor', 0.9)  # Default 90% of WTP
            sector = self.context.get('sector', 'tech')  # Get seller's sector
            avg_prefs = self.extract_buyer_preferences(sector)
            # Convert marketing attributes to average attribute vector for WTP calculation
            avg_attrs = get_average_attribute_vector(marketing_attrs, self.market_state.buyers_state, self.market_state.attribute_order)
            wtp = self._estimate_willingness_to_pay_for_attrs(avg_attrs, avg_prefs)
            return wtp * wtp_pricing_factor
        else:
            # With competition - price below similar competitors
            return self._competitive_pricing(marketing_attrs, competition)
            
    def _estimate_willingness_to_pay_for_attrs(self, attrs: List[float], buyer_prefs: List[float]) -> float:
        """
        Estimate buyer willingness-to-pay for specific attributes.
        
        Args:
            attrs: Attribute vector
            buyer_prefs: Buyer preference vector
            
        Returns:
            Estimated willingness-to-pay
        """
        if len(attrs) >= 1 and len(buyer_prefs) >= 1:
            # Calculate weighted utility
            wtp = buyer_prefs[0] * attrs[0] + buyer_prefs[1] * attrs[1]
            return wtp
        else:
            # Get default WTP from context
            default_wtp = self.context.get('default_wtp', 0.5)
            return default_wtp
    
    def _estimate_willingness_to_pay(self, attrs: List[float]) -> float:
        """
        Estimate buyer willingness-to-pay for given attributes.
        
        Args:
            attrs: Attribute vector
            
        Returns:
            Estimated willingness-to-pay
        """
        sector = self.context.get('sector', 'tech')  # Get seller's sector
        avg_prefs = self.extract_buyer_preferences(sector)
        # Get scaling factor from context for realistic pricing
        wtp_scaling_factor = self.context.get('wtp_scaling_factor', 100.0)  # Default 100x scaling
        return self._estimate_willingness_to_pay_for_attrs(attrs, avg_prefs) * wtp_scaling_factor
            
    def _competitive_pricing(self, marketing_attrs: Dict, competition: List[Offer]) -> float:
        """
        Set competitive pricing considering similar offers.
        
        Args:
            marketing_attrs: Proposed marketing attributes dict
            competition: Competing offers
            
        Returns:
            Competitive price
        """
        # Get pricing parameters from context
        similarity_threshold = self.context.get('similarity_threshold', 0.7)
        competitor_discount = self.context.get('competitor_discount', 0.95)  # Default 5% below competitors
        
        # Find similar competitors using attribute similarity
        similar_prices = []
        
        for offer in competition:
            similarity = self.calculate_attribute_similarity(marketing_attrs, offer.marketing_attributes)
            if similarity >= similarity_threshold:
                similar_prices.append(offer.price)
                
        if similar_prices:
            # Price below average competitor price
            avg_competitor_price = np.mean(similar_prices)
            return avg_competitor_price * competitor_discount
        else:
            # No similar competition - fall back to WTP pricing
            avg_attrs = get_average_attribute_vector(marketing_attrs, self.market_state.buyers_state, self.market_state.attribute_order)
            return self._estimate_willingness_to_pay(avg_attrs)
            
    def calculate_attribute_similarity(self, marketing_attrs1: Dict, marketing_attrs2: Dict) -> float:
        """
        Calculate similarity between two marketing attribute sets using average buyer perceptions.
        
        Args:
            marketing_attrs1: First marketing attributes dict (e.g., {"innovation": "high", "risk_score": 25})
            marketing_attrs2: Second marketing attributes dict
            
        Returns:
            Similarity score [0, 1] where 1 = identical, 0 = maximally different
        """
        buyers = self.market_state.buyers_state
        attribute_order = self.market_state.attribute_order
        
        # Convert marketing attributes to average attribute vectors using market perception
        attrs1 = get_average_attribute_vector(marketing_attrs1, buyers, attribute_order)
        attrs2 = get_average_attribute_vector(marketing_attrs2, buyers, attribute_order)
        
        if not attrs1 or not attrs2 or len(attrs1) != len(attrs2):
            return 0.0
            
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum([(a1 - a2)**2 for a1, a2 in zip(attrs1, attrs2)]))
        
        # Convert to similarity (max distance for unit vectors is sqrt(2))
        max_distance = np.sqrt(len(attrs1))
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, similarity)
        
    def create_and_post_offer(self, good_id: str, marketing_attrs: Dict, price: float) -> Offer:
        """
        Create offer and post it to market state.
        
        Args:
            good_id: Forecast ID to be offered
            marketing_attrs: Chosen marketing attributes dict
            price: Chosen price
            
        Returns:
            Created Offer object
        """
        offer = Offer(
            good_id=good_id,
            price=price,
            seller=self.seller_id,
            marketing_attributes=marketing_attrs
        )
        
        # Add to market state
        self.market_state.offers.append(offer)
        
        return offer


def assign_forecasts_to_sellers(market_model: MarketModel, sellers: List[AkerlofSeller], 
                               period: int, config: Dict[str, Any]) -> None:
    """
    System-level forecast assignment to sellers based on their quality types.
    
    This function replaces the individual seller production decisions with
    centralized forecast assignment based on agent quality types.
    
    Args:
        market_model: The market model containing state and tools
        sellers: List of AkerlofSeller instances to assign forecasts to
        period: Current simulation period
        config: Configuration containing quality type parameters
    """
    sector = config.get('sector', 'tech')
    horizon = config.get('horizon', 1)
    
    # Quality type effort distributions
    effort_distributions = config.get('effort_distributions', {
        'high_quality': {'mean': 7.0, 'std': 1.0},    # ~85% forecast accuracy
        'medium_quality': {'mean': 4.0, 'std': 1.0},  # ~70% forecast accuracy  
        'low_quality': {'mean': 1.5, 'std': 0.5}      # ~55% forecast accuracy
    })
    
    # Clear existing forecasts for this period
    current_forecast_ids = [fid for fid in market_model.state.knowledge_good_forecasts.keys() 
                           if f"_r{period}_" in fid]
    for fid in current_forecast_ids:
        del market_model.state.knowledge_good_forecasts[fid]
    
    # Assign forecasts to each seller
    for seller in sellers:
        # Determine seller quality type from context or seller_id
        quality_type = seller.context.get('quality_type')
        if not quality_type:
            # Infer from seller_id if not explicitly set
            if 'high' in seller.seller_id.lower():
                quality_type = 'high_quality'
            elif 'medium' in seller.seller_id.lower() or 'med' in seller.seller_id.lower():
                quality_type = 'medium_quality'
            else:
                quality_type = 'low_quality'
        
        # Sample effort level from quality type distribution
        effort_params = effort_distributions.get(quality_type, effort_distributions['low_quality'])
        effort_numeric = np.random.normal(effort_params['mean'], effort_params['std'])
        effort_numeric = max(0.1, effort_numeric)  # Ensure positive effort
        
        # Generate forecast using sector_forecast_impl
        forecast_response = sector_forecast_impl(
            market_model,
            config_data=seller.context,
            sector=sector,
            horizon=horizon,
            effort=effort_numeric
        )
        
        # Create unique forecast ID for this seller and period
        forecast_id = f"{seller.seller_id}_r{period}_{quality_type}"
        
        # Store forecast in market state
        market_model.state.knowledge_good_forecasts[forecast_id] = forecast_response.forecast
        
        # Update seller context with assigned forecast info
        if 'assigned_forecasts' not in seller.context:
            seller.context['assigned_forecasts'] = {}
        seller.context['assigned_forecasts'][period] = {
            'forecast_id': forecast_id,
            'effort_used': effort_numeric,
            'quality_type': quality_type,
            'quality_tier': forecast_response.quality_tier
        }


def create_marketing_akerlof_sellers(market_model: MarketModel, seller_configs: List[Dict[str, Any]]) -> List[AkerlofSeller]:
    """
    Create AkerlofSeller instances configured for marketing-focused simulation.
    
    Args:
        market_model: The market model
        seller_configs: List of seller configuration dictionaries
        
    Returns:
        List of configured AkerlofSeller instances
    """
    sellers = []
    
    for config in seller_configs:
        seller_id = config['seller_id']
        quality_type = config.get('quality_type', 'medium_quality')
        
        # Set up context with marketing-focused parameters
        context = {
            'quality_type': quality_type,
            'sector': config.get('sector', 'tech'),
            'horizon': config.get('horizon', 1),
            'research_budget': config.get('research_budget', 7.5),  # Budget for market research tools
            'agent_id': seller_id,
            'seller_id': seller_id,
            
            # Tool parameters for research tools
            'tool_parameters': config.get('tool_parameters', {
                'analyze_historical_performance': {
                    'effort_thresholds': {'high': 3.0, 'medium': 1.5}
                },
                'analyze_buyer_preferences': {
                    'effort_thresholds': {'high': 3.0, 'medium': 1.5}
                },
                'research_competitive_pricing': {
                    'effort_thresholds': {'high': 2.5, 'medium': 1.2}
                },
                'sector_forecast': {
                    'effort_thresholds': {'high': 5.0, 'medium': 2.0}
                }
            }),
            
            # Marketing strategy parameters
            'default_strategy': config.get('default_strategy', {"methodology": "standard", "coverage": 0.5}),
            'wtp_pricing_factor': config.get('wtp_pricing_factor', 0.9),
            'wtp_scaling_factor': config.get('wtp_scaling_factor', 100.0),
            'choice_model': config.get('choice_model', 'greedy')
        }
        
        # Create seller instance
        seller = AkerlofSeller(seller_id, market_model, context)
        sellers.append(seller)
    
    return sellers


def run_marketing_simulation_period(market_model: MarketModel, sellers: List[AkerlofSeller], 
                                   period: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single period of marketing-focused simulation.
    
    Args:
        market_model: The market model
        sellers: List of AkerlofSeller instances
        period: Current period number
        config: Simulation configuration
        
    Returns:
        Dictionary containing period results
    """
    # Step 1: Assign forecasts to sellers based on quality types
    assign_forecasts_to_sellers(market_model, sellers, period, config)
    
    # Step 2: Clear previous offers
    market_model.state.offers = []
    
    # Step 3: Each seller makes marketing decisions
    period_offers = []
    seller_strategies = {}
    
    for seller in sellers:
        try:
            # Seller researches market and creates marketing strategies
            offers = seller.marketing_strategy_decision()
            period_offers.extend(offers)
            
            # Record seller strategy for analysis
            seller_strategies[seller.seller_id] = {
                'num_offers': len(offers),
                'offers': [
                    {
                        'forecast_id': offer.good_id,
                        'price': offer.price,
                        'marketing_attributes': offer.marketing_attributes
                    }
                    for offer in offers
                ]
            }
            
        except Exception as e:
            # Handle seller errors gracefully
            seller_strategies[seller.seller_id] = {
                'error': str(e),
                'num_offers': 0,
                'offers': []
            }
    
    # Step 4: Update market state with new offers
    market_model.state.offers = period_offers
    
    # Return period summary
    return {
        'period': period,
        'num_sellers': len(sellers),
        'num_offers': len(period_offers),
        'seller_strategies': seller_strategies,
        'total_forecasts_assigned': len([fid for fid in market_model.state.knowledge_good_forecasts.keys() 
                                        if f"_r{period}_" in fid])
    }
