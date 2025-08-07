"""
AkerlofSeller implementation for demonstrating Akerlof's "Market for Lemons" dynamics.

This implementation creates a two-stage decision process where sellers:
1. Stage 1: Choose effort level (production quality with sunk costs)
2. Stage 2: Choose marketing strategy (attribute claims + competitive pricing)

The asymmetric information between true quality and observable attributes creates
conditions for adverse selection and potential market failure.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, Offer, ForecastData, TradeData, RegimeParameters,
    BuyerState, SellerState, build_confusion_matrix, generate_forecast_signal,
    categorical_draw
)
from multi_agent_economics.tools.implementations.economic import sector_forecast_impl


class AkerlofSeller:
    """
    AkerlofSeller implementing two-stage decision making in markets with asymmetric information.
    
    Stage 1: Effort allocation (production quality choice with sunk costs)
    Stage 2: Marketing (attribute claims + pricing for already-produced goods)
    
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

        # Effort cost structure - represents production quality trade-offs
        # Get from context with sensible defaults
        self.effort_costs = self.context.get('effort_costs', {
            'high': 100.0,    # Extensive research, premium data, sophisticated models
            'medium': 60.0,   # Standard research, decent data, basic models  
            'low': 30.0       # Minimal research, public data, simple heuristics
        })
        
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
        
        # Get performance tolerance from context
        performance_tolerance = self.context.get('performance_tolerance', 0.05)  # Default 5% tolerance
        
        for trade in trades:
            if trade.good_id in self.performance_data:
                actual_performance = self.performance_data[trade.good_id]
                # Consider performances within tolerance range as "similar"
                if abs(actual_performance - performance_level) <= performance_tolerance:
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
    
    def stage1_effort_decision(self, historical_trades: List[TradeData], round_num: int) -> Tuple[str, Tuple[str, ForecastData]]:
        """
        Stage 1: Choose effort level to maximize expected profit.
        
        Economic logic:
        1. Analyze historical data to estimate revenue by performance level
        2. Calculate expected profit for each effort level (revenue - cost)
        3. Choose effort level with highest expected profit
        
        Args:
            historical_trades: Historical trading data for learning
            round_num: Current round number for forecast ID generation
            
        Returns:
            Tuple of (effort_level, (forecast_id, forecast_data))
        """
        # Calculate expected profits for each effort level
        effort_profits = {}
        
        for effort, cost in self.effort_costs.items():
            # Map effort to expected performance level
            expected_performance = self.effort_quality_mapping[effort]
            
            # Estimate revenue based on historical data
            historical_revenues = self.analyze_historical_revenues(expected_performance, historical_trades)
            expected_revenue = np.mean(historical_revenues) if historical_revenues else 0.0
            
            # Calculate expected profit
            expected_profit = expected_revenue - cost
            effort_profits[effort] = expected_profit
            
        # Choose effort level with maximum expected profit
        optimal_effort = max(effort_profits.keys(), key=lambda x: effort_profits[x])
        
        # Convert effort string to numeric value for economic tools
        effort_numeric = self._convert_effort_to_numeric(optimal_effort)
        
        # Generate forecast with chosen effort level
        sector = self.context.get('sector', 'tech')  # Get sector from context, default to 'tech'
        horizon = self.context.get('horizon', 1)     # Get forecast horizon from context, default to 1
        
        forecast_data = sector_forecast_impl(
            self.market_model,
            config_data=self.context,
            sector=sector,
            horizon=horizon,
            effort=effort_numeric
        )
        forecast_id = f"{self.seller_id}_forecast_r{round_num}_{optimal_effort}"
        
        # Store forecast in market state for buyer access
        self.market_state.knowledge_good_forecasts[forecast_id] = forecast_data

        return optimal_effort, (forecast_id, forecast_data)

    def generate_forecast_with_effort(self, effort_level: str, round_num: int) -> Tuple[str, ForecastData]:
        """
        Generate forecast with specified effort level for testing purposes.
        
        Args:
            effort_level: String effort level ('high', 'medium', 'low')
            round_num: Round number for forecast ID
            
        Returns:
            Tuple of (forecast_id, forecast_data)
        """
        # Convert effort string to numeric value for economic tools
        effort_numeric = self._convert_effort_to_numeric(effort_level)
        
        # Generate forecast with chosen effort level
        sector = self.context.get('sector', 'tech')
        horizon = self.context.get('horizon', 1)
        
        forecast_data = sector_forecast_impl(
            self.market_model,
            config_data=self.context,
            sector=sector,
            horizon=horizon,
            effort=effort_numeric
        )
        forecast_id = f"{self.seller_id}_forecast_r{round_num}_{effort_level}"
        
        # Store forecast in market state for buyer access
        self.market_state.knowledge_good_forecasts[forecast_id] = forecast_data
        
        return forecast_id, forecast_data


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
        """
        # Extract buyer preferences for optimal attribute selection
        avg_buyer_prefs = self.extract_buyer_preferences()
        
        # Choose attribute claims (can be independent of true quality - asymmetric info!)
        # For now, use a simple strategy: claim high attributes if profitable
        proposed_attrs = self.choose_optimal_attributes(avg_buyer_prefs, market_competition)
        
        # Set competitive price
        price = self.estimate_competitive_price(proposed_attrs, market_competition)
        
        # Create and post offer
        offer = self.create_and_post_offer(good_id, proposed_attrs, price)
        
        return offer
        
    def extract_buyer_preferences(self) -> List[float]:
        """
        Extract average buyer preferences from market state.
        
        Returns:
            List of average attribute preferences across all buyers
        """
        if not self.market_state.buyers_state:
            # Get default preferences from context
            default_prefs = self.context.get('default_buyer_preferences', [0.5, 0.5])
            return default_prefs
            
        # Calculate average preferences across all buyers
        all_prefs = [buyer.attr_mu for buyer in self.market_state.buyers_state if buyer.attr_mu]
        
        if not all_prefs:
            # Get default preferences from context
            default_prefs = self.context.get('default_buyer_preferences', [0.5, 0.5])
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
            Chosen attribute vector [attr1, attr2]
        """
        # Get marketing strategies from context, with defaults
        strategies = self.context.get('marketing_strategies', {
            'premium_claims': [0.9, 0.8],    # High claims, high marketing cost (10)
            'standard_claims': [0.7, 0.6],   # Medium claims, medium marketing cost (5)
            'basic_claims': [0.5, 0.4]       # Low claims, low marketing cost (2)
        })
        
        # Get pricing parameters from context
        wtp_pricing_factor = self.context.get('wtp_pricing_factor', 0.9)  # Default 90% of WTP
        
        best_strategy = None
        best_profit = -float('inf')
        
        for strategy_name, attributes in strategies.items():
            # Estimate revenue from this strategy
            wtp = self._estimate_willingness_to_pay_for_attrs(attributes, avg_buyer_prefs)
            marketing_cost = self.marketing_costs[strategy_name]
            
            # Calculate expected profit (revenue - marketing cost)
            expected_profit = wtp * wtp_pricing_factor - marketing_cost
            
            if expected_profit > best_profit:
                best_profit = expected_profit
                best_strategy = attributes
        
        # Get default fallback from context
        default_strategy = self.context.get('default_strategy', [0.7, 0.6])
        return best_strategy if best_strategy else default_strategy
        
    def estimate_competitive_price(self, proposed_attrs: List[float], competition: List[Offer]) -> float:
        """
        Estimate competitive price for given attributes.
        
        Args:
            proposed_attrs: Proposed attribute vector
            competition: List of competing offers
            
        Returns:
            Competitive price estimate
        """
        if not competition:
            # No competition - price based on buyer willingness-to-pay
            wtp_pricing_factor = self.context.get('wtp_pricing_factor', 0.9)  # Default 90% of WTP
            avg_prefs = self.extract_buyer_preferences()
            wtp = self._estimate_willingness_to_pay_for_attrs(proposed_attrs, avg_prefs)
            return wtp * wtp_pricing_factor
        else:
            # With competition - price below similar competitors
            return self._competitive_pricing(proposed_attrs, competition)
            
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
        avg_prefs = self.extract_buyer_preferences()
        # Get scaling factor from context for realistic pricing
        wtp_scaling_factor = self.context.get('wtp_scaling_factor', 100.0)  # Default 100x scaling
        return self._estimate_willingness_to_pay_for_attrs(attrs, avg_prefs) * wtp_scaling_factor
            
    def _competitive_pricing(self, attrs: List[float], competition: List[Offer]) -> float:
        """
        Set competitive pricing considering similar offers.
        
        Args:
            attrs: Proposed attribute vector
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
            similarity = self.calculate_attribute_similarity(attrs, offer.attr_vector)
            if similarity >= similarity_threshold:
                similar_prices.append(offer.price)
                
        if similar_prices:
            # Price below average competitor price
            avg_competitor_price = np.mean(similar_prices)
            return avg_competitor_price * competitor_discount
        else:
            # No similar competition - fall back to WTP pricing
            return self._estimate_willingness_to_pay(attrs)
            
    def calculate_attribute_similarity(self, attrs1: List[float], attrs2: List[float]) -> float:
        """
        Calculate similarity between two attribute vectors using Euclidean distance.
        
        Args:
            attrs1: First attribute vector
            attrs2: Second attribute vector
            
        Returns:
            Similarity score [0, 1] where 1 = identical, 0 = maximally different
        """
        if len(attrs1) != len(attrs2):
            return 0.0
            
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum([(a1 - a2)**2 for a1, a2 in zip(attrs1, attrs2)]))
        
        # Convert to similarity (max distance for unit vectors is sqrt(2))
        max_distance = np.sqrt(len(attrs1))
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, similarity)
        
    def create_and_post_offer(self, good_id: str, attrs: List[float], price: float) -> Offer:
        """
        Create offer and post it to market state.
        
        Args:
            good_id: Forecast ID to be offered
            attrs: Chosen attribute vector
            price: Chosen price
            
        Returns:
            Created Offer object
        """
        offer = Offer(
            good_id=good_id,
            price=price,
            seller=self.seller_id,
            attr_vector=attrs
        )
        
        # Add to market state
        self.market_state.offers.append(offer)
        
        return offer
