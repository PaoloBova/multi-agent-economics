"""
Economic models and data structures.

This module contains economic models, data structures, and mathematical
representations used throughout the simulation framework.
"""

# Import available modules
from .market_for_finance import (
    transition_regimes,
    generate_regime_returns,
    build_regime_covariance,
    build_confusion_matrix,
    generate_forecast_signal,
    update_agent_beliefs,
    compute_portfolio_moments,
    optimize_portfolio,
    MarketState,
    MarketModel,
    Offer
)

# TODO: Import these when implemented
# from .portfolio import Portfolio
# from .order import Order, OrderType  
# from .market_data import MarketData, PriceHistory

__all__ = [
    "transition_regimes",
    "generate_regime_returns", 
    "build_regime_covariance",
    "build_confusion_matrix",
    "generate_forecast_signal",
    "update_agent_beliefs",
    "compute_portfolio_moments",
    "optimize_portfolio",
    "MarketState",
    "MarketModel",
    "Offer"
]
