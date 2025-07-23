#!/usr/bin/env python3
"""
Test script for the new effort-based economic tools.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.core.economic_tools import (
    sector_forecast, monte_carlo_var, price_note, reflect
)
from multi_agent_economics.core.tool_wrapper import (
    SimulationState, create_autogen_tools, load_tool_config
)


def test_economic_tools():
    """Test the new economic tools with effort-based interface."""
    print("=== Testing Economic Tools ===\n")
    
    # Initialize simulation state
    simulation_state = SimulationState()
    
    # Set up agent budgets
    agents = ["SellerBank1.analyst", "SellerBank1.trader", "BuyerFund1.risk_officer"]
    for agent in agents:
        simulation_state.budgets[agent] = 20.0
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "data" / "config" / "effort_based_tools.json"
    config_data = load_tool_config(str(config_path))
    
    print(f"✓ Loaded configuration from: {config_path}")
    print(f"✓ Budget settings: {config_data.get('budget_settings', {})}")
    print()
    
    # Test sector forecast with different effort levels
    print("--- Testing Sector Forecast ---")
    simulation_state.set_current_agent("SellerBank1.analyst")
    
    # High effort forecast
    result_high = sector_forecast(
        sector="tech", 
        horizon=6, 
        effort=6.0,
        _state=simulation_state,
        _config=config_data
    )
    print(f"High effort (6.0): Quality={result_high['quality_tier']}, Confidence={result_high['confidence']}")
    
    # Medium effort forecast
    result_med = sector_forecast(
        sector="finance",
        horizon=4,
        effort=3.0,
        _state=simulation_state,
        _config=config_data
    )
    print(f"Medium effort (3.0): Quality={result_med['quality_tier']}, Confidence={result_med['confidence']}")
    
    # Low effort forecast
    result_low = sector_forecast(
        sector="healthcare",
        horizon=3,
        effort=1.0,
        _state=simulation_state,
        _config=config_data
    )
    print(f"Low effort (1.0): Quality={result_low['quality_tier']}, Confidence={result_low['confidence']}")
    print()
    
    # Test budget constraints
    print("--- Testing Budget Constraints ---")
    current_budget = simulation_state.get_budget("SellerBank1.analyst")
    print(f"Current budget: {current_budget}")
    
    # Try to spend more than available
    result_over = sector_forecast(
        sector="energy",
        horizon=2,
        effort=15.0,  # More than remaining budget
        _state=simulation_state,
        _config=config_data
    )
    print(f"Over-budget request (15.0): Used={result_over['effort_used']}, Warnings={result_over['warnings']}")
    print()
    
    # Test Monte Carlo VaR
    print("--- Testing Monte Carlo VaR ---")
    simulation_state.set_current_agent("BuyerFund1.risk_officer")
    
    portfolio = {
        "value": 1000000,
        "volatility": 0.20,
        "confidence": 0.95
    }
    
    var_result = monte_carlo_var(
        portfolio=portfolio,
        effort=5.0,
        _state=simulation_state,
        _config=config_data
    )
    print(f"VaR Analysis: Quality={var_result['quality_tier']}, VaR_95=${var_result['var_95']:,.0f}")
    print(f"Simulations: {var_result['n_simulations']:,}")
    print()
    
    # Test pricing tool
    print("--- Testing Price Note ---")
    simulation_state.set_current_agent("SellerBank1.trader")
    
    payoff_fn = {"notional": 100}
    forecast_data = {"forecast": [0.08, 0.09, 0.07, 0.10]}
    discount_curve = {"rate": 0.03}
    
    price_result = price_note(
        payoff_fn=payoff_fn,
        forecast=forecast_data,
        discount_curve=discount_curve,
        effort=4.0,
        _state=simulation_state,
        _config=config_data
    )
    print(f"Pricing: Quality={price_result['quality_tier']}, Fair=${price_result['fair_price']:.2f}, Quoted=${price_result['quoted_price']:.2f}")
    print(f"Accuracy: {price_result['pricing_accuracy']:.3f}")
    print()
    
    # Test reflection tool
    print("--- Testing Reflection ---")
    reflection_result = reflect(
        topic="market strategy for next round",
        effort=2.5,
        _state=simulation_state,
        _config=config_data
    )
    print(f"Reflection: Depth={reflection_result['reflection_depth']}, Confidence={reflection_result['confidence']:.2f}")
    print(f"Insights: {len(reflection_result['insights'])}, Actions: {len(reflection_result['next_actions'])}")
    print()
    
    # Show final budget states
    print("--- Final Budget States ---")
    for agent in agents:
        budget = simulation_state.get_budget(agent)
        usage = len(simulation_state.tool_usage.get(agent, []))
        print(f"{agent}: ${budget:.1f} remaining, {usage} tool calls")
    
    print("\n✅ All tool tests completed successfully!")


def test_autogen_integration():
    """Test AutoGen integration (without actually using AutoGen)."""
    print("\n=== Testing AutoGen Integration ===\n")
    
    # Setup simulation
    simulation_state = SimulationState()
    config_data = load_tool_config()
    
    # List of tool functions
    tool_functions = [sector_forecast, monte_carlo_var, price_note, reflect]
    
    # Create wrapped tools
    wrapped_tools = create_autogen_tools(tool_functions, simulation_state, config_data)
    
    print(f"✓ Created {len(wrapped_tools)} AutoGen-ready tools")
    print(f"✓ Tool names: {[f.__name__ for f in wrapped_tools]}")
    
    # Test wrapped tool (simulates what AutoGen would do)
    simulation_state.budgets["test_agent"] = 10.0
    simulation_state.set_current_agent("test_agent")
    
    wrapped_forecast = wrapped_tools[0]  # sector_forecast
    result = wrapped_forecast(sector="tech", horizon=3, effort=2.0)
    
    print(f"✓ Wrapped tool test: Quality={result['quality_tier']}, Effort used={result['effort_used']}")
    print("\n✅ AutoGen integration test completed!")


if __name__ == "__main__":
    test_economic_tools()
    test_autogen_integration()
