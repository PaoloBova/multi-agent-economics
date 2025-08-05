#!/usr/bin/env python3
"""
Demonstration of the new functional tool integration system.

This example shows how to:
1. Set up a MarketModel with MarketState
2. Create economic tools using the functional approach
3. Use tools with proper budget management and state access
4. Show AutoGen schema generation working correctly
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multi_agent_economics.models.market_for_finance import MarketModel, MarketState, RegimeParameters
from multi_agent_economics.core.autogen_tools import create_economic_tools, setup_agent_with_tools


def create_demo_market_model():
    """Create a demo market model with realistic regime-switching parameters."""
    
    # Set up regime parameters for different sectors
    regime_parameters = {
        "tech": {
            0: RegimeParameters(mu=0.12, sigma=0.18),  # Bull market: high return, high vol
            1: RegimeParameters(mu=0.02, sigma=0.25)   # Bear market: low return, higher vol
        },
        "finance": {
            0: RegimeParameters(mu=0.08, sigma=0.15),  # Bull market
            1: RegimeParameters(mu=-0.01, sigma=0.22)  # Bear market: negative return
        },
        "healthcare": {
            0: RegimeParameters(mu=0.06, sigma=0.12),  # Defensive: steady growth
            1: RegimeParameters(mu=0.03, sigma=0.15)   # Defensive: less affected by bear market
        }
    }
    
    # Create market state
    market_state = MarketState(
        prices={},
        offers=[],
        trades=[],
        demand_profile={},
        supply_profile={},
        index_values={"tech": 105.2, "finance": 98.7, "healthcare": 102.1},
        current_regimes={"tech": 0, "finance": 0, "healthcare": 0},
        regime_parameters=regime_parameters,
        current_period=0,
        risk_free_rate=0.03,
        # Agent context fields (new)
        current_agent_id="demo_agent",
        budgets={"demo_agent": 25.0, "analyst_1": 30.0, "trader_1": 20.0},
        tool_usage={}
    )
    
    # Create market model
    market_model = MarketModel(
        id=1,
        name="Demo Economic Market",
        agents=[],
        state=market_state,
        step=lambda: None,  # Dummy step function
        collect_stats=lambda: {}  # Dummy stats function
    )
    
    return market_model


def create_demo_config():
    """Create demo configuration for tools."""
    return {
        "tool_parameters": {
            "sector_forecast": {
                "effort_thresholds": {"high": 5.0, "medium": 2.0},
                "quality_tiers": {
                    "high": {"noise_factor": 0.1, "confidence": 0.9},
                    "medium": {"noise_factor": 0.3, "confidence": 0.7},
                    "low": {"noise_factor": 0.6, "confidence": 0.5}
                }
            },
            "monte_carlo_var": {
                "effort_thresholds": {"high": 5.0, "medium": 2.0},
                "simulation_sizes": {"high": 50000, "medium": 10000, "low": 1000}
            },
            "price_note": {
                "effort_thresholds": {"high": 4.0, "medium": 2.0},
                "pricing_error_std": {"high": 0.01, "medium": 0.05, "low": 0.10}
            },
            "reflect": {
                "confidence_formula": {"base": 0.8, "effort_multiplier": 0.05}
            }
        },
        "market_data": {
            "sectors": {
                "tech": {"mean": 0.10, "std": 0.20},
                "finance": {"mean": 0.06, "std": 0.18},
                "healthcare": {"mean": 0.05, "std": 0.12},
                "energy": {"mean": 0.04, "std": 0.22}
            },
            "default_sector": {"mean": 0.06, "std": 0.15}
        }
    }


async def demo_tool_usage():
    """Demonstrate the functional tool system."""
    print("=== Functional Tool Integration Demo ===\n")
    
    # 1. Create market model and configuration
    print("1. Setting up market model and configuration...")
    market_model = create_demo_market_model()
    config_data = create_demo_config()
    
    print(f"   Market state initialized with:")
    print(f"   - Current regimes: {market_model.state.current_regimes}")
    print(f"   - Index values: {market_model.state.index_values}")
    print(f"   - Agent budgets: {market_model.state.budgets}")
    print()
    
    # 2. Create tools using functional approach
    print("2. Creating economic tools using functional approach...")
    tools = create_economic_tools(market_model, config_data)
    
    print(f"   Created {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    print()
    
    # 3. Demonstrate tool schema generation
    print("3. AutoGen FunctionTool schema generation:")
    print("   Tool schemas (automatically generated from function signatures):")
    for tool in tools:
        schema = tool.schema
        print(f"\n   {tool.name}:")
        print(f"     Description: {schema.get('description', 'N/A')}")
        if 'parameters' in schema and 'properties' in schema['parameters']:
            print("     Parameters:")
            for param_name, param_info in schema['parameters']['properties'].items():
                param_type = param_info.get('type', 'unknown')
                param_desc = param_info.get('description', 'No description')
                print(f"       - {param_name} ({param_type}): {param_desc}")
    print()
    
    # 4. Test tool execution with different effort levels
    print("4. Testing tool execution with different effort levels:")
    
    # Set agent context
    market_model.state.current_agent_id = "demo_agent"
    initial_budget = market_model.state.budgets["demo_agent"]
    print(f"   Agent 'demo_agent' starting budget: {initial_budget}")
    print()
    
    # Test sector forecast with low effort
    print("   a) Sector forecast with low effort (1.0):")
    sector_tool = tools[0]  # sector_forecast
    result = await sector_tool.run_json({
        "sector": "tech",
        "horizon": 3,
        "effort": 1.0
    }, None)  # CancellationToken can be None for demo
    
    print(f"      Result: {result}")
    print(f"      Quality tier: {result.get('quality_tier')}")
    print(f"      Effort used: {result.get('effort_used')}")
    remaining_budget = market_model.state.budgets["demo_agent"]
    print(f"      Remaining budget: {remaining_budget}")
    print()
    
    # Test Monte Carlo VaR with high effort
    print("   b) Monte Carlo VaR with high effort (6.0):")
    var_tool = tools[1]  # monte_carlo_var
    result = await var_tool.run_json({
        "portfolio_value": 1000000.0,
        "volatility": 0.15,
        "confidence_level": 0.95,
        "effort": 6.0
    }, None)
    
    print(f"      Result summary:")
    print(f"        VaR estimate: ${result.get('var_estimate', 0):,.2f}")
    print(f"        Quality tier: {result.get('quality_tier')}")
    print(f"        Simulations: {result.get('n_simulations'):,}")
    print(f"        Effort used: {result.get('effort_used')}")
    remaining_budget = market_model.state.budgets["demo_agent"]
    print(f"      Remaining budget: {remaining_budget}")
    print()
    
    # Test pricing with insufficient budget
    print("   c) Price note with insufficient budget (25.0 effort, but limited budget):")
    price_tool = tools[2]  # price_note
    result = await price_tool.run_json({
        "notional": 100000.0,
        "payoff_type": "autocall",
        "underlying_forecast": [0.08, 0.06, 0.09],
        "discount_rate": 0.03,
        "effort": 25.0  # More than remaining budget
    }, None)
    
    print(f"      Result summary:")
    print(f"        Fair value: ${result.get('fair_value', 0):,.2f}")
    print(f"        Quoted price: ${result.get('quoted_price', 0):,.2f}")
    print(f"        Quality tier: {result.get('quality_tier')}")
    print(f"        Effort requested: {result.get('effort_requested')}")
    print(f"        Effort used: {result.get('effort_used')}")
    print(f"        Warnings: {result.get('warnings', [])}")
    remaining_budget = market_model.state.budgets["demo_agent"]
    print(f"      Final budget: {remaining_budget}")
    print()
    
    # 5. Show tool usage history
    print("5. Tool usage history:")
    usage_history = market_model.state.tool_usage.get("demo_agent", [])
    print(f"   Agent 'demo_agent' used {len(usage_history)} tools:")
    for i, usage in enumerate(usage_history, 1):
        print(f"     {i}. {usage['tool']} - effort: {usage['effort']}")
    print()
    
    # 6. Test reflection tool
    print("6. Testing reflection tool:")
    reflect_tool = tools[3]  # reflect
    result = await reflect_tool.run_json({
        "context": "Current portfolio shows high tech sector concentration with recent volatility increase. Need to decide on rebalancing strategy.",
        "goals": ["minimize risk", "maintain growth potential", "diversify sectors"],
        "effort": 2.5
    }, None)
    
    print(f"   Reflection result:")
    print(f"     Analysis depth: {result.get('analysis_depth')}")
    print(f"     Confidence: {result.get('confidence'):.2f}")
    print(f"     Next actions: {result.get('next_actions')}")
    print(f"     Effort used: {result.get('effort_used')}")
    print()
    
    print("=== Demo Complete ===")
    print(f"Final agent budget: {market_model.state.budgets['demo_agent']:.2f}")
    print(f"Total tools used: {len(market_model.state.tool_usage.get('demo_agent', []))}")


async def demo_agent_setup():
    """Demonstrate setting up an agent with the new tool system."""
    print("\n=== Agent Setup Demo ===")
    
    # Create market model and config
    market_model = create_demo_market_model()
    config_data = create_demo_config()
    
    print("Note: This demo shows the agent setup process.")
    print("In a real scenario, you would provide an actual model_client like:")
    print("  from autogen_ext.models.openai import OpenAIChatCompletionClient")
    print("  model_client = OpenAIChatCompletionClient(model='gpt-4')")
    print()
    
    # Mock model client for demo (real usage would require actual API keys)
    class MockModelClient:
        def __init__(self):
            self.model = "mock-gpt-4"
        
        def __str__(self):
            return f"MockModelClient(model={self.model})"
    
    mock_client = MockModelClient()
    
    try:
        agent = setup_agent_with_tools(
            market_model=market_model,
            config_data=config_data,
            agent_name="EconomicAnalyst_1",
            agent_role="analyst",
            organization="TechBank",
            model_client=mock_client,
            initial_budget=25.0
        )
        
        print(f"Agent created successfully:")
        print(f"  Name: {agent.name}")
        print(f"  Tools available: {len(agent.tools)}")
        print(f"  Agent ID in market: TechBank.analyst")
        print(f"  Initial budget: {market_model.state.budgets.get('TechBank.analyst', 0)}")
        print(f"  Model client: {mock_client}")
        
    except Exception as e:
        print(f"Agent setup demo completed with expected error (no real model client): {e}")
        print("This is expected in demo mode without actual AutoGen dependencies.")


if __name__ == "__main__":
    # Run the demo
    print("Starting functional tool integration demonstration...")
    print("This demo requires autogen-core and autogen-agentchat packages.")
    print()
    
    try:
        # Test tool creation and usage
        asyncio.run(demo_tool_usage())
        
        # Test agent setup
        asyncio.run(demo_agent_setup())
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nThis demo requires AutoGen packages to be installed:")
        print("  pip install autogen-agentchat autogen-ext[openai]")
        print("\nBut the core functionality demonstration above should work without them.")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("This might be expected if AutoGen packages are not properly installed.")