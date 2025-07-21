#!/usr/bin/env python3
"""
Simple example demonstrating the structured note lemons simulation.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.scenarios import run_flagship_scenario


def create_simple_config():
    """Create a simple configuration for testing."""
    return {
        "rounds": 3,  # Short test run
        "seller_banks": ["SellerBank1"],
        "buyer_funds": ["BuyerFund1"],
        "initial_budgets": {
            "seller_bank": 20,
            "buyer_fund": 10
        },
        "market_parameters": {
            "base_notional": 100,
            "discount_rate": 0.03,
            "volatility": 0.15
        }
    }


async def run_example():
    """Run a simple example simulation."""
    print("=== Structured Note Lemons - Simple Example ===\n")
    
    # Setup
    workspace_dir = Path("example_output")
    config = create_simple_config()
    
    print("Configuration:")
    print(f"- Rounds: {config['rounds']}")
    print(f"- Seller banks: {config['seller_banks']}")
    print(f"- Buyer funds: {config['buyer_funds']}")
    print(f"- Initial budgets: Sellers={config['initial_budgets']['seller_bank']}, Buyers={config['initial_budgets']['buyer_fund']}")
    
    try:
        print("\nRunning simulation...")
        results = await run_flagship_scenario(workspace_dir, config)
        
        print("\n=== RESULTS ===")
        summary = results.get("summary", {})
        print(f"✓ Completed {summary.get('total_rounds', 0)} rounds")
        print(f"✓ Total trades: {summary.get('total_trades', 0)}")
        print(f"✓ Budget utilization:")
        
        budget_summary = summary.get("budget_summary", {})
        for agent, balance in budget_summary.get("agents", {}).items():
            print(f"  - {agent}: {balance:.1f} credits remaining")
        
        print(f"\n📁 Detailed results saved to: {workspace_dir}")
        
        # Show sample of the action log
        action_metrics = summary.get("action_metrics", {})
        if action_metrics:
            print(f"\n📊 Action Summary:")
            print(f"  - Total internal actions: {action_metrics.get('total_internal_actions', 0)}")
            print(f"  - Total external actions: {action_metrics.get('total_external_actions', 0)}")
            
            tools_used = action_metrics.get("tools", {})
            if tools_used:
                print(f"  - Tools used: {', '.join(tools_used.keys())}")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        print("\nThis is expected if AutoGen is not installed.")
        print("The simulation will run in mock mode for testing.")


def main():
    """Main example function."""
    print("This example demonstrates the structured note lemons scenario.")
    print("It shows how economic agents with different quality strategies")
    print("interact in a market with information asymmetries.\n")
    
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
