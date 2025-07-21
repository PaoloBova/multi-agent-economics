#!/usr/bin/env python3
"""
Run the structured note lemons simulation using the new framework.
"""

import asyncio
import json
import yaml
import pandas as pd
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.scenarios import run_flagship_scenario


def load_prepared_data():
    """Load the prepared scenario data."""
    scenario_config_path = Path("data/interim/scenario_config.json")
    agent_configs_path = Path("data/interim/agent_configs.json")
    market_data_path = Path("data/interim/market_data.json")
    
    if not all(p.exists() for p in [scenario_config_path, agent_configs_path, market_data_path]):
        raise FileNotFoundError("Prepared data not found. Run prepare_scenario.py first.")
    
    with open(scenario_config_path, 'r') as f:
        scenario_config = json.load(f)
    
    with open(agent_configs_path, 'r') as f:
        agent_configs = json.load(f)
    
    with open(market_data_path, 'r') as f:
        market_data = json.load(f)
    
    return scenario_config, agent_configs, market_data


def create_simulation_config(scenario_config, agent_configs, market_data):
    """Create the configuration for the simulation."""
    return {
        "rounds": scenario_config["rounds"],
        "seller_banks": [org for org, config in agent_configs.items() if config["type"] == "seller_bank"],
        "buyer_funds": [org for org, config in agent_configs.items() if config["type"] == "buyer_fund"],
        "initial_budgets": {
            "seller_bank": next(config["budget"] for config in agent_configs.values() if config["type"] == "seller_bank"),
            "buyer_fund": next(config["budget"] for config in agent_configs.values() if config["type"] == "buyer_fund")
        },
        "market_parameters": scenario_config["market_parameters"],
        "market_data": market_data
    }


def export_simulation_output(results):
    """Export simulation results in the expected DVC format."""
    # Create output directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/action_logs").mkdir(parents=True, exist_ok=True)
    Path("data/processed/artifacts").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # Export main results
    with open("data/processed/simulation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create CSV output for compatibility
    simulation_data = []
    for round_result in results.get("rounds", []):
        round_num = round_result["round"]
        
        # Add trades to simulation data
        for trade in round_result.get("trades", []):
            simulation_data.append({
                "round": round_num,
                "seller": trade["seller"],
                "buyer": trade["buyer"],
                "instrument": trade["instrument"],
                "price": trade.get("price", 0),
                "quality": trade.get("quality", "unknown")
            })
        
        # Add round summary even if no trades
        if not round_result.get("trades"):
            simulation_data.append({
                "round": round_num,
                "seller": None,
                "buyer": None,
                "instrument": None,
                "price": None,
                "quality": None
            })
    
    # Export CSV
    if simulation_data:
        df = pd.DataFrame(simulation_data)
        df.to_csv("results/simulation_output.csv", index=False)
    else:
        # Create empty CSV with headers
        df = pd.DataFrame(columns=["round", "seller", "buyer", "instrument", "price", "quality"])
        df.to_csv("results/simulation_output.csv", index=False)
    
    print(f"Simulation complete! Exported {len(simulation_data)} records to results/simulation_output.csv")


async def main():
    """Main simulation function."""
    print("Starting structured note lemons simulation...")
    
    try:
        # Load prepared data
        scenario_config, agent_configs, market_data = load_prepared_data()
        print(f"Loaded scenario: {scenario_config['scenario_type']}")
        print(f"Organizations: {len(agent_configs)}")
        
        # Create simulation configuration
        sim_config = create_simulation_config(scenario_config, agent_configs, market_data)
        
        # Set up workspace
        workspace_dir = Path("data/processed")
        
        # Run the simulation
        print(f"Running {sim_config['rounds']} rounds...")
        results = await run_flagship_scenario(workspace_dir, sim_config)
        
        # Export results
        export_simulation_output(results)
        
        # Print summary
        summary = results.get("summary", {})
        print("\n=== SIMULATION SUMMARY ===")
        print(f"Total rounds: {summary.get('total_rounds', 0)}")
        print(f"Total trades: {summary.get('total_trades', 0)}")
        print(f"Average trades per round: {summary.get('average_trades_per_round', 0):.2f}")
        
        budget_summary = summary.get("budget_summary", {})
        print(f"Total budget allocated: {budget_summary.get('total_allocated', 0)}")
        print(f"Total spent: {budget_summary.get('total_spent', 0)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python scripts/prepare_scenario.py' first.")
        sys.exit(1)
    except Exception as e:
        print(f"Simulation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
