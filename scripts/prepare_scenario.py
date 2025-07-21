#!/usr/bin/env python3
"""
Prepare scenario configuration and market data for the structured note lemons simulation.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
import numpy as np


def load_simulation_config():
    """Load the simulation configuration."""
    with open("simulation_config.yaml", 'r') as f:
        return yaml.safe_load(f)


def generate_market_data(config):
    """Generate synthetic market data for the simulation."""
    sectors = config["economic"]["sectors"]
    rounds = config["simulation"]["rounds"]
    
    market_data = {
        "sectors": sectors,
        "discount_curve": {
            "rate": config["economic"]["market_parameters"]["discount_rate"],
            "term_structure": [0.025, 0.03, 0.035, 0.04]
        },
        "time_series": {}
    }
    
    # Generate time series data for each sector
    for sector, params in sectors.items():
        np.random.seed(config["simulation"]["random_seed"])
        
        # Generate daily returns for the simulation period
        daily_returns = np.random.normal(
            params["mean_growth"] / 252,  # Daily mean
            params["volatility"] / np.sqrt(252),  # Daily volatility
            rounds * 5  # 5 data points per round
        )
        
        # Convert to cumulative prices
        prices = 100 * np.cumprod(1 + daily_returns)
        
        market_data["time_series"][sector] = {
            "prices": prices.tolist(),
            "returns": daily_returns.tolist(),
            "volatility": params["volatility"],
            "mean_growth": params["mean_growth"]
        }
    
    return market_data


def create_agent_configs(config):
    """Create individual agent configurations."""
    agent_configs = {}
    
    # Create seller bank configurations
    for bank in config["agents"]["seller_banks"]:
        agent_configs[bank] = {
            "organization": bank,
            "type": "seller_bank",
            "roles": ["Analyst", "Structurer", "PM"],
            "budget": config["agents"]["initial_budgets"]["seller_bank"],
            "capabilities": {
                "Analyst": ["sector_forecast", "reflect"],
                "Structurer": ["price_note", "doc_generate", "reflect"],
                "PM": ["kanban_update", "reflect", "share_artifact"]
            }
        }
    
    # Create buyer fund configurations  
    for fund in config["agents"]["buyer_funds"]:
        agent_configs[fund] = {
            "organization": fund,
            "type": "buyer_fund", 
            "roles": ["Risk-Officer", "Trader"],
            "budget": config["agents"]["initial_budgets"]["buyer_fund"],
            "capabilities": {
                "Risk-Officer": ["monte_carlo_var", "reflect"],
                "Trader": ["reflect"]
            }
        }
    
    return agent_configs


def create_scenario_config(config):
    """Create the scenario-specific configuration."""
    return {
        "scenario_type": config["scenario"]["type"],
        "description": config["scenario"]["description"],
        "rounds": config["simulation"]["rounds"],
        "quality_thresholds": config["quality"]["thresholds"],
        "market_parameters": config["economic"]["market_parameters"],
        "tools_config": config.get("tools", {})
    }


def main():
    """Main preparation function."""
    print("Preparing scenario configuration and data...")
    
    # Load configuration
    config = load_simulation_config()
    
    # Create output directories
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    
    # Generate market data
    print("Generating synthetic market data...")
    market_data = generate_market_data(config)
    with open("data/interim/market_data.json", 'w') as f:
        json.dump(market_data, f, indent=2)
    
    # Create agent configurations
    print("Creating agent configurations...")
    agent_configs = create_agent_configs(config)
    with open("data/interim/agent_configs.json", 'w') as f:
        json.dump(agent_configs, f, indent=2)
    
    # Create scenario configuration
    print("Creating scenario configuration...")
    scenario_config = create_scenario_config(config)
    with open("data/interim/scenario_config.json", 'w') as f:
        json.dump(scenario_config, f, indent=2)
    
    print("Preparation complete!")
    print(f"- Market data: {len(market_data['sectors'])} sectors")
    print(f"- Agent configs: {len(agent_configs)} organizations")
    print(f"- Scenario: {scenario_config['scenario_type']}")


if __name__ == "__main__":
    main()
