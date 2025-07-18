#!/usr/bin/env python3
"""
Simulation runner for multi-agent economics simulation.
This script will use AutoGen to run the actual multi-agent simulation.
"""

import os
import json
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "simulation_config.yaml") -> dict:
    """Load simulation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_simulation():
    """Run the multi-agent economics simulation."""
    config = load_config()
    
    # Load prepared data
    agents_df = pd.read_csv('data/processed/initial_agents.csv')
    
    print(f"Starting simulation with {len(agents_df)} agents")
    print(f"Simulation steps: {config['simulation']['steps']}")
    
    # TODO: Implement actual AutoGen simulation here
    # For now, create mock output
    
    simulation_steps = config['simulation']['steps']
    output_frequency = config['simulation']['output_frequency']
    
    # Mock simulation results
    results = []
    agent_behaviors = {}
    
    for step in range(0, simulation_steps, output_frequency):
        # Mock market data
        market_price = 100 + (step * 0.1) + (step % 10) * 2  # Simple trend + noise
        total_trades = len(agents_df) * 0.3  # 30% of agents trade each period
        
        results.append({
            'time_step': step,
            'market_price': market_price,
            'total_trades': total_trades,
            'market_volume': total_trades * market_price
        })
    
    # Mock agent behavior data
    for _, agent in agents_df.iterrows():
        agent_id = agent['agent_id']
        agent_behaviors[str(agent_id)] = {
            'type': agent['agent_type'],
            'final_wealth': agent['initial_wealth'] * (0.8 + 0.4 * (agent_id % 10) / 10),
            'total_transactions': 10 + (agent_id % 50),
            'avg_transaction_value': 50 + (agent_id % 100)
        }
    
    # Ensure output directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/simulation_output.csv', index=False)
    
    with open('results/agent_behaviors.json', 'w') as f:
        json.dump(agent_behaviors, f, indent=2)
    
    print(f"Simulation completed. Results saved to results/")
    print(f"Final market price: ${results_df['market_price'].iloc[-1]:.2f}")


if __name__ == "__main__":
    run_simulation()
