#!/usr/bin/env python3
"""
Data preparation script for multi-agent economics simulation.
This script processes raw data and prepares it for simulation.
"""

import os
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "simulation_config.yaml") -> dict:
    """Load simulation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_initial_data():
    """Prepare initial data for simulation."""
    config = load_config()
    
    # Create initial agent data
    agent_count = config['agents']['count']
    initial_wealth = config['agents']['initial_wealth']
    
    agents_data = {
        'agent_id': range(agent_count),
        'agent_type': ['consumer'] * (agent_count // 3) + 
                     ['producer'] * (agent_count // 3) + 
                     ['trader'] * (agent_count - 2 * (agent_count // 3)),
        'initial_wealth': [initial_wealth] * agent_count,
        'position_x': [0.0] * agent_count,  # Will be randomized in simulation
        'position_y': [0.0] * agent_count,
    }
    
    agents_df = pd.DataFrame(agents_data)
    
    # Ensure output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    agents_df.to_csv('data/processed/initial_agents.csv', index=False)
    
    print(f"Prepared data for {agent_count} agents")
    print(f"Agent types: {agents_df['agent_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    prepare_initial_data()
