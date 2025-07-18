#!/usr/bin/env python3
"""
Analysis script for multi-agent economics simulation results.
This script analyzes simulation outputs and generates visualizations.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_results():
    """Analyze simulation results and create visualizations."""
    
    # Load results
    results_df = pd.read_csv('results/simulation_output.csv')
    
    with open('results/agent_behaviors.json', 'r') as f:
        agent_behaviors = json.load(f)
    
    # Ensure output directories exist
    os.makedirs('results/analysis', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Analysis 1: Market trends
    print("=== Market Analysis ===")
    print(f"Initial market price: ${results_df['market_price'].iloc[0]:.2f}")
    print(f"Final market price: ${results_df['market_price'].iloc[-1]:.2f}")
    print(f"Price change: {((results_df['market_price'].iloc[-1] / results_df['market_price'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Average trading volume: {results_df['market_volume'].mean():.2f}")
    
    # Analysis 2: Agent wealth distribution
    agent_wealth = [behavior['final_wealth'] for behavior in agent_behaviors.values()]
    initial_wealth = 1000  # From config
    
    print("\n=== Agent Analysis ===")
    print(f"Average final wealth: ${pd.Series(agent_wealth).mean():.2f}")
    print(f"Wealth std deviation: ${pd.Series(agent_wealth).std():.2f}")
    print(f"Wealth range: ${min(agent_wealth):.2f} - ${max(agent_wealth):.2f}")
    
    # Create plots
    plt.style.use('default')
    
    # Plot 1: Economic indicators over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(results_df['time_step'], results_df['market_price'], 'b-', linewidth=2)
    ax1.set_title('Market Price Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Market Price ($)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results_df['time_step'], results_df['market_volume'], 'g-', linewidth=2)
    ax2.set_title('Trading Volume Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Trading Volume ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/economic_indicators.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Agent wealth distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agent_wealth, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(initial_wealth, color='red', linestyle='--', linewidth=2, 
                label=f'Initial Wealth (${initial_wealth})')
    plt.axvline(pd.Series(agent_wealth).mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean Final Wealth (${pd.Series(agent_wealth).mean():.0f})')
    plt.title('Distribution of Final Agent Wealth')
    plt.xlabel('Final Wealth ($)')
    plt.ylabel('Number of Agents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/plots/agent_wealth_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary analysis
    analysis_summary = {
        'market_performance': {
            'initial_price': float(results_df['market_price'].iloc[0]),
            'final_price': float(results_df['market_price'].iloc[-1]),
            'price_change_percent': float(((results_df['market_price'].iloc[-1] / results_df['market_price'].iloc[0]) - 1) * 100),
            'average_volume': float(results_df['market_volume'].mean())
        },
        'agent_performance': {
            'initial_wealth': initial_wealth,
            'average_final_wealth': float(pd.Series(agent_wealth).mean()),
            'wealth_std_dev': float(pd.Series(agent_wealth).std()),
            'min_wealth': float(min(agent_wealth)),
            'max_wealth': float(max(agent_wealth))
        }
    }
    
    with open('results/analysis/summary.json', 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"\nAnalysis completed. Results saved to results/analysis/ and results/plots/")


if __name__ == "__main__":
    analyze_results()
