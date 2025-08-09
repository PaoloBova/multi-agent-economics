#!/usr/bin/env python3
"""
Script to generate regime history data for test fixtures.
Run once to generate reproducible test data.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_agent_economics.models.market_for_finance import (
    generate_regime_history, RegimeParameters
)

# Set random seed for reproducibility
np.random.seed(42)

# Define regime parameters in dictionary format for the function
regime_parameters = {
    "tech": {
        0: {"mu": 0.12, "sigma": 0.18},  # High growth, high volatility
        1: {"mu": 0.04, "sigma": 0.25}   # Low growth, higher volatility
    },
    "finance": {
        0: {"mu": 0.08, "sigma": 0.15},  # Moderate growth, moderate volatility
        1: {"mu": 0.02, "sigma": 0.22}   # Low growth, higher volatility
    },
    "healthcare": {
        0: {"mu": 0.10, "sigma": 0.20},  # Steady growth, moderate volatility
        1: {"mu": 0.05, "sigma": 0.28}   # Low growth, high volatility
    }
}

# Define transition matrices
transition_matrices = {
    "tech": np.array([
        [0.75, 0.25],  # From regime 0: 75% stay in 0, 25% switch to 1
        [0.30, 0.70]   # From regime 1: 30% switch to 0, 70% stay in 1
    ]),
    "finance": np.array([
        [0.80, 0.20],  # More persistent than tech
        [0.25, 0.75]
    ]),
    "healthcare": np.array([
        [0.85, 0.15],  # Most persistent sector
        [0.20, 0.80]
    ])
}

# Initial regimes
initial_regimes = {"tech": 0, "finance": 1, "healthcare": 0}

# Generate 20 periods of history
history = generate_regime_history(
    regimes=initial_regimes,
    regime_params=regime_parameters,
    transition_matrices=transition_matrices,
    history_length=20
)

print("# Generated regime history data for test fixtures")
print("# Copy this into the test file as a static fixture")
print()
print("regime_history_data = [")
for i, period_data in enumerate(history):
    print(f"    # Period {i}")
    print(f"    PeriodData(")
    print(f"        period={period_data['period']},")
    print(f"        returns={period_data['returns']},")
    print(f"        regimes={period_data['regimes']},")
    print(f"        index_values={period_data['index_values']}")
    print(f"    ),")
print("]")