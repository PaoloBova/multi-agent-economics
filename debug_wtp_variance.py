"""Debug script to understand WTP variance in buyer preference analysis."""

import numpy as np
from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, BuyerState, RegimeParameters
)
from multi_agent_economics.tools.implementations.economic import analyze_buyer_preferences_impl

# Create buyers with varied preferences
buyers = []
for i in range(30):
    methodology_weight = 0.3 + 0.4 * (i / 29)  # Range from 0.3 to 0.7
    coverage_weight = 1.0 - methodology_weight
    
    buyer = BuyerState(
        buyer_id=f"buyer_{i+1}",
        regime_beliefs={"tech": [0.6, 0.4]},
        budget=100.0,
        conversion_functions={
            "methodology": {"categorical": {"basic": 0.3, "advanced": 0.7, "cutting_edge": 1.0}},
            "coverage": {"numeric": {"scale_factor": 1.0}}
        },
        attr_weights={"tech": [methodology_weight, coverage_weight]}
    )
    buyer.ensure_sector_exists("tech", 2)  # Ensure sector is properly initialized
    buyers.append(buyer)

state = MarketState(
    offers=[],
    trades=[],
    index_values={"tech": 100.0},
    buyers_state=buyers,
    current_regimes={"tech": 0},
    regime_parameters={"tech": {0: RegimeParameters(mu=0.05, sigma=0.15)}},
    attribute_order=["methodology", "coverage"],
    sector_order=["tech"],
    marketing_attribute_definitions={
        "methodology": {"type": "qualitative", "values": ["basic", "advanced", "cutting_edge"]},
        "coverage": {"type": "numeric", "range": [0.0, 1.0]}
    }
)

market_model = MarketModel(id=1, name="Test", agents=[], state=state, step=lambda: None, collect_stats=lambda: {})

config_data = {
    "tool_parameters": {
        "analyze_buyer_preferences": {
            "effort_thresholds": {"high": 3.0, "medium": 1.5},
            "high_effort_num_buyers": 30,
            "high_effort_num_test_offers": 12,
            "high_effort_analyze_by_attribute": True
        }
    }
}

np.random.seed(42)
result = analyze_buyer_preferences_impl(market_model, config_data, "tech", effort=4.0)

print(f"Analysis method: {result.analysis_method}")
print(f"Total observations: {result.total_observations}")
print(f"Sample size: {result.sample_size}")
print(f"Quality tier: {result.quality_tier}")
print(f"Warnings: {result.warnings}")

# Check WTP variance
if result.raw_wtp_data:
    wtp_values = [point.willingness_to_pay for point in result.raw_wtp_data]
    wtp_variance = np.var(wtp_values)
    print(f"WTP variance: {wtp_variance}")
    print(f"WTP min: {min(wtp_values)}")
    print(f"WTP max: {max(wtp_values)}")
    print(f"WTP range: {max(wtp_values) - min(wtp_values)}")
    
    # Check if variance meets threshold
    print(f"Variance > 1e-10: {wtp_variance > 1e-10}")