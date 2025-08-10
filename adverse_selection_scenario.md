# Adverse Selection Crisis Scenario Model

## Executive Summary

This scenario models how technological disruption in forecasting markets can trigger a gradual transition from high-quality equilibrium to adverse selection crisis. The key insight is that corner-cutting (quality reduction while maintaining marketing claims) can become individually rational while being collectively destructive, leading to market collapse during crisis periods.

## Initial Conditions & Technology Shock

### Phase 0: High-Quality Equilibrium Setup
- **Seller Track Records**: All sellers start with 85%+ accuracy ratings
- **Market Structure**: Established sellers with high forecasting costs, limited entry
- **Buyer Preferences**: Strong willingness-to-pay premiums (2-3x) for quality marketing attributes
- **Forecasting Environment**: Moderate difficulty, standard regime transition patterns

### Technology Disruption Trigger
- **Cost Reduction**: New ML tools reduce forecasting costs by ~50%
- **Easy Wins Period**: Current regimes show clear transition signals, making basic forecasts more accurate
- **Market Entry**: Lower barriers attract new sellers
- **Demand Growth**: Increased buyer interest due to perceived quality improvements

## Detailed Phase Evolution

### Phase 1: Technology Disruption (Periods 1-5)
**Key Dynamics:**
- Some sellers (particularly high-cost, low-productivity) discover corner-cutting opportunities
- Corner-cutting involves reducing effort while maintaining marketing attribute claims
- Easy forecasting period masks quality differences initially
- New seller entry increases competitive pressure

**Seller Decision Process:**
```
Corner-cutting becomes viable when:
marginal_cost_savings > expected_reputation_penalty × discount_factor
```

**Implementation Notes:**
- Model forecasting costs as productivity parameters: `cost = effort / productivity`
- "Easy wins" implemented as temporarily higher base forecast accuracy
- Track record updates remain lagged by 2-3 periods

### Phase 2: Gradual Corner-Cutting Spread (Periods 6-15)
**Feedback Mechanisms:**
1. **Profit Advantage**: Corner-cutters maintain revenues while reducing costs
2. **Competitive Pressure**: Increased supply erodes quality premiums by 10-20%
3. **Detection Lag**: Reputation system delays mean short-term corner-cutting pays off
4. **Social Learning**: Sellers observe others' success with corner-cutting

**Buyer Experience:**
- Begin receiving forecasts that underperform expectations
- Individual disappointments not yet systematic enough to trigger belief updates
- Accumulated portfolio positions based on corner-cut advice building up

**Critical Implementation:**
- Track accumulated sector positions influenced by each seller's historical advice
- Weight attribution by recency: `attribution_weight = exp(-decay_rate × periods_ago)`

### Phase 3: Adverse Selection Acceleration (Periods 16-25)
**Buyer Learning Process:**
- Bayesian updating of marketing attribute preferences based on economic impacts
- `attr_mu[sector][attribute]` decreases as "premium" forecasts disappoint
- Willingness-to-pay for quality claims falls 30-50%

**Market Response:**
- Lower quality premiums make honest production less viable
- Honest sellers face exit-or-cheat decision
- Market share shifts toward corner-cutters
- Average market quality deteriorates

**Price Competition Dynamics:**
```python
# Sellers estimate demand elasticity from historical data
elasticity = estimate_price_elasticity(price_history, quantity_history)
optimal_price = marginal_cost / (1 - 1/elasticity)  # Standard monopolistic competition
```

### Phase 4: Crisis Trigger (Periods 26-27)
**Crisis Event**: Rare regime transition causes large negative returns (e.g., tech sector crash)

**Crisis Mechanism:**
1. **Accumulated Position Effects**: Buyers suffer major losses on positions built up over multiple periods
2. **Quality Gap Exposure**: Corner-cut forecasts perform especially poorly in difficult conditions
3. **Attribution Process**: Losses attributed to accumulated advice, not just marginal purchases
4. **Mass Belief Updating**: Systematic revision of all marketing attribute preferences

**Portfolio Loss Attribution:**
```python
def attribute_crisis_losses(buyer_positions, historical_advice, sector_crash_returns):
    total_loss = buyer_positions[sector] * crash_returns[sector]
    # Attribute loss to advice sources based on historical influence
    for seller_id, advice_weight in historical_advice[sector].items():
        attributed_loss[seller_id] = total_loss * advice_weight
    return attributed_loss
```

### Phase 5: Market Collapse/Recovery (Periods 28+)
**Immediate Collapse:**
- Track record updates finally hit corner-cutters (with lag)
- Quality premiums collapse across all marketing attributes
- Mass seller exit or desperate strategy switching

**Recovery Scenarios:**
1. **Market Consolidation**: Few quality producers survive, rebuild reputation slowly
2. **Regulatory Response**: External verification mechanisms introduced
3. **Persistent Damage**: Market remains stuck in low-quality equilibrium

## Key Economic Feedback Loops

### Primary Feedback Dynamics

```
Technology Shock
      ↓
Corner-Cutting Becomes Profitable
      ↓
Competitive Pressure Increases → Quality Premiums Fall
      ↓                              ↓
More Corner-Cutting Incentive ← Honest Sellers Exit/Switch
      ↓
Average Quality Deteriorates
      ↓
Buyer Disappointment Increases
      ↓
Preference Updates → Lower Willingness-to-Pay
      ↓
[REINFORCING SPIRAL]
      ↓
Crisis Period Hits
      ↓
Quality Gap Exposed → Mass Belief Revision
      ↓
Market Collapse
```

### Secondary Feedback Loops

1. **Entry/Exit Dynamics**: Lower quality → lower consumer surplus → some buyers exit → remaining buyers more price-sensitive → further pressure on quality premiums

2. **Reputation Lag Amplification**: Delayed track record updates allow corner-cutting to spread before reputation consequences hit

3. **Accumulation Effects**: Portfolio positions build up based on biased advice → larger crisis impact → stronger preference revision

## Implementation Requirements

### Core Mechanism Extensions

1. **Effort-to-Quality Function**:
```python
def effort_to_quality(effort, productivity, base_quality=0.55, max_gain=0.35):
    return base_quality + max_gain * (1 - np.exp(-2 * productivity * effort))
```

2. **Historical Position Tracking**:
```python
# Track influence of each advice source on buyer positions
buyer_position_attribution[buyer_id][sector][seller_id] = influence_weight
```

3. **Dynamic Cost Structure**:
```python
# Technology shock reduces costs over time
forecasting_cost[period] = initial_cost * cost_reduction_factor[period]
```

4. **Price Elasticity Estimation**:
```python
def estimate_elasticity(seller_history):
    # Estimate from seller's own price/quantity history
    return -(delta_quantity/delta_price) * (mean_price/mean_quantity)
```

### Calibration Parameters

| Parameter | Initial Value | Crisis Value | Notes |
|-----------|---------------|--------------|-------|
| Base Forecast Accuracy | 55% | 55% | With zero effort |
| Max Achievable Accuracy | 90% | 90% | With max effort & productivity |
| Quality Premium | 2.5x | 0.8x | Buyer willingness-to-pay multiplier |
| Track Record Lag | 3 periods | 3 periods | Reputation update delay |
| Cost Reduction | 0% | 50% | From technology shock |
| Crisis Probability | 0% | 100% | Rare regime transition |

## Areas for Future Development

### High Priority
1. **Sector Correlation Effects**: How do crises in one sector affect belief updating in others?
2. **Buyer Heterogeneity**: Different buyer types with varying sophistication levels
3. **Regulatory Response**: Modeling potential policy interventions

### Medium Priority  
1. **Dynamic Entry/Exit**: More sophisticated seller entry/exit decisions
2. **Learning Speed Variation**: Different buyers learn at different rates
3. **Network Effects**: Reputation spillovers between related sellers

### Technical Challenges
1. **Computational Efficiency**: Tracking full attribution history may be expensive
2. **Parameter Sensitivity**: Model behavior may be sensitive to calibration
3. **Equilibrium Multiplicity**: Multiple possible outcomes depending on initial conditions

## Expected Outcomes

### Successful Demonstration Should Show:
1. **Gradual Transition**: Smooth evolution from high-quality to low-quality equilibrium
2. **Individual Rationality**: Each seller's corner-cutting decision makes sense given their information
3. **Collective Irrationality**: Market outcome is worse for everyone than coordinated quality maintenance
4. **Crisis Amplification**: Difficult periods expose accumulated problems and trigger rapid adjustment
5. **Path Dependence**: Market may not recover to original high-quality state

### Policy Implications:
- Information asymmetries can persist even with reputation systems
- Technology shocks can destabilize quality equilibria
- Crisis periods serve as "truth-telling" moments that reveal accumulated problems
- Early intervention may be more effective than post-crisis regulation