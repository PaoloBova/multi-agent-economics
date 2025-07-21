# Structured Note Lemons - Implementation Guide

## Overview

This implementation follows your task and tool playbook to create a sophisticated multi-agent economics simulation. The flagship scenario simulates a financial market where internal quality drives adverse selection in structured note trading.

## Architecture

### Core Infrastructure (`multi_agent_economics/core/`)

1. **Artifacts System** (`artifacts.py`)
   - Structured resources shared between agents
   - Workspace-based access control
   - Private, shared, and organization buckets

2. **Tools System** (`tools.py`)
   - Credit-priced functions with quality tiers
   - Built-in finance tools (sector_forecast, price_note, monte_carlo_var)
   - Extensible registry system

3. **Actions Tracking** (`actions.py`)
   - Internal actions (tool calls, reasoning)
   - External actions (market trades, price posts)
   - Complete audit trail

4. **Budget Management** (`budget.py`)
   - Credit allocation and tracking
   - Transaction history
   - Spending pattern analysis

5. **Quality System** (`quality.py`)
   - Quality production functions
   - Tool usage → quality mapping
   - Adverse selection mechanics

### Agent Framework (`multi_agent_economics/agents/`)

- **EconomicAgent**: Enhanced AutoGen agent with workspace access
- **Tool Integration**: Automatic credit management and artifact creation
- **Role-based**: Specialized agents (Analyst, Structurer, PM, Risk-Officer, Trader)

### Flagship Scenario (`multi_agent_economics/scenarios/`)

- **Structured Note Lemons**: Finance scenario with quality-driven adverse selection
- **Graph-based Topology**: Uses AutoGen GraphGroupChat for sophisticated interactions
- **Multi-round Simulation**: Tracks quality patterns and market efficiency

## Key Features

### 1. Quality Production Function
```
High Quality = Tool Spend ≥ Threshold
- sector_forecast (tier=high): 3 credits
- price_note (tier=high): 4 credits  
- Total for high quality: ≥6 credits
```

### 2. Collaboration Mechanics
- Artifacts stored externally to agent messages
- Share costs time but not credits
- Visibility controls via workspace buckets

### 3. Market Dynamics
- Sequential round-robin trading
- Quality hidden from buyers ex-ante
- Payoff realization drives learning

## Getting Started

### 1. Installation
```bash
# Install AutoGen (optional for testing)
pip install autogen-agentchat autogen-ext[openai]

# Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Test Infrastructure
```bash
python test_infrastructure.py
```

### 3. Run DVC Pipeline
```bash
# Prepare scenario
dvc repro prepare_scenario

# Run simulation  
dvc repro run_simulation

# Analyze results
dvc repro analyze_results
```

### 4. Simple Example
```bash
python examples/simple_lemons_example.py
```

## Configuration

### Simulation Config (`simulation_config.yaml`)
```yaml
scenario:
  type: "structured_note_lemons"
  
agents:
  seller_banks: ["SellerBank1", "SellerBank2"]
  buyer_funds: ["BuyerFund1", "BuyerFund2"]
  initial_budgets:
    seller_bank: 20
    buyer_fund: 10

simulation:
  rounds: 10
  
quality:
  thresholds:
    structured_note:
      high_quality: 6.0
      medium_quality: 3.0
```

### Tools Config (`configs/tools.json`)
```json
{
  "tools": [
    {
      "id": "sector_forecast",
      "cost": 3,
      "precision_tiers": {
        "high": {"noise_factor": 0.1},
        "med": {"noise_factor": 0.3},
        "low": {"noise_factor": 0.6}
      }
    }
  ]
}
```

## Output Structure

```
data/processed/
├── simulation_results.json     # Main results
├── action_logs/               # Detailed action tracking
└── artifacts/                 # Agent workspaces

results/
├── simulation_output.csv      # Trade data
├── analysis/                  # Quality metrics
└── plots/                     # Visualizations
```

## Key Metrics

### Quality Metrics
- `tool_spend_high_vs_low`: Separates quality levels
- `high_q_trade_share`: Market efficiency indicator
- `cost_quality_correlation`: Quality investment effectiveness

### Collaboration Metrics  
- `artifact_share_latency`: Team coordination efficiency
- `reflect_calls / tool_calls`: Metacognition intensity
- `budget_utilization`: Resource allocation patterns

### Economic Metrics
- `profit_variance`: Lemons market collapse indicator
- `avg_price_by_quality`: Price discrimination
- `trade_frequency`: Market activity

## Research Applications

### 1. Adverse Selection
- Compare high vs low quality sellers
- Track price discrimination over time
- Measure market unraveling

### 2. Coordination Patterns
- Analyze artifact sharing networks
- Measure team efficiency vs individual work
- Study delegation patterns

### 3. LLM Economic Behavior
- Tool choice under budget constraints
- Quality vs cost trade-offs
- Strategic reasoning patterns

## Extension Points

### 1. New Domains
- Add healthcare, insurance, infrastructure scenarios
- Create domain-specific tool registries
- Implement sector-specific quality functions

### 2. Advanced Interactions
- Add regulatory agents
- Implement M&A dynamics
- Create dynamic topology changes

### 3. Behavioral Models
- Add learning mechanisms
- Implement behavioral biases
- Create adaptive strategies

## MockMode (Without AutoGen)

The system includes mock implementations for testing without AutoGen:
- Simulated agent interactions
- Dummy tool execution
- Sample output generation

This allows development and testing of the core infrastructure independently.

---

This implementation provides a complete framework for studying economic interactions with LLM agents, focusing on quality production, adverse selection, and multi-agent coordination patterns.
