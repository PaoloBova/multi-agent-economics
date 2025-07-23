# Introduction for Collaborators: Multi-Agent LLM Economics Framework

**A comprehensive guide for collaborators to understand and extend the multi-agent llm economics simulation platform**

---

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Architecture Overview](#architecture-overview)
3. [Core Systems Deep Dive](#core-systems-deep-dive)
4. [Scenario Development Guide](#scenario-development-guide)
5. [Extension Points](#extension-points)
6. [Advanced Features](#advanced-features)
7. [Implementation Examples](#implementation-examples)

---

## Theoretical Foundation

### Economic Theory Implementation

This project implements a framework for running economic simulations with agents powered by LLMs. The agents typically act within organizations that produce and exchange goods and services. Once agents have made their choices for a given round, we run our simulation backend to update our economy.

We draw upon microeconomic theory to simulate our economy. Agents typically choose to allocate resources to different tasks under a resource constraint (that is typically set by the organisation that employs them). These allocations are treated as inputs to some production function. A production function may cover not just the production of consumption or capital goods, but also product quality, or effort to produce knowledge. Agents may then exchange their goods or services on the market, typically by posting a price and labelling their product. Interested consumers may then purchase the product based on their willingness to pay, what they believe the product is worth, and the range of alternative offers available. We also allow agents to form contracts that lead to exchange happening in the future contingent on success (e.g. take on orders at a prespecified price).

Our motive in providing such a simulation is to separate assessment of domain-specific capabilities from the metacognitive and strategic reasoning capabilities that are required for agents to perform well in an economic setting, such as reflection on which approach to follow and collaboration (see Zhang et al. 2025).

Our framework is certainly more involved than simply getting the llms to roleplay a textbook economic model. However, we argue that only when embedded in a sufficiently rich environment with a sufficiently nuanced task will agents have the need to engage the relevant reasoning capabilities. In short, textbook models clue in llm agents too much to the tradeoffs they must make, and grant an unearned awareness of the economic situation they are in; textbook cues will not usually be available in the real economy.

We implement a number of mechanics to help simulate the cognitive demands of the market:

#### 1. **Credit-Based Resource Constraints**
- Every action (tool usage, analysis, computation) costs credits from a finite budget
- Unused credits are usually returned to the organisation
- Forces agents to make strategic trade-offs between depth of analysis and cost
- Creates realistic scarcity that drives economic decision-making

#### 2. **Organizational Coordination Costs**
- Agents within organizations may share artifacts to coordinate
- Sharing incurs time costs (latency) but not credit costs
- Creates tension between coordination benefits and efficiency

#### 3. **Multi-Organization Competition**
- Different organizations compete in the same market
- Each has internal teams with specialized roles
- Strategic interactions emerge from budget constraints and information asymmetries

#### 4. **Imperfect information**
- Agents do not always observe most variables in the market. For example, agents typically have to learn a mapping from expenditure on a task to the quality of the output. They also typically have no credibly way to show that the quality of their output is high.
- Quality may be hidden from buyers (information asymmetry)
- Adverse selection and moral hazard are allowed to occur.

### Supported Economic Scenarios

The framework is designed to model various economic sectors:

- **Finance**: Structured products, derivatives pricing, investment decisions
- **Healthcare**: Treatment bundling, quality vs. cost trade-offs
- **Insurance**: Risk assessment, contract design, claims processing  
- **Infrastructure**: Capacity planning, maintenance scheduling, demand forecasting

---

## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT LLM ECONOMICS                    │
├─────────────────────────────────────────────────────────────────┤
│  External Data Layer (JSON/Markdown configs)                    │
│  ├── Tool Definitions          ├── Quality Thresholds           │
│  ├── Market Data              ├── Agent Prompts                 │
│  └── Interaction Topologies   └── Role Definitions              │
├─────────────────────────────────────────────────────────────────┤
│  Core Infrastructure                                            │
│  ├── Artifact Management      ├── Budget Management             │
│  ├── Tool Registry           ├── Quality Functions              │
│  ├── Action Logging                                             │
├─────────────────────────────────────────────────────────────────┤
│  Agent Framework (AutoGen 0.6.4 + Extensions)                   │
│  ├── EconomicAgent           ├── WorkspaceMemory                │
│  ├── Role Specialization     └── Tool Integration               │
├─────────────────────────────────────────────────────────────────┤
│  Scenario Layer                                                 │
│  ├── DiGraphBuilder Topology ├── Market Dynamics                │
│  ├── Multi-Round Simulation  └── DVC Pipeline for Analysis      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Data/Code Separation**: All configurations externalized to JSON/Markdown files
2. **Backward Compatibility**: Fallback mechanisms for missing configurations
3. **Extensibility**: Plugin architecture for tools, scenarios, and topologies
4. **AutoGen Integration**: Leverage latest version of AutoGen
5. **Economic Realism**: Every action has costs, creating authentic resource constraints

---

## Core Systems Deep Dive

### 1. Artifact Management System

**Purpose**: Enables collaboration with tangible time costs

```python
# Workspace structure per organization
workspace/
├── private/     # Agent's personal artifacts
├── shared/      # Artifacts shared with agent  
└── org/         # Organization-wide artifacts
```

**Key Features**:
- **Access Control Lists (ACLs)** on artifacts
- **Bucket-based organization** (private/shared/org)
- **Version tracking** and caching
- **Shareable handles** across agent boundaries

**Implementation**: `multi_agent_economics.core.artifacts`

### 2. Tool Registry & Execution

**Purpose**: Credit-based tools with quality tiers

```json
{
  "id": "sector_forecast",
  "cost": 3,
  "inputs": ["sector", "horizon"],
  "outputs": ["forecast_report"],
  "precision_tiers": {
    "high": {"noise_factor": 0.1},
    "med": {"noise_factor": 0.3}, 
    "low": {"noise_factor": 0.6}
  }
}
```

**Key Features**:
- **Precision tiers**: Higher cost = lower error variance
- **External configuration**: Tools defined in JSON files
- **Market data integration**: Tools access real economic data
- **Automatic logging**: All tool usage tracked for analysis

**Implementation**: `multi_agent_economics.core.tools`

### 3. Quality Functions

**Purpose**: Maps tool spending to output quality

```python
# Quality calculation with weighted tool contributions
quality_score = sum(
    tool_usage[tool] * weights[tool] 
    for tool in quality_tools
)

quality_level = (
    "high" if score >= high_threshold else
    "medium" if score >= medium_threshold else
    "low"
)
```

**Key Features**:
- **Weighted scoring**: Different tools contribute differently to quality
- **Product-specific thresholds**: Configurable per market/product type
- **Hidden from buyers**: Information asymmetry enforcement
- **External configuration**: Thresholds defined in JSON

**Implementation**: `multi_agent_economics.core.quality`

### 4. Budget Management

**Purpose**: Credit-based resource allocation

**Key Features**:
- **Organization-level budgets** with per-agent tracking
- **Debit/credit operations** for tool usage
- **Transaction logging** for economic analysis
- **Budget exhaustion handling**

**Implementation**: `multi_agent_economics.core.budget`

### 5. Action Logging

**Purpose**: Comprehensive tracking of agent decisions

**Two Action Types**:
- **Internal Actions**: Tool calls, planning, reflection (logged privately)
- **External Actions**: Market-facing decisions (posted prices, trades)

**Implementation**: `multi_agent_economics.core.actions`

### 6. Workspace Memory

**Purpose**: Intelligent artifact loading for AutoGen agents

**Key Features**:
- **On-demand payload injection**: Only load artifacts when needed
- **TTL caching**: Prevents redundant artifact loading
- **Version awareness**: Tracks artifact updates across rounds
- **Context size management**: Respects token limits

**Implementation**: `multi_agent_economics.core.workspace_memory`

---

## Scenario Development Guide

### Creating a New Scenario

To implement a new economic scenario (e.g., infrastructure markets, healthcare systems), follow this pattern:

#### 1. **Define the Economic Structure**

```python
class YourScenario:
    def __init__(self, workspace_dir: Path, config: Dict[str, Any]):
        # Initialize core systems
        self.artifact_manager = ArtifactManager(workspace_dir / "artifacts")
        self.tool_registry = ToolRegistry(config_path=your_tools_config)
        self.budget_manager = BudgetManager()
        # ... etc
```

#### 2. **Create Agent Organizations**

```python
async def setup_agents(self):
    # Create organizations with specialized roles
    for provider_name in self.config["providers"]:
        provider_agents = {}
        for role in ["analyst", "engineer", "manager"]:
            agent = create_agent(
                name=f"{provider_name}_{role}",
                role=role,
                organization=provider_name,
                # ... core managers
            )
            provider_agents[role] = agent
        self.agents[provider_name] = provider_agents
```

#### 3. **Design Interaction Topology**

```python
def create_interaction_topology(self):
    builder = DiGraphBuilder()
    
    # Add agents as nodes
    all_agents = [agent for org in self.agents.values() 
                  for agent in org.values()]
    for agent in all_agents:
        builder.add_node(agent)
    
    # Define interaction patterns
    # Intra-org: full mesh within teams
    # Inter-org: role-based connections (engineers ↔ engineers)
    
    return builder.build()
```

#### 4. **Implement Market Dynamics**

```python
async def run_round(self, round_number: int):
    # Update market state
    market_message = self._create_market_message()
    
    # Create GraphFlow team
    team = GraphFlow(
        participants=all_agents,
        termination_condition=MaxMessageTermination(max_messages=20)
    )
    
    # Run conversation
    async for message in team.run_stream(task=market_message):
        # Process messages, extract external actions
        pass
    
    # Analyze results
    return self._process_round_results()
```

### Example: Infrastructure Capacity Market

**Participants**:
- **Grid Operators**: `demand_forecaster`, `capacity_planner`, `operations_manager`
- **Generators**: `maintenance_scheduler`, `bid_optimizer`, `risk_assessor`

**Tools**:
- `forecast_demand(region, horizon)` - Cost: 4 credits
- `schedule_maintenance(asset, timeframe)` - Cost: 3 credits  
- `simulate_outage(scenario)` - Cost: 5 credits
- `optimize_capacity_bid(forecast, costs)` - Cost: 3 credits

**External Actions**:
- `commit_capacity(amount, price, timeframe)`
- `request_capacity(demand_curve, region)`

**Quality Mapping**:
- High-quality capacity commitments require `forecast_demand` + `simulate_outage`
- Low-quality commitments skip expensive simulation
- Buyers see reliability failures only ex-post

---

## Extension Points

### 1. **Tool Development**

Add new domain-specific tools by:

**A. Define Tool in JSON**:
```json
{
  "id": "medical_diagnosis",
  "cost": 6,
  "inputs": ["patient_data", "symptom_history"],
  "outputs": ["diagnosis_report"],
  "precision_tiers": {
    "specialist": {"accuracy": 0.95},
    "general": {"accuracy": 0.85},
    "basic": {"accuracy": 0.70}
  }
}
```

**B. Implement Tool Function**:
```python
def medical_diagnosis_tool(patient_data: dict, symptom_history: list, 
                          precision: str = "general") -> Artifact:
    # Your domain logic here
    accuracy = tool_config["precision_tiers"][precision]["accuracy"]
    diagnosis = your_diagnostic_algorithm(patient_data, accuracy)
    
    return Artifact(
        id=f"diagnosis_{uuid.uuid4()}",
        type="medical_report",
        payload={"diagnosis": diagnosis, "confidence": accuracy},
        created_by="medical_diagnosis"
    )
```

**C. Register in ToolRegistry**:
```python
tool_registry.register_tool("medical_diagnosis", medical_diagnosis_tool)
```

### 2. **Agent Roles & Specialization**

Create specialized agent types:

```python
class SpecialistAgent(EconomicAgent):
    def __init__(self, specialty: str, **kwargs):
        self.specialty = specialty
        super().__init__(**kwargs)
    
    def _build_system_message(self) -> str:
        base_message = super()._build_system_message()
        specialist_guidance = self._load_specialist_knowledge()
        return base_message + "\n\n" + specialist_guidance
```

### 3. **Market Mechanisms**

Implement new market structures:

- **Auction Systems**: Dutch auctions, sealed-bid auctions
- **Matching Markets**: Two-sided markets with preferences
- **Network Markets**: Platform dynamics with network effects
- **Dynamic Pricing**: Real-time price discovery mechanisms

### 4. **Advanced Topologies**

Create complex interaction patterns:

```python
class NetworkTopologyBuilder:
    def create_small_world_network(self, agents: List[Agent], 
                                  p_rewire: float = 0.3):
        # Small-world network for information diffusion studies
        pass
    
    def create_scale_free_network(self, agents: List[Agent], 
                                 preferential_attachment: bool = True):
        # Scale-free networks for hub-based coordination
        pass
```

---

## Advanced Features

### 1. **Multi-Round Learning**

Agents can learn and adapt across rounds:

```python
class LearningAgent(EconomicAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory = PersistentMemory()
        self.strategy_tracker = StrategyTracker()
    
    def update_strategy(self, round_results: Dict[str, Any]):
        # Update internal models based on observed outcomes
        success_rate = self.analyze_past_performance(round_results)
        self.strategy_tracker.update_weights(success_rate)
```

### 2. **Information Networks**

Model information flow and asymmetries:

```python
class InformationNetwork:
    def __init__(self, agents: List[Agent], topology: nx.Graph):
        self.agents = agents
        self.topology = topology
    
    def propagate_information(self, source: Agent, 
                            information: Artifact, 
                            decay_rate: float = 0.1):
        # Model information spread with noise/decay
        pass
```

### 3. **Regulatory Environments**

Add regulatory constraints and compliance costs:

```python
class RegulatoryFramework:
    def __init__(self, rules: List[Rule]):
        self.rules = rules
    
    def check_compliance(self, action: ExternalAction) -> ComplianceResult:
        # Verify action against regulatory rules
        violations = []
        for rule in self.rules:
            if not rule.check(action):
                violations.append(rule)
        return ComplianceResult(violations)
```

### 4. **Risk Management**

Sophisticated risk modeling:

```python
class RiskManager:
    def calculate_portfolio_risk(self, positions: List[Position], 
                               market_data: MarketData) -> RiskMetrics:
        # VaR, Expected Shortfall, stress testing
        pass
    
    def simulate_market_shocks(self, scenarios: List[Scenario]) -> Results:
        # Monte Carlo simulation of extreme events
        pass
```

---

## Implementation Examples

### Example 1: Healthcare Treatment Markets

**Scenario**: Hospitals compete to provide treatment bundles

**Agents**:
- **Hospitals**: `diagnostician`, `treatment_planner`, `resource_manager`
- **Patients/Insurers**: `care_coordinator`, `cost_analyst`

**Tools**:
- `patient_assessment(symptoms, history)` - Cost: 4
- `treatment_planning(assessment, guidelines)` - Cost: 6  
- `resource_allocation(treatment_plan, capacity)` - Cost: 3
- `outcome_prediction(treatment, patient_profile)` - Cost: 5

**Quality Function**:
```json
{
  "treatment_bundle": {
    "high_quality_threshold": 10.0,
    "quality_tools": ["patient_assessment", "treatment_planning", "outcome_prediction"],
    "weights": {
      "patient_assessment": 0.3,
      "treatment_planning": 0.5,
      "outcome_prediction": 0.2
    }
  }
}
```

**Market Dynamics**:
- Hospitals post treatment bundles with price and claimed quality
- Patients/insurers choose based on price and reputation
- Actual outcomes revealed post-treatment
- Quality shortcuts lead to adverse outcomes and reputation damage

### Example 2: Insurance Contract Design

**Scenario**: Insurance companies design and price policies

**Agents**:
- **Insurers**: `actuary`, `underwriter`, `product_manager`
- **Customers**: `risk_assessor`, `policy_shopper`

**Tools**:
- `risk_modeling(customer_profile, historical_data)` - Cost: 5
- `contract_design(risk_model, competitive_analysis)` - Cost: 4
- `pricing_optimization(contract, market_conditions)` - Cost: 3
- `regulatory_compliance(contract, regulations)` - Cost: 2

**Adverse Selection**:
- High-risk customers attracted to under-priced policies
- Insurers who skip expensive risk modeling face selection problems
- Market unraveling as careful insurers exit

### Example 3: Infrastructure Investment

**Scenario**: Power companies invest in generation capacity

**Agents**:
- **Utilities**: `demand_analyst`, `technology_assessor`, `financial_planner`
- **Regulators**: `market_monitor`, `reliability_coordinator`

**Tools**:
- `demand_forecasting(region, scenario, horizon)` - Cost: 6
- `technology_assessment(options, performance_data)` - Cost: 5
- `financial_modeling(investment, revenue_projections)` - Cost: 4
- `reliability_analysis(capacity_mix, demand_uncertainty)` - Cost: 7

**Strategic Considerations**:
- Long-term capacity decisions under uncertainty
- Coordination between competing utilities
- Regulatory approval processes
- Stranded asset risks from technology changes

---

## Getting Started: Quick Setup

### 1. **Environment Setup**
```bash
# Clone repository
git clone <repo-url>
cd multi-agent-economics

# Setup Python environment
mamba create -n multi-agent-economics python=3.12
mamba activate multi-agent-economics

# Install dependencies
poetry install

# Verify setup
python test_infrastructure.py
```

### 2. **Create Your First Scenario**
```bash
# Copy flagship scenario as template
cp -r multi_agent_economics/scenarios/structured_note_lemons.py \
      multi_agent_economics/scenarios/your_scenario.py

# Modify configurations
# - data/config/your_tools.json
# - data/config/your_quality_thresholds.json
# - data/prompts/your_role_definitions.json

# Test scenario
python scripts/run_simulation.py --scenario your_scenario
```

### 3. **Configuration Files to Modify**

**Tool Definitions** (`data/config/enhanced_tools.json`):
```json
{
  "tools": [
    {
      "id": "your_tool",
      "cost": 3,
      "inputs": ["input1", "input2"],
      "outputs": ["output1"],
      "precision_tiers": {"high": {}, "med": {}, "low": {}}
    }
  ]
}
```

**Quality Thresholds** (`data/config/quality_thresholds.json`):
```json
{
  "product_types": {
    "your_product": {
      "high_quality_threshold": 8.0,
      "quality_tools": ["tool1", "tool2"],
      "weights": {"tool1": 0.6, "tool2": 0.4}
    }
  }
}
```

**Agent Roles** (`data/prompts/role_definitions.json`):
```json
{
  "roles": {
    "your_role": {
      "description": "Role responsibilities...",
      "tools_focus": ["tool1", "tool2"],
      "coordination_style": "collaborative"
    }
  }
}
```

---

## Conclusion

This framework provides a sophisticated foundation for studying multi-agent economics with realistic constraints and incentives. The combination of:

- **Credit-based resource management**
- **Quality production functions** 
- **Information asymmetries**
- **Organizational coordination costs**
- **AutoGen integration**

Creates a rich environment for studying economic phenomena like adverse selection, moral hazard, coordination failures, and market efficiency.

The externalized configuration system makes it easy to create new scenarios, tools, and agent types without modifying core code. The AutoGen 0.6.4 integration provides powerful conversation flow capabilities for complex multi-agent interactions.

Start with the flagship Structured-Note Lemons scenario to understand the patterns, then extend to your domain of interest using the tools and patterns described above.

**Happy simulating!** 🚀
