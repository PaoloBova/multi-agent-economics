"""
Structured-Note Lemons flagship scenario implementation.

This module implements the finance domain scenario where internal quality 
drives adverse selection in structured note markets.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

import networkx as nx

from ..core import (
    ArtifactManager, ToolRegistry, ActionLogger, 
    BudgetManager, QualityTracker, QualityFunction
)
from ..agents import create_agent


class StructuredNoteLemonsScenario:
    """Flagship scenario: Structured-Note Lemons market simulation."""
    
    def __init__(self, workspace_dir: Path, config: Optional[Dict[str, Any]] = None,
                 data_dir: Optional[Path] = None):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directory for configurations
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        
        # Load configuration
        self.config = config or self._default_config()
        
        # Initialize core systems with data paths
        self.artifact_manager = ArtifactManager(self.workspace_dir / "artifacts")
        
        # Initialize tool registry with data paths
        tool_config_path = self.data_dir / "config" / "enhanced_tools.json"
        market_data_path = self.data_dir / "market_data" / "sector_growth_rates.json"
        tool_params_path = self.data_dir / "config" / "tool_parameters.json"
        
        self.tool_registry = ToolRegistry(
            config_path=tool_config_path if tool_config_path.exists() else None,
            market_data_path=market_data_path if market_data_path.exists() else None,
            tool_params_path=tool_params_path if tool_params_path.exists() else None
        )
        
        self.action_logger = ActionLogger(self.workspace_dir / "logs")
        self.budget_manager = BudgetManager()
        
        # Initialize quality function with configuration
        quality_config_path = self.data_dir / "config" / "quality_thresholds.json"
        self.quality_function = QualityFunction(
            config_path=quality_config_path if quality_config_path.exists() else None
        )
        self.quality_tracker = QualityTracker(self.quality_function)
        
        # Market state
        self.market_state = self._initialize_market_state()
        
        # Agents and teams
        self.agents: Dict[str, Any] = {}
        self.teams: Dict[str, Any] = {}
        
        # Results tracking
        self.round_results: List[Dict[str, Any]] = []
        self.current_round = 0
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the scenario."""
        return {
            "rounds": 10,
            "seller_banks": ["SellerBank1", "SellerBank2"],
            "buyer_funds": ["BuyerFund1", "BuyerFund2"],
            "initial_budgets": {
                "seller_bank": 20,
                "buyer_fund": 10
            },
            "market_parameters": {
                "base_notional": 100,
                "discount_rate": 0.03,
                "volatility": 0.15
            }
        }
    
    def _initialize_market_state(self) -> Dict[str, Any]:
        """Initialize the market state with synthetic data."""
        return {
            "round": 0,
            "market_data": {
                "sectors": {
                    "tech": {"mean_growth": 0.08, "volatility": 0.15},
                    "finance": {"mean_growth": 0.06, "volatility": 0.12},
                    "healthcare": {"mean_growth": 0.07, "volatility": 0.10}
                },
                "discount_curve": {"rate": 0.03, "term_structure": [0.025, 0.03, 0.035]},
                "market_sentiment": "neutral"
            },
            "posted_instruments": [],
            "trade_history": [],
            "competitor_actions": []
        }
    
    async def setup_agents(self):
        """Create all agents for the scenario."""
        # Initialize model client
        model_client = OpenAIChatCompletionClient(model="gpt-4")
        
        # Create seller bank teams
        for bank_name in self.config["seller_banks"]:
            # Initialize budget for the organization
            self.budget_manager.initialize_budget(bank_name, self.config["initial_budgets"]["seller_bank"])
            
            # Create agents for this bank
            prompt_templates_path = self.data_dir / "prompts"
            role_definitions_path = self.data_dir / "prompts" / "role_definitions.json"
            
            bank_agents = {
                "analyst": create_agent(
                    name=f"{bank_name}_Analyst",
                    role="Analyst",
                    organization=bank_name,
                    model_client=model_client,
                    artifact_manager=self.artifact_manager,
                    tool_registry=self.tool_registry,
                    budget_manager=self.budget_manager,
                    action_logger=self.action_logger,
                    quality_tracker=self.quality_tracker,
                    prompt_templates_path=prompt_templates_path,
                    role_definitions_path=role_definitions_path
                ),
                "structurer": create_agent(
                    name=f"{bank_name}_Structurer",
                    role="Structurer", 
                    organization=bank_name,
                    model_client=model_client,
                    artifact_manager=self.artifact_manager,
                    tool_registry=self.tool_registry,
                    budget_manager=self.budget_manager,
                    action_logger=self.action_logger,
                    quality_tracker=self.quality_tracker,
                    prompt_templates_path=prompt_templates_path,
                    role_definitions_path=role_definitions_path
                ),
                "pm": create_agent(
                    name=f"{bank_name}_PM",
                    role="PM",
                    organization=bank_name,
                    model_client=model_client,
                    artifact_manager=self.artifact_manager,
                    tool_registry=self.tool_registry,
                    budget_manager=self.budget_manager,
                    action_logger=self.action_logger,
                    quality_tracker=self.quality_tracker,
                    prompt_templates_path=prompt_templates_path,
                    role_definitions_path=role_definitions_path
                )
            }
            
            self.agents[bank_name] = bank_agents
        
        # Create buyer fund teams
        for fund_name in self.config["buyer_funds"]:
            # Initialize budget for the organization
            self.budget_manager.initialize_budget(fund_name, self.config["initial_budgets"]["buyer_fund"])
            
            # Create agents for this fund
            prompt_templates_path = self.data_dir / "prompts"
            role_definitions_path = self.data_dir / "prompts" / "role_definitions.json"
            
            fund_agents = {
                "risk_officer": create_agent(
                    name=f"{fund_name}_RiskOfficer",
                    role="Risk-Officer",
                    organization=fund_name,
                    model_client=model_client,
                    artifact_manager=self.artifact_manager,
                    tool_registry=self.tool_registry,
                    budget_manager=self.budget_manager,
                    action_logger=self.action_logger,
                    quality_tracker=self.quality_tracker,
                    prompt_templates_path=prompt_templates_path,
                    role_definitions_path=role_definitions_path
                ),
                "trader": create_agent(
                    name=f"{fund_name}_Trader",
                    role="Trader",
                    organization=fund_name,
                    model_client=model_client,
                    artifact_manager=self.artifact_manager,
                    tool_registry=self.tool_registry,
                    budget_manager=self.budget_manager,
                    action_logger=self.action_logger,
                    quality_tracker=self.quality_tracker,
                    prompt_templates_path=prompt_templates_path,
                    role_definitions_path=role_definitions_path
                )
            }
            
            self.agents[fund_name] = fund_agents
    
    def create_interaction_topology(self):
        """Create interaction graph for GraphFlow using external topology configuration."""
        builder = DiGraphBuilder()
        
        # Load topology configuration
        topology_config = self._load_topology_config()
        
        # Add all agents as nodes
        all_agents = []
        for org_agents in self.agents.values():
            all_agents.extend(org_agents.values())
        
        for agent in all_agents:
            builder.add_node(agent)
        
        # Create intra-organization edges based on configuration
        self._create_intra_org_edges(builder, topology_config)
        
        # Create inter-organization edges based on configuration
        self._create_inter_org_edges(builder, topology_config)
        
        return builder.build(), all_agents
    
    def _load_topology_config(self) -> Dict[str, Any]:
        """Load interaction topology configuration from external file."""
        topology_config_path = self.data_dir / "config" / "interaction_topology.json"
        
        # Default fallback configuration (current hardcoded behavior)
        default_config = {
            "default_topology": "structured_note_lemons",
            "topologies": {
                "structured_note_lemons": {
                    "intra_organization": {"connectivity": "full_mesh"},
                    "inter_organization": {
                        "connectivity": "role_based",
                        "rules": [{
                            "source_orgs": "seller_banks",
                            "source_roles": ["pm"],
                            "target_orgs": "buyer_funds",
                            "target_roles": ["trader"],
                            "bidirectional": True
                        }]
                    }
                }
            }
        }
        
        if topology_config_path.exists():
            try:
                with open(topology_config_path, 'r') as f:
                    config = json.load(f)
                print(f"Loaded interaction topology from {topology_config_path}")
                return config
            except Exception as e:
                print(f"Warning: Could not load topology config from {topology_config_path}: {e}")
                print("Using default topology configuration")
        else:
            print(f"No topology config found at {topology_config_path}, using defaults")
        
        return default_config
    
    def _create_intra_org_edges(self, builder: DiGraphBuilder, topology_config: Dict[str, Any]):
        """Create edges within organizations based on topology configuration."""
        topology_name = topology_config.get("default_topology", "structured_note_lemons")
        topology = topology_config["topologies"][topology_name]
        intra_config = topology["intra_organization"]
        
        connectivity = intra_config.get("connectivity", "full_mesh")
        default_weight = intra_config.get("default_weight", 1.0)
        role_weights = intra_config.get("role_weights", {})
        
        for org_name, org_agents in self.agents.items():
            if connectivity == "full_mesh":
                # Full connectivity within organization
                agent_list = list(org_agents.values())
                for i, agent1 in enumerate(agent_list):
                    for j, agent2 in enumerate(agent_list):
                        if i != j:
                            # Determine edge weight based on roles
                            role1 = self._get_agent_role(agent1)
                            role2 = self._get_agent_role(agent2)
                            edge_key = f"{role1}->{role2}"
                            weight = role_weights.get(edge_key, default_weight)
                            
                            builder.add_edge(agent1, agent2, weight=weight)
            
            elif connectivity == "hub_and_spoke":
                # Hub and spoke within organization
                hub_role = intra_config.get("hub_role", "pm")
                hub_weight = intra_config.get("hub_weight", 0.9)
                spoke_weight = intra_config.get("spoke_weight", 0.6)
                
                if hub_role in org_agents:
                    hub_agent = org_agents[hub_role]
                    for role, agent in org_agents.items():
                        if role != hub_role:
                            builder.add_edge(hub_agent, agent, weight=hub_weight)
                            builder.add_edge(agent, hub_agent, weight=spoke_weight)
    
    def _get_agent_role(self, agent) -> str:
        """Extract the role from an agent's name."""
        agent_name = agent.name
        # Agent names are like "SellerBank1_Analyst"
        if "_" in agent_name:
            return agent_name.split("_")[-1].lower()
        return "unknown"
    
    def _create_inter_org_edges(self, builder: DiGraphBuilder, topology_config: Dict[str, Any]):
        """Create edges between organizations based on topology configuration."""
        topology_name = topology_config.get("default_topology", "structured_note_lemons")
        topology = topology_config["topologies"][topology_name]
        inter_config = topology["inter_organization"]
        
        connectivity = inter_config.get("connectivity", "role_based")
        
        if connectivity == "full_mesh":
            # Everyone can talk to everyone across organizations
            all_agents = []
            for org_agents in self.agents.values():
                all_agents.extend(org_agents.values())
            
            for i, agent1 in enumerate(all_agents):
                for j, agent2 in enumerate(all_agents):
                    if i != j and self._get_agent_org(agent1) != self._get_agent_org(agent2):
                        builder.add_edge(agent1, agent2)
        
        elif connectivity == "role_based":
            # Role-based connectivity rules
            rules = inter_config.get("rules", [])
            for rule in rules:
                self._apply_connectivity_rule(builder, rule)
    
    def _apply_connectivity_rule(self, builder: DiGraphBuilder, rule: Dict[str, Any]):
        """Apply a single connectivity rule between organizations."""
        source_org_type = rule["source_orgs"]  # e.g., "seller_banks"
        source_roles = rule["source_roles"]    # e.g., ["pm"]
        target_org_type = rule["target_orgs"]  # e.g., "buyer_funds" 
        target_roles = rule["target_roles"]    # e.g., ["trader"]
        bidirectional = rule.get("bidirectional", True)
        weight = rule.get("weight", 1.0)
        
        # Get source agents
        source_agents = []
        for org_name, org_agents in self.agents.items():
            if org_name in self.config.get(source_org_type, []):
                for role in source_roles:
                    if role in org_agents:
                        source_agents.append(org_agents[role])
        
        # Get target agents
        target_agents = []
        for org_name, org_agents in self.agents.items():
            if org_name in self.config.get(target_org_type, []):
                for role in target_roles:
                    if role in org_agents:
                        target_agents.append(org_agents[role])
        
        # Create edges with weights
        for source_agent in source_agents:
            for target_agent in target_agents:
                builder.add_edge(source_agent, target_agent, weight=weight)
                if bidirectional:
                    builder.add_edge(target_agent, source_agent, weight=weight)
    
    def _get_agent_org(self, agent) -> str:
        """Get the organization name for an agent."""
        agent_name = agent.name
        for org_name, org_agents in self.agents.items():
            for role_agent in org_agents.values():
                if role_agent.name == agent_name:
                    return org_name
        return "unknown"
    
    async def run_round(self, round_number: int) -> Dict[str, Any]:
        """Run a single round of the simulation."""
        self.current_round = round_number
        self.market_state["round"] = round_number
        
        print(f"\n=== ROUND {round_number} ===")
        
        # Create market situation message
        market_message = self._create_market_message()
        
        round_results = {
            "round": round_number,
            "market_state": dict(self.market_state),
            "trades": [],
            "quality_metrics": {},
            "budget_changes": {},
            "actions": {"internal": [], "external": []}
        }
        
        # Create interaction topology
        graph, all_agents = self.create_interaction_topology()
        
        # Create GraphFlow team
        team = GraphFlow(
            participants=all_agents,
            termination_condition=MaxMessageTermination(max_messages=20)
        )
        
        print(f"Market Update: {market_message}")
        
        # Run the conversation with termination condition
        console = Console()
        messages = []
        
        # Simple termination after max messages or when ROUND_COMPLETE is mentioned
        max_messages = 20
        message_count = 0
        
        async for message in team.run_stream(task=market_message):
            print(f"{message.source}: {message.content}")
            messages.append({
                "source": message.source,
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            })
            
            message_count += 1
            if message_count >= max_messages or "ROUND_COMPLETE" in message.content:
                break
        
        # Process round results
        round_results["conversation"] = messages
        round_results = await self._process_round_results(round_results)
        
        return round_results
    
    def _create_market_message(self) -> str:
        """Create the market situation message for the round."""
        return f"""
MARKET UPDATE - Round {self.current_round}

Current Market Conditions:
- Tech sector outlook: Growth potential with moderate volatility
- Finance sector: Stable growth expectations
- Healthcare: Steady performance expected
- Base discount rate: {self.market_state['market_data']['discount_curve']['rate']}

Your organizations should now:
1. SELLERS: Analyze market conditions, price structured notes, and post offerings
2. BUYERS: Assess available instruments and make investment decisions

Remember:
- Quality analysis requires appropriate tool usage
- Budget constraints limit your analytical capabilities
- Market actions are visible to competitors

Begin your analysis and decision-making process. End with ROUND_COMPLETE when ready.
"""
    
    async def _process_round_results(self, round_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process the results of a round."""
        # Collect all actions from the logger
        round_start = datetime.now().replace(hour=0, minute=0, second=0)
        round_end = datetime.now()
        
        actions = self.action_logger.get_actions_by_timeframe(round_start, round_end)
        round_results["actions"] = actions
        
        # Process external actions to identify trades
        trades = []
        posted_instruments = {}
        
        for action in actions["external"]:
            if action["action"] == "post_price":
                posted_instruments[action["actor"]] = {
                    "price": action.get("price"),
                    "instrument": action.get("good"),
                    "actor": action["actor"]
                }
            elif action["action"] == "accept":
                # Find corresponding posted instrument
                for seller, instrument in posted_instruments.items():
                    trade = {
                        "seller": seller.split('.')[0],
                        "buyer": action["actor"].split('.')[0],
                        "instrument": instrument["instrument"],
                        "price": instrument["price"],
                        "timestamp": action["timestamp"]
                    }
                    trades.append(trade)
        
        round_results["trades"] = trades
        
        # Calculate quality metrics
        quality_metrics = self.quality_tracker.get_quality_distribution()
        round_results["quality_metrics"] = quality_metrics
        
        # Update market state
        self.market_state["posted_instruments"].extend(list(posted_instruments.values()))
        self.market_state["trade_history"].extend(trades)
        
        return round_results
    
    async def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        print("Setting up Structured-Note Lemons simulation...")
        
        await self.setup_agents()
        
        print(f"Running {self.config['rounds']} rounds...")
        
        simulation_results = {
            "config": self.config,
            "rounds": [],
            "summary": {}
        }
        
        # Run all rounds
        for round_num in range(1, self.config["rounds"] + 1):
            round_result = await self.run_round(round_num)
            simulation_results["rounds"].append(round_result)
            self.round_results.append(round_result)
        
        # Generate final summary
        simulation_results["summary"] = self._generate_summary()
        
        # Export results
        await self._export_results(simulation_results)
        
        return simulation_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate simulation summary metrics."""
        total_trades = sum(len(r["trades"]) for r in self.round_results)
        
        # Quality distribution across all rounds
        all_quality_data = []
        for round_result in self.round_results:
            if "quality_metrics" in round_result and "distribution" in round_result["quality_metrics"]:
                all_quality_data.append(round_result["quality_metrics"]["distribution"])
        
        # Budget analysis
        budget_summary = self.budget_manager.get_budget_summary()
        
        # Action analysis
        action_metrics = self.action_logger.calculate_metrics()
        
        return {
            "total_rounds": len(self.round_results),
            "total_trades": total_trades,
            "average_trades_per_round": total_trades / max(len(self.round_results), 1),
            "budget_summary": budget_summary,
            "action_metrics": action_metrics,
            "quality_patterns": self.quality_tracker.identify_quality_patterns()
        }
    
    async def _export_results(self, results: Dict[str, Any]):
        """Export simulation results to files."""
        # Main results file
        results_file = self.workspace_dir / "simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Export action logs
        self.action_logger.export_summary(self.workspace_dir / "action_summary.json")
        
        # Export quality report
        self.quality_tracker.export_quality_report(self.workspace_dir / "quality_report.json")
        
        print(f"Results exported to {self.workspace_dir}")


async def run_flagship_scenario(workspace_dir: Path, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Main entry point for running the flagship scenario."""
    scenario = StructuredNoteLemonsScenario(workspace_dir, config)
    return await scenario.run_simulation()
