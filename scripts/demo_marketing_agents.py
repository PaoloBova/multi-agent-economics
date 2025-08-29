"""
Demonstration of marketing-focused multi-agent economics simulation.

This demo shows how the transformed system works where agents receive
pre-assigned forecasts and focus entirely on marketing decisions.
"""

import numpy as np
from pathlib import Path
import asyncio
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime



        
from autogen_agentchat.agents import AssistantAgent
import autogen_agentchat.conditions as conditions
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import multi_agent_economics.core.abm as abm
import multi_agent_economics.models.market_for_finance as market_for_finance
from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, BuyerState, RegimeParameters
)
import multi_agent_economics.models.scenario_templates as scenario_templates
from multi_agent_economics.tools.schemas import PostToMarketResponse
from multi_agent_economics.core.artifacts import ArtifactManager
from multi_agent_economics.core.workspace_memory import WorkspaceMemory
from multi_agent_economics.tools.artifacts import create_artifact_tools
from multi_agent_economics.tools.economic import create_economic_tools
        
def setup_logging(log_file: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite log file each run
            # logging.StreamHandler()  # Also log to console
        ]
    )

    # Reduce log volume from noisy components (filter out DEBUG/INFO, keep WARNING+)
    logging.getLogger('autogen_core.events').setLevel(logging.WARNING)  # Reduces 335+ message passing logs
    logging.getLogger('httpcore').setLevel(logging.WARNING)             # Reduces HTTP connection details
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)  # Reduces OpenAI request/response details
    logging.getLogger('httpx').setLevel(logging.WARNING)                # Reduces HTTP request logs
    logging.getLogger('asyncio').setLevel(logging.WARNING)              # Reduces async task logs

    # Keep useful logs at INFO level (artifact operations, demo progress, errors)
    logger = logging.getLogger('demo_artifact_tools')
    logging.getLogger('artifacts.tools').setLevel(logging.INFO)         # Artifact tool operations
    logging.getLogger('autogen_core').setLevel(logging.INFO)            # Core AutoGen operations (non-events)

def termination_condition(messages: str) -> bool:
    """Determine if the chat should terminate based on the message."""
    try:
        posted_offer = PostToMarketResponse.model_validate_json(messages[-1])
        return posted_offer.success == True
    except Exception:
        return False

def create_market_state():
    """Create a market state for demonstration."""

    # Create buyers with diverse preferences
    buyers = []
    for i in range(1000):
        # Vary buyer preferences and budgets
        methodology_weight = 0.4 + (i % 3) * 0.2  # 0.4, 0.6, 0.8
        coverage_weight = 0.3 + (i % 2) * 0.4     # 0.3, 0.7
        budget = 80 + i * 10                      # 80-130
        
        buyer = BuyerState(
            buyer_id=f"buyer_{i}",
            regime_beliefs={"tech": [0.6, 0.4]},
            budget=budget,
            attr_weights={"tech": [methodology_weight, coverage_weight]},
            attr_mu={"tech": [methodology_weight, coverage_weight]},
            attr_sigma2={"tech": [0.2, 0.2]},
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.1, "scale": 0.8}
            }
        )
        buyers.append(buyer)
    
    # Regime parameters for forecasting
    regime_params = {
        "tech": {
            0: RegimeParameters(mu=0.06, sigma=0.15),  # Normal market
            1: RegimeParameters(mu=-0.02, sigma=0.25)  # Volatile market
        }
    }
    
    scenario = scenario_templates.generate_boom_scenario(sectors=["tech"], growth_factor=1.1)
    
    max_periods = 1000
    regime_history = market_for_finance.generate_regime_history_from_scenario(scenario, max_periods)
    
    # Create market state
    market_state = MarketState(
        regime_history=regime_history,
        offers=[],
        trades=[],
        index_values={"tech": 100.0},
        all_trades=[],
        buyers_state=buyers,
        sellers_state=[],
        current_regimes={"tech": 0},
        regime_parameters=regime_params,
        regime_correlations=np.array([[1.0]]),
        current_period=0,
        knowledge_good_forecasts={},
        knowledge_good_impacts={},
        attribute_order=["methodology", "coverage"],
        sector_order=["tech"],
        marketing_attribute_definitions={
            "methodology": {
                "type": "qualitative",
                "values": ["basic", "standard", "premium"],
                "descriptions": {
                    "premium": "Advanced machine learning methods",
                    "standard": "Traditional econometric approaches", 
                    "basic": "Simple heuristic methods"
                }
            },
            "coverage": {
                "type": "numeric",
                "range": [0.0, 1.0],
                "description": "Fraction of available data sources utilized"
            }
        },
        risk_free_rate=0.03
    )
    
    return market_state

def create_model_client_openai(args) -> OpenAIChatCompletionClient:
    """Create an OpenAI model client."""
        # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add your OpenAI API key to .env file")
        raise ValueError("OPENAI_API_KEY not found in environment")
    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model=args["model_name"],
        api_key=api_key
    )
    return model_client

def create_model_client(args):
    """Create a model client based on the specified type."""
    if args["model_type"] == "openai":
        return create_model_client_openai(args)
    else:
        raise ValueError(f"Unsupported model type: {args['model_type']}")

def create_agents(model, config):
    """Create agents and added context variables."""

    quality_distribution = config['quality_distribution']
    quality_types = list(config['type_quality_mapping'].keys())
    org_ids = config['org_ids']
    sectors = config['sectors']
    model_clients = config['model_clients']
    n_agents = config['n_agents']
    
    agents = []
    agent_metadata = {}

    base_assistant_agent_config = {
        "reflect_on_tool_use": True,
        "max_tool_iterations": 10,
    }
    artifact_manager_path = Path(f"./demo_marketing_agents_workspace")
    artifact_manager = ArtifactManager(artifact_manager_path)
    
    for agent_id in range(n_agents):
        model_name = np.random.choice(model_clients)
        org_id = np.random.choice(org_ids)
        quality_type = np.random.choice(quality_types, p=quality_distribution)
        sector = np.random.choice(sectors)
        agent_name = f"seller_{agent_id}_{org_id}"
        
        model_client = create_model_client({"model_type": "openai", "model_name": model_name})
        
        system_message = Path("./scripts/prompt_templates/system_prompt_marketing_task.md").read_text().format(
            org_name=org_id,
            org_description="A leading provider of financial market forecasts."
        )
        reflect_on_tool_use = base_assistant_agent_config["reflect_on_tool_use"]
        max_tool_iterations = base_assistant_agent_config["max_tool_iterations"]

        agent_workspace = artifact_manager.create_workspace(f"agent_{agent_id}_{agent_name}")
        workspace_memory = WorkspaceMemory(name=f"{agent_name} personal workspace", workspace=agent_workspace)
        base_context = {
            'budget_costs': {
                'tool:load_artifact': 0.0,
                'tool:write_artifact': 0.0, 
                'tool:share_artifact': 0.0,
                'tool:unload_artifact': 0.0,
                'tool:list_artifacts': 0.0
            }
        }
        agent_context = {**base_context,
                         **config,
                         'org_id': org_id,
                         'agent_name': agent_name,
                         'workspace_memory': workspace_memory}
        tools = create_artifact_tools(agent_context)
        tools += create_economic_tools(model, agent_context)
        
        agent = AssistantAgent(name=agent_name,
                               model_client=model_client,
                               tools=tools,
                               memory=[workspace_memory],
                               reflect_on_tool_use=reflect_on_tool_use,
                               max_tool_iterations=max_tool_iterations,
                               system_message=system_message)
        agents.append(agent)
        agent_metadata[agent_name] = {
            'agent_index': agent_id,
            'agent_name': agent_name,
            'org_id': org_id,
            'quality_type': quality_type,
            'sector': sector,
            'model_name': model_name,
        }
    
    print(f"Created {len(agents)} agents")
    print(f"Quality distribution: {quality_distribution}")
    
    model.state.budgets = {org_id: 50.0 for org_id in org_ids}
    
    return agents, agent_metadata

def collect_stats_demo(model, config):
    """Collect statistics for demonstration."""
    
    period = model.state.current_period
    print(f"--- Period {period + 1} ---")
    print(f"Offers posted (all-time): {len(model.state.offers)}")
    print(f"Trades executed: {len(model.state.trades)}")
    
    # Logging in this case means both to log to file and to collect in-memory
    # for summary statistics at the end of the simulation by storing in the
    # models model_results (and agent_results) attributes, appended as a dict
    # for each period (and each agent).
    
    # Log stats for the forecast features generated by each agent this period
    # Count by band of confidence level
    
    # Log stats for the offers posted by each agent this period
    # Count by band of marketing characteristics and price ranges
    
    # Log stats for the trades executed this period
    # Count by band of marketing characteristics and price ranges
    
    # Collect trade counts by quality type
    quality_types = list(config['type_quality_mapping'].keys())
    trade_counts = {qtype: 0 for qtype in quality_types}
    for trade in model.state.trades:
        trade.org_id = trade.seller_id
        for _agent_name, metadata in model.agent_metadata.items():
            trade_counts[metadata['quality_type']] += 1
    stats = {"trade_counts": trade_counts}
    model.model_results.append(stats)

    return

def assign_forecasts_to_agents(market_model: MarketModel, config: dict):
    """Assign forecasts to agents based on their quality type."""
    true_next_regime = market_model.state.regime_history[
        market_model.state.current_period + 1].regimes['tech']
    for _agent_id, agent_metadata in market_model.agent_metadata.items():
        # Assign forecasts based on quality type
        quality_type = agent_metadata['quality_type']
        forecast_quality = config['type_quality_mapping'][quality_type]
        confusion_matrix = market_for_finance.build_confusion_matrix(forecast_quality, K=2)
        assigned_forecast = market_for_finance.generate_forecast_signal(
            true_next_regime,
            confusion_matrix
        )
        agent_metadata['assigned_forecast'] = assigned_forecast
    
    return

def create_ai_chats(market_model: MarketModel, config: dict):
    """Create AI chats to persist agent conversations."""
    
        
    # In this demo, agents are sellers powered by LLMs to make marketing decisions.

    # Each LLM works independently on behalf of a unique seller to market their
    # pre-assigned forecasts.
    
    # AI agents typically work in teams to evaluate and reason about how to
    # allocate their resources to different tools. In this demo, resources are
    # spent on better information and analysis about the market and buyers.
    
    # We use a simple Augoten group chat to allow the agents to build up a
    # shared understanding of the market and their strategy over time.
    
    def group_agents_by_metadata_key(model, key):
        """Group agents by a specified metadata key."""
        groups = {}
        for metadata in zip(model.agent_metadata):
            agent = model.agents[metadata['agent_index']]
            group = metadata[key]
            if group not in groups:
                groups[group] = []
            groups[group].append(agent)
        return groups

    # Group agents by organization id
    # We have a chat per organization
    chats = {}

    for org_id, group in group_agents_by_metadata_key(market_model, 'org_id').items():
        # Create a chat for the organization
        termination_condition = conditions.FunctionalTermination(termination_condition)
        chat_id = f"org_chat_{org_id}"
        chats[chat_id] = RoundRobinGroupChat(agents= group,
                                             max_turns=config["max_chat_turns"],
                                             termination_condition=termination_condition)
    
    # We also have a single-agent chat per agent that we can use if necessary.
    # Typically, we let agents terminate these chats themselves by calling a
    # stop_chat tool they have access to. If they don't have access to that tool,
    # then we give them a max turn limit.
    # TODO: Consider setting a max token limit instead of a max turn limit.
    
    for agent in market_model.agents:
        chat_id = f"single_agent_chat_{agent.name}"
        chats[chat_id] = RoundRobinGroupChat(agents=[agent],
                                   max_turns=config["max_chat_turns_single_agent"],
                                   termination_condition=conditions.StopMessageTermination())
    
    market_model.chats = chats
    return

def derive_org_performance(market_model: MarketModel, org_id: str) -> str:
    """Derive organization-specific performance information."""
    last_round_trades = [trade for trade in market_model.state.trades
                          if trade.seller_id == org_id]
    last_round_revenue = sum(trade.price * trade.quantity for trade in last_round_trades)
    return (f"Total trades last round: {len(last_round_trades)}, "
            f"Revenue last round: {last_round_revenue:.2f}, ")

def derive_public_market_info(market_model: MarketModel, org_id: str) -> str:
    """Derive public market information excluding org-specific data."""
    index_info = ", ".join([f"{sector}: {value:.2f}"
                            for sector, value in market_model.state.index_values.items()])
    return (f"Current index values by sector: {index_info}, ")

def run_ai_agents(market_model: MarketModel, config: dict):
    """Run the AI agents to make marketing decisions."""
    
    # Start with group chats only. Run through each chat in sequence
    
    for chat_id, chat in market_model.chats.items():
        if chat_id .starts_with("org_chat_"):
            print(f"Running chat {chat_id} with {len(chat.agents)} agents.")
            budget_balance = 10.0  # For demo, we just refill balance each period
            org_id = market_model.agent_metadata[chat.agents[0].name]['org_id']
            org_performance = derive_org_performance(market_model, org_id)
            market_info_public = derive_public_market_info(market_model, org_id)
            current_forecast = market_model.agent_metadata[chat.agents[0].name]['assigned_forecast']
            task_prompt = Path("./scripts/prompt_templates/marketing_task.md").read_text().format(
                budget_balance=budget_balance,
                org_performance=org_performance,
                market_info_public=market_info_public,
                current_forecast=current_forecast
            )
            chat.run(task=task_prompt)
            print(f"Chat {chat_id} completed.")
    
    # We are only running org group chats which perfectly seperate agents. So,
    # we have no need to save and load state between chats and between rounds.
    
    # TODO: Validate that all groups have posted an offer. If not, raise a
    # warning.
    
    return

def run_model_step(market_model: MarketModel, config: dict):
    """Run a single step of the market model."""
    
    assign_forecasts_to_agents(market_model, config)

    run_ai_agents(market_model, config)

    # Run the market simulation step
    market_for_finance.model_step(market_model, config)
    
    # Each LLM seller should have posted a trade offer by now. If they haven't,
    # raise a warning.
    for agent in market_model.agents:
        org_id = market_model.agent_metadata[agent.name]['org_id']
        if not any(offer.seller_id == org_id
                   for offer in market_model.state.offers):
            print(f"WARNING: Seller {org_id} did not post any offers this period.")
    
    
    return

def run_demo_simulation():
    """Run a demonstration of the marketing-focused simulation."""
    
    print("=== Marketing-Focused Multi-Agent Economics Demo ===\n")
    
    # Load environment variables
    load_dotenv()
    # Setup logging for debugging
    log_file = Path("./demo_marketing_agents.log")
    setup_logging(log_file)
    
    # Initialise model

    config = {
        'sector': 'tech',
        'horizon': 1,
        'effort_distributions': {
            'high_quality': {'mean': 7.0, 'std': 1.0},
            'medium_quality': {'mean': 4.0, 'std': 1.0},
            'low_quality': {'mean': 1.5, 'std': 0.5}
        },
        'type_quality_mapping': {
            'high_quality': 0.9,
            'medium_quality': 0.7,
            'low_quality': 0.5
        },
        
        "quality_distribution": [0.2, 0.3, 0.5],  # Probabilities for high, medium, low quality
        "org_ids": ['forest_forecasts', 'reuters_analytics'],
        "sectors": ['tech'],
        "model_clients": ['gpt-4o-mini'],
        "n_agents": 4,

        "tool_parameters": {
            "sector_forecast": {
                "effort_thresholds": {"high": 5.0, "medium": 2.0},
                "effort_level_quality_mapping": {"high": 0.9, "medium": 0.7, "low": 0.5},
                "default_num_regimes": 2,
                "base_forecast_quality": 0.6,
                "default_regime_persistence": 0.8
            },
            "analyze_historical_performance": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "high_effort_max_trades": 100,
                "high_effort_noise_factor": 0.05,
                "medium_effort_max_trades": 50,
                "medium_effort_noise_factor": 0.1,
                "low_effort_max_trades": 20,
                "low_effort_noise_factor": 0.2
            },
            "analyze_buyer_preferences": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "high_effort_num_buyers": 30,
                "high_effort_num_test_offers": 12,
                "high_effort_analyze_by_attribute": True,
                "medium_effort_num_buyers": 15,
                "medium_effort_num_test_offers": 6,
                "medium_effort_analyze_by_attribute": False,
                "low_effort_num_buyers": 5,
                "low_effort_num_test_offers": 3,
                "low_effort_analyze_by_attribute": False
            },
            "research_competitive_pricing": {
                "effort_thresholds": {"high": 2.5, "medium": 1.2},
                "high_effort_num_buyers": 30,
                "high_effort_price_points": 12,
                "high_effort_lookback_trades": 50,
                "medium_effort_num_buyers": 15,
                "medium_effort_price_points": 6,
                "medium_effort_lookback_trades": 20,
                "low_effort_num_buyers": 8,
                "low_effort_price_points": 4,
                "low_effort_lookback_trades": 10
            },    
        },
        "max_chat_turns": 20,
        "max_chat_turns_single_agent": 10,
        "market_config": market_for_finance.MarketConfig()
    }
    
    market_state = create_market_state()
    model = MarketModel(
      id=1,
      num_rounds=1,
      name="marketing_demo",
      agents=[],
      agent_metadata={},
      state=market_state,
      step=run_model_step,
      collect_stats=collect_stats_demo)
    
    agents, agent_metadata = create_agents(model, config)
    model.agents = agents
    model.agent_metadata = agent_metadata

    abm.run(model, config)

    print("=== Simulation Summary ===")
    
    model_results = model.model_results
    agent_results = model.agent_results
    
    print("=== Simulation completed. ===")
    
    # Plot trend over time in total trades executed by quality type
    print("Plotting trade trends by quality type...")
    
    import matplotlib.pyplot as plt
    
    quality_types = ['high_quality', 'medium_quality', 'low_quality']
    # To convert model_results to a DataFrame, we need to flatten the nested
    # dictionaries using pd.json_normalize
    import pandas as pd
    plot_df = pd.json_normalize(model_results, sep='_')
    plt.figure(figsize=(10, 6))
    for qtype in quality_types:
        trade_counts = plot_df[f"trade_counts_{qtype}"]
        plt.plot(plot_df.index + 1, trade_counts, label=qtype)
    plt.xlabel("Period")
    plt.ylabel("Number of Trades Executed")
    plt.title("Trades Executed Over Time by Quality Type")
    plt.legend()


if __name__ == "__main__":
    run_demo_simulation()