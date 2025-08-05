"""
Example usage of the new economic tools with AutoGen.
"""

from typing import List
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .economic_tools import sector_forecast, monte_carlo_var, price_note, reflect
from .tool_wrapper import (
    SimulationState, 
    create_autogen_tools, 
    load_tool_config
)


def setup_economic_simulation(config_path: str = None) -> tuple:
    """
    Set up the economic simulation with tools and state.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (simulation_state, autogen_tools, config_data)
    """
    # Initialize simulation state
    simulation_state = SimulationState()
    
    # Load configuration
    config_data = load_tool_config(config_path)
    
    # List of pure tool functions
    tool_functions = [
        sector_forecast,
        monte_carlo_var, 
        price_note,
    ]
    
    # Create AutoGen-ready tools
    autogen_tools = create_autogen_tools(tool_functions, simulation_state, config_data)
    
    return simulation_state, autogen_tools, config_data


def create_economic_agent(name: str, role: str, organization: str,
                         autogen_tools: List, model_client,
                         simulation_state: SimulationState,
                         initial_budget: float = 20.0) -> AssistantAgent:
    """
    Create an AutoGen agent with economic tools.
    
    Args:
        name: Agent name
        role: Agent role (analyst, trader, etc.)
        organization: Organization name
        autogen_tools: List of wrapped tool functions
        model_client: AutoGen model client
        simulation_state: Simulation state for budget tracking
        initial_budget: Starting budget for the agent
        
    Returns:
        Configured AssistantAgent
    """
    agent_id = f"{organization}.{role}"
    
    # Set initial budget
    simulation_state.budgets[agent_id] = initial_budget
    
    # Create system message
    system_message = f"""You are {role} at {organization}, an economic agent in a financial simulation.

Your responsibilities:
- Make strategic decisions using available tools
- Manage your budget efficiently (you start with {initial_budget} credits)
- Consider effort allocation when using tools - more effort generally yields better results

Available tools:
- sector_forecast: Generate market forecasts (effort determines accuracy)
- monte_carlo_var: Calculate portfolio risk (effort determines simulation quality)
- price_note: Price financial instruments (effort determines pricing accuracy)
- reflect: Strategic planning and analysis (effort determines depth)

Budget Management:
- Each tool call requires an 'effort' parameter (float) representing credits to spend
- Higher effort generally produces better quality results
- You'll receive warnings if you try to spend more than your available budget
- Plan your tool usage strategically to maximize value

Remember: Quality of your work depends on the effort you allocate to each task.
"""
    
    # Create agent with tools
    agent = AssistantAgent(
        name=name,
        model_client=model_client,
        system_message=system_message,
        tools=autogen_tools
    )
    
    return agent


def set_agent_context(simulation_state: SimulationState, agent_id: str):
    """
    Set the current agent context for tool calls.
    
    This should be called before an agent makes tool calls to ensure
    budget tracking works correctly.
    
    Args:
        simulation_state: Simulation state
        agent_id: ID of the agent making tool calls
    """
    simulation_state.set_current_agent(agent_id)


# Example usage
def example_usage():
    """Example of how to set up and use the economic tools."""
    
    # Setup simulation
    simulation_state, autogen_tools, config_data = setup_economic_simulation()
    
    # Create model client (would need actual API key)
    # model_client = OpenAIChatCompletionClient(model="gpt-4")
    
    # Create agents
    # analyst = create_economic_agent(
    #     name="Analyst_1",
    #     role="analyst", 
    #     organization="SellerBank1",
    #     autogen_tools=autogen_tools,
    #     model_client=model_client,
    #     simulation_state=simulation_state,
    #     initial_budget=25.0
    # )
    
    # Set agent context before tool calls
    # set_agent_context(simulation_state, "SellerBank1.analyst")
    
    # Now agents can use tools with effort parameters:
    # result = sector_forecast(sector="tech", horizon=6, effort=5.0)
    
    print("Economic simulation setup complete!")
    print(f"Available tools: {[f.__name__ for f in autogen_tools]}")
    print(f"Configuration loaded: {list(config_data.keys())}")
    
    return simulation_state, autogen_tools, config_data


if __name__ == "__main__":
    example_usage()
