"""
Tool wrapper system for providing simulation state and configuration access.
"""

from typing import Dict, Any, Callable, List
from functools import wraps


class SimulationState:
    """
    Simple simulation state interface for tools.
    This will be replaced by the actual simulation backend.
    """
    
    def __init__(self):
        self.budgets: Dict[str, float] = {}
        self.tool_usage: Dict[str, List[Dict[str, Any]]] = {}
        self.current_agent_id: str = "unknown"
    
    def get_budget(self, agent_id: str) -> float:
        """Get current budget for an agent."""
        return self.budgets.get(agent_id, 0.0)
    
    def deduct_budget(self, agent_id: str, amount: float) -> None:
        """Deduct amount from agent's budget."""
        if agent_id in self.budgets:
            self.budgets[agent_id] -= amount
        else:
            self.budgets[agent_id] = -amount
    
    def record_tool_usage(self, agent_id: str, tool_name: str, effort: float) -> None:
        """Record tool usage for analysis."""
        if agent_id not in self.tool_usage:
            self.tool_usage[agent_id] = []
        
        self.tool_usage[agent_id].append({
            "tool": tool_name,
            "effort": effort,
            "timestamp": "now"  # Would use actual timestamp
        })
    
    def set_current_agent(self, agent_id: str) -> None:
        """Set the current agent context for tool calls."""
        self.current_agent_id = agent_id


def create_economic_tool_wrapper(simulation_state: SimulationState, 
                                config_data: Dict[str, Any]):
    """
    Create a wrapper function that injects simulation state and config into tools.
    
    Args:
        simulation_state: Current simulation state
        config_data: Configuration data for tools
    
    Returns:
        Wrapper function that can be applied to tool functions
    """
    
    def wrapper(tool_func: Callable) -> Callable:
        """
        Wrap a tool function to provide simulation state and config access.
        
        Args:
            tool_func: The original tool function
            
        Returns:
            Wrapped function with state and config injected
        """
        
        @wraps(tool_func)
        def wrapped_tool(*args, **kwargs):
            # Inject simulation state and config
            kwargs['_state'] = simulation_state
            kwargs['_config'] = config_data
            
            # Call the original tool function
            return tool_func(*args, **kwargs)
        
        return wrapped_tool
    
    return wrapper


def create_autogen_tools(tool_functions: List[Callable], 
                        simulation_state: SimulationState,
                        config_data: Dict[str, Any]) -> List[Callable]:
    """
    Create AutoGen-ready tool functions with simulation state and config access.
    
    Args:
        tool_functions: List of pure tool functions
        simulation_state: Current simulation state
        config_data: Configuration data
        
    Returns:
        List of wrapped functions ready for AutoGen registration
    """
    wrapper = create_economic_tool_wrapper(simulation_state, config_data)
    return [wrapper(func) for func in tool_functions]


def load_tool_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load tool configuration from file or return defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    from pathlib import Path
    
    default_config = {
        "market_data": {
            "sectors": {
                "tech": {"mean": 0.08, "std": 0.15},
                "finance": {"mean": 0.06, "std": 0.12},
                "healthcare": {"mean": 0.07, "std": 0.10},
                "energy": {"mean": 0.04, "std": 0.20}
            },
            "default_sector": {"mean": 0.05, "std": 0.15}
        },
        "tool_parameters": {
            "sector_forecast_thresholds": {
                "high": 5.0,
                "medium": 2.0
            },
            "monte_carlo_thresholds": {
                "high": 4.0,
                "medium": 2.0
            },
            "price_note_thresholds": {
                "high": 6.0,
                "medium": 3.0
            },
            "reflect_thresholds": {
                "high": 2.0,
                "medium": 1.0
            }
        }
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            # Merge with defaults
            for key, value in file_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration")
    
    return default_config
