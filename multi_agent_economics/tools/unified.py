"""
Simple unified tool creation.

Basic function to combine economic and artifact tools without complex agent setup.
"""

from typing import Dict, List, Any, Optional
from autogen_core.tools import FunctionTool

from .economic import create_economic_tools  
from .artifacts import create_artifact_tools


def create_all_tools(market_model, config_data: Dict[str, Any]) -> List[FunctionTool]:
    """
    Create both economic and artifact tools.
    
    Args:
        market_model: MarketModel instance for economic tools
        config_data: Configuration dictionary for economic tools  
        agent: Agent object for artifact tools (optional)
        
    Returns:
        List of all FunctionTool instances
    """
    # Always create economic tools
    economic_tools = create_economic_tools(market_model, config_data)
    
    # Add artifact tools if agent provided
    artifact_tools = create_artifact_tools(context=config_data)

    return economic_tools + artifact_tools