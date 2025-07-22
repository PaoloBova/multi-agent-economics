"""
Tool system for agent actions with cost and quality mechanics.
"""

import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from .artifacts import Artifact


@dataclass
class Tool:
    """Definition of a tool that agents can use."""
    id: str
    cost: float
    inputs: List[str]
    outputs: List[str]
    latency: int
    precision_tiers: Optional[Dict[str, Dict[str, Any]]] = None
    description: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """Create tool from dictionary."""
        return cls(**data)


class ToolRegistry:
    """Registry of all available tools, loaded from JSON configuration."""
    
    def __init__(self, config_path: Optional[Path] = None, 
                 market_data_path: Optional[Path] = None,
                 tool_params_path: Optional[Path] = None):
        self.tools: Dict[str, Tool] = {}
        self.tool_functions: Dict[str, Callable] = {}
        
        # Load configurations
        self.market_data = self._load_market_data(market_data_path)
        self.tool_params = self._load_tool_parameters(tool_params_path)
        
        if config_path and config_path.exists():
            self.load_from_file(config_path)
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _load_market_data(self, market_data_path: Optional[Path]) -> Dict[str, Any]:
        """Load market data from file or use defaults."""
        if market_data_path and market_data_path.exists():
            with open(market_data_path, 'r') as f:
                return json.load(f)
        
        # Default fallback data
        return {
            "sectors": {
                "tech": {"mean": 0.08, "std": 0.15},
                "finance": {"mean": 0.06, "std": 0.12},
                "healthcare": {"mean": 0.07, "std": 0.10},
                "energy": {"mean": 0.04, "std": 0.20}
            },
            "default_sector": {"mean": 0.05, "std": 0.15}
        }
    
    def _load_tool_parameters(self, tool_params_path: Optional[Path]) -> Dict[str, Any]:
        """Load tool parameters from file or use defaults."""
        if tool_params_path and tool_params_path.exists():
            with open(tool_params_path, 'r') as f:
                return json.load(f)
        
        # Default fallback parameters
        return {
            "precision_tiers": {
                "high": {"noise_factor": 0.1},
                "med": {"noise_factor": 0.3},
                "low": {"noise_factor": 0.6}
            },
            "error_factors": {
                "high": 0.01,
                "med": 0.05,
                "low": 0.10
            },
            "noise_multipliers": {
                "high": 0.1,
                "med": 0.3,
                "low": 0.6
            }
        }
    
    def load_from_file(self, config_path: Path):
        """Load tool definitions from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Load additional configuration parameters if present
        if "monte_carlo_config" in data:
            self.monte_carlo_config = data["monte_carlo_config"]
        else:
            self.monte_carlo_config = {
                "n_simulations": 10000,
                "default_volatility": 0.15,
                "confidence_level": 0.95
            }
        
        if "pricing_config" in data:
            self.pricing_config = data["pricing_config"]
        else:
            self.pricing_config = {
                "default_notional": 100,
                "default_discount_rate": 0.03
            }
        
        for tool_data in data.get("tools", []):
            tool = Tool.from_dict(tool_data)
            self.tools[tool.id] = tool
    
    def register_tool(self, tool: Tool, func: Callable):
        """Register a tool with its implementation function."""
        self.tools[tool.id] = tool
        self.tool_functions[tool.id] = func
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool definition by ID."""
        return self.tools.get(tool_id)
    
    def get_tool_function(self, tool_id: str) -> Optional[Callable]:
        """Get a tool's implementation function."""
        return self.tool_functions.get(tool_id)
    
    def _register_builtin_tools(self):
        """Register built-in tools for the finance scenario."""
        
        # Sector forecast tool
        def sector_forecast(sector: str, horizon: int, tier: str = "med") -> Dict[str, Any]:
            """Generate sector growth forecast with configurable precision."""
            # Use loaded market data
            sectors_data = self.market_data.get("sectors", {})
            default_data = self.market_data.get("default_sector", {"mean": 0.05, "std": 0.15})
            base_params = sectors_data.get(sector, default_data)
            
            # Add noise based on precision tier using loaded parameters
            noise_multipliers = self.tool_params.get("noise_multipliers", {"high": 0.1, "med": 0.3, "low": 0.6})
            noise_factor = noise_multipliers.get(tier, 0.3)
            
            # Generate forecast
            forecast = []
            for _ in range(horizon):
                base_rate = np.random.normal(base_params["mean"], base_params["std"])
                noise = np.random.normal(0, base_params["std"] * noise_factor)
                forecast.append(float(base_rate + noise))
            
            return {
                "sector": sector,
                "horizon": horizon,
                "forecast": forecast,
                "tier": tier,
                "confidence": 1.0 - noise_factor
            }
        
        # Monte Carlo VaR tool
        def monte_carlo_var(portfolio: Dict[str, Any], confidence: float = 0.95) -> Dict[str, Any]:
            """Calculate Value at Risk using Monte Carlo simulation."""
            # Use configuration from loaded data
            config = getattr(self, 'monte_carlo_config', {
                "n_simulations": 10000,
                "default_volatility": 0.15,
                "confidence_level": 0.95
            })
            
            n_simulations = config.get("n_simulations", 10000)
            portfolio_value = portfolio.get("value", 1000000)
            volatility = portfolio.get("volatility", config.get("default_volatility", 0.15))
            
            # Generate random returns
            returns = np.random.normal(0, volatility, n_simulations)
            portfolio_values = portfolio_value * (1 + returns)
            losses = portfolio_value - portfolio_values
            
            # Calculate VaR
            var_95 = float(np.percentile(losses, confidence * 100))
            
            return {
                "var_95": var_95,
                "portfolio_value": portfolio_value,
                "confidence": confidence,
                "max_loss": float(np.max(losses)),
                "expected_loss": float(np.mean(losses))
            }
        
        # Price note tool
        def price_note(payoff_fn: Dict[str, Any], forecast: Dict[str, Any], 
                      discount_curve: Dict[str, Any], tier: str = "med") -> Dict[str, Any]:
            """Price a structured note given payoff function and market forecast."""
            # Simplified pricing model
            base_price = payoff_fn.get("notional", 100)
            growth_factor = np.mean(forecast.get("forecast", [0.05]))
            discount_rate = discount_curve.get("rate", 0.03)
            
            # Add pricing error based on tier using loaded parameters
            error_factors = self.tool_params.get("error_factors", {"high": 0.01, "med": 0.05, "low": 0.10})
            error = np.random.normal(0, error_factors.get(tier, 0.05))
            
            fair_price = base_price * (1 + growth_factor) / (1 + discount_rate)
            final_price = fair_price * (1 + error)
            
            return {
                "fair_price": float(fair_price),
                "quoted_price": float(final_price),
                "tier": tier,
                "growth_factor": float(growth_factor),
                "discount_rate": discount_rate,
                "pricing_error": float(error)
            }
        
        # Share artifact tool
        def share_artifact(artifact_id: str, target: str) -> Dict[str, Any]:
            """Share an artifact with another agent/role."""
            return {
                "artifact_id": artifact_id,
                "target": target,
                "shared_at": "now",  # Would use actual timestamp
                "status": "success"
            }
        
        # Reflect tool
        def reflect(scratchpad: str) -> Dict[str, Any]:
            """Generate a reflection/planning artifact."""
            # This would integrate with LLM for actual reflection
            return {
                "reflection": f"Analyzed situation: {scratchpad[:100]}...",
                "next_actions": ["gather_data", "analyze_risk", "make_decision"],
                "confidence": 0.8
            }
        
        # Register all tools
        tools_config = [
            {
                "id": "sector_forecast",
                "cost": 3,
                "inputs": ["sector", "horizon"],
                "outputs": ["forecast_report"],
                "latency": 1,
                "precision_tiers": {"high": {}, "med": {}, "low": {}},
                "description": "Generate sector growth forecast"
            },
            {
                "id": "monte_carlo_var",
                "cost": 2,
                "inputs": ["portfolio"],
                "outputs": ["var_report"],
                "latency": 1,
                "description": "Calculate portfolio VaR"
            },
            {
                "id": "price_note",
                "cost": 4,
                "inputs": ["payoff_fn", "forecast", "discount_curve"],
                "outputs": ["pricing_report"],
                "latency": 1,
                "precision_tiers": {"high": {}, "med": {}},
                "description": "Price structured financial instrument"
            },
            {
                "id": "share_artifact",
                "cost": 0.2,
                "inputs": ["artifact_id", "target"],
                "outputs": ["share_confirmation"],
                "latency": 1,
                "description": "Share artifact with another agent"
            },
            {
                "id": "reflect",
                "cost": 0.5,
                "inputs": ["scratchpad"],
                "outputs": ["reflection"],
                "latency": 1,
                "description": "Generate strategic reflection"
            }
        ]
        
        tool_functions = {
            "sector_forecast": sector_forecast,
            "monte_carlo_var": monte_carlo_var,
            "price_note": price_note,
            "share_artifact": share_artifact,
            "reflect": reflect
        }
        
        for tool_config in tools_config:
            tool = Tool.from_dict(tool_config)
            func = tool_functions[tool.id]
            self.register_tool(tool, func)


def call_tool(tool_id: str, registry: ToolRegistry, agent_id: str, 
              **kwargs) -> Artifact:
    """Execute a tool and return the resulting artifact."""
    tool = registry.get_tool(tool_id)
    tool_func = registry.get_tool_function(tool_id)
    
    if not tool or not tool_func:
        raise ValueError(f"Tool {tool_id} not found")
    
    # Execute the tool function
    result = tool_func(**kwargs)
    
    # Create artifact from result
    artifact = Artifact.create(
        artifact_type=tool.outputs[0] if tool.outputs else "generic_output",
        payload=result,
        created_by=agent_id,
        visibility=[agent_id],  # Private by default
        metadata={
            "tool_id": tool_id,
            "cost": tool.cost,
            "latency": tool.latency,
            "inputs": kwargs
        }
    )
    
    return artifact
