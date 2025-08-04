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
        
        # Default fallback data - should load from config files
        return {}
    
    def _load_tool_parameters(self, tool_params_path: Optional[Path]) -> Dict[str, Any]:
        """Load tool parameters from file or use defaults."""
        if tool_params_path and tool_params_path.exists():
            with open(tool_params_path, 'r') as f:
                return json.load(f)
        
        # Default fallback parameters - should load from config files
        return {}
    
    def load_from_file(self, config_path: Path):
        """Load tool definitions from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Load additional configuration parameters if present
        if "monte_carlo_config" in data:
            self.monte_carlo_config = data["monte_carlo_config"]
        else:
            self.monte_carlo_config = {}
        
        if "pricing_config" in data:
            self.pricing_config = data["pricing_config"]
        else:
            self.pricing_config = {}
        
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
        
        # Sector forecast tool - Enhanced with regime prediction
        def sector_forecast(sector: str, horizon: int, tier: str = "med", 
                           current_regime: int = None, next_regime: int = None) -> Dict[str, Any]:
            """
            Generate sector forecast with regime transition predictions.
            
            This enhanced version provides regime predictions using confusion matrices,
            making forecasts economically valuable for portfolio optimization.
            """
            # Import regime-switching functions
            from ..models.market_for_finance import build_confusion_matrix, generate_forecast_signal
            
            # Load regime-switching configuration
            from pathlib import Path
            import json
            
            regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
            if regime_config_path.exists():
                with open(regime_config_path) as f:
                    regime_config = json.load(f)
            else:
                regime_config = {}
            
            # Use loaded market data for regime parameters
            sectors_data = self.market_data.get("sectors", {})
            default_data = self.market_data.get("default_sector", {})
            base_params = sectors_data.get(sector, default_data)
            
            if not base_params:
                return {"error": "No market data available for sector and no configuration loaded"}
            
            # Get regime parameters from config
            regime_params_config = regime_config.get("regime_parameters", {})
            quality_mapping = regime_params_config.get("confusion_matrix", {}).get("quality_mapping", {"high": 0.8, "med": 0.5, "low": 0.2})
            forecast_quality = quality_mapping.get(tier, 0.5)
            
            # Number of regimes from config
            K = regime_params_config.get("num_regimes", 2)
            
            # If no current regime specified, use default from config
            if current_regime is None:
                current_regime = regime_params_config.get("default_current_regime", 0)
                
            # If no next regime specified, simulate transition
            if next_regime is None:
                transition_prob = regime_params_config.get("default_transition_probability", 0.3)
                next_regime = 1 - current_regime if np.random.random() < transition_prob else current_regime
            
            # Build confusion matrix and generate forecast signal
            base_quality = regime_params_config.get("confusion_matrix", {}).get("base_quality", 0.6)
            confusion_matrix = build_confusion_matrix(forecast_quality, K, base_quality=base_quality)
            forecast_signal = generate_forecast_signal(next_regime, confusion_matrix)
            
            # Generate regime-dependent returns from config
            regime_definitions = regime_params_config.get("regime_definitions", {})
            regime_params = {}
            for regime_id, regime_def in regime_definitions.items():
                regime_id = int(regime_id)
                return_mult = regime_def.get("return_multiplier", 1.0)
                vol_mult = regime_def.get("volatility_multiplier", 1.0)
                regime_params[regime_id] = {
                    "mu": base_params["mean"] * return_mult,
                    "sigma": base_params["std"] * vol_mult
                }
            
            # Generate forecasted returns based on predicted regime
            forecast = []
            predicted_regime = forecast_signal
            regime_mu = regime_params[predicted_regime]["mu"] 
            regime_sigma = regime_params[predicted_regime]["sigma"]
            
            for _ in range(horizon):
                forecasted_return = np.random.normal(regime_mu, regime_sigma)
                forecast.append(float(forecasted_return))
            
            # Calculate forecast accuracy (for economic value measurement)
            accuracy = confusion_matrix[next_regime, forecast_signal]
            
            return {
                "sector": sector,
                "horizon": horizon,
                "forecast": forecast,
                "tier": tier,
                "predicted_regime": int(predicted_regime),
                "true_regime": int(next_regime),
                "regime_accuracy": float(accuracy),
                "forecast_quality": forecast_quality,
                "confidence": float(accuracy),
                "regime_signal": int(forecast_signal),
                "attr_vector_component": float(forecast_quality * accuracy)  # For offer attributes
            }
        
        # Monte Carlo VaR tool
        def monte_carlo_var(portfolio: Dict[str, Any], confidence: float = 0.95) -> Dict[str, Any]:
            """Calculate Value at Risk using Monte Carlo simulation."""
            # Load regime-switching configuration
            from pathlib import Path
            import json
            
            regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
            if regime_config_path.exists():
                with open(regime_config_path) as f:
                    regime_config = json.load(f)
                mc_config = regime_config.get("tool_quality_parameters", {}).get("monte_carlo_var", {})
            else:
                mc_config = {}
            
            # Use configuration from loaded data
            config = getattr(self, 'monte_carlo_config', mc_config)
            
            n_simulations = config.get("simulation_sizes", {}).get("medium", 10000)
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
            base_price = payoff_fn.get("notional", pricing_config.get("default_notional", 100))
            growth_factor = np.mean(forecast.get("forecast", [0.05]))
            discount_rate = discount_curve.get("rate", pricing_config.get("default_discount_rate", 0.03))
            
            # Load regime-switching configuration for pricing
            from pathlib import Path
            import json
            
            regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
            if regime_config_path.exists():
                with open(regime_config_path) as f:
                    regime_config = json.load(f)
                pricing_config = regime_config.get("tool_quality_parameters", {}).get("price_note", {})
            else:
                pricing_config = {}
            
            # Add pricing error based on tier using loaded parameters
            error_factors = pricing_config.get("pricing_error_std", {"high": 0.01, "med": 0.05, "low": 0.10})
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
            # Load regime-switching configuration for reflection
            from pathlib import Path
            import json
            
            regime_config_path = Path(__file__).parent.parent.parent / "data" / "config" / "regime_switching.json"
            if regime_config_path.exists():
                with open(regime_config_path) as f:
                    regime_config = json.load(f)
                reflect_config = regime_config.get("tool_quality_parameters", {}).get("reflect", {})
            else:
                reflect_config = {}
            
            # This would integrate with LLM for actual reflection
            return {
                "reflection": f"Analyzed situation: {scratchpad[:100]}...",
                "next_actions": ["gather_data", "analyze_risk", "make_decision"],
                "confidence": reflect_config.get("confidence_formula", {}).get("base", 0.8)
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
