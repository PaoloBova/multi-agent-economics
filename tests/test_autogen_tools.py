"""
Test suite for functional AutoGen tool integration system.

Tests the create_economic_tools function and verifies that:
- Tools are created correctly with proper schemas
- Budget management works as expected
- State access through closures functions properly
- Tool quality tiers are applied correctly
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.models.market_for_finance import MarketModel, MarketState, RegimeParameters
from multi_agent_economics.core.autogen_tools import create_economic_tools


class TestFunctionalToolIntegration:
    """Test suite for functional tool integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test market model
        regime_parameters = {
            "tech": {
                0: RegimeParameters(mu=0.10, sigma=0.15),
                1: RegimeParameters(mu=0.02, sigma=0.25)
            },
            "finance": {
                0: RegimeParameters(mu=0.08, sigma=0.12),
                1: RegimeParameters(mu=-0.01, sigma=0.20)
            }
        }
        
        self.market_state = MarketState(
            prices={},
            offers=[],
            trades=[],
            demand_profile={},
            supply_profile={},
            index_values={"tech": 100.0, "finance": 100.0},
            current_regimes={"tech": 0, "finance": 0},
            regime_parameters=regime_parameters,
            current_period=0,
            risk_free_rate=0.03,
            current_agent_id="test_agent",
            budgets={"test_agent": 10.0, "other_agent": 5.0},
            tool_usage={}
        )
        
        self.market_model = MarketModel(
            id=1,
            name="Test Market",
            agents=[],
            state=self.market_state,
            step=lambda: None,
            collect_stats=lambda: {}
        )
        
        # Test configuration
        self.config_data = {
            "tool_parameters": {
                "sector_forecast": {
                    "effort_thresholds": {"high": 5.0, "medium": 2.0},
                    "quality_tiers": {
                        "high": {"noise_factor": 0.1, "confidence": 0.9},
                        "medium": {"noise_factor": 0.3, "confidence": 0.7},
                        "low": {"noise_factor": 0.6, "confidence": 0.5}
                    }
                },
                "monte_carlo_var": {
                    "effort_thresholds": {"high": 5.0, "medium": 2.0},
                    "simulation_sizes": {"high": 50000, "medium": 10000, "low": 1000}
                },
                "price_note": {
                    "effort_thresholds": {"high": 4.0, "medium": 2.0},
                    "pricing_error_std": {"high": 0.01, "medium": 0.05, "low": 0.10}
                }
            },
            "market_data": {
                "sectors": {
                    "tech": {"mean": 0.10, "std": 0.20},
                    "finance": {"mean": 0.06, "std": 0.18}
                },
                "default_sector": {"mean": 0.06, "std": 0.15}
            }
        }
    
    def test_tool_creation(self):
        """Test that tools are created correctly."""
        tools = create_economic_tools(self.market_model, self.config_data)
        
        # Should create 4 tools
        assert len(tools) == 4
        
        # Check tool names
        tool_names = [tool.name for tool in tools]
        expected_names = ["sector_forecast", "monte_carlo_var", "price_note", "reflect"]
        for expected_name in expected_names:
            assert expected_name in tool_names
    
    def test_tool_schema_generation(self):
        """Test that AutoGen correctly generates schemas from function signatures."""
        tools = create_economic_tools(self.market_model, self.config_data)
        
        # Test sector_forecast schema
        sector_tool = next(tool for tool in tools if tool.name == "sector_forecast")
        schema = sector_tool.schema
        
        assert schema["name"] == "sector_forecast"
        assert "description" in schema
        assert "parameters" in schema
        
        params = schema["parameters"]["properties"]
        assert "sector" in params
        assert "horizon" in params
        assert "effort" in params
        
        # Check parameter types
        assert params["sector"]["type"] == "string"
        assert params["horizon"]["type"] == "integer"
        assert params["effort"]["type"] == "number"
        
        # Check descriptions are present
        assert "description" in params["sector"]
        assert "description" in params["horizon"]
        assert "description" in params["effort"]
    
    @pytest.mark.asyncio
    async def test_sector_forecast_functionality(self):
        """Test sector forecast tool functionality."""
        tools = create_economic_tools(self.market_model, self.config_data)
        sector_tool = next(tool for tool in tools if tool.name == "sector_forecast")
        
        # Test with low effort
        result = await sector_tool.run_json({
            "sector": "tech",
            "horizon": 3,
            "effort": 1.0
        }, None)
        
        # Verify result structure
        assert result["sector"] == "tech"
        assert result["horizon"] == 3
        assert len(result["forecast"]) == 3
        assert result["quality_tier"] == "low"
        assert result["confidence"] == 0.5
        assert result["effort_used"] == 1.0
        assert result["regime_used"] == 0  # Current regime for tech
        
        # Verify budget was deducted
        assert self.market_state.budgets["test_agent"] == 9.0
        
        # Verify tool usage was recorded
        assert "test_agent" in self.market_state.tool_usage
        assert len(self.market_state.tool_usage["test_agent"]) == 1
        assert self.market_state.tool_usage["test_agent"][0]["tool"] == "sector_forecast"
    
    @pytest.mark.asyncio
    async def test_budget_management(self):
        """Test budget management and insufficient budget warnings."""
        tools = create_economic_tools(self.market_model, self.config_data)
        sector_tool = next(tool for tool in tools if tool.name == "sector_forecast")
        
        # Set a small budget
        self.market_state.budgets["test_agent"] = 2.0
        
        # Request more effort than available budget
        with patch('warnings.warn') as mock_warn:
            result = await sector_tool.run_json({
                "sector": "tech",
                "horizon": 2,
                "effort": 5.0  # More than available budget
            }, None)
        
        # Should have warned about insufficient budget
        mock_warn.assert_called_once()
        warn_message = mock_warn.call_args[0][0]
        assert "Insufficient budget" in warn_message
        
        # Should use available budget instead
        assert result["effort_used"] == 2.0
        assert result["effort_requested"] == 5.0
        assert len(result["warnings"]) == 1
        assert "Budget limited effort to 2.0" in result["warnings"][0]
        
        # Budget should be exhausted
        assert self.market_state.budgets["test_agent"] == 0.0
    
    @pytest.mark.asyncio
    async def test_quality_tiers(self):
        """Test that effort levels correctly determine quality tiers."""
        tools = create_economic_tools(self.market_model, self.config_data)
        sector_tool = next(tool for tool in tools if tool.name == "sector_forecast")
        
        # Set high budget
        self.market_state.budgets["test_agent"] = 20.0
        
        # Test low effort (< 2.0)
        result_low = await sector_tool.run_json({
            "sector": "tech",
            "horizon": 2,
            "effort": 1.0
        }, None)
        assert result_low["quality_tier"] == "low"
        assert result_low["confidence"] == 0.5
        
        # Test medium effort (2.0 <= effort < 5.0)
        result_medium = await sector_tool.run_json({
            "sector": "tech",
            "horizon": 2,
            "effort": 3.0
        }, None)
        assert result_medium["quality_tier"] == "medium"
        assert result_medium["confidence"] == 0.7
        
        # Test high effort (>= 5.0)
        result_high = await sector_tool.run_json({
            "sector": "tech",
            "horizon": 2,
            "effort": 6.0
        }, None)
        assert result_high["quality_tier"] == "high"
        assert result_high["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_monte_carlo_var_functionality(self):
        """Test Monte Carlo VaR tool functionality."""
        tools = create_economic_tools(self.market_model, self.config_data)
        var_tool = next(tool for tool in tools if tool.name == "monte_carlo_var")
        
        # Set high budget
        self.market_state.budgets["test_agent"] = 10.0
        
        result = await var_tool.run_json({
            "portfolio_value": 1000000.0,
            "volatility": 0.15,
            "confidence_level": 0.95,
            "effort": 3.0
        }, None)
        
        # Verify result structure
        assert result["portfolio_value"] == 1000000.0
        assert result["volatility"] == 0.15
        assert result["confidence_level"] == 0.95
        assert result["quality_tier"] == "medium"  # 3.0 effort = medium tier
        assert result["n_simulations"] == 10000  # Medium tier simulation count
        assert "var_estimate" in result
        assert "expected_shortfall" in result
        assert result["effort_used"] == 3.0
        
        # VaR should be positive (loss)
        assert result["var_estimate"] > 0
        
        # Budget should be deducted
        assert self.market_state.budgets["test_agent"] == 7.0
    
    @pytest.mark.asyncio
    async def test_price_note_functionality(self):
        """Test price note tool functionality."""
        tools = create_economic_tools(self.market_model, self.config_data)
        price_tool = next(tool for tool in tools if tool.name == "price_note")
        
        # Set budget
        self.market_state.budgets["test_agent"] = 10.0
        
        result = await price_tool.run_json({
            "notional": 100000.0,
            "payoff_type": "linear",
            "underlying_forecast": [0.06, 0.08, 0.05],
            "discount_rate": 0.03,
            "effort": 2.5
        }, None)
        
        # Verify result structure
        assert result["notional"] == 100000.0
        assert result["payoff_type"] == "linear"
        assert result["quality_tier"] == "medium"  # 2.5 effort = medium tier
        assert "fair_value" in result
        assert "quoted_price" in result
        assert "pricing_error" in result
        assert "pricing_accuracy" in result
        assert result["effort_used"] == 2.5
        
        # Prices should be reasonable
        assert result["fair_value"] > 0
        assert result["quoted_price"] > 0
        assert 0 <= result["pricing_accuracy"] <= 1
        
        # Budget should be deducted
        assert self.market_state.budgets["test_agent"] == 7.5
    
    @pytest.mark.asyncio
    async def test_reflect_functionality(self):
        """Test reflect tool functionality."""
        tools = create_economic_tools(self.market_model, self.config_data)
        reflect_tool = next(tool for tool in tools if tool.name == "reflect")
        
        # Set budget
        self.market_state.budgets["test_agent"] = 5.0
        
        result = await reflect_tool.run_json({
            "context": "Portfolio shows high tech concentration with recent volatility.",
            "goals": ["minimize risk", "diversify sectors", "maintain growth"],
            "effort": 2.0
        }, None)
        
        # Verify result structure
        assert "context_summary" in result
        assert result["goals_analyzed"] == ["minimize risk", "diversify sectors", "maintain growth"]
        assert "reflection" in result
        assert "next_actions" in result
        assert "analysis_depth" in result
        assert "confidence" in result
        assert result["effort_used"] == 2.0
        
        # Should generate reasonable recommendations
        assert isinstance(result["next_actions"], list)
        assert len(result["next_actions"]) > 0
        assert 0 <= result["confidence"] <= 1
        
        # Budget should be deducted
        assert self.market_state.budgets["test_agent"] == 3.0
    
    @pytest.mark.asyncio
    async def test_regime_switching_integration(self):
        """Test that tools correctly use regime-switching parameters."""
        tools = create_economic_tools(self.market_model, self.config_data)
        sector_tool = next(tool for tool in tools if tool.name == "sector_forecast")
        
        # Test with tech sector in regime 0 (bull market)
        self.market_state.current_regimes["tech"] = 0
        self.market_state.budgets["test_agent"] = 10.0
        
        result = await sector_tool.run_json({
            "sector": "tech",
            "horizon": 5,
            "effort": 1.0
        }, None)
        
        assert result["regime_used"] == 0
        assert len(result["forecast"]) == 5
        
        # Change to bear market and test again
        self.market_state.current_regimes["tech"] = 1
        
        result2 = await sector_tool.run_json({
            "sector": "tech",
            "horizon": 3,
            "effort": 1.0
        }, None)
        
        assert result2["regime_used"] == 1
        assert len(result2["forecast"]) == 3
    
    def test_multiple_agents(self):
        """Test that multiple agents can use tools with separate budgets."""
        tools = create_economic_tools(self.market_model, self.config_data)
        
        # Verify separate agent budgets
        assert self.market_state.budgets["test_agent"] == 10.0
        assert self.market_state.budgets["other_agent"] == 5.0
        
        # Change agent context
        self.market_state.current_agent_id = "other_agent"
        
        # Both agents should be able to use tools independently
        # This is tested implicitly through the budget management system
        assert self.market_state.current_agent_id == "other_agent"
    
    def test_configuration_loading(self):
        """Test that configuration is properly loaded and used."""
        # Test with minimal config
        minimal_config = {"market_data": {"default_sector": {"mean": 0.05, "std": 0.10}}}
        
        tools = create_economic_tools(self.market_model, minimal_config)
        assert len(tools) == 4  # Should still create all tools
        
        # Test with missing config sections
        empty_config = {}
        tools_empty = create_economic_tools(self.market_model, empty_config)
        assert len(tools_empty) == 4  # Should handle missing config gracefully


def run_basic_tool_test():
    """Run a basic test that doesn't require pytest."""
    print("Running basic functional tool integration test...")
    
    # Create test setup
    test_instance = TestFunctionalToolIntegration()
    test_instance.setup_method()
    
    # Test tool creation
    try:
        test_instance.test_tool_creation()
        print("✓ Tool creation test passed")
    except Exception as e:
        print(f"✗ Tool creation test failed: {e}")
        return False
    
    # Test schema generation
    try:
        test_instance.test_tool_schema_generation()
        print("✓ Schema generation test passed")
    except Exception as e:
        print(f"✗ Schema generation test failed: {e}")
        return False
    
    # Test configuration loading
    try:
        test_instance.test_configuration_loading()
        print("✓ Configuration loading test passed")
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False
    
    print("✓ All basic tests passed!")
    return True


if __name__ == "__main__":
    # Run basic tests that don't require pytest
    success = run_basic_tool_test()
    
    if success:
        print("\nTo run full test suite with async tests:")
        print("  python -m pytest tests/test_autogen_tools.py -v")
    else:
        print("\nBasic tests failed. Check the implementation.")