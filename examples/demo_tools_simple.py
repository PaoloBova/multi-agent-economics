#!/usr/bin/env python3
"""
Simple demonstration of the enhanced tool integration system.

This demo bypasses package-level imports and directly tests the new tool system.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Direct imports to bypass package import issues
from multi_agent_economics.models.market_for_finance import MarketModel, MarketState, RegimeParameters
from multi_agent_economics.tools.economic import create_economic_tools
from multi_agent_economics.tools.artifacts import create_artifact_tools_for_agent
from multi_agent_economics.tools.unified import create_all_tools
from multi_agent_economics.tools.schemas import SectorForecastResponse, MonteCarloVarResponse


def create_test_market_model():
    """Create a test market model."""
    regime_parameters = {
        "tech": {
            0: RegimeParameters(mu=0.10, sigma=0.15),
            1: RegimeParameters(mu=0.02, sigma=0.25)
        }
    }
    
    market_state = MarketState(
        offers=[],
        trades=[],
        demand_profile={},
        supply_profile={},
        index_values={"tech": 100.0},
        current_regimes={"tech": 0},
        regime_parameters=regime_parameters,
        current_period=0,
        risk_free_rate=0.03,
        current_agent_id="test_agent",
        budgets={"test_agent": 25.0},
        tool_usage={}
    )
    
    return MarketModel(
        id=1,
        name="Test Market",
        agents=[],
        state=market_state,
        step=lambda: None,
        collect_stats=lambda: {}
    )


def create_test_config():
    """Create test configuration."""
    return {
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
            }
        },
        "market_data": {
            "sectors": {
                "tech": {"mean": 0.10, "std": 0.20}
            },
            "default_sector": {"mean": 0.06, "std": 0.15}
        }
    }


async def test_economic_tools():
    """Test enhanced economic tools with Pydantic responses."""
    print("=== Testing Enhanced Economic Tools ===\n")
    
    # Setup
    market_model = create_test_market_model()
    config_data = create_test_config()
    
    print("1. Creating economic tools...")
    tools = create_economic_tools(market_model, config_data)
    print(f"   Created {len(tools)} tools")
    
    # Test sector forecast
    print("\n2. Testing sector_forecast tool...")
    sector_tool = tools[0]
    
    result = await sector_tool.run_json({
        "sector": "tech",
        "horizon": 3,
        "effort": 5.0
    }, None)
    
    # Verify it's a proper structured response
    forecast_response = SectorForecastResponse(**result)
    
    print(f"   ✓ Structured response created successfully")
    print(f"   - Sector: {forecast_response.sector}")
    print(f"   - Quality Tier: {forecast_response.quality_tier}")
    print(f"   - Confidence: {forecast_response.confidence}")
    print(f"   - Effort Used: {forecast_response.effort_used}")
    print(f"   - Forecast Length: {len(forecast_response.forecast)}")
    
    # Test Monte Carlo VaR
    print("\n3. Testing monte_carlo_var tool...")
    var_tool = tools[1]
    
    result = await var_tool.run_json({
        "portfolio_value": 1000000.0,
        "volatility": 0.15,
        "confidence_level": 0.95,
        "effort": 3.0
    }, None)
    
    # Verify structured response
    var_response = MonteCarloVarResponse(**result)
    
    print(f"   ✓ Structured response created successfully")
    print(f"   - VaR Estimate: ${var_response.var_estimate:,.2f}")
    print(f"   - Quality Tier: {var_response.quality_tier}")
    print(f"   - Simulations: {var_response.n_simulations:,}")
    print(f"   - Effort Used: {var_response.effort_used}")
    
    # Check budget was deducted
    remaining_budget = market_model.state.budgets["test_agent"]
    used_budget = 25.0 - remaining_budget
    print(f"\n4. Budget tracking:")
    print(f"   - Budget used: {used_budget}")
    print(f"   - Budget remaining: {remaining_budget}")
    
    print("\n✅ Economic tools test completed successfully!")


class MockAgent:
    """Mock agent for artifact tool testing."""
    def __init__(self):
        self.name = "TestAgent"
        self.workspace_memory = MockWorkspaceMemory()
        self.budget_manager = MockBudgetManager()
        self.action_logger = MockActionLogger()


class MockWorkspaceMemory:
    """Mock workspace memory."""
    def __init__(self):
        self.name = "test_workspace"
        self.loaded = []
        
    def load_artifact(self, artifact_id):
        self.loaded.append(artifact_id)
        return True
        
    def unload_artifact(self, artifact_id):
        if artifact_id in self.loaded:
            self.loaded.remove(artifact_id)
        return True
        
    def get_loaded_artifacts(self):
        return self.loaded.copy()
        
    def get_artifact_status(self, artifact_id):
        return {"loaded": artifact_id in self.loaded}
        
    def build_context_additions(self):
        return ["[workspace] test_artifact_1, test_artifact_2 available"]


class MockBudgetManager:
    """Mock budget manager."""
    def charge_credits(self, org, amount):
        pass  # Success


class MockActionLogger:
    """Mock action logger."""
    def log_internal_action(self, actor, action, details):
        print(f"   [LOG] {actor} -> {action}")


async def test_artifact_tools():
    """Test artifact tools with per-agent closure pattern."""
    print("\n=== Testing Artifact Tools ===\n")
    
    # Create mock agent
    agent = MockAgent()
    print("1. Creating artifact tools for agent...")
    
    tools = create_artifact_tools_for_agent(agent)
    print(f"   Created {len(tools)} artifact tools")
    
    # Test list_artifacts
    print("\n2. Testing list_artifacts...")
    list_tool = tools[4]  # list_artifacts is last
    
    result = await list_tool.run_json({}, None)
    print(f"   Status: {result['status']}")
    print(f"   Available: {result['workspace_listing']}")
    
    # Test load_artifact
    print("\n3. Testing load_artifact...")
    load_tool = tools[0]  # load_artifact is first
    
    result = await load_tool.run_json({
        "artifact_id": "test_artifact_1"
    }, None)
    
    print(f"   Status: {result['status']}")
    print(f"   Message: {result['message']}")
    print(f"   Loaded artifacts: {agent.workspace_memory.loaded}")
    
    print("\n✅ Artifact tools test completed successfully!")


async def test_unified_system():
    """Test the unified tool creation system."""
    print("\n=== Testing Unified Tool System ===\n")
    
    # Setup
    market_model = create_test_market_model()
    config_data = create_test_config()
    agent = MockAgent()
    
    print("1. Creating unified tool set...")
    all_tools = create_all_tools(market_model, config_data, agent)
    
    economic_count = 0
    artifact_count = 0
    
    for tool in all_tools:
        tool_name = getattr(tool, 'name', 'unknown')
        if tool_name in ['sector_forecast', 'monte_carlo_var', 'price_note']:
            economic_count += 1
        elif 'artifact' in tool_name:
            artifact_count += 1
    
    print(f"   Total tools: {len(all_tools)}")
    print(f"   Economic tools: {economic_count}")
    print(f"   Artifact tools: {artifact_count}")
    
    # Test that both tool types work
    print("\n2. Testing integration...")
    
    # Test economic tool
    market_model.state.current_agent_id = "test_agent"
    economic_tool = all_tools[0]
    
    result = await economic_tool.run_json({
        "sector": "tech",
        "horizon": 2,
        "effort": 2.0
    }, None)
    
    print(f"   Economic tool: {result['sector']} forecast generated")
    
    # Test artifact tool
    artifact_tool = next(t for t in all_tools if getattr(t, 'name', '') == 'list_artifacts')
    result = await artifact_tool.run_json({}, None)
    
    print(f"   Artifact tool: {result['status']}")
    
    print("\n✅ Unified system test completed successfully!")


async def main():
    """Run all tests."""
    print("Enhanced Tool System - Simple Test Suite")
    print("=" * 45)
    
    try:
        await test_economic_tools()
        await test_artifact_tools()
        await test_unified_system()
        
        print("\n" + "=" * 45)
        print("🎉 All tests passed!")
        print("\nKey features demonstrated:")
        print("• Pydantic response models for type safety")
        print("• Decoupled implementations")
        print("• Per-agent artifact tools via closures")
        print("• Budget tracking and state management")
        print("• Unified tool creation system")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())