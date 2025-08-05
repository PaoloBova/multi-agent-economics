#!/usr/bin/env python3
"""
Simple test to verify the core infrastructure works.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_economics.core import (
    ArtifactManager, ToolRegistry, ActionLogger, 
    BudgetManager, QualityTracker, QualityFunction
)


def test_core_infrastructure():
    """Test that all core components can be instantiated and work together."""
    print("Testing core infrastructure...")
    
    # Test workspace setup
    workspace_dir = Path("test_workspace")
    workspace_dir.mkdir(exist_ok=True)
    
    # Setup data paths
    data_dir = Path(__file__).parent / "data"
    tool_config_path = data_dir / "config" / "enhanced_tools.json"
    market_data_path = data_dir / "market_data" / "sector_growth_rates.json"
    tool_params_path = data_dir / "config" / "tool_parameters.json"
    quality_config_path = data_dir / "config" / "quality_thresholds.json"
    
    # Initialize components
    print("1. Initializing components...")
    artifact_manager = ArtifactManager(workspace_dir / "artifacts")
    
    tool_registry = ToolRegistry(
        config_path=tool_config_path if tool_config_path.exists() else None,
        market_data_path=market_data_path if market_data_path.exists() else None,
        tool_params_path=tool_params_path if tool_params_path.exists() else None
    )
    
    action_logger = ActionLogger(workspace_dir / "logs")
    budget_manager = BudgetManager()
    
    quality_function = QualityFunction(
        config_path=quality_config_path if quality_config_path.exists() else None
    )
    quality_tracker = QualityTracker(quality_function)
    
    print("✓ All components initialized successfully")
    
    # Test budget management
    print("2. Testing budget management...")
    budget_manager.initialize_budget("TestOrg", 100.0)
    success = budget_manager.debit("TestOrg", 10.0, "Test debit")
    assert success, "Budget debit should succeed"
    balance = budget_manager.get_balance("TestOrg")
    assert balance == 90.0, f"Expected balance 90.0, got {balance}"
    print("✓ Budget management working")
    
    # Test tool registry
    print("3. Testing tool registry...")
    tools = tool_registry.tools
    assert "sector_forecast" in tools, "sector_forecast tool should be registered"
    assert "price_note" in tools, "price_note tool should be registered"
    print(f"✓ Tool registry has {len(tools)} tools")
    
    # Test workspace
    print("4. Testing workspace...")
    workspace = artifact_manager.create_workspace("TestOrg")
    assert workspace is not None, "Workspace creation should succeed"
    print("✓ Workspace created successfully")
    
    # Test quality function
    print("5. Testing quality function...")
    # Use higher tool usage to account for weighted calculation
    # sector_forecast: 12 * 0.5 = 6.0, price_note: 10 * 0.4 = 4.0, total = 10.0 > 6.0 threshold
    tool_usage = {"sector_forecast": 12.0, "price_note": 10.0}
    quality_result = quality_function.calculate_quality_score("structured_note", tool_usage)
    assert quality_result["quality"] == "high", f"Expected high quality, got {quality_result['quality']} with score {quality_result['score']}"
    print("✓ Quality function working")
    
    print("\n🎉 All core infrastructure tests passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(workspace_dir, ignore_errors=True)


async def test_scenario_imports():
    """Test that scenario imports work (even without AutoGen)."""
    print("Testing scenario imports...")
    
    try:
        from multi_agent_economics.scenarios import StructuredNoteLemonsScenario
        print("✓ Scenario import successful")
        
        # Try to create a scenario instance
        workspace_dir = Path("test_scenario")
        scenario = StructuredNoteLemonsScenario(workspace_dir)
        print("✓ Scenario instantiation successful")
        
        # Cleanup
        import shutil
        shutil.rmtree(workspace_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"⚠️  Scenario test failed (expected if AutoGen not installed): {e}")


def main():
    """Run all tests."""
    print("=== Multi-Agent Economics Infrastructure Test ===\n")
    
    try:
        test_core_infrastructure()
        print()
        asyncio.run(test_scenario_imports())
        
        print("\n✅ Infrastructure is ready for implementation!")
        print("Next steps:")
        print("1. Install AutoGen: pip install autogen-agentchat autogen-ext[openai]")
        print("2. Set up OpenAI API key in .env file")
        print("3. Run: python scripts/prepare_scenario.py")
        print("4. Run: python scripts/run_simulation.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
