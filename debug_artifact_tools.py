#!/usr/bin/env python3
"""Debug script to understand why artifact tools are failing."""

import asyncio
import tempfile
import shutil
from pathlib import Path

# Import actual classes
from multi_agent_economics.core.artifacts import Workspace, ArtifactManager
from multi_agent_economics.core.workspace_memory import WorkspaceMemory
from multi_agent_economics.core.budget import BudgetManager
from multi_agent_economics.core.actions import ActionLogger
from multi_agent_economics.tools.artifacts import create_artifact_tools_for_agent


class DebugAgent:
    def __init__(self, name: str, workspace: Workspace, budget_manager: BudgetManager):
        self.name = name
        self.workspace_memory = WorkspaceMemory(name, workspace, max_chars=2000)
        self.budget_manager = budget_manager
        self.action_logger = ActionLogger()
        
        # Initialize budget
        budget_manager.initialize_budget("artifact_ops", 50.0)
        print(f"Initialized budget: {budget_manager.get_balance('artifact_ops')}")


async def debug_artifact_tools():
    """Debug what's going wrong with artifact tools."""
    print("=== Debugging Artifact Tools ===\n")
    
    # Create temporary workspace
    temp_dir = tempfile.mkdtemp()
    try:
        # Set up components
        artifact_manager = ArtifactManager(Path(temp_dir))
        workspace = artifact_manager.create_workspace("TestOrg")
        budget_manager = BudgetManager()
        agent = DebugAgent("TestAgent", workspace, budget_manager)
        
        print(f"Agent budget balance: {agent.budget_manager.get_balance('artifact_ops')}")
        
        # Create artifact tools
        tools = create_artifact_tools_for_agent(agent)
        print(f"Created {len(tools)} artifact tools")
        
        # Test load artifact
        print("\n1. Testing load_artifact...")
        load_tool = tools[0]
        try:
            result = await load_tool.run_json({"artifact_id": "test_artifact"}, None)
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
            print(f"Status: {result.status}")
            print(f"Message: {result.message}")
        except Exception as e:
            print(f"Error during load: {e}")
            import traceback
            traceback.print_exc()
        
        # Test list artifacts
        print("\n2. Testing list_artifacts...")
        list_tool = tools[4]
        try:
            result = await list_tool.run_json({}, None)
            print(f"Result type: {type(result)}")
            print(f"Status: {result.status}")
            print(f"Message: {result.message}")
        except Exception as e:
            print(f"Error during list: {e}")
            import traceback
            traceback.print_exc()
            
        # Check budget after operations
        print(f"\nFinal budget balance: {agent.budget_manager.get_balance('artifact_ops')}")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    asyncio.run(debug_artifact_tools())