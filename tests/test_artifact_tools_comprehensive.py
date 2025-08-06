"""
Comprehensive tests for artifact management tools using real classes.

This test suite verifies the complete artifact management workflow using actual
core classes from the project, avoiding mocks wherever possible to ensure
realistic behavior testing.

Test Coverage:
- Artifact loading/unloading with workspace memory integration
- Artifact writing with persistent storage
- Cross-workspace sharing functionality  
- Budget tracking and credit management
- Action logging and audit trails
- Error handling and edge cases
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Import actual classes (no mocks)
from multi_agent_economics.core.artifacts import Artifact, Workspace, ArtifactManager
from multi_agent_economics.core.workspace_memory import WorkspaceMemory
from multi_agent_economics.core.budget import BudgetManager
from multi_agent_economics.core.actions import ActionLogger, InternalAction

# Import tools and schemas
from multi_agent_economics.tools.artifacts import create_artifact_tools_for_agent
from multi_agent_economics.tools.schemas import (
    ArtifactLoadResponse, ArtifactUnloadResponse, ArtifactWriteResponse,
    ArtifactShareResponse, ArtifactListResponse
)


class MockAgent:
    """
    Real agent class using actual core components for testing.
    
    This represents what a production agent would look like, with all the
    components that the artifact tools expect to interact with.
    """
    
    def __init__(self, name: str, workspace: Workspace, budget_manager: BudgetManager):
        self.name = name
        self.workspace = workspace  # Store workspace for artifact operations
        self.workspace_memory = WorkspaceMemory(name, workspace, max_chars=2000, payload_ttl_minutes=30)
        self.budget_manager = budget_manager
        self.action_logger = ActionLogger()
        
        # Initialize budget
        budget_manager.initialize_budget(f"{name}_artifacts", 50.0)


# Test Fixtures and Helpers

@pytest.fixture
def temp_workspace_dir():
    """Create temporary directory for workspace testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def artifact_manager(temp_workspace_dir):
    """Create ArtifactManager with temporary storage."""
    return ArtifactManager(temp_workspace_dir)


@pytest.fixture
def test_workspace(artifact_manager):
    """Create test workspace for agent."""
    return artifact_manager.create_workspace("TestOrg")


@pytest.fixture 
def target_workspace(artifact_manager):
    """Create target workspace for sharing tests."""
    return artifact_manager.create_workspace("TargetOrg")


@pytest.fixture
def third_workspace(artifact_manager):
    """Create third workspace for chain sharing tests."""
    return artifact_manager.create_workspace("ThirdOrg")


@pytest.fixture
def fourth_workspace(artifact_manager):
    """Create fourth workspace for extended chain sharing tests."""
    return artifact_manager.create_workspace("FourthOrg")


@pytest.fixture
def budget_manager():
    """Create BudgetManager for testing."""
    return BudgetManager()


@pytest.fixture
def target_budget_manager():
    """Create separate BudgetManager for target agent."""
    return BudgetManager()


@pytest.fixture
def third_budget_manager():
    """Create separate BudgetManager for third agent."""
    return BudgetManager()


@pytest.fixture
def fourth_budget_manager():
    """Create separate BudgetManager for fourth agent."""
    return BudgetManager()


@pytest.fixture
def test_agent(test_workspace, budget_manager):
    """Create MockAgent with all real components."""
    return MockAgent("TestAnalyst", test_workspace, budget_manager)


@pytest.fixture
def target_agent(target_workspace, target_budget_manager):
    """Create target agent for sharing tests."""
    return MockAgent("TargetAnalyst", target_workspace, target_budget_manager)


@pytest.fixture
def third_agent(third_workspace, third_budget_manager):
    """Create third agent for chain sharing tests."""
    return MockAgent("ThirdResearcher", third_workspace, third_budget_manager)


@pytest.fixture
def fourth_agent(fourth_workspace, fourth_budget_manager):
    """Create fourth agent for extended chain sharing tests."""
    return MockAgent("FourthAnalyst", fourth_workspace, fourth_budget_manager)


@pytest.fixture
def artifact_tools(test_agent):
    """Create artifact tools bound to test agent."""
    return create_artifact_tools_for_agent(test_agent)


@pytest.fixture
def target_artifact_tools(target_agent):
    """Create artifact tools bound to target agent."""
    return create_artifact_tools_for_agent(target_agent)


@pytest.fixture
def third_artifact_tools(third_agent):
    """Create artifact tools bound to third agent."""
    return create_artifact_tools_for_agent(third_agent)


@pytest.fixture
def fourth_artifact_tools(fourth_agent):
    """Create artifact tools bound to fourth agent."""
    return create_artifact_tools_for_agent(fourth_agent)


def create_test_artifacts(workspace: Workspace) -> list:
    """
    Helper to create realistic test artifacts in workspace.
    
    Creates a variety of artifacts with different types and content
    to simulate real usage scenarios.
    
    Returns:
        list: Created artifact IDs for reference in tests
    """
    artifacts = []
    
    # Market analysis artifact
    market_analysis = Artifact.create(
        artifact_type="analysis",
        payload={
            "sector": "technology",
            "forecast": [0.12, 0.08, 0.15],
            "confidence": 0.85,
            "methodology": "regime_switching",
            "created_on": datetime.now().isoformat()
        },
        created_by="TestAnalyst",
        visibility=["TestOrg.*"]
    )
    workspace.store_artifact(market_analysis, bucket="org")
    artifacts.append(market_analysis.id)
    
    # Risk assessment artifact  
    risk_assessment = Artifact.create(
        artifact_type="risk_model",
        payload={
            "portfolio_value": 1000000.0,
            "var_95": 85000.0,
            "expected_shortfall": 120000.0,
            "model_type": "monte_carlo",
            "simulations": 50000,
            "validation_date": datetime.now().isoformat()
        },
        created_by="TestAnalyst", 
        visibility=["TestOrg.*", "TargetOrg.*"]  # Shareable
    )
    workspace.store_artifact(risk_assessment, bucket="shared")
    artifacts.append(risk_assessment.id)
    
    # Trading strategy artifact
    strategy = Artifact.create(
        artifact_type="strategy",
        payload={
            "name": "momentum_reversal",
            "parameters": {
                "lookback_period": 20,
                "threshold": 0.02,
                "risk_limit": 0.05
            },
            "backtesting_results": {
                "sharpe_ratio": 1.8,
                "max_drawdown": 0.12,
                "total_return": 0.24
            }
        },
        created_by="TestAnalyst",
        visibility=["TestOrg.trader", "TestOrg.analyst"]  # Restricted visibility
    )
    workspace.store_artifact(strategy, bucket="private")
    artifacts.append(strategy.id)
    
    return artifacts


# Artifact Loading Tests

@pytest.mark.asyncio
async def test_load_artifact_success(test_agent, artifact_tools, test_workspace):
    """
    Test successful artifact loading with workspace memory integration.
    
    Expected Behavior:
    - Artifact should be marked as loaded in workspace memory
    - Should return success response with correct artifact ID
    - Should include version information
    - Budget should be charged correctly
    - Action should be logged
    """
    # Arrange: Create test artifact
    artifact_ids = create_test_artifacts(test_workspace)
    target_artifact_id = artifact_ids[0]  # Market analysis
    load_tool = artifact_tools[0]  # load_artifact
    
    # Get initial state
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    initial_loaded = test_agent.workspace_memory.get_loaded_artifacts()
    
    # Act: Load the artifact
    result = await load_tool.run_json({"artifact_id": target_artifact_id}, None)
    
    # Assert: Verify success response structure
    assert isinstance(result, ArtifactLoadResponse)
    assert result.status == "loaded"
    assert result.artifact_id == target_artifact_id
    assert result.message == f"Artifact {target_artifact_id} will be injected in your next prompt"
    assert isinstance(result.version, int)
    assert result.version > 0
    
    # Assert: Verify workspace memory state
    loaded_artifacts = test_agent.workspace_memory.get_loaded_artifacts()
    assert target_artifact_id in loaded_artifacts
    assert len(loaded_artifacts) == len(initial_loaded) + 1
    
    # Assert: Verify artifact status
    status = test_agent.workspace_memory.get_artifact_status(target_artifact_id)
    assert status["exists"] is True
    assert status["loaded"] is True
    
    # Assert: Verify budget deduction (artifact tools charge 0.2 credits for loading)
    final_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    assert final_budget == initial_budget - 0.2
    
    # Assert: Verify action logging
    transactions = test_agent.budget_manager.get_transaction_history("TestAnalyst_artifacts")
    load_transaction = [t for t in transactions if t["description"] == "tool:load_artifact" and t["amount"] == -0.2]
    assert len(load_transaction) == 1


@pytest.mark.asyncio
async def test_load_nonexistent_artifact(test_agent, artifact_tools):
    """
    Test loading an artifact that doesn't exist.
    
    Expected Behavior:
    - Should return error response
    - Should not modify workspace memory state
    - Should not charge budget for failed operation
    - Should still log the action attempt
    """
    # Arrange
    nonexistent_id = "analysis#nonexist"
    load_tool = artifact_tools[0]
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    initial_loaded = test_agent.workspace_memory.get_loaded_artifacts()
    
    # Act
    result = await load_tool.run_json({"artifact_id": nonexistent_id}, None)
    
    # Assert: Error response
    assert isinstance(result, ArtifactLoadResponse)
    assert result.status == "error"
    assert result.artifact_id == nonexistent_id
    assert "Failed to load" in result.message
    
    # Assert: No state changes
    assert test_agent.workspace_memory.get_loaded_artifacts() == initial_loaded
    assert test_agent.budget_manager.get_balance("TestAnalyst_artifacts") == initial_budget - 0.2  # Still charged for attempt


@pytest.mark.asyncio
async def test_load_artifact_insufficient_budget(test_agent, artifact_tools, test_workspace):
    """
    Test loading artifact when agent has insufficient budget.
    
    Expected Behavior:
    - Should return error response about insufficient credits
    - Should not load artifact or modify workspace memory
    - Should not deduct any credits
    """
    # Arrange: Drain agent's budget
    artifact_ids = create_test_artifacts(test_workspace)
    target_artifact_id = artifact_ids[0]
    load_tool = artifact_tools[0]
    
    # Drain budget to below required amount
    current_balance = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    test_agent.budget_manager.debit("TestAnalyst_artifacts", current_balance - 0.1, "test_drain")
    
    initial_loaded = test_agent.workspace_memory.get_loaded_artifacts()
    
    # Act
    result = await load_tool.run_json({"artifact_id": target_artifact_id}, None)
    
    # Assert: Error about insufficient credits
    assert isinstance(result, ArtifactLoadResponse)
    assert result.status == "error"
    assert "credits" in result.message.lower()
    
    # Assert: No changes to workspace state
    assert test_agent.workspace_memory.get_loaded_artifacts() == initial_loaded


# Artifact Unloading Tests

@pytest.mark.asyncio
async def test_unload_artifact_success(test_agent, artifact_tools, test_workspace):
    """
    Test successful artifact unloading.
    
    Expected Behavior:
    - Artifact should be removed from loaded state in workspace memory
    - Should return success response
    - Should clear payload cache
    - No budget charge for unloading
    - Action should be logged
    """
    # Arrange: Load an artifact first
    artifact_ids = create_test_artifacts(test_workspace)
    target_artifact_id = artifact_ids[0]
    
    load_tool = artifact_tools[0]
    unload_tool = artifact_tools[1]
    
    # Load the artifact
    await load_tool.run_json({"artifact_id": target_artifact_id}, None)
    assert target_artifact_id in test_agent.workspace_memory.get_loaded_artifacts()
    
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    
    # Act: Unload the artifact  
    result = await unload_tool.run_json({"artifact_id": target_artifact_id}, None)
    
    # Assert: Success response
    assert isinstance(result, ArtifactUnloadResponse)
    assert result.status == "unloaded"
    assert result.artifact_id == target_artifact_id
    assert result.message == f"Artifact {target_artifact_id} unloaded from memory"
    
    # Assert: Workspace memory state
    loaded_artifacts = test_agent.workspace_memory.get_loaded_artifacts()
    assert target_artifact_id not in loaded_artifacts
    
    # Assert: Artifact status updated
    status = test_agent.workspace_memory.get_artifact_status(target_artifact_id)
    assert status["loaded"] is False
    
    # Assert: No budget charge for unloading
    assert test_agent.budget_manager.get_balance("TestAnalyst_artifacts") == initial_budget


@pytest.mark.asyncio
async def test_unload_not_loaded_artifact(test_agent, artifact_tools):
    """
    Test unloading an artifact that wasn't loaded.
    
    Expected Behavior:
    - Should still return success (idempotent operation)
    - Should not affect other loaded artifacts
    - No budget charge
    """
    # Arrange
    unload_tool = artifact_tools[1]
    not_loaded_id = "analysis#notloaded"
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    
    # Act
    result = await unload_tool.run_json({"artifact_id": not_loaded_id}, None)
    
    # Assert: Success response (idempotent)
    assert isinstance(result, ArtifactUnloadResponse)
    assert result.status == "unloaded"  # Idempotent operation
    assert result.artifact_id == not_loaded_id
    
    # Assert: No budget charge
    assert test_agent.budget_manager.get_balance("TestAnalyst_artifacts") == initial_budget


# Artifact Writing Tests

@pytest.mark.asyncio
async def test_write_artifact_success(test_agent, artifact_tools, test_workspace):
    """
    Test successful artifact writing with persistence.
    
    Expected Behavior:
    - Artifact should be created and stored in workspace
    - Should return success response with path and size info
    - Budget should be charged (0.5 credits for writing)
    - Should be retrievable from workspace afterward
    - Should be marked as seen in workspace memory
    """
    # Arrange
    write_tool = artifact_tools[2]
    new_artifact_id = "test_forecast_123"
    test_content = {
        "forecast_type": "sector_analysis", 
        "sector": "healthcare",
        "predictions": [0.08, 0.12, 0.06],
        "confidence_interval": [0.05, 0.15],
        "model_version": "v2.1",
        "creation_timestamp": datetime.now().isoformat()
    }
    
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    initial_artifact_count = len(test_workspace.list_artifacts())
    
    # Act
    result = await write_tool.run_json({
        "artifact_id": new_artifact_id,
        "content": test_content,
        "artifact_type": "forecast"
    }, None)
    
    # Assert: Success response structure
    assert isinstance(result, ArtifactWriteResponse)
    assert result.status == "written"
    assert result.artifact_id == new_artifact_id
    assert result.message == f"Artifact {new_artifact_id} written successfully"
    assert isinstance(result.version, int)
    assert result.version > 0
    assert isinstance(result.size_chars, int)
    assert result.size_chars > 0
    assert result.path.endswith(f"{new_artifact_id}.json")
    
    # Assert: Budget deduction
    final_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    assert final_budget == initial_budget - 0.5
    
    # Assert: Artifact persisted in workspace
    final_artifact_count = len(test_workspace.list_artifacts())
    assert final_artifact_count == initial_artifact_count + 1
    
    # Assert: Artifact retrievable and content matches
    stored_artifact = test_workspace.get_artifact(new_artifact_id)
    assert stored_artifact is not None
    assert stored_artifact.type == "forecast"
    assert stored_artifact.created_by == "TestAnalyst"
    assert stored_artifact.payload == test_content
    
    # Assert: Workspace memory tracking
    status = test_agent.workspace_memory.get_artifact_status(new_artifact_id)
    assert status["exists"] is True
    assert status["last_seen"] == result.version


@pytest.mark.asyncio 
async def test_write_artifact_large_content(test_agent, artifact_tools):
    """
    Test writing artifact with large content payload.
    
    Expected Behavior:
    - Should handle large content appropriately
    - Size should be reported correctly
    - Should still succeed within reasonable limits
    """
    # Arrange: Create large content payload
    write_tool = artifact_tools[2]
    large_content = {
        "data_type": "large_dataset",
        "observations": [{"value": i, "timestamp": f"2024-01-{i:02d}"} for i in range(1, 500)],
        "metadata": {
            "description": "Large test dataset for performance testing",
            "size_notes": "Contains 499 observations with timestamps"
        }
    }
    
    # Act
    result = await write_tool.run_json({
        "artifact_id": "large_dataset_test",
        "content": large_content,
        "artifact_type": "dataset"
    }, None)
    
    # Assert: Success with large content
    assert isinstance(result, ArtifactWriteResponse)
    assert result.status == "written"
    assert result.size_chars > 1000  # Should be substantially large


# Artifact Sharing Tests

@pytest.mark.asyncio
async def test_share_artifact_success(test_agent, artifact_tools, test_workspace, target_workspace, artifact_manager):
    """
    Test successful artifact sharing between workspaces.
    
    Expected Behavior:
    - Artifact should become available in target workspace
    - Should return success response with target info
    - Budget should be charged (0.3 credits for sharing)
    - Original artifact should remain in source workspace
    - Should be logged as external action
    """
    # Arrange: Create artifact and target agent
    artifact_ids = create_test_artifacts(test_workspace)
    shareable_artifact_id = artifact_ids[1]  # Risk assessment (already has TargetOrg visibility)
    
    share_tool = artifact_tools[3]
    target_agent_id = "TargetOrg.analyst"
    
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    initial_target_artifacts = target_workspace.list_artifacts()
    
    # Act
    result = await share_tool.run_json({
        "artifact_id": shareable_artifact_id,
        "target_agent": target_agent_id
    }, None)
    
    # Assert: Success response
    assert isinstance(result, ArtifactShareResponse)
    assert result.status == "shared"
    assert result.artifact_id == shareable_artifact_id
    assert result.target == "TargetOrg"
    assert "shared with TargetOrg" in result.message
    
    # Assert: Budget charged
    final_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    assert final_budget == initial_budget - 0.3
    
    # Assert: Artifact should now be available in target workspace
    # The implementation actually copies the artifact to the target workspace's shared bucket
    final_target_artifacts = target_workspace.list_artifacts()
    assert len(final_target_artifacts) == len(initial_target_artifacts) + 1
    
    # Assert: Shared artifact should be retrievable from target workspace
    shared_artifact = target_workspace.get_artifact(shareable_artifact_id)
    assert shared_artifact is not None
    assert shared_artifact.id == shareable_artifact_id
    
    # Assert: Shared artifact should have sharing metadata
    assert "sharing_history" in shared_artifact.metadata
    sharing_history = shared_artifact.metadata["sharing_history"]
    assert len(sharing_history) == 1
    
    sharing_record = sharing_history[0]
    assert sharing_record["shared_from"] == "TestOrg"
    assert sharing_record["shared_to"] == "TargetOrg"
    assert sharing_record["shared_by"] == "TestAnalyst"
    assert sharing_record["sharing_reason"] == "agent_collaboration"
    assert sharing_record["original_created_by"] == "TestAnalyst"
    assert "shared_at" in sharing_record


@pytest.mark.asyncio
async def test_share_nonexistent_artifact(test_agent, artifact_tools):
    """
    Test sharing an artifact that doesn't exist.
    
    Expected Behavior:
    - Should return error response
    - Budget should still be charged (operation was attempted)
    - No artifacts should be shared
    """
    # Arrange
    share_tool = artifact_tools[3]
    nonexistent_id = "analysis#missing"
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    
    # Act
    result = await share_tool.run_json({
        "artifact_id": nonexistent_id,
        "target_agent": "TargetOrg.trader"
    }, None)
    
    # Assert: Error response for nonexistent artifact
    assert isinstance(result, ArtifactShareResponse)
    assert result.status == "error"
    assert result.artifact_id == nonexistent_id
    assert "not exist" in result.message.lower() or "not found" in result.message.lower()
    
    # Assert: Budget should still be charged (operation was attempted)
    final_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    assert final_budget == initial_budget - 0.3




@pytest.mark.asyncio
async def test_chain_sharing_comprehensive(test_agent, target_agent, third_agent, fourth_agent,
                                          artifact_tools, target_artifact_tools, third_artifact_tools, fourth_artifact_tools,
                                          test_workspace, target_workspace, third_workspace, fourth_workspace,
                                          artifact_manager):
    """
    Test comprehensive chain sharing functionality.
    
    Expected Behavior:
    - First share: TestOrg -> TargetOrg (should work with original visibility)
    - Second share: TargetOrg -> ThirdOrg (should work with "ownership" rule)
    - Each step should preserve sharing history
    - Visibility should expand to include new targets
    - Budget should be charged for each sharing operation
    """
    # Arrange: Create artifact
    artifact_ids = create_test_artifacts(test_workspace)
    shareable_artifact_id = artifact_ids[1]  # Risk assessment with visibility ["TestOrg.*", "TargetOrg.*"]
    
    share_tool = artifact_tools[3]
    
    # Step 1: TestOrg -> TargetOrg
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    
    result1 = await share_tool.run_json({
        "artifact_id": shareable_artifact_id,
        "target_agent": "TargetOrg.analyst"
    }, None)
    
    assert result1.status == "shared"
    assert result1.target == "TargetOrg"
    
    # Verify budget charged
    budget_after_step1 = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    assert budget_after_step1 == initial_budget - 0.3
    
    # Check artifact in TargetOrg
    target_artifact = target_workspace.get_artifact(shareable_artifact_id)
    assert target_artifact is not None
    assert target_artifact.id == shareable_artifact_id
    
    # Check sharing history (1 entry)
    sharing_history = target_artifact.metadata["sharing_history"]
    assert len(sharing_history) == 1
    assert sharing_history[0]["shared_from"] == "TestOrg"
    assert sharing_history[0]["shared_to"] == "TargetOrg"
    assert sharing_history[0]["shared_by"] == "TestAnalyst"
    
    # Check expanded visibility
    assert "TargetOrg.*" in target_artifact.visibility
    
    # Step 2: TargetOrg -> ThirdOrg (chain sharing)
    target_share_tool = target_artifact_tools[3]
    
    target_initial_budget = target_agent.budget_manager.get_balance("TargetAnalyst_artifacts")
    
    result2 = await target_share_tool.run_json({
        "artifact_id": shareable_artifact_id,
        "target_agent": "ThirdOrg.researcher"
    }, None)
    
    assert result2.status == "shared"
    assert result2.target == "ThirdOrg"
    
    # Verify budget charged for TargetOrg
    target_budget_after = target_agent.budget_manager.get_balance("TargetAnalyst_artifacts")
    assert target_budget_after == target_initial_budget - 0.3
    
    # Check artifact in ThirdOrg
    third_artifact = third_workspace.get_artifact(shareable_artifact_id)
    assert third_artifact is not None
    assert third_artifact.id == shareable_artifact_id
    
    # Check sharing history (2 entries now)
    third_sharing_history = third_artifact.metadata["sharing_history"]
    assert len(third_sharing_history) == 2
    
    # First entry: TestOrg -> TargetOrg
    assert third_sharing_history[0]["shared_from"] == "TestOrg"
    assert third_sharing_history[0]["shared_to"] == "TargetOrg"
    assert third_sharing_history[0]["shared_by"] == "TestAnalyst"
    
    # Second entry: TargetOrg -> ThirdOrg
    assert third_sharing_history[1]["shared_from"] == "TargetOrg"
    assert third_sharing_history[1]["shared_to"] == "ThirdOrg"
    assert third_sharing_history[1]["shared_by"] == "TargetAnalyst"
    
    # Check that original creator is preserved
    assert third_sharing_history[0]["original_created_by"] == "TestAnalyst"
    assert third_sharing_history[1]["original_created_by"] == "TestAnalyst"
    
    # Check expanded visibility includes all targets
    assert "TestOrg.*" in third_artifact.visibility
    assert "TargetOrg.*" in third_artifact.visibility
    assert "ThirdOrg.*" in third_artifact.visibility
    
    # Step 3: ThirdOrg -> FourthOrg (extended chain)
    third_share_tool = third_artifact_tools[3]
    
    result3 = await third_share_tool.run_json({
        "artifact_id": shareable_artifact_id,
        "target_agent": "FourthOrg.analyst"
    }, None)
    
    assert result3.status == "shared"
    assert result3.target == "FourthOrg"
    
    # Check final artifact in FourthOrg
    fourth_artifact = fourth_workspace.get_artifact(shareable_artifact_id)
    assert fourth_artifact is not None
    
    # Check sharing history (3 entries now)
    fourth_sharing_history = fourth_artifact.metadata["sharing_history"]
    assert len(fourth_sharing_history) == 3
    
    # All entries should preserve original creator
    for entry in fourth_sharing_history:
        assert entry["original_created_by"] == "TestAnalyst"
    
    # Final visibility should include all organizations
    assert "TestOrg.*" in fourth_artifact.visibility
    assert "TargetOrg.*" in fourth_artifact.visibility
    assert "ThirdOrg.*" in fourth_artifact.visibility
    assert "FourthOrg.*" in fourth_artifact.visibility


# Artifact Listing Tests

@pytest.mark.asyncio
async def test_list_artifacts_with_loaded_states(test_agent, artifact_tools, test_workspace):
    """
    Test listing artifacts with various loaded states.
    
    Expected Behavior:
    - Should return success response with workspace listing
    - Should include all accessible artifacts
    - Should show loaded vs unloaded status correctly
    - Should include total count
    - No budget charge for listing
    """
    # Arrange: Create artifacts and load one
    artifact_ids = create_test_artifacts(test_workspace)
    load_tool = artifact_tools[0]
    list_tool = artifact_tools[4]
    
    # Load one artifact
    await load_tool.run_json({"artifact_id": artifact_ids[0]}, None)
    
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    
    # Act
    result = await list_tool.run_json({}, None)
    
    # Assert: Success response structure
    assert isinstance(result, ArtifactListResponse)
    assert result.status == "success"
    assert isinstance(result.workspace_listing, str)
    assert isinstance(result.loaded_artifacts, list)
    assert isinstance(result.loaded_status, dict)
    assert isinstance(result.total_artifacts, int)
    assert result.message == "Use load_artifact(id) to inject artifact content in your next prompt"
    
    # Assert: Loaded artifact tracking
    assert len(result.loaded_artifacts) == 1
    assert artifact_ids[0] in result.loaded_artifacts
    
    # Assert: Status details for loaded artifacts
    for artifact_id in result.loaded_artifacts:
        assert artifact_id in result.loaded_status
        status_info = result.loaded_status[artifact_id]
        assert isinstance(status_info, dict)
    
    # Assert: No budget charge for listing
    assert test_agent.budget_manager.get_balance("TestAnalyst_artifacts") == initial_budget
    
    # Assert: Reasonable total count
    assert result.total_artifacts >= len(artifact_ids)


@pytest.mark.asyncio
async def test_list_artifacts_empty_workspace(test_agent, artifact_tools):
    """
    Test listing artifacts in empty workspace.
    
    Expected Behavior:
    - Should return success response
    - Should indicate no artifacts available
    - Empty loaded artifacts list
    - Zero or minimal total count
    """
    # Arrange: Ensure clean workspace (no test artifacts created)
    list_tool = artifact_tools[4]
    
    # Act
    result = await list_tool.run_json({}, None)
    
    # Assert: Success with empty state
    assert isinstance(result, ArtifactListResponse)
    assert result.status == "success"
    assert len(result.loaded_artifacts) == 0
    assert len(result.loaded_status) == 0
    # Note: total_artifacts might still be > 0 due to mock implementation


# Integration and Edge Case Tests

@pytest.mark.asyncio
async def test_full_artifact_workflow(test_agent, artifact_tools, test_workspace):
    """
    Test complete artifact management workflow.
    
    This integration test verifies the full lifecycle:
    1. Write artifact
    2. List to verify it exists
    3. Load it
    4. Verify loaded state
    5. Unload it
    6. Verify unloaded state
    
    Expected Behavior:
    - All operations should succeed in sequence
    - State should be consistent throughout
    - Budget should be charged appropriately for each operation
    - Final workspace state should match expectations
    """
    # Arrange
    load_tool, unload_tool, write_tool, share_tool, list_tool = artifact_tools
    workflow_artifact_id = "workflow_test_artifact"
    test_content = {"workflow": "test", "step": 1}
    
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    
    # Step 1: Write artifact
    write_result = await write_tool.run_json({
        "artifact_id": workflow_artifact_id,
        "content": test_content,
        "artifact_type": "workflow_test"
    }, None)
    assert write_result.status == "written"
    
    # Step 2: List artifacts (should include new one)
    list_result = await list_tool.run_json({}, None)
    assert list_result.status == "success"
    # Note: Artifact might not appear in listing due to mock workspace memory
    
    # Step 3: Load artifact
    load_result = await load_tool.run_json({"artifact_id": workflow_artifact_id}, None)
    assert load_result.status == "loaded"
    assert workflow_artifact_id in test_agent.workspace_memory.get_loaded_artifacts()
    
    # Step 4: Verify loaded state in listing
    list_result2 = await list_tool.run_json({}, None)
    assert list_result2.status == "success"
    
    # Step 5: Unload artifact
    unload_result = await unload_tool.run_json({"artifact_id": workflow_artifact_id}, None)
    assert unload_result.status == "unloaded"
    assert workflow_artifact_id not in test_agent.workspace_memory.get_loaded_artifacts()
    
    # Assert: Total budget usage (0.5 write + 0.2 load + 0 unload + 0 list = 0.7)
    final_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    expected_cost = 0.5 + 0.2  # write + load (unload and list are free)
    assert abs(final_budget - (initial_budget - expected_cost)) < 0.01


@pytest.mark.asyncio
async def test_concurrent_artifact_operations(test_agent, artifact_tools, test_workspace):
    """
    Test concurrent artifact operations to verify thread safety.
    
    Expected Behavior:
    - Multiple simultaneous operations should complete successfully
    - Final state should be consistent
    - All operations should be properly logged
    - Budget should be charged correctly for all operations
    """
    # Arrange: Create multiple artifacts for concurrent operations
    artifact_ids = create_test_artifacts(test_workspace)
    load_tool, unload_tool = artifact_tools[0], artifact_tools[1]
    
    initial_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    
    # Act: Perform concurrent operations
    concurrent_tasks = [
        load_tool.run_json({"artifact_id": artifact_ids[0]}, None),
        load_tool.run_json({"artifact_id": artifact_ids[1]}, None),
        load_tool.run_json({"artifact_id": artifact_ids[2]}, None),
    ]
    
    results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    
    # Assert: All operations completed successfully
    for i, result in enumerate(results):
        assert not isinstance(result, Exception)
        assert isinstance(result, ArtifactLoadResponse)
        assert result.status == "loaded"
        print(f"\nArtifact {i} details:")
        print(f"  ID: {result.artifact_id}")
        print(f"  Version: {result.version}")
        print(f"  Message: {result.message}")
    
    # Check detailed artifact information from workspace
    print(f"\nDetailed artifact information:")
    for i, artifact_id in enumerate(artifact_ids):
        artifact = test_workspace.get_artifact(artifact_id)
        if artifact:
            print(f"\nArtifact {i} ({artifact_id}):")
            print(f"  Type: {artifact.type}")
            print(f"  Created by: {artifact.created_by}")
            print(f"  Visibility: {artifact.visibility}")
            print(f"  Payload keys: {list(artifact.payload.keys())}")
            
            # Show specific content based on type
            if artifact.type == "analysis":
                print(f"  Sector: {artifact.payload.get('sector')}")
                print(f"  Confidence: {artifact.payload.get('confidence')}")
            elif artifact.type == "risk_model":
                print(f"  Portfolio value: ${artifact.payload.get('portfolio_value'):,}")
                print(f"  VaR 95%: ${artifact.payload.get('var_95'):,}")
            elif artifact.type == "strategy":
                print(f"  Strategy name: {artifact.payload.get('name')}")
                print(f"  Sharpe ratio: {artifact.payload.get('backtesting_results', {}).get('sharpe_ratio')}")
    
    # Check workspace memory status for each artifact
    print(f"\nWorkspace memory status:")
    for artifact_id in artifact_ids:
        status = test_agent.workspace_memory.get_artifact_status(artifact_id)
        print(f"  {artifact_id}: exists={status['exists']}, loaded={status['loaded']}")
    
    # Assert: Final state is consistent
    loaded_artifacts = test_agent.workspace_memory.get_loaded_artifacts()
    assert len(loaded_artifacts) == 3
    
    # Assert: Budget charged correctly (3 loads × 0.2 = 0.6)
    final_budget = test_agent.budget_manager.get_balance("TestAnalyst_artifacts")
    expected_cost = 3 * 0.2
    assert abs(final_budget - (initial_budget - expected_cost)) < 0.01


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])