"""
Unit tests for WorkspaceMemory and artifact tools functionality.
"""

import unittest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from multi_agent_economics.core import (
    Workspace, Artifact, WorkspaceMemory, BudgetManager, ActionLogger,
    load_artifact, unload_artifact, write_artifact, share_artifact, list_artifacts
)


class MockAgent:
    """Mock agent for testing context-based tools."""
    def __init__(self, workspace_memory=None, budget_manager=None, action_logger=None):
        self.workspace_memory = workspace_memory
        self.budget_manager = budget_manager
        self.action_logger = action_logger


class TestWorkspaceMemory(unittest.TestCase):
    """Test the WorkspaceMemory class and artifact tools."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test workspace
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Workspace(workspace_id="test", workspace_dir=self.temp_dir)
        
        # Create test memory instance
        self.memory = WorkspaceMemory(
            name="test_agent", 
            workspace=self.workspace,
            max_chars=100,
            payload_ttl_minutes=5
        )
        
        # Create mock dependencies
        self.budget_manager = Mock()
        self.budget_manager.charge_credits = Mock(return_value=True)
        self.action_logger = Mock()
        self.action_logger.log_internal_action = Mock(return_value=True)
        
        # Create mock agent with dependencies
        self.agent = MockAgent(
            workspace_memory=self.memory,
            budget_manager=self.budget_manager,
            action_logger=self.action_logger
        )
        
        # Create test context
        self.context = {"caller": self.agent}

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_memory_initialization(self):
        """Test WorkspaceMemory initialization."""
        self.assertEqual(self.memory.name, "test_agent")
        self.assertEqual(self.memory.workspace, self.workspace)
        self.assertEqual(self.memory.max_chars, 100)
        self.assertEqual(self.memory.meta, {})
        self.assertEqual(self.memory.payload_cache, {})

    def test_clear_memory(self):
        """Test clearing memory state."""
        # Add some test data
        self.memory.meta["test"] = {"last_seen": 123, "loaded": True}
        self.memory.payload_cache["test"] = ("payload", datetime.now())
        
        # Clear and verify
        self.memory.clear()
        self.assertEqual(self.memory.meta, {})
        self.assertEqual(self.memory.payload_cache, {})

    def test_build_context_additions_empty_workspace(self):
        """Test context building with empty workspace."""
        additions = self.memory.build_context_additions()
        self.assertEqual(additions, [])

    def test_load_unload_artifact_memory_operations(self):
        """Test loading and unloading artifacts in memory."""
        # Test loading
        result = self.memory.load_artifact("test_artifact")
        self.assertTrue(result)
        self.assertTrue(self.memory.meta["test_artifact"]["loaded"])
        
        # Test unloading
        result = self.memory.unload_artifact("test_artifact")
        self.assertTrue(result)
        self.assertFalse(self.memory.meta["test_artifact"]["loaded"])

    def test_mark_artifact_seen(self):
        """Test marking artifacts as seen."""
        self.memory.mark_artifact_seen("test_artifact", 12345)
        self.assertEqual(self.memory.meta["test_artifact"]["last_seen"], 12345)
        self.assertFalse(self.memory.meta["test_artifact"]["loaded"])

    def test_get_loaded_artifacts(self):
        """Test getting list of loaded artifacts."""
        # Load some artifacts
        self.memory.load_artifact("artifact1")
        self.memory.load_artifact("artifact2")
        self.memory.load_artifact("artifact3")
        
        # Unload one
        self.memory.unload_artifact("artifact2")
        
        loaded = self.memory.get_loaded_artifacts()
        self.assertEqual(set(loaded), {"artifact1", "artifact3"})

    def test_get_artifact_status_nonexistent(self):
        """Test getting status of non-existent artifact."""
        status = self.memory.get_artifact_status("nonexistent")
        self.assertFalse(status["exists"])

    def test_get_artifact_status_existing(self):
        """Test getting status of existing artifact."""
        self.memory.load_artifact("test_artifact")
        status = self.memory.get_artifact_status("test_artifact")
        
        self.assertTrue(status["exists"])
        self.assertTrue(status["loaded"])
        self.assertEqual(status["last_seen"], -1)
        self.assertFalse(status["cached"])

    def test_build_context_with_loaded_artifact(self):
        """Test context building with a loaded artifact."""
        # Create a test artifact
        artifact = Artifact.create(
            artifact_type="test",
            payload={"message": "test content"},
            created_by="test_agent",
            visibility=["test.*"]
        )
        
        # Store in workspace
        artifact_path = self.workspace.store_artifact(artifact, bucket="private")
        artifact_id = os.path.basename(artifact_path).replace('.json', '')
        
        # Load in memory
        self.memory.load_artifact(artifact_id)
        
        # Build context
        additions = self.memory.build_context_additions()
        
        # Should have workspace listing and artifact payload
        self.assertEqual(len(additions), 2)
        self.assertTrue(additions[0].startswith("[workspace]"))
        self.assertTrue(additions[1].startswith(f"[artifact:{artifact_id}@v"))

    def test_payload_truncation(self):
        """Test that long payloads are truncated."""
        # Create artifact with long content
        long_content = "x" * 200  # Longer than max_chars (100)
        artifact = Artifact.create(
            artifact_type="test",
            payload={"message": long_content},
            created_by="test_agent",
            visibility=["test.*"]
        )
        
        # Store and load
        artifact_path = self.workspace.store_artifact(artifact, bucket="private")
        artifact_id = os.path.basename(artifact_path).replace('.json', '')
        self.memory.load_artifact(artifact_id)
        
        # Build context
        additions = self.memory.build_context_additions()
        
        # Find the artifact payload line
        payload_line = None
        for line in additions:
            if line.startswith(f"[artifact:{artifact_id}"):
                payload_line = line
                break
        
        self.assertIsNotNone(payload_line)
        self.assertTrue("TRUNCATED" in payload_line)

    def test_version_detection(self):
        """Test that version changes are detected."""
        # Create and store an artifact
        artifact = Artifact.create(
            artifact_type="test",
            payload={"version": 1},
            created_by="test_agent",
            visibility=["test.*"]
        )
        
        artifact_path = self.workspace.store_artifact(artifact, bucket="private")
        artifact_id = os.path.basename(artifact_path).replace('.json', '')
        
        # Load and mark as seen
        self.memory.load_artifact(artifact_id)
        
        # First build should show artifact as new (*)
        additions = self.memory.build_context_additions()
        workspace_line = additions[0]
        self.assertTrue("*" in workspace_line)

    # Tool function tests with context

    def test_load_artifact_tool_success(self):
        """Test load_artifact tool function."""
        result = load_artifact("test_artifact", self.context)
        
        self.assertEqual(result["status"], "loaded")
        self.assertEqual(result["artifact_id"], "test_artifact")
        self.budget_manager.charge_credits.assert_called_once_with("test", 0.2)
        self.action_logger.log_internal_action.assert_called_once()

    def test_load_artifact_tool_no_context(self):
        """Test load_artifact tool without context."""
        result = load_artifact("test_artifact")
        self.assertEqual(result["status"], "error")
        self.assertIn("No agent context", result["message"])

    def test_load_artifact_tool_insufficient_credits(self):
        """Test load_artifact tool with insufficient credits."""
        self.budget_manager.charge_credits.side_effect = Exception("Insufficient credits")
        
        result = load_artifact("test_artifact", self.context)
        self.assertEqual(result["status"], "error")
        self.assertIn("Insufficient credits", result["message"])

    def test_unload_artifact_tool_success(self):
        """Test unload_artifact tool function."""
        # First load an artifact
        self.memory.load_artifact("test_artifact")
        
        # Then unload it
        result = unload_artifact("test_artifact", self.context)
        
        self.assertEqual(result["status"], "unloaded")
        self.assertEqual(result["artifact_id"], "test_artifact")
        self.action_logger.log_internal_action.assert_called()

    def test_list_artifacts_tool(self):
        """Test list_artifacts tool function."""
        result = list_artifacts(self.context)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("workspace_listing", result)
        self.assertIn("loaded_artifacts", result)
        self.assertIn("message", result)

    def test_write_artifact_tool_success(self):
        """Test write_artifact tool function."""
        content = {"analysis": "test results", "confidence": 0.8}
        
        result = write_artifact("test_analysis", content, "analysis", None, self.context)
        
        self.assertEqual(result["status"], "written")
        self.assertEqual(result["artifact_id"], "test_analysis")
        self.budget_manager.charge_credits.assert_called_with("test", 0.5)
        self.action_logger.log_internal_action.assert_called()

    def test_share_artifact_tool_no_artifact(self):
        """Test share_artifact tool with non-existent artifact."""
        result = share_artifact("nonexistent", "target_org", self.context)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def test_tool_with_no_workspace_memory(self):
        """Test tool behavior when agent has no workspace memory."""
        agent_no_memory = MockAgent()
        context_no_memory = {"caller": agent_no_memory}
        
        result = load_artifact("test", context_no_memory)
        self.assertEqual(result["status"], "error")
        self.assertIn("no workspace memory", result["message"])


if __name__ == '__main__':
    unittest.main()
