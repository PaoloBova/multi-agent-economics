"""
Artifact tool factory - simplified wrappers that only handle budget and logging.

Wrappers call implementations directly and return their results unchanged.
"""

from typing import Dict, List, Any, Optional
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool

from .implementations.artifacts import (
    load_artifact_impl, unload_artifact_impl, write_artifact_impl,
    share_artifact_impl, list_artifacts_impl
)
from .schemas import (
    ArtifactLoadResponse, ArtifactUnloadResponse, ArtifactWriteResponse,
    ArtifactShareResponse, ArtifactListResponse
)
from ..core.actions import InternalAction


def create_artifact_tools_for_agent(agent) -> List[FunctionTool]:
    """Create artifact tools with simple wrappers."""
    
    # Tool 1: Load Artifact
    async def load_artifact(
        artifact_id: Annotated[str, "ID of the artifact to load into memory"]
    ) -> ArtifactLoadResponse:
        """Load an artifact into memory for injection in next prompt."""
        
        # Budget management (simple credit deduction)
        budget_manager = getattr(agent, 'budget_manager', None)
        if budget_manager:
            # Use agent-specific budget category
            budget_category = f"{agent.name}_artifacts"
            success = budget_manager.charge_credits(budget_category, 0.2, "tool:load_artifact")
            if not success:
                return ArtifactLoadResponse(
                    status="error",
                    artifact_id=artifact_id, 
                    message="Insufficient credits"
                )
        
        # Record action
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action = InternalAction(
                actor=agent.name,
                action="load_artifact",
                inputs={"artifact_id": artifact_id},
                tool="load_artifact",
                cost=0.2
            )
            action_logger.log_internal_action(action)
        
        # Call implementation - it handles ALL the work
        return load_artifact_impl(agent, artifact_id)
    
    
    # Tool 2: Unload Artifact
    async def unload_artifact(
        artifact_id: Annotated[str, "ID of the artifact to unload from memory"]
    ) -> ArtifactUnloadResponse:
        """Unload an artifact from memory."""
        
        # Record action (no cost for unloading)
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action = InternalAction(
                actor=agent.name,
                action="unload_artifact",
                inputs={"artifact_id": artifact_id},
                tool="unload_artifact",
                cost=0.0
            )
            action_logger.log_internal_action(action)
        
        # Call implementation
        return unload_artifact_impl(agent, artifact_id)
    
    
    # Tool 3: Write Artifact
    async def write_artifact(
        artifact_id: Annotated[str, "ID for the new or updated artifact"],
        content: Annotated[Dict[str, Any], "Content/payload of the artifact"],
        artifact_type: Annotated[str, "Type of artifact"] = "analysis"
    ) -> ArtifactWriteResponse:
        """Write or update an artifact in the workspace."""
        
        # Budget management
        budget_manager = getattr(agent, 'budget_manager', None)
        if budget_manager:
            # Use agent-specific budget category
            budget_category = f"{agent.name}_artifacts"
            success = budget_manager.charge_credits(budget_category, 0.5, "tool:write_artifact")
            if not success:
                return ArtifactWriteResponse(
                    status="error",
                    artifact_id=artifact_id,
                    message="Insufficient credits"
                )
        
        # Record action
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action = InternalAction(
                actor=agent.name,
                action="write_artifact",
                inputs={"artifact_id": artifact_id, "artifact_type": artifact_type},
                tool="write_artifact",
                cost=0.5
            )
            action_logger.log_internal_action(action)
        
        # Call implementation
        return write_artifact_impl(agent, artifact_id, content, artifact_type)
    
    
    # Tool 4: Share Artifact
    async def share_artifact(
        artifact_id: Annotated[str, "ID of the artifact to share"],
        target_agent: Annotated[str, "Agent to share with (format: org.role)"]
    ) -> ArtifactShareResponse:
        """Share an artifact with another agent."""
        
        # Budget management
        budget_manager = getattr(agent, 'budget_manager', None)
        if budget_manager:
            # Use agent-specific budget category
            budget_category = f"{agent.name}_artifacts"
            success = budget_manager.charge_credits(budget_category, 0.3, "tool:share_artifact")
            if not success:
                return ArtifactShareResponse(
                    status="error",
                    artifact_id=artifact_id,
                    message="Insufficient credits"
                )
        
        # Record action
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action = InternalAction(
                actor=agent.name,
                action="share_artifact",
                inputs={"artifact_id": artifact_id, "target_agent": target_agent},
                tool="share_artifact",
                cost=0.3
            )
            action_logger.log_internal_action(action)
        
        # Call implementation
        return share_artifact_impl(agent, artifact_id, target_agent)
    
    
    # Tool 5: List Artifacts
    async def list_artifacts() -> ArtifactListResponse:
        """List all artifacts available in the workspace."""
        
        # Record action (no cost for listing)
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action = InternalAction(
                actor=agent.name,
                action="list_artifacts",
                inputs={},
                tool="list_artifacts",
                cost=0.0
            )
            action_logger.log_internal_action(action)
        
        # Call implementation
        return list_artifacts_impl(agent)
    
    
    # Return tools
    return [
        FunctionTool(load_artifact, description="Load artifact into memory"),
        FunctionTool(unload_artifact, description="Unload artifact from memory"),
        FunctionTool(write_artifact, description="Write artifact to workspace"),
        FunctionTool(share_artifact, description="Share artifact with another agent"),
        FunctionTool(list_artifacts, description="List available artifacts")
    ]