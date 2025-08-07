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


def create_artifact_tools_for_agent(agent, context={}) -> List[FunctionTool]:
    """Create artifact tools with simple wrappers."""
    
    DEFAULT_COSTS = {
        "tool:load_artifact": 0.2,
        "tool:write_artifact": 0.5,
        "tool:share_artifact": 0.3,
        "tool:unload_artifact": 0.0,
        "tool:list_artifacts": 0.0
    }

    def _handle_budget_and_logging(agent, context, tool_name, inputs, 
    error_response_class, **error_kwargs):
        """Handle budget charging and action logging with consistent error 
    handling."""

        # Budget management
        budget_manager = context.get('budget_manager', None)
        budget_cost = context.get('budget_costs', {}).get(tool_name, DEFAULT_COSTS.get(tool_name, 0.0))

        if budget_manager and budget_cost > 0:
            budget_category = f"{agent.name}_artifacts"
            success = budget_manager.charge_credits(budget_category, budget_cost, tool_name)
            if not success:
                return error_response_class(
                    status="error",
                    message="Insufficient credits",
                    **error_kwargs
                )

        # Action logging
        action_logger = context.get('action_logger', None)
        if action_logger:
            action = InternalAction(
                actor=agent.name,
                action=tool_name.replace("tool:", ""),
                inputs=inputs,
                tool=tool_name.replace("tool:", ""),
                cost=budget_cost
            )
            action_logger.log_internal_action(action)

        return None  # No error

    # Tool 1: Load Artifact
    async def load_artifact(
        artifact_id: Annotated[str, "ID of the artifact to load into memory"]
    ) -> ArtifactLoadResponse:
        """Load an artifact into memory for injection in next prompt."""
        # Handle budget and logging
        error_response = _handle_budget_and_logging(
            agent,
            context,
            "tool:load_artifact",
            {"artifact_id": artifact_id},
            ArtifactLoadResponse,
            artifact_id=artifact_id
        )
        if error_response:
            return error_response

        # Call implementation - it handles ALL the work
        return load_artifact_impl(agent, artifact_id)
    
    
    # Tool 2: Unload Artifact
    async def unload_artifact(
        artifact_id: Annotated[str, "ID of the artifact to unload from memory"]
    ) -> ArtifactUnloadResponse:
        """Unload an artifact from memory."""
        # Handle budget and logging
        error_response = _handle_budget_and_logging(
            agent,
            context,
            "tool:unload_artifact",
            {"artifact_id": artifact_id},
            ArtifactUnloadResponse,
            artifact_id=artifact_id
        )
        if error_response:
            return error_response

        # Call implementation
        return unload_artifact_impl(agent, artifact_id)
    
    
    # Tool 3: Write Artifact
    async def write_artifact(
        artifact_id: Annotated[str, "ID for the new or updated artifact"],
        content: Annotated[Dict[str, Any], "Content/payload of the artifact"],
        artifact_type: Annotated[str, "Type of artifact"] = "analysis"
    ) -> ArtifactWriteResponse:
        """Write or update an artifact in the workspace."""
        # Handle budget and logging
        error_response = _handle_budget_and_logging(
            agent,
            context,
            "tool:write_artifact",
            {
                "artifact_id": artifact_id,
                "content": content,
                "artifact_type": artifact_type
            },
            ArtifactWriteResponse,
            artifact_id=artifact_id
        )
        if error_response:
            return error_response
        # Call implementation
        return write_artifact_impl(agent, artifact_id, content, artifact_type)
    
    
    # Tool 4: Share Artifact
    async def share_artifact(
        artifact_id: Annotated[str, "ID of the artifact to share"],
        target_agent: Annotated[str, "Agent to share with (format: org.role)"]
    ) -> ArtifactShareResponse:
        """Share an artifact with another agent."""
        
        # Handle budget and logging
        error_response = _handle_budget_and_logging(
            agent,
            context,
            "tool:share_artifact",
            {
                "artifact_id": artifact_id,
                "target_agent": target_agent
            },
            ArtifactShareResponse,
            artifact_id=artifact_id,
            target_agent=target_agent
        )
        if error_response:
            return error_response
        # Call implementation
        return share_artifact_impl(agent, artifact_id, target_agent)
    
    
    # Tool 5: List Artifacts
    async def list_artifacts() -> ArtifactListResponse:
        """List all artifacts available in the workspace."""
        
        # Handle budget and logging
        error_response = _handle_budget_and_logging(
            agent,
            context,
            "tool:list_artifacts",
            {},
            ArtifactListResponse
        )
        if error_response:
            return error_response
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