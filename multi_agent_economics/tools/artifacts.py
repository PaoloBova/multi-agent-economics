"""
Artifact tool factory - simplified wrappers that only handle budget and logging.

Wrappers call implementations directly and return their results unchanged.
"""

from typing import Dict, List, Any, Optional
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool
from pydantic import BaseModel

from .implementations.artifacts import (
    load_artifact_impl, unload_artifact_impl, write_artifact_impl,
    list_artifacts_impl
)
from .schemas import (
    ArtifactLoadResponse, ArtifactUnloadResponse, ArtifactWriteResponse,
    ArtifactListResponse
)
from ..core.actions import InternalAction


def create_artifact_tools(context) -> List[FunctionTool]:
    """Create artifact tools with simple wrappers."""
    
    DEFAULT_COSTS = {
        "tool:load_artifact": 0.2,
        "tool:write_artifact": 0.5,
        "tool:unload_artifact": 0.0,
        "tool:list_artifacts": 0.0
    }

    def _handle_budget_and_logging(agent_name, context, tool_name, inputs, 
    error_response_class, **error_kwargs):
        """Handle budget charging and action logging with consistent error 
    handling."""

        # Budget management
        budget_manager = context.get('budget_manager', None)
        budget_cost = context.get('budget_costs', {}).get(tool_name, DEFAULT_COSTS.get(tool_name, 0.0))

        if budget_manager and budget_cost > 0:
            budget_category = f"{agent_name}_artifacts"
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
                actor=agent_name,
                action=tool_name.replace("tool:", ""),
                inputs=inputs,
                tool=tool_name.replace("tool:", ""),
                cost=budget_cost
            )
            action_logger.log_internal_action(action)

        return None  # No error

    # Tool 1: Load Artifact
    async def load_artifact(
        artifact_id: Annotated[str, "Unique identifier for the artifact to load into memory. Example: 'market_analysis_v1'"]
    ) -> ArtifactLoadResponse:
        """Load an artifact into memory for injection in next prompt."""
        # Handle budget and logging
        agent_name = context.get('agent_name', 'unknown')
        error_response = _handle_budget_and_logging(
            agent_name,
            context,
            "tool:load_artifact",
            {"artifact_id": artifact_id},
            ArtifactLoadResponse,
            artifact_id=artifact_id
        )
        if error_response:
            return error_response

        # Call implementation - it handles ALL the work
        workspace_memory = context.get('workspace_memory')
        return load_artifact_impl(workspace_memory, artifact_id)
    
    
    # Tool 2: Unload Artifact
    async def unload_artifact(
        artifact_id: Annotated[str, "Unique identifier for the artifact to unload from memory. Example: 'market_analysis_v1'"]
    ) -> ArtifactUnloadResponse:
        """Unload an artifact from memory."""
        # Handle budget and logging
        agent_name = context.get('agent_name', 'unknown')
        error_response = _handle_budget_and_logging(
            agent_name,
            context,
            "tool:unload_artifact",
            {"artifact_id": artifact_id},
            ArtifactUnloadResponse,
            artifact_id=artifact_id
        )
        if error_response:
            return error_response

        # Call implementation
        workspace_memory = context.get('workspace_memory')
        return unload_artifact_impl(workspace_memory, artifact_id)
    
    
    # Tool 3: Write Artifact  
    async def write_artifact(
        artifact_id: Annotated[str, "Unique identifier for the new or updated artifact. Use descriptive names like 'poem_sunset', 'market_analysis', 'financial_report'"],
        content: Annotated[Dict[str, Any], """REQUIRED: The actual content to write to the artifact as a dictionary. 
        
        Examples:
        - For poems: {'title': 'Sunset Dreams', 'text': 'Golden rays dance...', 'theme': 'nature', 'style': 'free verse'}
        - For analysis: {'title': 'Market Analysis', 'summary': 'Key findings...', 'data': [1,2,3], 'conclusion': 'The market shows...'}
        - For reports: {'title': 'Q1 Report', 'sections': {'intro': '...', 'findings': '...'}, 'metrics': {'revenue': 100000}}
        
        Always provide a structured dictionary with your content, never just a string."""],
        artifact_type: Annotated[str, "Type/category of the artifact. Examples: 'poem', 'analysis', 'report', 'data', 'summary'"] = "analysis"
    ) -> ArtifactWriteResponse:
        """Write or update an artifact in the workspace."""
        # Handle budget and logging
        agent_name = context.get('agent_name', 'unknown')
        error_response = _handle_budget_and_logging(
            agent_name,
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
        workspace_memory = context.get('workspace_memory')
        agent_name = context.get('agent_name', 'unknown')
        return write_artifact_impl(workspace_memory, artifact_id, content, artifact_type, agent_name)
    
    
    # Tool 4: List Artifacts
    async def list_artifacts() -> ArtifactListResponse:
        """List all artifacts available in the workspace."""
        
        # Handle budget and logging
        agent_name = context.get('agent_name', 'unknown')
        error_response = _handle_budget_and_logging(
            agent_name,
            context,
            "tool:list_artifacts",
            {},
            ArtifactListResponse
        )
        if error_response:
            return error_response
        # Call implementation
        workspace_memory = context.get('workspace_memory')
        return list_artifacts_impl(workspace_memory)
    

    # Return tools
    return [
        FunctionTool(load_artifact, description="Load artifact into memory"),
        FunctionTool(unload_artifact, description="Unload artifact from memory"),
        FunctionTool(write_artifact, description="Write artifact to workspace"),
        FunctionTool(list_artifacts, description="List available artifacts")
    ]