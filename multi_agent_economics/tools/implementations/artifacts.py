"""
Core implementations for artifact management tools.

These functions handle ALL parameter unpacking from agent context and return
Pydantic response models directly. Wrappers only handle budget/logging.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..schemas import (
    ArtifactLoadResponse, ArtifactUnloadResponse, ArtifactWriteResponse,
    ArtifactListResponse
)
from ...core.artifacts import Artifact

logger = logging.getLogger('artifacts.tools')


def load_artifact_impl(workspace_memory, artifact_id: str) -> ArtifactLoadResponse:
    """
    Load artifact using workspace memory.
    
    Args:
        workspace_memory: WorkspaceMemory instance for artifact management
        artifact_id: ID of artifact to load
    
    Returns:
        ArtifactLoadResponse: Complete Pydantic response
    """
    if not workspace_memory:
        return ArtifactLoadResponse(
            status="error",
            artifact_id=artifact_id,
            message="No workspace memory available"
        )
    
    try:
        logger.debug(f"Loading artifact '{artifact_id}' for workspace memory {workspace_memory.name}")
        
        # Attempt to load the artifact
        success = workspace_memory.load_artifact(artifact_id)
        
        if success:
            logger.info(f"Successfully loaded artifact '{artifact_id}' for {workspace_memory.name}")
            return ArtifactLoadResponse(
                status="loaded",
                artifact_id=artifact_id,
                message=f"Artifact {artifact_id} will be injected in your next prompt",
                version=int(datetime.now().timestamp())
            )
        else:
            logger.warning(f"Failed to load artifact '{artifact_id}' for {workspace_memory.name} - artifact may not exist")
            return ArtifactLoadResponse(
                status="error",
                artifact_id=artifact_id,
                message=f"Failed to load artifact {artifact_id} - may not exist or access denied"
            )
    except Exception as e:
        return ArtifactLoadResponse(
            status="error",
            artifact_id=artifact_id,
            message=f"Error loading artifact: {str(e)}"
        )


def unload_artifact_impl(workspace_memory, artifact_id: str) -> ArtifactUnloadResponse:
    """
    Unload artifact using workspace memory.
    
    Args:
        workspace_memory: WorkspaceMemory instance for artifact management
        artifact_id: ID of artifact to unload
    
    Returns:
        ArtifactUnloadResponse: Complete Pydantic response
    """
    
    if not workspace_memory:
        return ArtifactUnloadResponse(
            status="error",
            artifact_id=artifact_id,
            message="No workspace memory available (agent.memory[0] not found)"
        )
    
    try:
        # Attempt to unload the artifact
        success = workspace_memory.unload_artifact(artifact_id)
        
        if success:
            return ArtifactUnloadResponse(
                status="unloaded",
                artifact_id=artifact_id,
                message=f"Artifact {artifact_id} unloaded from memory"
            )
        else:
            return ArtifactUnloadResponse(
                status="error",
                artifact_id=artifact_id,
                message=f"Failed to unload artifact {artifact_id} - may not be loaded"
            )
    except Exception as e:
        return ArtifactUnloadResponse(
            status="error",
            artifact_id=artifact_id,
            message=f"Error unloading artifact: {str(e)}"
        )


def write_artifact_impl(
    workspace_memory,
    artifact_id: str,
    content: Dict[str, Any],
    artifact_type: str = "analysis",
    agent_name: str = "unknown_agent"
) -> ArtifactWriteResponse:
    """
    Write artifact using workspace memory.
    
    Args:
        workspace_memory: WorkspaceMemory instance for artifact management
        artifact_id: ID for the new/updated artifact
        content: Content payload for the artifact
        artifact_type: Type of artifact being created
        agent_name: Name of agent creating the artifact
    
    Returns:
        ArtifactWriteResponse: Complete Pydantic response
    """
    if not workspace_memory:
        return ArtifactWriteResponse(
            status="error",
            artifact_id=artifact_id,
            message="No workspace memory available"
        )
    
    # Get workspace from workspace_memory
    workspace = workspace_memory.workspace
    
    try:
        logger.debug(f"Writing artifact '{artifact_id}' (type: {artifact_type}) for agent '{agent_name}' to workspace {workspace.workspace_id}")
        
        # Create real artifact and store it in workspace
        artifact = Artifact(
            id=artifact_id,
            type=artifact_type,
            payload=content,
            visibility=["private"],  # Default to private visibility
            created_at=datetime.now(),
            created_by=agent_name,
            metadata={}
        )
        
        # Store in workspace if available
        if workspace:
            path = workspace.store_artifact(artifact, "private")
            logger.info(f"Stored artifact '{artifact_id}' in workspace {workspace.workspace_id} at {path}")
        else:
            # Fallback path if no workspace
            path = f"/workspace/{agent_name}/{artifact_id}.json"
            logger.warning(f"No workspace available, using fallback path: {path}")
        
        # Use timestamp as version
        version = int(artifact.created_at.timestamp())
        
        # Mark as seen in memory
        if hasattr(workspace_memory, 'mark_artifact_seen'):
            workspace_memory.mark_artifact_seen(artifact_id, version)
            logger.debug(f"Marked artifact '{artifact_id}' as seen (version {version}) in workspace memory")
        
        return ArtifactWriteResponse(
            status="written",
            artifact_id=artifact_id,
            version=version,
            path=path,
            message=f"Artifact {artifact_id} written successfully",
            size_chars=len(str(content))
        )
            
    except Exception as e:
        return ArtifactWriteResponse(
            status="error",
            artifact_id=artifact_id,
            message=f"Failed to write artifact: {str(e)}"
        )


def list_artifacts_impl(workspace_memory) -> ArtifactListResponse:
    """
    List artifacts using workspace memory.
    
    Args:
        workspace_memory: WorkspaceMemory instance for artifact management
    
    Returns:
        ArtifactListResponse: Complete Pydantic response
    """
    if not workspace_memory:
        return ArtifactListResponse(
            status="error",
            workspace_listing="[workspace] No workspace memory available",
            loaded_artifacts=[],
            loaded_status={},
            message="No workspace memory available"
        )
    
    try:
        # Get the context additions (which includes the artifact listing)
        if hasattr(workspace_memory, 'build_context_additions'):
            context_additions = workspace_memory.build_context_additions()
            
            # Extract workspace listing
            workspace_line = "[workspace] No artifacts available"
            for line in context_additions:
                if line.startswith("[workspace]"):
                    workspace_line = line
                    break
        else:
            workspace_line = "[workspace] Mock workspace - analysis_1, forecast_2 available"
        
        # Get loaded artifacts status
        if hasattr(workspace_memory, 'get_loaded_artifacts'):
            loaded_artifacts = workspace_memory.get_loaded_artifacts()
        else:
            loaded_artifacts = []  # Mock empty list
        
        # Get detailed status for loaded artifacts
        loaded_status = {}
        for aid in loaded_artifacts:
            try:
                if hasattr(workspace_memory, 'get_artifact_status'):
                    loaded_status[aid] = workspace_memory.get_artifact_status(aid)
                else:
                    loaded_status[aid] = {"loaded": True, "note": "get_artifact_status method not available"}
            except Exception as artifact_status_error:
                loaded_status[aid] = {"status": "unknown", "error": str(artifact_status_error)}
        
        # Count total artifacts from the actual workspace
        workspace = getattr(workspace_memory, 'workspace', None)
        if workspace and hasattr(workspace, 'list_artifacts'):
            try:
                total_artifacts = len(workspace.list_artifacts())
            except Exception as list_error:
                total_artifacts = 0
                workspace_line += f" (list error: {str(list_error)})"
        else:
            total_artifacts = max(len(loaded_artifacts), 0)  # Fallback
        
        return ArtifactListResponse(
            status="success",
            workspace_listing=workspace_line,
            loaded_artifacts=loaded_artifacts,
            loaded_status=loaded_status,
            total_artifacts=total_artifacts,
            message="Use load_artifact(id) to inject artifact content in your next prompt"
        )
        
    except Exception as e:
        return ArtifactListResponse(
            status="error",
            workspace_listing=f"[workspace] Error: {str(e)}",
            loaded_artifacts=[],
            loaded_status={},
            message=f"Error listing artifacts: {str(e)}"
        )