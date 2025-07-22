"""
Artifact management tools for AutoGen agents.

These tools work with WorkspaceMemory through the agent context system,
following the original vision where tools receive context with caller information.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from ..core.artifacts import Artifact
from ..core.workspace_memory import WorkspaceMemory


def load_artifact(artifact_id: str, _context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load an artifact into memory for injection in next prompt.
    
    Args:
        artifact_id: The ID of the artifact to load
        _context: Agent context containing caller information
        
    Returns:
        Dict with status information
    """
    if not _context or "caller" not in _context:
        return {"status": "error", "message": "No agent context provided"}
    
    agent = _context["caller"]
    
    # Get workspace memory from agent
    workspace_memory = getattr(agent, 'workspace_memory', None)
    if not workspace_memory:
        return {"status": "error", "message": "Agent has no workspace memory"}
    
    # Get budget manager from agent context
    budget_manager = getattr(agent, 'budget_manager', None)
    if budget_manager:
        try:
            org_name = workspace_memory.name.split('_')[0] 
            budget_manager.charge_credits(org_name, 0.2)
        except Exception as e:
            return {"status": "error", "message": f"Insufficient credits: {e}"}
    
    # Load the artifact in memory
    success = workspace_memory.load_artifact(artifact_id)
    
    if success:
        # Log the action if agent has action logger
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action_logger.log_internal_action(
                actor=workspace_memory.name,
                action="load_artifact",
                details={"artifact_id": artifact_id}
            )
        
        return {
            "status": "loaded",
            "artifact_id": artifact_id,
            "message": f"Artifact {artifact_id} will be injected in your next prompt"
        }
    else:
        return {
            "status": "error", 
            "message": f"Failed to load artifact {artifact_id}"
        }


def unload_artifact(artifact_id: str, _context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Unload an artifact from memory (stop injecting in prompts).
    
    Args:
        artifact_id: The ID of the artifact to unload
        _context: Agent context containing caller information
        
    Returns:
        Dict with status information
    """
    if not _context or "caller" not in _context:
        return {"status": "error", "message": "No agent context provided"}
    
    agent = _context["caller"]
    workspace_memory = getattr(agent, 'workspace_memory', None)
    if not workspace_memory:
        return {"status": "error", "message": "Agent has no workspace memory"}
    
    # No cost for unloading
    success = workspace_memory.unload_artifact(artifact_id)
    
    if success:
        # Log the action if agent has action logger
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action_logger.log_internal_action(
                actor=workspace_memory.name,
                action="unload_artifact", 
                details={"artifact_id": artifact_id}
            )
        
        return {
            "status": "unloaded",
            "artifact_id": artifact_id,
            "message": f"Artifact {artifact_id} unloaded from memory"
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to unload artifact {artifact_id}"
        }


def write_artifact(artifact_id: str, content: Dict[str, Any], 
                   artifact_type: str = "analysis", 
                   visibility: Optional[str] = None,
                   _context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Write/update an artifact in the workspace.
    
    Args:
        artifact_id: The ID of the artifact to write
        content: The content/payload of the artifact
        artifact_type: Type of artifact (analysis, forecast, etc.)
        visibility: Who can see this artifact (defaults to organization)
        _context: Agent context containing caller information
        
    Returns:
        Dict with status information
    """
    if not _context or "caller" not in _context:
        return {"status": "error", "message": "No agent context provided"}
    
    agent = _context["caller"]
    workspace_memory = getattr(agent, 'workspace_memory', None)
    if not workspace_memory:
        return {"status": "error", "message": "Agent has no workspace memory"}
    
    # Charge cost for writing
    budget_manager = getattr(agent, 'budget_manager', None)
    if budget_manager:
        try:
            org_name = workspace_memory.name.split('_')[0]
            budget_manager.charge_credits(org_name, 0.5)
        except Exception as e:
            return {"status": "error", "message": f"Insufficient credits: {e}"}
    
    # Determine visibility
    if visibility is None:
        org_name = workspace_memory.name.split('_')[0]
        visibility_list = [f"{org_name}.*"]  # Organization-wide visibility
    else:
        visibility_list = [visibility]
    
    # Create or update artifact
    artifact = Artifact.create(
        artifact_type=artifact_type,
        payload=content,
        created_by=workspace_memory.name,
        visibility=visibility_list
    )
    
    # Store in workspace
    try:
        artifact_path = workspace_memory.workspace.store_artifact(artifact, bucket="org")
        
        # Mark as seen in memory (use timestamp as version)
        version = int(artifact.created_at.timestamp())
        workspace_memory.mark_artifact_seen(artifact_id, version)
        
        # Log the action
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action_logger.log_internal_action(
                actor=workspace_memory.name,
                action="write_artifact",
                details={
                    "artifact_id": artifact_id,
                    "type": artifact_type,
                    "size_chars": len(str(content))
                }
            )
        
        return {
            "status": "written",
            "artifact_id": artifact_id,
            "version": version,
            "path": artifact_path,
            "message": f"Artifact {artifact_id} written successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to write artifact: {e}"
        }


def share_artifact(artifact_id: str, target_organization: str, 
                   _context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Share an artifact with another organization.
    
    Args:
        artifact_id: The ID of the artifact to share
        target_organization: The organization to share with
        _context: Agent context containing caller information
        
    Returns:
        Dict with status information
    """
    if not _context or "caller" not in _context:
        return {"status": "error", "message": "No agent context provided"}
    
    agent = _context["caller"]
    workspace_memory = getattr(agent, 'workspace_memory', None)
    if not workspace_memory:
        return {"status": "error", "message": "Agent has no workspace memory"}
    
    # Charge cost for sharing
    budget_manager = getattr(agent, 'budget_manager', None)
    if budget_manager:
        try:
            org_name = workspace_memory.name.split('_')[0]
            budget_manager.charge_credits(org_name, 0.3)
        except Exception as e:
            return {"status": "error", "message": f"Insufficient credits: {e}"}
    
    # Get the artifact
    artifact = workspace_memory.workspace.get_artifact(artifact_id)
    if not artifact:
        return {
            "status": "error",
            "message": f"Artifact {artifact_id} not found"
        }
    
    # Update visibility to include target organization
    if f"{target_organization}.*" not in artifact.visibility:
        artifact.visibility.append(f"{target_organization}.*")
        
        # Re-store the artifact with updated visibility
        workspace_memory.workspace.store_artifact(artifact, bucket="shared")
        
        # Log the action
        action_logger = getattr(agent, 'action_logger', None)
        if action_logger:
            action_logger.log_external_action(
                actor=workspace_memory.name,
                action="share_artifact",
                target=target_organization,
                details={"artifact_id": artifact_id}
            )
        
        return {
            "status": "shared",
            "artifact_id": artifact_id,
            "target": target_organization,
            "message": f"Artifact {artifact_id} shared with {target_organization}"
        }
    else:
        return {
            "status": "already_shared",
            "message": f"Artifact {artifact_id} already shared with {target_organization}"
        }


def list_artifacts(_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    List all artifacts available in the workspace.
    
    Args:
        _context: Agent context containing caller information
        
    Returns:
        Dict with artifact information
    """
    if not _context or "caller" not in _context:
        return {"status": "error", "message": "No agent context provided"}
    
    agent = _context["caller"]
    workspace_memory = getattr(agent, 'workspace_memory', None)
    if not workspace_memory:
        return {"status": "error", "message": "Agent has no workspace memory"}
    
    # Get the context additions (which includes the artifact listing)
    context_additions = workspace_memory.build_context_additions()
    
    # Extract just the workspace listing
    workspace_line = None
    for line in context_additions:
        if line.startswith("[workspace]"):
            workspace_line = line
            break
    
    # Get detailed status for loaded artifacts
    loaded_artifacts = workspace_memory.get_loaded_artifacts()
    loaded_status = {
        aid: workspace_memory.get_artifact_status(aid) 
        for aid in loaded_artifacts
    }
    
    return {
        "status": "success",
        "workspace_listing": workspace_line or "[workspace] No artifacts available",
        "loaded_artifacts": loaded_artifacts,
        "loaded_status": loaded_status,
        "message": "Use load_artifact(id) to inject artifact content in your next prompt"
    }


# Export tool functions as a list for easy registration
ARTIFACT_TOOLS = [
    load_artifact,
    unload_artifact,
    write_artifact,
    share_artifact,
    list_artifacts
]
