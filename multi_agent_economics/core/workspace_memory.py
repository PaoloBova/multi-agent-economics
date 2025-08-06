"""
WorkspaceMemory implementation for AutoGen agents with artifact management.

This module provides a memory system that integrates with our artifact workspace,
allowing agents to load/unload artifacts on-demand without polluting chat history.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from .artifacts import Workspace, Artifact


class WorkspaceMemory:
    """
    Memory that manages artifact awareness and on-demand payload injection.
    
    Keeps per-artifact state for ONE agent:
    - last_seen_ver: the newest version the agent is aware of
    - loaded: bool, inject payload into next prompt if True
    """
    
    def __init__(self, name: str, workspace: Workspace, 
                 max_chars: int = 4000, payload_ttl_minutes: int = 60):
        """
        Initialize workspace memory for an agent.
        
        Args:
            name: Memory identifier (typically agent name)
            workspace: The workspace this agent has access to
            max_chars: Maximum characters to inject per artifact payload
            payload_ttl_minutes: How long to cache payloads before refreshing
        """
        self.name = name
        self.workspace = workspace
        self.max_chars = max_chars
        self.ttl = timedelta(minutes=payload_ttl_minutes)
        
        # Per-artifact metadata: {artifact_id: {"last_seen": int, "loaded": bool}}
        self.meta: Dict[str, Dict[str, Any]] = {}
        
        # Payload cache: {artifact_id: (payload_text, timestamp)}
        self.payload_cache: Dict[str, Tuple[str, datetime]] = {}
    
    def clear(self) -> None:
        """Clear all memory state."""
        self.meta.clear()
        self.payload_cache.clear()
    
    def build_context_additions(self) -> List[str]:
        """
        Build context additions for agent prompts.
        
        Returns:
            List of strings to inject into the agent's system prompt
        """
        # Get current workspace state
        live_artifacts = self._get_workspace_listing()
        
        # Build artifact listing and collect payloads to inject
        now = datetime.now()
        listing = []
        injected_payloads = []
        
        for artifact_id, current_version in live_artifacts.items():
            # Get or create metadata for this artifact
            artifact_meta = self.meta.setdefault(artifact_id, {
                "last_seen": -1,
                "loaded": False
            })
            
            # Check if artifact is new BEFORE potentially loading it
            is_new = current_version > artifact_meta["last_seen"]
            
            # Check if artifact is loaded and needs injection
            if artifact_meta["loaded"]:
                payload = self._get_cached_payload(artifact_id, current_version, now)
                if payload:
                    injected_payloads.append(
                        f"[artifact:{artifact_id}@v{current_version}]\n{payload}"
                    )
            
            # Mark unseen artifacts with *
            tag = f"{artifact_id}{'*' if is_new else ''}"
            listing.append(tag)
        
        # Build context additions
        context_additions = []
        
        # Add workspace listing
        if listing:
            context_additions.append(f"[workspace] Available artifacts: {', '.join(listing)}")
        
        # Add loaded artifact payloads
        context_additions.extend(injected_payloads)
        
        return context_additions
    
    def _get_workspace_listing(self) -> Dict[str, int]:
        """Get current artifacts and their versions from workspace."""
        artifact_listing = {}
        
        # List all artifacts from all accessible buckets
        artifact_ids = self.workspace.list_artifacts(bucket="all")
        
        for artifact_id in artifact_ids:
            artifact = self.workspace.get_artifact(artifact_id)
            if artifact:
                # Use created_at timestamp as version (convert to int)
                version = int(artifact.created_at.timestamp())
                artifact_listing[artifact_id] = version
        
        return artifact_listing
    
    def _get_cached_payload(self, artifact_id: str, current_version: int, now: datetime) -> Optional[str]:
        """Get cached payload, refreshing if needed."""
        # Check if we have a cached payload
        cached_payload, cached_time = self.payload_cache.get(artifact_id, ("", datetime.min))
        
        # Get last seen version for this artifact
        last_seen = self.meta[artifact_id]["last_seen"]
        
        # Refresh if version changed or cache expired
        if current_version != last_seen or now - cached_time > self.ttl:
            artifact = self.workspace.get_artifact(artifact_id)
            if artifact:
                # Convert payload to string (truncate if needed)
                payload_str = str(artifact.payload)
                if len(payload_str) > self.max_chars:
                    payload_str = payload_str[:self.max_chars] + "... [TRUNCATED]"
                
                # Update cache and mark as seen
                self.payload_cache[artifact_id] = (payload_str, now)
                self.meta[artifact_id]["last_seen"] = current_version
                
                return payload_str
        
        return cached_payload if cached_payload else None
    
    # Public methods for tools to manipulate memory state
    
    def load_artifact(self, artifact_id: str) -> bool:
        """Mark an artifact for loading in next prompt."""
        # First check if the artifact actually exists in the workspace
        if self.workspace.get_artifact(artifact_id) is None:
            return False  # Artifact doesn't exist
        
        artifact_meta = self.meta.setdefault(artifact_id, {
            "last_seen": -1,
            "loaded": False
        })
        artifact_meta["loaded"] = True
        return True
    
    def unload_artifact(self, artifact_id: str) -> bool:
        """Stop loading an artifact in prompts."""
        if artifact_id in self.meta:
            self.meta[artifact_id]["loaded"] = False
        
        # Clear from payload cache
        self.payload_cache.pop(artifact_id, None)
        return True
    
    def mark_artifact_seen(self, artifact_id: str, version: int) -> None:
        """Mark an artifact as seen (e.g., after writing)."""
        artifact_meta = self.meta.setdefault(artifact_id, {
            "last_seen": -1,
            "loaded": False
        })
        artifact_meta["last_seen"] = version
    
    def get_loaded_artifacts(self) -> List[str]:
        """Get list of currently loaded artifact IDs."""
        return [aid for aid, meta in self.meta.items() if meta.get("loaded", False)]
    
    def get_artifact_status(self, artifact_id: str) -> Dict[str, Any]:
        """Get status information for an artifact."""
        if artifact_id not in self.meta:
            return {"exists": False}
        
        meta = self.meta[artifact_id]
        return {
            "exists": True,
            "loaded": meta.get("loaded", False),
            "last_seen": meta.get("last_seen", -1),
            "cached": artifact_id in self.payload_cache
        }
