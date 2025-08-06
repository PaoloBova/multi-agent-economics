"""
Artifact system for storing and sharing structured resources between agents.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Artifact:
    """Structured resource that can be stored and shared between agents."""
    id: str
    type: str
    payload: Dict[str, Any]
    visibility: List[str]
    created_at: datetime
    created_by: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, artifact_type: str, payload: Dict[str, Any], 
               created_by: str, visibility: List[str], 
               metadata: Optional[Dict[str, Any]] = None) -> "Artifact":
        """Create a new artifact with auto-generated ID."""
        return cls(
            id=f"{artifact_type}#{uuid.uuid4().hex[:8]}",
            type=artifact_type,
            payload=payload,
            visibility=visibility,
            created_at=datetime.now(),
            created_by=created_by,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary for JSON serialization."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create artifact from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class Workspace:
    """Manages artifact storage and access for an agent or organization."""
    
    def __init__(self, workspace_id: str, workspace_dir: Path, artifact_manager: Optional["ArtifactManager"] = None):
        self.workspace_id = workspace_id
        self.workspace_dir = Path(workspace_dir)
        self.artifact_manager = artifact_manager  # Reference to parent manager
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bucket directories
        self.private_bucket = self.workspace_dir / "private"
        self.shared_bucket = self.workspace_dir / "shared"
        self.org_bucket = self.workspace_dir / "org"
        
        for bucket in [self.private_bucket, self.shared_bucket, self.org_bucket]:
            bucket.mkdir(exist_ok=True)
    
    def store_artifact(self, artifact: Artifact, bucket: str = "private") -> str:
        """Store an artifact in the specified bucket."""
        bucket_path = getattr(self, f"{bucket}_bucket")
        artifact_file = bucket_path / f"{artifact.id}.json"
        
        with open(artifact_file, 'w') as f:
            json.dump(artifact.to_dict(), f, indent=2)
        
        return str(artifact_file)
    
    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID from any accessible bucket."""
        for bucket in ["private", "shared", "org"]:
            bucket_path = getattr(self, f"{bucket}_bucket")
            artifact_file = bucket_path / f"{artifact_id}.json"
            
            if artifact_file.exists():
                with open(artifact_file, 'r') as f:
                    data = json.load(f)
                return Artifact.from_dict(data)
        
        return None
    
    def list_artifacts(self, bucket: str = "all") -> List[str]:
        """List all artifact IDs in the specified bucket(s)."""
        artifact_ids = []
        
        buckets = ["private", "shared", "org"] if bucket == "all" else [bucket]
        
        for bucket_name in buckets:
            bucket_path = getattr(self, f"{bucket_name}_bucket")
            for artifact_file in bucket_path.glob("*.json"):
                artifact_ids.append(artifact_file.stem)
        
        return artifact_ids
    
    def share_artifact(self, artifact_id: str, target_workspace: "Workspace", 
                      shared_by: str = "unknown", sharing_reason: str = "collaboration") -> bool:
        """
        Share an artifact with another workspace with enhanced tracking.
        
        Args:
            artifact_id: ID of artifact to share
            target_workspace: Target workspace to share with
            shared_by: Agent/user who initiated the sharing
            sharing_reason: Reason for sharing (e.g., "collaboration", "review", "handoff")
        
        Returns:
            bool: True if sharing was successful
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return False
        
        # Add sharing metadata to track the sharing transaction
        sharing_metadata = {
            "shared_from": self.workspace_id,
            "shared_to": target_workspace.workspace_id,
            "shared_by": shared_by,
            "shared_at": datetime.now().isoformat(),
            "sharing_reason": sharing_reason,
            "original_created_by": artifact.created_by
        }
        
        # Create a copy of the artifact with sharing metadata
        existing_sharing_history = (artifact.metadata or {}).get("sharing_history", [])
        new_sharing_history = existing_sharing_history + [sharing_metadata]
        
        shared_artifact = Artifact(
            id=artifact.id,
            type=artifact.type,
            payload=artifact.payload.copy(),  # Deep copy of payload
            visibility=artifact.visibility.copy(),
            created_at=artifact.created_at,
            created_by=artifact.created_by,
            metadata={
                **(artifact.metadata or {}),
                "sharing_history": new_sharing_history
            }
        )
        
        # Store in target's shared bucket
        target_workspace.store_artifact(shared_artifact, bucket="shared")
        return True


class ArtifactManager:
    """Global manager for all workspaces and cross-workspace operations."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces: Dict[str, Workspace] = {}
    
    def create_workspace(self, workspace_id: str) -> Workspace:
        """Create a new workspace."""
        workspace_dir = self.base_dir / workspace_id
        workspace = Workspace(workspace_id, workspace_dir, self)  # Pass self as artifact_manager
        self.workspaces[workspace_id] = workspace
        return workspace
    
    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get an existing workspace."""
        return self.workspaces.get(workspace_id)
    
    def share_artifact(self, artifact_id: str, from_workspace: str, 
                      to_workspace: str) -> bool:
        """Share an artifact between workspaces."""
        source = self.get_workspace(from_workspace)
        target = self.get_workspace(to_workspace)
        
        if not source or not target:
            return False
        
        return source.share_artifact(artifact_id, target)
    
    def check_access(self, agent_id: str, artifact_id: str) -> bool:
        """Check if an agent has access to a specific artifact."""
        # Extract workspace from agent_id (e.g., "Seller.Trader" -> "Seller")
        workspace_id = agent_id.split('.')[0]
        workspace = self.get_workspace(workspace_id)
        
        if not workspace:
            return False
        
        artifact = workspace.get_artifact(artifact_id)
        if not artifact:
            return False
        
        # Check visibility permissions
        return agent_id in artifact.visibility or f"{workspace_id}.*" in artifact.visibility
