"""
Core infrastructure for multi-agent economics simulation.

This module provides the fundamental building blocks for the simulation:
- Artifacts and workspace mechanics  
- Action logging and tracking
- Credit and budget management
- Workspace memory management
"""

from .artifacts import Artifact, ArtifactManager, Workspace
from .actions import InternalAction, ExternalAction, ActionLogger
from .budget import BudgetManager, CreditTracker
from .workspace_memory import WorkspaceMemory

__all__ = [
    "Artifact", "ArtifactManager", "Workspace",
    "InternalAction", "ExternalAction", "ActionLogger",
    "BudgetManager", "CreditTracker",
    "WorkspaceMemory"
]
