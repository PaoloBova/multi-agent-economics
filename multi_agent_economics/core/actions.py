"""
Action logging and tracking for multi-agent economics simulation.

This module provides classes for tracking internal and external actions
taken by agents during simulation.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class InternalAction:
    """Represents an internal action (tool usage, artifact management)."""
    actor: str
    action: str
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    tool: Optional[str] = None
    cost: float = 0.0
    latency: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class ExternalAction:
    """Represents an external action (market interactions)."""
    actor: str
    action: str
    target: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ActionLogger:
    """Logs and tracks actions taken by agents."""
    
    def __init__(self):
        self.internal_actions = []
        self.external_actions = []
    
    def log_internal_action(self, action: InternalAction):
        """Log an internal action."""
        self.internal_actions.append(action)
    
    def log_external_action(self, action: ExternalAction):
        """Log an external action."""
        self.external_actions.append(action)
    
    def get_actions_by_actor(self, actor: str):
        """Get all actions by a specific actor."""
        internal = [a for a in self.internal_actions if a.actor == actor]
        external = [a for a in self.external_actions if a.actor == actor]
        return {"internal": internal, "external": external}
    
    def get_all_actions(self):
        """Get all actions (internal and external)."""
        return self.internal_actions + self.external_actions
    
    def clear(self):
        """Clear all logged actions."""
        self.internal_actions.clear()
        self.external_actions.clear()