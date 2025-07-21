"""
Action logging and tracking system for internal and external agent actions.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class InternalAction:
    """Represents an internal action (tool call or reasoning step)."""
    actor: str
    action: str
    tool: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    cost: float = 0.0
    latency: int = 0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class ExternalAction:
    """Represents an external action that changes market state."""
    actor: str
    action: str
    good: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class ActionLogger:
    """Logs and tracks all agent actions for analysis."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.internal_actions: List[InternalAction] = []
        self.external_actions: List[ExternalAction] = []
        
        # Separate log files
        self.internal_log_file = self.log_dir / "internal_actions.jsonl"
        self.external_log_file = self.log_dir / "external_actions.jsonl"
    
    def log_internal_action(self, action: InternalAction):
        """Log an internal action."""
        self.internal_actions.append(action)
        self._append_to_file(self.internal_log_file, action.to_dict())
    
    def log_external_action(self, action: ExternalAction):
        """Log an external action."""
        self.external_actions.append(action)
        self._append_to_file(self.external_log_file, action.to_dict())
    
    def _append_to_file(self, file_path: Path, data: Dict[str, Any]):
        """Append a single JSON line to log file."""
        with open(file_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def get_agent_actions(self, agent_id: str, action_type: str = "both") -> List[Dict[str, Any]]:
        """Get all actions for a specific agent."""
        actions = []
        
        if action_type in ["internal", "both"]:
            actions.extend([
                action.to_dict() for action in self.internal_actions 
                if action.actor == agent_id
            ])
        
        if action_type in ["external", "both"]:
            actions.extend([
                action.to_dict() for action in self.external_actions 
                if action.actor == agent_id
            ])
        
        return sorted(actions, key=lambda x: x["timestamp"])
    
    def get_actions_by_timeframe(self, start_time: datetime, end_time: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Get all actions within a time frame."""
        internal = [
            action.to_dict() for action in self.internal_actions
            if start_time <= action.timestamp <= end_time
        ]
        
        external = [
            action.to_dict() for action in self.external_actions
            if start_time <= action.timestamp <= end_time
        ]
        
        return {
            "internal": internal,
            "external": external
        }
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate key metrics from logged actions."""
        metrics = {
            "total_internal_actions": len(self.internal_actions),
            "total_external_actions": len(self.external_actions),
            "agents": {},
            "tools": {},
            "costs": {}
        }
        
        # Per-agent metrics
        agents = set()
        for action in self.internal_actions + self.external_actions:
            agents.add(action.actor)
        
        for agent in agents:
            agent_internal = [a for a in self.internal_actions if a.actor == agent]
            agent_external = [a for a in self.external_actions if a.actor == agent]
            
            total_cost = sum(a.cost for a in agent_internal)
            tool_calls = len([a for a in agent_internal if a.tool])
            reflect_calls = len([a for a in agent_internal if a.action == "reflect"])
            
            metrics["agents"][agent] = {
                "internal_actions": len(agent_internal),
                "external_actions": len(agent_external),
                "total_cost": total_cost,
                "tool_calls": tool_calls,
                "reflect_calls": reflect_calls,
                "reflect_ratio": reflect_calls / max(tool_calls, 1)
            }
        
        # Tool usage metrics
        for action in self.internal_actions:
            if action.tool:
                if action.tool not in metrics["tools"]:
                    metrics["tools"][action.tool] = {"count": 0, "total_cost": 0}
                metrics["tools"][action.tool]["count"] += 1
                metrics["tools"][action.tool]["total_cost"] += action.cost
        
        # Cost distribution
        all_costs = [a.cost for a in self.internal_actions if a.cost > 0]
        if all_costs:
            metrics["costs"] = {
                "mean": sum(all_costs) / len(all_costs),
                "min": min(all_costs),
                "max": max(all_costs),
                "total": sum(all_costs)
            }
        
        return metrics
    
    def export_summary(self, output_file: Path):
        """Export a summary of all actions and metrics."""
        summary = {
            "metrics": self.calculate_metrics(),
            "timeline": {
                "internal_actions": [a.to_dict() for a in self.internal_actions],
                "external_actions": [a.to_dict() for a in self.external_actions]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
