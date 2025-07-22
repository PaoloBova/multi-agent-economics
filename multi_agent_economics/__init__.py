"""
Multi-Agent Economics Simulation Package

A sophisticated framework for simulating economic interactions using LLM-powered agents.
Built on Microsoft AutoGen with advanced tool-based interaction patterns.
"""

__version__ = "0.1.0"
__author__ = "Paolo Bova"
__email__ = "paolobova@proton.me"

# Core infrastructure
from .core import (
    ArtifactManager, ToolRegistry, ActionLogger,
    BudgetManager, QualityTracker, QualityFunction,
    Artifact, Tool, InternalAction, ExternalAction,
    WorkspaceMemory,
    load_artifact, unload_artifact, write_artifact, 
    share_artifact, list_artifacts
)

# Agent framework
from .agents import EconomicAgent, create_agent

# Scenarios
from .scenarios import StructuredNoteLemonsScenario, run_flagship_scenario

__all__ = [
    # Core infrastructure
    "ArtifactManager", "ToolRegistry", "ActionLogger",
    "BudgetManager", "QualityTracker", "QualityFunction",
    "Artifact", "Tool", "InternalAction", "ExternalAction",
    "WorkspaceMemory",
    "load_artifact", "unload_artifact", "write_artifact", 
    "share_artifact", "list_artifacts",
    
    # Agent framework
    "EconomicAgent", "create_agent",
    
    # Scenarios
    "StructuredNoteLemonsScenario", "run_flagship_scenario"
]
