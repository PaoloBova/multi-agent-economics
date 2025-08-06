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
    ArtifactManager, ActionLogger,
    BudgetManager, 
    Artifact, InternalAction, ExternalAction,
    WorkspaceMemory
)

# Agent framework (use AutoGen AssistantAgent directly with tools)
# from .agents import EconomicAgent, create_agent

# Scenarios - TODO: Update for new tool system
# from .scenarios import StructuredNoteLemonsScenario, run_flagship_scenario

__all__ = [
    # Core infrastructure
    "ArtifactManager", "ActionLogger",
    "BudgetManager",
    "Artifact", "InternalAction", "ExternalAction",
    "WorkspaceMemory",
    
    # Agent framework (use AutoGen AssistantAgent directly with tools)
    # "EconomicAgent", "create_agent",
    
    # Scenarios - TODO: Update for new tool system
    # "StructuredNoteLemonsScenario", "run_flagship_scenario"
]
